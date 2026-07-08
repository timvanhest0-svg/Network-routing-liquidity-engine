"""
Streamlit simulation engine for Chapter 8 liquidity-risk simulations
====================================================================

This app consolidates the logic from the Chapter 8 simulation scripts:
- shifted-lognormal fallback parameters for topology-exponent paths;
- direct and indirect network-liquidity routing-capacity simulation;
- estimated-EWI emulator with target recall and precision;
- policy support as additional routing capacity;
- support-sensitivity, network-size efficiency, and EWI-quality tables.

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------------------------------------------------------
# Defaults calibrated to the Chapter 8 scripts
# -----------------------------------------------------------------------------

FALLBACK_LOGNORM_PARAMS = (0.3190514240176254, 0.0, 1.0899155671153902)
INVESTMENT_DEFAULT = 100.0
BUFFER_NORMAL_DEFAULT = 40.0


@dataclass(frozen=True)
class SimulationConfig:
    n_nodes: int = 24
    scenarios: int = 1000
    trading_days: int = 200
    investment: float = INVESTMENT_DEFAULT
    buffer_normal_pct: float = BUFFER_NORMAL_DEFAULT
    liquidity_risk_q: float = 0.025
    seed: int = 42
    target_recall: float = 0.70
    target_precision: float = 0.25
    ewi_lead_window: int = 5
    support_days: int = 10
    support_pct: float = 10.0
    lognorm_sigma: float = FALLBACK_LOGNORM_PARAMS[0]
    lognorm_loc: float = FALLBACK_LOGNORM_PARAMS[1]
    lognorm_scale: float = FALLBACK_LOGNORM_PARAMS[2]


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def make_liquidity_multiplier_grids(n_nodes: int, grid_size: int = 5000):
    """
    Create interpolation grids for direct and indirect liquidity multipliers.

    Direct multiplier:
        E[k]

    Indirect multiplier:
        E[k^2] / E[k] - 1
    """
    degree = np.arange(1, n_nodes + 1, dtype=float)
    gamma_grid = np.linspace(0.01, 20.0, grid_size)

    direct_grid = np.empty_like(gamma_grid)
    indirect_grid = np.empty_like(gamma_grid)

    for i, gamma in enumerate(gamma_grid):
        pk = degree ** (-gamma)
        pk /= pk.sum()

        expected_k = float((pk * degree).sum())
        expected_k2 = float((pk * degree * degree).sum())

        direct_grid[i] = expected_k
        indirect_grid[i] = (
            expected_k2 / expected_k - 1.0
            if expected_k > 0
            else np.nan
        )

    return gamma_grid, direct_grid, indirect_grid


@st.cache_data(show_spinner=False)
def simulate_base_paths(
    n_nodes: int,
    scenarios: int,
    trading_days: int,
    investment: float,
    buffer_normal_pct: float,
    liquidity_risk_q: float,
    seed: int,
    lognorm_sigma: float,
    lognorm_loc: float,
    lognorm_scale: float,
):
    """
    Simulate shifted-lognormal topology-exponent paths and direct/indirect
    network-liquidity routing-capacity paths.
    """
    rng = np.random.default_rng(seed)

    gamma_grid, direct_lm_grid, indirect_lm_grid = make_liquidity_multiplier_grids(
        n_nodes
    )

    gamma_paths = lognorm_loc + rng.lognormal(
        mean=np.log(lognorm_scale),
        sigma=lognorm_sigma,
        size=(scenarios, trading_days),
    )

    direct_lm = np.empty_like(gamma_paths)
    indirect_lm = np.empty_like(gamma_paths)

    for s in range(scenarios):
        direct_lm[s] = np.interp(gamma_paths[s], gamma_grid, direct_lm_grid)
        indirect_lm[s] = np.interp(gamma_paths[s], gamma_grid, indirect_lm_grid)

    baseline_available = investment * (1.0 - buffer_normal_pct / 100.0)

    direct_liquidity = baseline_available * direct_lm
    indirect_liquidity = baseline_available * indirect_lm

    risk_threshold = float(np.nanquantile(direct_liquidity, liquidity_risk_q))
    event_day = direct_liquidity < risk_threshold

    return {
        "gamma_paths": gamma_paths,
        "direct_lm": direct_lm,
        "indirect_lm": indirect_lm,
        "direct_liquidity": direct_liquidity,
        "indirect_liquidity": indirect_liquidity,
        "risk_threshold": risk_threshold,
        "event_day": event_day,
    }


def future_event_mask(event_day: np.ndarray, lead_window: int) -> np.ndarray:
    """
    Mark days that are followed by a liquidity-risk day within the lead window.

    This is used to build a cleaner false-positive pool. A false-positive
    candidate should not be followed by a risk day within the lead window.
    """
    out = np.zeros_like(event_day, dtype=bool)

    for h in range(1, lead_window + 1):
        out[:, :-h] |= event_day[:, h:]

    return out


def build_nested_ewi_design(
    event_day: np.ndarray,
    lead_window: int = 5,
    seed: int = 42,
):
    """
    Create EWI scenario-design positions.

    The design contains:
    - potential true-signal positions before evaluable event days;
    - false-positive positions not followed by an event within the lead window.
    """
    rng = np.random.default_rng(seed)

    evaluable = event_day.copy()
    evaluable[:, :lead_window] = False

    event_positions = np.argwhere(evaluable)
    if len(event_positions) > 0:
        event_positions = event_positions[rng.permutation(len(event_positions))]

    future_event = future_event_mask(event_day, lead_window)

    false_positive_candidates = np.argwhere(~event_day & ~future_event)
    if len(false_positive_candidates) > 0:
        false_positive_candidates = false_positive_candidates[
            rng.permutation(len(false_positive_candidates))
        ]

    true_signal_positions = []

    for s, t in event_positions:
        possible_days = np.arange(max(0, t - lead_window), t)

        if len(possible_days) > 0:
            signal_t = int(rng.choice(possible_days))
            true_signal_positions.append((int(s), signal_t))

    # Deduplicate true-signal positions while preserving order.
    # This prevents several nearby events from being assigned to the same
    # boolean EWI signal day and counted multiple times.
    seen = set()
    unique_true_signal_positions = []

    for pos in true_signal_positions:
        if pos not in seen:
            unique_true_signal_positions.append(pos)
            seen.add(pos)

    return {
        "n_evaluable_events": int(len(event_positions)),
        "true_signal_positions": unique_true_signal_positions,
        "false_positive_positions": [
            (int(s), int(t)) for s, t in false_positive_candidates
        ],
    }


def create_nested_ewi(
    event_day: np.ndarray,
    design: dict,
    target_recall: float,
    target_precision: float,
    lead_window: int = 5,
):
    """
    Create an estimated-EWI signal process matching target recall and precision
    as closely as possible.
    """
    ewi = np.zeros_like(event_day, dtype=bool)

    n_events = design["n_evaluable_events"]
    true_positions = design["true_signal_positions"]
    fp_positions = design["false_positive_positions"]

    n_true = int(round(target_recall * n_events)) if n_events else 0
    n_true = min(n_true, len(true_positions))

    selected_true = true_positions[:n_true]

    for s, t in selected_true:
        ewi[s, t] = True

    if target_precision > 0 and n_true > 0:
        target_total_signals = int(round(n_true / target_precision))
        n_fp = max(0, target_total_signals - n_true)
    else:
        n_fp = 0

    n_fp = min(n_fp, len(fp_positions))

    for s, t in fp_positions[:n_fp]:
        ewi[s, t] = True

    # A signal is a true positive if it is followed by a liquidity-risk day
    # within the lead window.
    tp_signal = np.zeros_like(event_day, dtype=bool)

    for s, t in np.argwhere(ewi):
        future = event_day[s, t + 1 : t + 1 + lead_window]
        tp_signal[s, t] = bool(future.any())

    signal_days = int(ewi.sum())
    true_positive_signal_days = int(tp_signal.sum())
    false_positive_signal_days = signal_days - true_positive_signal_days

    diagnostics = {
        "target_recall": target_recall * 100,
        "target_precision": target_precision * 100,
        "lead_window": lead_window,
        "evaluable_event_days": n_events,
        "detected_event_days": n_true,
        "achieved_recall": (n_true / n_events * 100) if n_events else np.nan,
        "signal_days": signal_days,
        "true_positive_signal_days": true_positive_signal_days,
        "false_positive_signal_days": false_positive_signal_days,
        "achieved_precision": (
            true_positive_signal_days / signal_days * 100
            if signal_days
            else np.nan
        ),
        "signal_day_rate": signal_days / event_day.size * 100,
        "false_positive_rate": false_positive_signal_days / event_day.size * 100,
    }

    return ewi, diagnostics


def forward_active_mask(flags: np.ndarray, duration: int):
    """Activate policy support for a fixed duration after an EWI signal."""
    if duration <= 0:
        return np.zeros_like(flags, dtype=bool)

    out = np.zeros_like(flags, dtype=bool)
    kernel = np.ones(duration, dtype=int)

    for s in range(flags.shape[0]):
        conv = np.convolve(flags[s].astype(int), kernel, mode="full")[
            : flags.shape[1]
        ]
        out[s] = conv > 0

    return out


def evaluate_liquidity(arr: np.ndarray, risk_threshold: float):
    """Evaluate liquidity-risk days, scenarios, and cumulative shortfall."""
    risk = arr < risk_threshold
    shortfall = np.maximum(0.0, risk_threshold - arr)

    return {
        "risk_days": int(risk.sum()),
        "risk_day_rate": risk.sum() / arr.size * 100,
        "risk_scenarios": int(risk.any(axis=1).sum()),
        "risk_scenario_rate": risk.any(axis=1).sum() / arr.shape[0] * 100,
        "total_shortfall": float(shortfall.sum()),
    }


def run_support_policy(
    direct_lm: np.ndarray,
    base_metrics: dict,
    risk_threshold: float,
    active_mask: np.ndarray,
    investment: float,
    buffer_normal_pct: float,
    support_pct: float,
):
    """Apply additional routing-capacity support and calculate policy metrics."""
    baseline_available = investment * (1.0 - buffer_normal_pct / 100.0)

    support = investment * (support_pct / 100.0) * active_mask

    support_intensity = (
        support.sum() / (baseline_available * direct_lm.size) * 100
        if baseline_available > 0
        else 0.0
    )

    policy_liquidity = (baseline_available + support) * direct_lm
    policy_metrics = evaluate_liquidity(policy_liquidity, risk_threshold)

    risk_reduction = (
        (base_metrics["risk_days"] - policy_metrics["risk_days"])
        / base_metrics["risk_days"]
        * 100
        if base_metrics["risk_days"]
        else np.nan
    )

    shortfall_reduction = (
        (base_metrics["total_shortfall"] - policy_metrics["total_shortfall"])
        / base_metrics["total_shortfall"]
        * 100
        if base_metrics["total_shortfall"]
        else np.nan
    )

    efficiency = (
        shortfall_reduction / support_intensity
        if support_intensity > 0
        else np.nan
    )

    extra = {
        "support_intensity_pct": support_intensity,
        "risk_reduction_pct": risk_reduction,
        "shortfall_reduction_pct": shortfall_reduction,
        "mitigation_efficiency": efficiency,
    }

    return policy_liquidity, policy_metrics, extra


def to_excel_download(sheets: dict) -> bytes:
    """Convert a dictionary of DataFrames to an Excel workbook in memory."""
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            safe_name = sheet_name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)

    return output.getvalue()


# -----------------------------------------------------------------------------
# Streamlit app
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Liquidity-risk routing capacity simulation engine",
    layout="wide",
)

st.title("Liquidity-risk routing capacity simulation engine")

st.caption(
    "Interactive version of the Chapter 8 simulation: shifted-lognormal topology paths, "
    "network-liquidity routing capacity, estimated-EWI signals, and routing-capacity "
    "policy support."
)


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

with st.sidebar:
    st.header("1. Simulation settings")

    n_nodes = st.slider(
        "Network size",
        min_value=10,
        max_value=150,
        value=24,
        step=1,
    )

    scenarios = st.number_input(
        "Simulation scenarios",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
    )

    trading_days = st.number_input(
        "Trading days",
        min_value=50,
        max_value=1000,
        value=200,
        step=10,
    )

    seed = st.number_input(
        "Random seed",
        min_value=1,
        max_value=999999,
        value=42,
        step=1,
    )

    investment = st.number_input(
        "Investment base",
        min_value=1.0,
        max_value=10000.0,
        value=INVESTMENT_DEFAULT,
        step=10.0,
    )

    buffer_normal_pct = st.slider(
        "Normal buffer / unavailable liquidity (%)",
        min_value=0.0,
        max_value=90.0,
        value=BUFFER_NORMAL_DEFAULT,
        step=1.0,
    )

    liquidity_risk_q = st.slider(
        "Liquidity-risk threshold quantile",
        min_value=0.001,
        max_value=0.100,
        value=0.025,
        step=0.001,
        format="%.3f",
    )

    st.header("2. EWI and policy settings")

    target_recall = st.slider(
        "Target EWI recall",
        min_value=0.0,
        max_value=1.0,
        value=0.70,
        step=0.05,
    )

    target_precision = st.slider(
        "Target EWI precision",
        min_value=0.05,
        max_value=1.0,
        value=0.25,
        step=0.05,
    )

    ewi_lead_window = st.slider(
        "EWI lead window, trading days",
        min_value=1,
        max_value=30,
        value=5,
        step=1,
    )

    support_days = st.slider(
        "Support duration after EWI",
        min_value=1,
        max_value=60,
        value=10,
        step=1,
    )

    support_pct = st.slider(
        "Additional routing-capacity support (%)",
        min_value=0.0,
        max_value=50.0,
        value=10.0,
        step=1.0,
    )

    st.header("3. Distribution settings")

    st.caption(
        "Fallback shifted-lognormal parameters used to draw the topology-exponent paths."
    )

    lognorm_sigma = st.number_input(
        "Lognormal sigma",
        min_value=0.01,
        max_value=5.0,
        value=FALLBACK_LOGNORM_PARAMS[0],
        step=0.01,
    )

    lognorm_loc = st.number_input(
        "Location shift",
        min_value=0.0,
        max_value=10.0,
        value=FALLBACK_LOGNORM_PARAMS[1],
        step=0.01,
    )

    lognorm_scale = st.number_input(
        "Lognormal scale",
        min_value=0.01,
        max_value=10.0,
        value=FALLBACK_LOGNORM_PARAMS[2],
        step=0.01,
    )


cfg = SimulationConfig(
    n_nodes=int(n_nodes),
    scenarios=int(scenarios),
    trading_days=int(trading_days),
    investment=float(investment),
    buffer_normal_pct=float(buffer_normal_pct),
    liquidity_risk_q=float(liquidity_risk_q),
    seed=int(seed),
    target_recall=float(target_recall),
    target_precision=float(target_precision),
    ewi_lead_window=int(ewi_lead_window),
    support_days=int(support_days),
    support_pct=float(support_pct),
    lognorm_sigma=float(lognorm_sigma),
    lognorm_loc=float(lognorm_loc),
    lognorm_scale=float(lognorm_scale),
)


# -----------------------------------------------------------------------------
# Run simulation
# -----------------------------------------------------------------------------

with st.spinner("Running simulation..."):
    sim = simulate_base_paths(
        cfg.n_nodes,
        cfg.scenarios,
        cfg.trading_days,
        cfg.investment,
        cfg.buffer_normal_pct,
        cfg.liquidity_risk_q,
        cfg.seed,
        cfg.lognorm_sigma,
        cfg.lognorm_loc,
        cfg.lognorm_scale,
    )

    base_metrics = evaluate_liquidity(
        sim["direct_liquidity"],
        sim["risk_threshold"],
    )

    design = build_nested_ewi_design(
        sim["event_day"],
        cfg.ewi_lead_window,
        cfg.seed,
    )

    ewi_flags, ewi_diag = create_nested_ewi(
        sim["event_day"],
        design,
        cfg.target_recall,
        cfg.target_precision,
        cfg.ewi_lead_window,
    )

    active_mask = forward_active_mask(
        ewi_flags,
        cfg.support_days,
    )

    policy_liquidity, policy_metrics, policy_extra = run_support_policy(
        sim["direct_lm"],
        base_metrics,
        sim["risk_threshold"],
        active_mask,
        cfg.investment,
        cfg.buffer_normal_pct,
        cfg.support_pct,
    )


# -----------------------------------------------------------------------------
# KPIs
# -----------------------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Liquidity-risk threshold",
    f"{sim['risk_threshold']:.2f}",
)

col2.metric(
    "Baseline risk-day rate",
    f"{base_metrics['risk_day_rate']:.2f}%",
)

col3.metric(
    "Policy risk-day rate",
    f"{policy_metrics['risk_day_rate']:.2f}%",
)

col4.metric(
    "Shortfall reduction",
    f"{policy_extra['shortfall_reduction_pct']:.2f}%",
)


# -----------------------------------------------------------------------------
# Summary tables
# -----------------------------------------------------------------------------

simulation_summary = pd.DataFrame(
    [
        ["Network size", cfg.n_nodes],
        ["Simulation scenarios", cfg.scenarios],
        ["Trading days per scenario", cfg.trading_days],
        ["Total scenario-days", cfg.scenarios * cfg.trading_days],
        ["Investment base", cfg.investment],
        ["Normal buffer / unavailable liquidity (%)", cfg.buffer_normal_pct],
        ["Liquidity-risk threshold quantile", cfg.liquidity_risk_q],
        ["Liquidity-risk threshold", round(sim["risk_threshold"], 4)],
        ["Liquidity-risk days", base_metrics["risk_days"]],
        ["Liquidity-risk day rate (%)", round(base_metrics["risk_day_rate"], 4)],
        ["Liquidity-risk scenarios", base_metrics["risk_scenarios"]],
        [
            "Liquidity-risk scenario rate (%)",
            round(base_metrics["risk_scenario_rate"], 4),
        ],
        ["Lognormal sigma", cfg.lognorm_sigma],
        ["Lognormal location shift", cfg.lognorm_loc],
        ["Lognormal scale", cfg.lognorm_scale],
    ],
    columns=["Metric", "Value"],
)

ewi_summary = pd.DataFrame(
    [[k.replace("_", " ").title(), v] for k, v in ewi_diag.items()],
    columns=["Metric", "Value"],
)

policy_summary = pd.DataFrame(
    [
        {
            "Policy": "No mitigation",
            "Support setting (%)": 0.0,
            "Policy support intensity (%)": 0.0,
            "Liquidity-risk scenario rate (%)": base_metrics["risk_scenario_rate"],
            "Liquidity-risk day rate (%)": base_metrics["risk_day_rate"],
            "Total routing shortfall": base_metrics["total_shortfall"],
            "Risk-day reduction (%)": 0.0,
            "Shortfall reduction (%)": 0.0,
            "Mitigation efficiency": np.nan,
        },
        {
            "Policy": "Routing-capacity support after EWI",
            "Support setting (%)": cfg.support_pct,
            "Policy support intensity (%)": policy_extra["support_intensity_pct"],
            "Liquidity-risk scenario rate (%)": policy_metrics[
                "risk_scenario_rate"
            ],
            "Liquidity-risk day rate (%)": policy_metrics["risk_day_rate"],
            "Total routing shortfall": policy_metrics["total_shortfall"],
            "Risk-day reduction (%)": policy_extra["risk_reduction_pct"],
            "Shortfall reduction (%)": policy_extra["shortfall_reduction_pct"],
            "Mitigation efficiency": policy_extra["mitigation_efficiency"],
        },
    ]
)


# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------

fig_tab, policy_tab, sensitivity_tab, network_tab, ewi_tab, downloads_tab = st.tabs(
    [
        "Liquidity paths",
        "Policy result",
        "Support sensitivity",
        "Network-size efficiency",
        "EWI quality",
        "Downloads",
    ]
)


# -----------------------------------------------------------------------------
# Liquidity paths tab
# -----------------------------------------------------------------------------

with fig_tab:
    st.subheader("Direct and indirect network-liquidity routing capacity")

    x = np.arange(1, cfg.trading_days + 1)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
        sharex=True,
    )

    for ax, arr, title, color in [
        (
            axes[0],
            sim["direct_liquidity"],
            "A. Direct network liquidity routing capacity",
            "darkblue",
        ),
        (
            axes[1],
            sim["indirect_liquidity"],
            "B. Indirect network liquidity routing capacity",
            "darkred",
        ),
    ]:
        mean = np.nanmean(arr, axis=0)
        q10 = np.nanquantile(arr, 0.10, axis=0)
        q90 = np.nanquantile(arr, 0.90, axis=0)
        q025 = np.nanquantile(arr, 0.025, axis=0)
        q975 = np.nanquantile(arr, 0.975, axis=0)

        ax.plot(
            x,
            mean,
            color=color,
            linewidth=2.0,
            label="Average capacity",
        )

        ax.fill_between(
            x,
            q10,
            q90,
            color=color,
            alpha=0.18,
            label="10-90% range",
        )

        ax.fill_between(
            x,
            q025,
            q975,
            color=color,
            alpha=0.08,
            label="2.5-97.5% range",
        )

        ax.axhline(
            sim["risk_threshold"],
            color="red",
            linestyle="--",
            linewidth=1.0,
            label="Q2.5 liquidity-risk day threshold",
        )

        y_max = max(
            np.nanmax(sim["direct_liquidity"]),
            np.nanmax(sim["indirect_liquidity"]),
        )

        y_upper = math.ceil((y_max * 1.05) / 100) * 100

        # User-requested behaviour:
        # start the y-axis at the selected investment amount.
        ax.set_ylim(cfg.investment, y_upper)

        ax.set_title(title)
        ax.set_xlabel("Trading day")
        ax.grid(True, linestyle="--", alpha=0.35)

    axes[0].set_ylabel("Direct network liquidity routing capacity")
    axes[0].legend(fontsize=8, loc="upper right")

    axes[1].set_ylabel("Indirect network liquidity routing capacity")
    axes[1].legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    st.pyplot(fig)


# -----------------------------------------------------------------------------
# Policy result tab
# -----------------------------------------------------------------------------

with policy_tab:
    st.subheader("Policy comparison")

    plot_df = policy_summary.copy()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.8),
    )

    colors = ["#333333", "#2e7d32"]

    axes[0].bar(
        plot_df["Policy"],
        plot_df["Liquidity-risk scenario rate (%)"],
        color=colors,
    )
    axes[0].set_title("A. Scenario rate")
    axes[0].set_ylabel("% of scenarios")

    axes[1].bar(
        plot_df["Policy"],
        plot_df["Liquidity-risk day rate (%)"],
        color=colors,
    )
    axes[1].set_title("B. Risk-day rate")
    axes[1].set_ylabel("% of trading days")

    denom = plot_df.loc[0, "Total routing shortfall"]

    shortfall_pct = (
        plot_df["Total routing shortfall"] / denom * 100
        if denom
        else np.nan
    )

    axes[2].bar(
        plot_df["Policy"],
        shortfall_pct,
        color=colors,
    )
    axes[2].axhline(
        100,
        color="grey",
        linestyle="--",
        linewidth=1,
    )
    axes[2].set_title("C. Cumulative routing shortfall")
    axes[2].set_ylabel("% of no mitigation")

    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()

    st.pyplot(fig)
    st.dataframe(policy_summary.round(3), use_container_width=True)


# -----------------------------------------------------------------------------
# Support sensitivity tab
# -----------------------------------------------------------------------------

with sensitivity_tab:
    st.subheader("Support sensitivity")

    support_grid = list(range(0, 45, 5))
    rows = []

    for sp in support_grid:
        if sp == 0:
            rows.append(
                {
                    "Support setting (%)": sp,
                    "Policy support intensity (%)": 0.0,
                    "Risk-day reduction (%)": 0.0,
                    "Shortfall reduction (%)": 0.0,
                    "Mitigation efficiency": np.nan,
                }
            )
        else:
            _, pm, extra = run_support_policy(
                sim["direct_lm"],
                base_metrics,
                sim["risk_threshold"],
                active_mask,
                cfg.investment,
                cfg.buffer_normal_pct,
                float(sp),
            )

            rows.append(
                {
                    "Support setting (%)": sp,
                    "Policy support intensity (%)": extra[
                        "support_intensity_pct"
                    ],
                    "Risk-day reduction (%)": extra["risk_reduction_pct"],
                    "Shortfall reduction (%)": extra[
                        "shortfall_reduction_pct"
                    ],
                    "Mitigation efficiency": extra[
                        "mitigation_efficiency"
                    ],
                }
            )

    support_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8.8, 5.0))

    ax.plot(
        support_df["Support setting (%)"],
        support_df["Risk-day reduction (%)"],
        marker="o",
        label="Risk-day reduction",
    )

    ax.plot(
        support_df["Support setting (%)"],
        support_df["Shortfall reduction (%)"],
        marker="s",
        label="Shortfall reduction",
    )

    ax.set_xlabel("Additional routing-capacity support (%)")
    ax.set_ylabel("Reduction relative to no mitigation (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    st.pyplot(fig)
    st.dataframe(support_df.round(3), use_container_width=True)


# -----------------------------------------------------------------------------
# Network-size efficiency tab
# -----------------------------------------------------------------------------

with network_tab:
    st.subheader("Mitigation efficiency over network size")

    st.caption(
        "Uses the current EWI settings and support-duration setting across "
        "a compact default grid."
    )

    network_grid = list(range(20, 101, 10))
    support_levels = [10, 20, 30, 40]
    rows = []

    for n in network_grid:
        sim_n = simulate_base_paths(
            int(n),
            cfg.scenarios,
            cfg.trading_days,
            cfg.investment,
            cfg.buffer_normal_pct,
            cfg.liquidity_risk_q,
            cfg.seed,
            cfg.lognorm_sigma,
            cfg.lognorm_loc,
            cfg.lognorm_scale,
        )

        bm_n = evaluate_liquidity(
            sim_n["direct_liquidity"],
            sim_n["risk_threshold"],
        )

        design_n = build_nested_ewi_design(
            sim_n["event_day"],
            cfg.ewi_lead_window,
            cfg.seed,
        )

        ewi_n, _ = create_nested_ewi(
            sim_n["event_day"],
            design_n,
            cfg.target_recall,
            cfg.target_precision,
            cfg.ewi_lead_window,
        )

        active_n = forward_active_mask(
            ewi_n,
            cfg.support_days,
        )

        for sp in support_levels:
            _, _, extra = run_support_policy(
                sim_n["direct_lm"],
                bm_n,
                sim_n["risk_threshold"],
                active_n,
                cfg.investment,
                cfg.buffer_normal_pct,
                float(sp),
            )

            rows.append(
                {
                    "Network size": n,
                    "Support setting (%)": sp,
                    "Policy support intensity (%)": extra[
                        "support_intensity_pct"
                    ],
                    "Shortfall reduction (%)": extra[
                        "shortfall_reduction_pct"
                    ],
                    "Mitigation efficiency": extra[
                        "mitigation_efficiency"
                    ],
                }
            )

    network_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8.8, 5.0))

    for sp in support_levels:
        sub = network_df[network_df["Support setting (%)"] == sp]

        ax.plot(
            sub["Network size"],
            sub["Mitigation efficiency"],
            marker="o",
            label=f"{sp}% support",
        )

    ax.set_xlabel("Network size")
    ax.set_ylabel("Shortfall reduction per 1% support intensity")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    st.pyplot(fig)
    st.dataframe(network_df.round(3), use_container_width=True)


# -----------------------------------------------------------------------------
# EWI-quality tab
# -----------------------------------------------------------------------------

with ewi_tab:
    st.subheader("EWI-quality impact under fixed 10% support")

    ewi_scenarios = [
        ("Low recall-precision", 0.50, 0.10),
        ("Baseline", 0.70, 0.25),
        ("Higher recall", 0.90, 0.25),
        ("Less targeted", 0.70, 0.10),
        ("More targeted", 0.70, 0.50),
    ]

    rows = []

    for label, rec, prec in ewi_scenarios:
        ewi_q, diag_q = create_nested_ewi(
            sim["event_day"],
            design,
            rec,
            prec,
            cfg.ewi_lead_window,
        )

        active_q = forward_active_mask(
            ewi_q,
            cfg.support_days,
        )

        _, pm_q, extra_q = run_support_policy(
            sim["direct_lm"],
            base_metrics,
            sim["risk_threshold"],
            active_q,
            cfg.investment,
            cfg.buffer_normal_pct,
            10.0,
        )

        rows.append(
            {
                "EWI setting": label,
                "Target recall (%)": rec * 100,
                "Target precision (%)": prec * 100,
                "Achieved recall (%)": diag_q["achieved_recall"],
                "Achieved precision (%)": diag_q["achieved_precision"],
                "Signal-day rate (%)": diag_q["signal_day_rate"],
                "Support setting (%)": 10.0,
                "Policy support intensity (%)": extra_q[
                    "support_intensity_pct"
                ],
                "Policy risk-day rate (%)": pm_q["risk_day_rate"],
                "Risk-day reduction (%)": extra_q["risk_reduction_pct"],
                "Shortfall reduction (%)": extra_q[
                    "shortfall_reduction_pct"
                ],
                "Mitigation efficiency": extra_q[
                    "mitigation_efficiency"
                ],
            }
        )

    ewi_quality_df = pd.DataFrame(rows)

    st.dataframe(ewi_quality_df.round(3), use_container_width=True)

    st.subheader("Current EWI diagnostics")
    st.dataframe(ewi_summary.round(3), use_container_width=True)


# -----------------------------------------------------------------------------
# Downloads tab
# -----------------------------------------------------------------------------

with downloads_tab:
    st.subheader("Download current outputs")

    all_outputs = {
        "Simulation summary": simulation_summary,
        "EWI summary": ewi_summary,
        "Policy summary": policy_summary,
    }

    excel_bytes = to_excel_download(all_outputs)

    st.download_button(
        "Download summary workbook (.xlsx)",
        data=excel_bytes,
        file_name="streamlit_simulation_engine_summary.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
    )

    st.download_button(
        "Download policy summary (.csv)",
        data=policy_summary.to_csv(index=False).encode("utf-8"),
        file_name="policy_summary.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download simulation summary (.csv)",
        data=simulation_summary.to_csv(index=False).encode("utf-8"),
        file_name="simulation_summary.csv",
        mime="text/csv",
    )


st.info(
    "Interpretation note: the policy channel is implemented as additional "
    "routing-capacity support after an estimated-EWI signal. The topology "
    "exponent paths are drawn from fallback shifted-lognormal parameters."
)