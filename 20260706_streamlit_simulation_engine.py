"""
Streamlit simulation engine for Chapter 8 liquidity-risk simulations
====================================================================

This app consolidates the logic from the attached Chapter 8 scripts:
- gamma distribution fitting / fallback lognormal parameters;
- direct and indirect network-liquidity routing-capacity simulation;
- estimated-EWI emulator with target recall and precision;
- policy support as additional routing capacity;
- support-sensitivity, network-size efficiency, and EWI-quality tables.

Run locally with:
    streamlit run streamlit_simulation_engine.py

Optional input:
    gamma_bestfit_summary_generated_in_script.csv
or upload a CSV with columns: distribution, window, params.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------------------------------------------------------
# Defaults calibrated to the attached scripts
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
    gamma_shape: float = FALLBACK_LOGNORM_PARAMS[0]
    gamma_loc: float = FALLBACK_LOGNORM_PARAMS[1]
    gamma_scale: float = FALLBACK_LOGNORM_PARAMS[2]


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def parse_fit_params(uploaded_file, fallback=FALLBACK_LOGNORM_PARAMS):
    """Load 2010-2018 lognormal parameters from an uploaded fit-summary CSV."""
    if uploaded_file is None:
        return fallback, "Fallback lognormal parameters"

    fit = pd.read_csv(uploaded_file)
    fit.columns = [c.strip() for c in fit.columns]
    required = {"distribution", "window", "params"}
    if not required.issubset(set(fit.columns)):
        st.warning("Uploaded fit file does not contain distribution, window and params columns. Using fallback parameters.")
        return fallback, "Fallback lognormal parameters"

    rows = fit[
        (fit["distribution"].astype(str).str.strip() == "lognorm")
        & (fit["window"].astype(str).str.strip() == "2010-2018")
    ]
    if rows.empty:
        st.warning("No 2010-2018 lognormal row found in the uploaded fit file. Using fallback parameters.")
        return fallback, "Fallback lognormal parameters"

    params = tuple(float(x) for x in ast.literal_eval(rows.iloc[0]["params"]))
    return params, "Uploaded 2010-2018 lognormal fit"


def to_excel_download(dfs: Dict[str, pd.DataFrame]) -> bytes:
    """Create an in-memory Excel workbook from one or more dataframes."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df in dfs.items():
            safe_name = sheet_name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)
    return output.getvalue()


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
    gamma_shape: float,
    gamma_loc: float,
    gamma_scale: float,
):
    """Simulate gamma paths and direct/indirect network-liquidity paths."""
    rng = np.random.default_rng(seed)
    gamma_grid, direct_lm_grid, indirect_lm_grid = make_liquidity_multiplier_grids(n_nodes)

    gamma_paths = gamma_loc + rng.lognormal(
        mean=np.log(gamma_scale), sigma=gamma_shape, size=(scenarios, trading_days)
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


def build_nested_ewi_design(event_day: np.ndarray, lead_window: int = 5, seed: int = 42):
    """Create a stable design so EWI scenarios are nested and comparable."""
    rng = np.random.default_rng(seed)
    evaluable = event_day.copy()
    evaluable[:, :lead_window] = False
    event_positions = np.argwhere(evaluable)
    if len(event_positions) > 0:
        event_positions = event_positions[rng.permutation(len(event_positions))]

    all_positions = np.argwhere(~event_day)
    if len(all_positions) > 0:
        all_positions = all_positions[rng.permutation(len(all_positions))]

    true_signal_positions = []
    for s, t in event_positions:
        possible_days = np.arange(max(0, t - lead_window), t)
        if len(possible_days) > 0:
            signal_t = int(rng.choice(possible_days))
            true_signal_positions.append((int(s), signal_t))

    return {
        "n_evaluable_events": int(len(event_positions)),
        "true_signal_positions": true_signal_positions,
        "false_positive_positions": [(int(s), int(t)) for s, t in all_positions],
    }


def create_nested_ewi(event_day: np.ndarray, design: dict, target_recall: float, target_precision: float):
    """Create an estimated-EWI signal process matching target recall and precision as closely as possible."""
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

    # Signal is true positive if followed by a risk day within the lead window.
    tp_signal = np.zeros_like(event_day, dtype=bool)
    for s, t in np.argwhere(ewi):
        future = event_day[s, t + 1 : t + 1 + st.session_state.get("lead_window_for_tp", 5)]
        tp_signal[s, t] = bool(future.any())

    detected_events = n_true
    signal_days = int(ewi.sum())
    true_positive_signal_days = int(tp_signal.sum())
    false_positive_signal_days = signal_days - true_positive_signal_days

    diagnostics = {
        "target_recall": target_recall * 100,
        "target_precision": target_precision * 100,
        "lead_window": st.session_state.get("lead_window_for_tp", 5),
        "evaluable_event_days": n_events,
        "detected_event_days": detected_events,
        "achieved_recall": (detected_events / n_events * 100) if n_events else np.nan,
        "signal_days": signal_days,
        "true_positive_signal_days": true_positive_signal_days,
        "false_positive_signal_days": false_positive_signal_days,
        "achieved_precision": (true_positive_signal_days / signal_days * 100) if signal_days else np.nan,
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
        conv = np.convolve(flags[s].astype(int), kernel, mode="full")[: flags.shape[1]]
        out[s] = conv > 0
    return out


def evaluate_liquidity(arr: np.ndarray, risk_threshold: float):
    """Evaluate liquidity-risk days, scenarios and cumulative shortfall."""
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
    support_intensity = support.sum() / (baseline_available * direct_lm.size) * 100 if baseline_available > 0 else 0.0
    policy_liquidity = (baseline_available + support) * direct_lm
    policy_metrics = evaluate_liquidity(policy_liquidity, risk_threshold)

    risk_reduction = (
        (base_metrics["risk_days"] - policy_metrics["risk_days"]) / base_metrics["risk_days"] * 100
        if base_metrics["risk_days"]
        else np.nan
    )
    shortfall_reduction = (
        (base_metrics["total_shortfall"] - policy_metrics["total_shortfall"]) / base_metrics["total_shortfall"] * 100
        if base_metrics["total_shortfall"]
        else np.nan
    )
    efficiency = shortfall_reduction / support_intensity if support_intensity > 0 else np.nan

    extra = {
        "support_intensity_pct": support_intensity,
        "risk_reduction_pct": risk_reduction,
        "shortfall_reduction_pct": shortfall_reduction,
        "mitigation_efficiency": efficiency,
    }
    return policy_liquidity, policy_metrics, extra


# -----------------------------------------------------------------------------
# Streamlit app
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Liquidity-risk simulation engine", layout="wide")
st.title("Liquidity-risk simulation engine")
st.caption(
    "Interactive version of the Chapter 8 simulation: gamma-driven network liquidity, estimated-EWI signals, "
    "and routing-capacity policy support."
)

with st.sidebar:
    st.header("1. Data and calibration")
    uploaded_fit = st.file_uploader("Optional gamma fit summary CSV", type=["csv"])
    params, params_source = parse_fit_params(uploaded_fit)
    st.write(f"Gamma parameters: {params_source}")

    st.header("2. Simulation settings")
    n_nodes = st.slider("Network size", 10, 150, 24, 1)
    scenarios = st.number_input("Simulation scenarios", 100, 10000, 1000, 100)
    trading_days = st.number_input("Trading days", 50, 1000, 200, 10)
    seed = st.number_input("Random seed", 1, 999999, 42, 1)
    investment = st.number_input("Investment base", 1.0, 10000.0, INVESTMENT_DEFAULT, 10.0)
    buffer_normal_pct = st.slider("Normal buffer / unavailable liquidity (%)", 0.0, 90.0, BUFFER_NORMAL_DEFAULT, 1.0)
    liquidity_risk_q = st.slider("Liquidity-risk threshold quantile", 0.001, 0.100, 0.025, 0.001, format="%.3f")

    st.header("3. EWI and policy settings")
    target_recall = st.slider("Target EWI recall", 0.0, 1.0, 0.70, 0.05)
    target_precision = st.slider("Target EWI precision", 0.05, 1.0, 0.25, 0.05)
    ewi_lead_window = st.slider("EWI lead window, trading days", 1, 30, 5, 1)
    support_days = st.slider("Support duration after EWI", 1, 60, 10, 1)
    support_pct = st.slider("Additional routing-capacity support (%)", 0.0, 50.0, 10.0, 1.0)

st.session_state["lead_window_for_tp"] = int(ewi_lead_window)

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
    gamma_shape=float(params[0]),
    gamma_loc=float(params[1]),
    gamma_scale=float(params[2]),
)

with st.spinner("Running simulation..."):
    sim = simulate_base_paths(
        cfg.n_nodes,
        cfg.scenarios,
        cfg.trading_days,
        cfg.investment,
        cfg.buffer_normal_pct,
        cfg.liquidity_risk_q,
        cfg.seed,
        cfg.gamma_shape,
        cfg.gamma_loc,
        cfg.gamma_scale,
    )
    base_metrics = evaluate_liquidity(sim["direct_liquidity"], sim["risk_threshold"])
    design = build_nested_ewi_design(sim["event_day"], cfg.ewi_lead_window, cfg.seed)
    ewi_flags, ewi_diag = create_nested_ewi(sim["event_day"], design, cfg.target_recall, cfg.target_precision)
    active_mask = forward_active_mask(ewi_flags, cfg.support_days)
    policy_liquidity, policy_metrics, policy_extra = run_support_policy(
        sim["direct_lm"],
        base_metrics,
        sim["risk_threshold"],
        active_mask,
        cfg.investment,
        cfg.buffer_normal_pct,
        cfg.support_pct,
    )

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Liquidity-risk threshold", f"{sim['risk_threshold']:.2f}")
col2.metric("Baseline risk-day rate", f"{base_metrics['risk_day_rate']:.2f}%")
col3.metric("Policy risk-day rate", f"{policy_metrics['risk_day_rate']:.2f}%")
col4.metric("Shortfall reduction", f"{policy_extra['shortfall_reduction_pct']:.2f}%")

# Summary tables
simulation_summary = pd.DataFrame(
    [
        ["Network size", cfg.n_nodes],
        ["Simulation scenarios", cfg.scenarios],
        ["Simulation trading days", cfg.scenarios * cfg.trading_days],
        ["Liquidity-risk threshold quantile", cfg.liquidity_risk_q],
        ["Liquidity-risk threshold", round(sim["risk_threshold"], 4)],
        ["Liquidity-risk days", base_metrics["risk_days"]],
        ["Liquidity-risk day rate (%)", round(base_metrics["risk_day_rate"], 4)],
        ["Liquidity-risk scenarios", base_metrics["risk_scenarios"]],
        ["Liquidity-risk scenario rate (%)", round(base_metrics["risk_scenario_rate"], 4)],
    ],
    columns=["Metric", "Value"],
)

ewi_summary = pd.DataFrame([[k.replace("_", " ").title(), v] for k, v in ewi_diag.items()], columns=["Metric", "Value"])

policy_summary = pd.DataFrame(
    [
        {
            "Policy": "No mitigation",
            "Support setting (%)": 0.0,
            "Policy support intensity (%)": 0.0,
            "Liquidity-risk scenario rate (%)": base_metrics["risk_scenario_rate"],
            "Liquidity-risk day rate (%)": base_metrics["risk_day_rate"],
            "Total shortfall": base_metrics["total_shortfall"],
            "Risk-day reduction (%)": 0.0,
            "Shortfall reduction (%)": 0.0,
            "Mitigation efficiency": np.nan,
        },
        {
            "Policy": "Routing-capacity support after EWI",
            "Support setting (%)": cfg.support_pct,
            "Policy support intensity (%)": policy_extra["support_intensity_pct"],
            "Liquidity-risk scenario rate (%)": policy_metrics["risk_scenario_rate"],
            "Liquidity-risk day rate (%)": policy_metrics["risk_day_rate"],
            "Total shortfall": policy_metrics["total_shortfall"],
            "Risk-day reduction (%)": policy_extra["risk_reduction_pct"],
            "Shortfall reduction (%)": policy_extra["shortfall_reduction_pct"],
            "Mitigation efficiency": policy_extra["mitigation_efficiency"],
        },
    ]
)

# Tabs
fig_tab, policy_tab, sensitivity_tab, network_tab, ewi_tab, downloads_tab = st.tabs(
    ["Liquidity paths", "Policy result", "Support sensitivity", "Network-size efficiency", "EWI quality", "Downloads"]
)

with fig_tab:
    st.subheader("Direct and indirect network-liquidity routing capacity")
    x = np.arange(1, cfg.trading_days + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    for ax, arr, title, color in [
        (axes[0], sim["direct_liquidity"], "A. Direct network liquidity", "darkblue"),
        (axes[1], sim["indirect_liquidity"], "B. Indirect network liquidity", "darkred"),
    ]:
        mean = np.nanmean(arr, axis=0)
        q10 = np.nanquantile(arr, 0.10, axis=0)
        q90 = np.nanquantile(arr, 0.90, axis=0)
        q025 = np.nanquantile(arr, 0.025, axis=0)
        q975 = np.nanquantile(arr, 0.975, axis=0)
        ax.plot(x, mean, color=color, linewidth=2.0, label="Mean")
        ax.fill_between(x, q10, q90, color=color, alpha=0.18, label="10-90% range")
        ax.fill_between(x, q025, q975, color=color, alpha=0.08, label="2.5-97.5% range")
        ax.axhline(sim["risk_threshold"], color="black", linestyle="--", linewidth=1.0, label="Direct risk threshold")
        ax.set_title(title)
        ax.set_xlabel("Trading day")
        ax.grid(True, linestyle="--", alpha=0.35)
    axes[0].set_ylabel("Liquidity routing capacity")
    axes[0].legend(fontsize=8)
    st.pyplot(fig)

with policy_tab:
    st.subheader("Policy comparison")
    plot_df = policy_summary.copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    colors = ["#333333", "#2e7d32"]
    axes[0].bar(plot_df["Policy"], plot_df["Liquidity-risk scenario rate (%)"], color=colors)
    axes[0].set_title("A. Scenario rate")
    axes[0].set_ylabel("% of scenarios")
    axes[1].bar(plot_df["Policy"], plot_df["Liquidity-risk day rate (%)"], color=colors)
    axes[1].set_title("B. Risk-day rate")
    axes[1].set_ylabel("% of trading days")
    shortfall_pct = plot_df["Total shortfall"] / plot_df.loc[0, "Total shortfall"] * 100 if plot_df.loc[0, "Total shortfall"] else np.nan
    axes[2].bar(plot_df["Policy"], shortfall_pct, color=colors)
    axes[2].axhline(100, color="grey", linestyle="--", linewidth=1)
    axes[2].set_title("C. Cumulative shortfall")
    axes[2].set_ylabel("% of no mitigation")
    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    st.pyplot(fig)
    st.dataframe(policy_summary.round(3), use_container_width=True)

with sensitivity_tab:
    st.subheader("Support sensitivity")
    support_grid = list(range(0, 45, 5))
    rows = []
    for sp in support_grid:
        if sp == 0:
            rows.append({
                "Support setting (%)": sp,
                "Policy support intensity (%)": 0.0,
                "Risk-day reduction (%)": 0.0,
                "Shortfall reduction (%)": 0.0,
                "Mitigation efficiency": np.nan,
            })
        else:
            _, pm, extra = run_support_policy(
                sim["direct_lm"], base_metrics, sim["risk_threshold"], active_mask,
                cfg.investment, cfg.buffer_normal_pct, float(sp)
            )
            rows.append({
                "Support setting (%)": sp,
                "Policy support intensity (%)": extra["support_intensity_pct"],
                "Risk-day reduction (%)": extra["risk_reduction_pct"],
                "Shortfall reduction (%)": extra["shortfall_reduction_pct"],
                "Mitigation efficiency": extra["mitigation_efficiency"],
            })
    support_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.plot(support_df["Support setting (%)"], support_df["Risk-day reduction (%)"], marker="o", label="Risk-day reduction")
    ax.plot(support_df["Support setting (%)"], support_df["Shortfall reduction (%)"], marker="s", label="Shortfall reduction")
    ax.set_xlabel("Additional routing-capacity support (%)")
    ax.set_ylabel("Reduction relative to no mitigation (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    st.pyplot(fig)
    st.dataframe(support_df.round(3), use_container_width=True)

with network_tab:
    st.subheader("Mitigation efficiency over network size")
    st.caption("Uses the current EWI settings and support-duration setting across a compact default grid.")
    network_grid = list(range(20, 101, 10))
    support_levels = [10, 20, 30, 40]
    rows = []
    for n in network_grid:
        sim_n = simulate_base_paths(
            int(n), cfg.scenarios, cfg.trading_days, cfg.investment, cfg.buffer_normal_pct,
            cfg.liquidity_risk_q, cfg.seed, cfg.gamma_shape, cfg.gamma_loc, cfg.gamma_scale
        )
        bm_n = evaluate_liquidity(sim_n["direct_liquidity"], sim_n["risk_threshold"])
        design_n = build_nested_ewi_design(sim_n["event_day"], cfg.ewi_lead_window, cfg.seed)
        ewi_n, _ = create_nested_ewi(sim_n["event_day"], design_n, cfg.target_recall, cfg.target_precision)
        active_n = forward_active_mask(ewi_n, cfg.support_days)
        for sp in support_levels:
            _, _, extra = run_support_policy(
                sim_n["direct_lm"], bm_n, sim_n["risk_threshold"], active_n,
                cfg.investment, cfg.buffer_normal_pct, float(sp)
            )
            rows.append({
                "Network size": n,
                "Support setting (%)": sp,
                "Policy support intensity (%)": extra["support_intensity_pct"],
                "Shortfall reduction (%)": extra["shortfall_reduction_pct"],
                "Mitigation efficiency": extra["mitigation_efficiency"],
            })
    network_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for sp in support_levels:
        sub = network_df[network_df["Support setting (%)"] == sp]
        ax.plot(sub["Network size"], sub["Mitigation efficiency"], marker="o", label=f"{sp}% support")
    ax.set_xlabel("Network size")
    ax.set_ylabel("Shortfall reduction per 1% support intensity")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    st.pyplot(fig)
    st.dataframe(network_df.round(3), use_container_width=True)

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
        ewi_q, diag_q = create_nested_ewi(sim["event_day"], design, rec, prec)
        active_q = forward_active_mask(ewi_q, cfg.support_days)
        _, pm_q, extra_q = run_support_policy(
            sim["direct_lm"], base_metrics, sim["risk_threshold"], active_q,
            cfg.investment, cfg.buffer_normal_pct, 10.0
        )
        rows.append({
            "EWI setting": label,
            "Target recall (%)": rec * 100,
            "Target precision (%)": prec * 100,
            "Achieved recall (%)": diag_q["achieved_recall"],
            "Achieved precision (%)": diag_q["achieved_precision"],
            "Signal-day rate (%)": diag_q["signal_day_rate"],
            "Support setting (%)": 10.0,
            "Policy support intensity (%)": extra_q["support_intensity_pct"],
            "Policy risk-day rate (%)": pm_q["risk_day_rate"],
            "Risk-day reduction (%)": extra_q["risk_reduction_pct"],
            "Shortfall reduction (%)": extra_q["shortfall_reduction_pct"],
            "Mitigation efficiency": extra_q["mitigation_efficiency"],
        })
    ewi_quality_df = pd.DataFrame(rows)
    st.dataframe(ewi_quality_df.round(3), use_container_width=True)
    st.subheader("Current EWI diagnostics")
    st.dataframe(ewi_summary.round(3), use_container_width=True)

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
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
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
    "Interpretation note: the policy channel is implemented as additional routing-capacity support after an estimated-EWI signal. "
)
