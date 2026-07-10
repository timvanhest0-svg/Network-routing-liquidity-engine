"""
Streamlit simulation engine for Chapter 8 routing liquidity-risk simulations
==========================================================================

This app consolidates the logic from the Chapter 8 simulation scripts:
- lognormal parameters for topology-exponent paths;
- direct and indirect network-liquidity routing-capacity simulation;
- estimated-EWI emulator with target recall and precision;
- routing-capacity policy support after EWI signals;
- support-sensitivity, network-size efficiency, and EWI-quality overviews.

Timing interpretation
---------------------
The EWI lead time is implemented as a fixed number of trading days before
a liquidity-risk event. For example, with a fixed lead time of X trading days,
a true EWI signal for an event on day t is placed on day t - X.

The support-start delay is a separate policy-implementation parameter.
If an EWI signal occurs on day s, policy support starts on day
s + support_start_delay and remains active for support_days.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass
from io import BytesIO
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------

FALLBACK_LOGNORM_PARAMS = (0.32, 0.0, 1.09)
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
    ewi_lead_time: int = 5
    support_days: int = 10
    support_start_delay: int = 5
    support_pct: float = 10.0
    lognorm_sigma: float = FALLBACK_LOGNORM_PARAMS[0]
    lognorm_loc: float = FALLBACK_LOGNORM_PARAMS[1]
    lognorm_scale: float = FALLBACK_LOGNORM_PARAMS[2]


# -----------------------------------------------------------------------------
# Simulation functions
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def make_liquidity_multiplier_grids(n_nodes: int, grid_size: int = 5000):
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
        indirect_grid[i] = expected_k2 / expected_k - 1.0 if expected_k > 0 else np.nan

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
    rng = np.random.default_rng(seed)
    gamma_grid, direct_lm_grid, indirect_lm_grid = make_liquidity_multiplier_grids(n_nodes)

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


# -----------------------------------------------------------------------------
# EWI design
# -----------------------------------------------------------------------------

def future_event_mask(event_day: np.ndarray, lead_time: int) -> np.ndarray:
    out = np.zeros_like(event_day, dtype=bool)
    if 0 < lead_time < event_day.shape[1]:
        out[:, :-lead_time] = event_day[:, lead_time:]
    return out


def detected_event_mask(event_day: np.ndarray, ewi: np.ndarray, lead_time: int) -> np.ndarray:
    detected = np.zeros_like(event_day, dtype=bool)
    if 0 < lead_time < event_day.shape[1]:
        detected[:, lead_time:] = event_day[:, lead_time:] & ewi[:, :-lead_time]
    return detected


def build_nested_ewi_design(event_day: np.ndarray, lead_time: int = 5, seed: int = 42):
    rng = np.random.default_rng(seed)
    evaluable = event_day.copy()
    evaluable[:, :lead_time] = False

    event_positions = np.argwhere(evaluable)
    if len(event_positions) > 0:
        event_positions = event_positions[rng.permutation(len(event_positions))]

    exact_pre_event = future_event_mask(event_day, lead_time)
    false_positive_candidates = np.argwhere(~event_day & ~exact_pre_event)
    if len(false_positive_candidates) > 0:
        false_positive_candidates = false_positive_candidates[rng.permutation(len(false_positive_candidates))]

    true_signal_positions = []
    signalable_event_count = 0

    for s, t in event_positions:
        signal_t = int(t - lead_time)
        if signal_t >= 0 and not event_day[s, signal_t]:
            signalable_event_count += 1
            true_signal_positions.append((int(s), signal_t))

    seen = set()
    unique_true_signal_positions = []
    for pos in true_signal_positions:
        if pos not in seen:
            unique_true_signal_positions.append(pos)
            seen.add(pos)

    return {
        "n_evaluable_events": int(len(event_positions)),
        "n_signalable_events": int(signalable_event_count),
        "true_signal_positions": unique_true_signal_positions,
        "false_positive_positions": [(int(s), int(t)) for s, t in false_positive_candidates],
    }


def create_nested_ewi(
    event_day: np.ndarray,
    design: dict,
    target_recall: float,
    target_precision: float,
    lead_time: int = 5,
):
    ewi = np.zeros_like(event_day, dtype=bool)
    n_events = design["n_evaluable_events"]
    n_signalable_events = design.get("n_signalable_events", n_events)
    true_positions = design["true_signal_positions"]
    fp_positions = design["false_positive_positions"]

    n_true_target = int(round(target_recall * n_events)) if n_events else 0
    n_true = min(n_true_target, len(true_positions))

    for s, t in true_positions[:n_true]:
        ewi[s, t] = True

    if target_precision > 0 and n_true > 0:
        target_total_signals = int(round(n_true / target_precision))
        n_fp = max(0, target_total_signals - n_true)
    else:
        n_fp = 0

    n_fp = min(n_fp, len(fp_positions))
    for s, t in fp_positions[:n_fp]:
        ewi[s, t] = True

    tp_signal = np.zeros_like(event_day, dtype=bool)
    if 0 < lead_time < event_day.shape[1]:
        tp_signal[:, :-lead_time] = ewi[:, :-lead_time] & event_day[:, lead_time:]

    signal_days = int(ewi.sum())
    true_positive_signal_days = int(tp_signal.sum())
    false_positive_signal_days = signal_days - true_positive_signal_days

    detected_events = detected_event_mask(event_day, ewi, lead_time)
    evaluable = event_day.copy()
    evaluable[:, :lead_time] = False
    detected_event_days = int((detected_events & evaluable).sum())

    diagnostics = {
        "target_recall": target_recall * 100,
        "target_precision": target_precision * 100,
        "fixed_lead_time_days": lead_time,
        "evaluable_event_days": n_events,
        "signalable_event_days": n_signalable_events,
        "selected_true_signal_days": n_true,
        "detected_event_days": detected_event_days,
        "achieved_recall": detected_event_days / n_events * 100 if n_events else np.nan,
        "achieved_recall_signalable_events": detected_event_days / n_signalable_events * 100 if n_signalable_events else np.nan,
        "signal_days": signal_days,
        "true_positive_signal_days": true_positive_signal_days,
        "false_positive_signal_days": false_positive_signal_days,
        "achieved_precision": true_positive_signal_days / signal_days * 100 if signal_days else np.nan,
        "signal_day_rate": signal_days / event_day.size * 100,
        "false_positive_rate": false_positive_signal_days / event_day.size * 100,
    }

    return ewi, diagnostics


# -----------------------------------------------------------------------------
# Policy functions
# -----------------------------------------------------------------------------

def forward_active_mask(flags: np.ndarray, duration: int, start_delay: int):
    if duration <= 0:
        return np.zeros_like(flags, dtype=bool)
    if start_delay < 1:
        raise ValueError("start_delay must be at least 1 trading day.")

    out = np.zeros_like(flags, dtype=bool)
    kernel = np.ones(duration, dtype=int)
    n_days = flags.shape[1]

    for s in range(flags.shape[0]):
        conv = np.convolve(flags[s].astype(int), kernel, mode="full")
        shifted = np.zeros(n_days, dtype=int)
        if start_delay < n_days:
            shifted[start_delay:] = conv[: n_days - start_delay]
        out[s] = shifted > 0

    return out


def evaluate_liquidity(arr: np.ndarray, risk_threshold: float):
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
    baseline_available = investment * (1.0 - buffer_normal_pct / 100.0)
    support = investment * (support_pct / 100.0) * active_mask

    support_intensity = (
        support.sum() / (baseline_available * direct_lm.size) * 100
        if baseline_available > 0
        else np.nan
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

    efficiency = shortfall_reduction / support_intensity if support_intensity > 0 else np.nan

    extra = {
        "support_intensity_pct": support_intensity,
        "risk_reduction_pct": risk_reduction,
        "shortfall_reduction_pct": shortfall_reduction,
        "mitigation_efficiency": efficiency,
    }

    return policy_liquidity, policy_metrics, extra


def random_active_mask_same_intensity(reference_active_mask: np.ndarray, seed: int) -> np.ndarray:
    """
    Create an untargeted benchmark with the same number of active support days
    as the EWI-triggered policy, but randomly allocated over scenario-days.
    """
    rng = np.random.default_rng(seed)
    n_active = int(reference_active_mask.sum())
    out = np.zeros_like(reference_active_mask, dtype=bool)

    if n_active <= 0:
        return out

    flat_size = reference_active_mask.size
    n_active = min(n_active, flat_size)
    selected = rng.choice(flat_size, size=n_active, replace=False)
    out.flat[selected] = True
    return out


# -----------------------------------------------------------------------------
# Download and plotting helpers
# -----------------------------------------------------------------------------

def to_excel_download(sheets: dict) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    return output.getvalue()


def fig_to_png_bytes(fig, dpi: int = 300) -> bytes:
    output = BytesIO()
    fig.savefig(output, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    output.seek(0)
    return output.getvalue()


def format_value_for_table_png(value):
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, (float, np.floating)):
        return f"{value:.3f}"
    return str(value)


def dataframe_to_png_bytes(df: pd.DataFrame, title: str, dpi: int = 300, font_size: int = 8) -> bytes:
    formatted_df = df.copy().astype(object).apply(lambda col: col.map(format_value_for_table_png))
    n_rows, n_cols = formatted_df.shape
    fig_width = max(8.0, min(26.0, n_cols * 1.8))
    fig_height = max(2.5, min(34.0, n_rows * 0.38 + 1.4))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    table = ax.table(
        cellText=formatted_df.values,
        colLabels=formatted_df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, 1.25)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row == 0:
            cell.set_facecolor("#f0f0f0")
            cell.set_text_props(weight="bold")

    fig.tight_layout()
    output = BytesIO()
    fig.savefig(output, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    output.seek(0)
    return output.getvalue()


def png_zip_download(png_files: dict) -> bytes:
    output = BytesIO()
    with ZipFile(output, "w") as zip_file:
        for file_name, png_bytes in png_files.items():
            zip_file.writestr(file_name, png_bytes)
    output.seek(0)
    return output.getvalue()


def add_bar_percent_labels(ax, bars, values, fontsize: int = 9):
    labels = [f"{v:.1f}%" if np.isfinite(v) else "" for v in values]
    ax.bar_label(bars, labels=labels, padding=3, fontsize=fontsize)


def set_positive_ylim(ax, values, pad: float = 1.15, min_upper: float = 1.0):
    finite = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
    if len(finite) == 0:
        ax.set_ylim(0, min_upper)
        return
    ax.set_ylim(0, max(float(finite.max()) * pad, min_upper))


# -----------------------------------------------------------------------------
# Definitions
# -----------------------------------------------------------------------------

GLOSSARY_ROWS = [
    {"Term": "Direct routing capacity", "Definition": "Network-liquidity capacity based on E[k]. This measure defines liquidity-risk events.", "Formula / implementation": "direct_liquidity = baseline_available * direct_lm"},
    {"Term": "Indirect routing capacity", "Definition": "Network-liquidity capacity based on E[k^2] / E[k] - 1. It is shown for comparison only.", "Formula / implementation": "indirect_liquidity = baseline_available * indirect_lm"},
    {"Term": "Liquidity-risk day", "Definition": "A scenario-day on which direct routing capacity falls below the liquidity-risk threshold.", "Formula / implementation": "direct_liquidity < risk_threshold"},
    {"Term": "Liquidity-risk scenario", "Definition": "A simulated path with at least one liquidity-risk day.", "Formula / implementation": "risk.any(axis=1)"},
    {"Term": "Fixed EWI lead time", "Definition": "Number of trading days before a liquidity-risk event at which a true EWI signal is placed.", "Formula / implementation": "signal_day = event_day - ewi_lead_time"},
    {"Term": "Target recall", "Definition": "Intended percentage of evaluable liquidity-risk event days that receive a true EWI signal.", "Formula / implementation": "target_recall * evaluable_event_days"},
    {"Term": "Target precision", "Definition": "Intended percentage of EWI signal days that should be true positives.", "Formula / implementation": "true_positive_signals / total_signals"},
    {"Term": "Policy support intensity", "Definition": "Realized support relative to baseline available capacity across all scenario-days.", "Formula / implementation": "sum of support / (baseline_available * scenario_days)"},
    {"Term": "Risk-day reduction", "Definition": "Percentage reduction in liquidity-risk days relative to baseline no support.", "Formula / implementation": "(baseline_risk_days - policy_risk_days) / baseline_risk_days * 100"},
    {"Term": "Shortfall reduction", "Definition": "Percentage reduction in cumulative routing shortfall relative to baseline no support.", "Formula / implementation": "(baseline_shortfall - policy_shortfall) / baseline_shortfall * 100"},
    {"Term": "Mitigation efficiency", "Definition": "Shortfall reduction per percentage point of realized support intensity.", "Formula / implementation": "shortfall_reduction_pct / support_intensity_pct"},
    {"Term": "EWI targeting value-added", "Definition": "Difference between EWI-triggered risk-day reduction and untargeted risk-day reduction at the same support intensity.", "Formula / implementation": "EWI risk-day reduction - random benchmark risk-day reduction"},
]


def glossary_dataframe() -> pd.DataFrame:
    return pd.DataFrame(GLOSSARY_ROWS)


def render_glossary_cards(df: pd.DataFrame):
    st.markdown("#### Search definitions")
    search = st.text_input(
        "Search by term or keyword",
        value="",
        placeholder="For example: recall, precision, value-added, support",
        label_visibility="collapsed",
    )

    display_df = df[["Term", "Definition", "Formula / implementation"]].fillna("")
    if search.strip():
        query = search.strip().lower()
        mask = (
            display_df["Term"].str.lower().str.contains(query, regex=False)
            | display_df["Definition"].str.lower().str.contains(query, regex=False)
            | display_df["Formula / implementation"].str.lower().str.contains(query, regex=False)
        )
        display_df = display_df[mask]

    st.caption(f"Showing {len(display_df)} definition(s).")
    for _, row in display_df.iterrows():
        with st.expander(str(row["Term"]), expanded=False):
            st.markdown(f"**Definition**  \n{row['Definition']}")
            st.markdown("**Formula / implementation**")
            st.code(str(row["Formula / implementation"]), language="python")


# -----------------------------------------------------------------------------
# Streamlit app
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Liquidity-risk routing capacity simulation engine", layout="wide")
st.title("Liquidity-risk routing capacity simulation engine")
st.caption(
    "Interactive Chapter 8 simulation: topology-exponent paths, network-liquidity "
    "routing capacity, fixed-lead EWI signals, and routing-capacity policy support."
)

with st.expander("Model definitions and timing interpretation", expanded=False):
    st.markdown(
        """
        This app separates the **warning system** from the **policy response**.

        - The **EWI lead time** determines when a valid warning signal is placed before a liquidity-risk event.
        - The **support-start delay** determines when policy support becomes active after an EWI signal.
        - A **liquidity-risk day** is defined using **direct network-liquidity routing capacity**.
        - **Recall** is event-based: it measures how many event days are detected.
        - **Precision** is signal-based: it measures how many EWI signals are true positives.
        - **EWI targeting value-added** compares EWI-triggered support with untargeted support of the same volume.
        """
    )
    st.dataframe(glossary_dataframe(), use_container_width=True, hide_index=True)


# Sidebar
with st.sidebar:
    st.header("1. Network simulation settings")
    n_nodes = st.slider("Network size", 10, 150, 24, 1)
    scenarios = st.number_input("Simulation scenarios", 100, 10000, 1000, 100)
    trading_days = st.number_input("Trading days", 50, 1000, 200, 10)
    seed = st.number_input("Random seed", 1, 999999, 42, 1)
    investment = st.number_input("Investment base", 1.0, 10000.0, INVESTMENT_DEFAULT, 10.0)
    buffer_normal_pct = st.slider("Normal buffer / unavailable liquidity (%)", 0.0, 90.0, BUFFER_NORMAL_DEFAULT, 10.0)
    liquidity_risk_q = st.slider("Liquidity-risk threshold quantile", 0.001, 0.100, 0.025, 0.001, format="%.3f")

    st.header("2. EWI and policy settings")
    target_recall = st.slider("Target EWI recall", 0.5, 1.0, 0.70, 0.05)
    target_precision = st.slider("Target EWI precision", 0.05, 1.0, 0.25, 0.05)
    ewi_lead_time = st.slider("Fixed EWI lead time before event, trading days", 1, 30, 5, 1)
    support_days = st.slider("Support duration after activation", 1, 60, 10, 1)
    support_start_delay = st.slider("Support start delay after EWI, trading days", 1, 10, 5, 1)
    support_pct = st.slider("Additional routing-capacity support (%)", 0.0, 100.0, 10.0, 1.0)

    st.header("3. Distribution settings")
    lognorm_sigma = st.number_input("Lognormal sigma", 0.01, 5.0, FALLBACK_LOGNORM_PARAMS[0], 0.01)
    lognorm_loc = st.number_input("Location shift", 0.0, 10.0, FALLBACK_LOGNORM_PARAMS[1], 0.01)
    lognorm_scale = st.number_input("Lognormal scale", 0.01, 10.0, FALLBACK_LOGNORM_PARAMS[2], 0.01)

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
    ewi_lead_time=int(ewi_lead_time),
    support_days=int(support_days),
    support_start_delay=int(support_start_delay),
    support_pct=float(support_pct),
    lognorm_sigma=float(lognorm_sigma),
    lognorm_loc=float(lognorm_loc),
    lognorm_scale=float(lognorm_scale),
)

if cfg.support_start_delay > cfg.ewi_lead_time:
    st.warning("Timing note: support starts after the expected liquidity-risk event.")
elif cfg.support_start_delay == cfg.ewi_lead_time:
    st.caption("Timing note: support starts on the expected liquidity-risk event day.")
else:
    st.caption("Timing note: support starts before the expected liquidity-risk event day.")

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
    base_metrics = evaluate_liquidity(sim["direct_liquidity"], sim["risk_threshold"])
    design = build_nested_ewi_design(sim["event_day"], cfg.ewi_lead_time, cfg.seed)
    ewi_flags, ewi_diag = create_nested_ewi(
        sim["event_day"], design, cfg.target_recall, cfg.target_precision, cfg.ewi_lead_time
    )
    active_mask = forward_active_mask(ewi_flags, cfg.support_days, cfg.support_start_delay)
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
col2.metric("Baseline routing risk-day rate", f"{base_metrics['risk_day_rate']:.2f}%")
col3.metric("Policy risk-day rate", f"{policy_metrics['risk_day_rate']:.2f}%")
col4.metric("Routing shortfall reduction", f"{policy_extra['shortfall_reduction_pct']:.2f}%")

simulation_summary = pd.DataFrame(
    [
        ["Network size", cfg.n_nodes],
        ["Simulation scenarios", cfg.scenarios],
        ["Trading days per scenario", cfg.trading_days],
        ["Total scenario-days", cfg.scenarios * cfg.trading_days],
        ["Investment base", cfg.investment],
        ["Normal buffer / unavailable liquidity (%)", cfg.buffer_normal_pct],
        ["Target recall (%)", cfg.target_recall * 100],
        ["Target precision (%)", cfg.target_precision * 100],
        ["Liquidity-risk threshold quantile", cfg.liquidity_risk_q],
        ["Liquidity-risk threshold", round(sim["risk_threshold"], 2)],
        ["Liquidity-risk days", base_metrics["risk_days"]],
        ["Liquidity-risk day rate (%)", round(base_metrics["risk_day_rate"], 2)],
        ["Liquidity-risk scenarios", base_metrics["risk_scenarios"]],
        ["Liquidity-risk scenario rate (%)", round(base_metrics["risk_scenario_rate"], 2)],
        ["Fixed EWI lead time before event", cfg.ewi_lead_time],
        ["Support duration after activation", cfg.support_days],
        ["Support start delay after EWI", cfg.support_start_delay],
        ["Support setting (%)", cfg.support_pct],
        ["Lognormal sigma", cfg.lognorm_sigma],
        ["Lognormal location shift", cfg.lognorm_loc],
        ["Lognormal scale", cfg.lognorm_scale],
    ],
    columns=["Metric", "Value"],
)

ewi_summary = pd.DataFrame([[k.replace("_", " ").title(), v] for k, v in ewi_diag.items()], columns=["Metric", "Value"])

policy_summary = pd.DataFrame(
    [
        {
            "Policy": "Baseline no support",
            "Liquidity-risk scenario rate (%)": round(base_metrics["risk_scenario_rate"], 2),
            "Liquidity-risk day rate (%)": round(base_metrics["risk_day_rate"], 2),
            "Total routing shortfall": round(base_metrics["total_shortfall"], 0),
            "Risk-day reduction (%)": 0.0,
            "Shortfall reduction (%)": 0.0,
        },
        {
            "Policy": "Routing-capacity support",
            "Liquidity-risk scenario rate (%)": round(policy_metrics["risk_scenario_rate"], 2),
            "Liquidity-risk day rate (%)": round(policy_metrics["risk_day_rate"], 2),
            "Total routing shortfall": round(policy_metrics["total_shortfall"], 0),
            "Risk-day reduction (%)": round(policy_extra["risk_reduction_pct"], 2),
            "Shortfall reduction (%)": round(policy_extra["shortfall_reduction_pct"], 2),
        },
    ]
)

fig_tab, policy_tab, sensitivity_tab, network_tab, ewi_tab, definitions_tab, downloads_tab = st.tabs(
    [
        "Liquidity routing paths",
        "Mitigation result",
        "Routing support impact",
        "Impact of network size",
        "EWI quality",
        "Definitions",
        "Downloads",
    ]
)

# Liquidity paths
with fig_tab:
    st.subheader("Direct and indirect network-liquidity routing capacity")
    x = np.arange(1, cfg.trading_days + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    for ax, arr, title, color in [
        (axes[0], sim["direct_liquidity"], "A. Direct network-liquidity routing capacity", "darkblue"),
        (axes[1], sim["indirect_liquidity"], "B. Indirect network-liquidity routing capacity", "darkred"),
    ]:
        mean = np.nanmean(arr, axis=0)
        q10 = np.nanquantile(arr, 0.10, axis=0)
        q90 = np.nanquantile(arr, 0.90, axis=0)
        q025 = np.nanquantile(arr, 0.025, axis=0)
        q975 = np.nanquantile(arr, 0.975, axis=0)

        ax.plot(x, mean, color=color, linewidth=2.0, label="Average capacity")
        ax.fill_between(x, q10, q90, color=color, alpha=0.18, label="10-90% range")
        ax.fill_between(x, q025, q975, color=color, alpha=0.08, label="2.5-97.5% range")
        
        y_max = max(np.nanmax(sim["direct_liquidity"]), np.nanmax(sim["indirect_liquidity"]))
        y_upper = math.ceil((y_max * 1.05) / 100) * 100
        min_y_upper = math.ceil((cfg.investment * 1.05) / 100) * 100
        ax.set_ylim(cfg.investment, max(y_upper, min_y_upper))
        ax.set_xlim(1, cfg.trading_days)
        ax.set_title(title)
        ax.set_xlabel("Trading day")
        ax.grid(True, linestyle="--", alpha=0.35)

    axes[0].set_ylabel("Direct routing capacity")
    axes[1].set_ylabel("Indirect routing capacity")
    axes[0].axhline(sim["risk_threshold"], color="red", linestyle="--", linewidth=1.0, label="Direct-capacity risk threshold")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[1].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.caption("Note: the y-axis starts at the selected investment base.")
    liquidity_paths_png = fig_to_png_bytes(fig)

# Mitigation result
with policy_tab:
    st.subheader("Mitigation impact comparison")
    plot_df = policy_summary.copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    colors = ["#1f77b4", "#ff7f0e"]

    bars_a = axes[0].bar(plot_df["Policy"], plot_df["Liquidity-risk scenario rate (%)"], color=colors, edgecolor="black", linewidth=0.5)
    add_bar_percent_labels(axes[0], bars_a, plot_df["Liquidity-risk scenario rate (%)"])
    axes[0].set_title("A. Routing risk scenario rate")
    axes[0].set_ylabel("Liquidity-risk scenario rate (%)")
    set_positive_ylim(axes[0], plot_df["Liquidity-risk scenario rate (%)"])

    bars_b = axes[1].bar(plot_df["Policy"], plot_df["Liquidity-risk day rate (%)"], color=colors, edgecolor="black", linewidth=0.5)
    add_bar_percent_labels(axes[1], bars_b, plot_df["Liquidity-risk day rate (%)"])
    axes[1].set_title("B. Routing risk-day rate")
    axes[1].set_ylabel("% of trading days")
    set_positive_ylim(axes[1], plot_df["Liquidity-risk day rate (%)"])

    denom = plot_df.loc[0, "Total routing shortfall"]
    shortfall_pct = plot_df["Total routing shortfall"] / denom * 100 if denom else pd.Series(np.nan, index=plot_df.index)
    bars_c = axes[2].bar(plot_df["Policy"], shortfall_pct, color=colors, edgecolor="black", linewidth=0.5)
    add_bar_percent_labels(axes[2], bars_c, shortfall_pct)
    axes[2].axhline(100, color="grey", linestyle="--", linewidth=1)
    axes[2].set_title("C. Cumulative routing shortfall")
    axes[2].set_ylabel("% of baseline no support")
    set_positive_ylim(axes[2], shortfall_pct)

    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.dataframe(policy_summary.round(2), use_container_width=True)
    policy_comparison_png = fig_to_png_bytes(fig)
    policy_summary_png = dataframe_to_png_bytes(policy_summary.round(2), title="Policy summary")

# Support sensitivity
with sensitivity_tab:
    st.subheader("Support sensitivity")
    support_grid = list(range(0, 45, 5))
    rows = []

    for sp in support_grid:
        if sp == 0:
            rows.append({"Support setting (%)": sp, "Policy support intensity (%)": 0.0, "Risk-day reduction (%)": 0.0, "Shortfall reduction (%)": 0.0, "Mitigation efficiency": np.nan})
        else:
            _, _, extra = run_support_policy(sim["direct_lm"], base_metrics, sim["risk_threshold"], active_mask, cfg.investment, cfg.buffer_normal_pct, float(sp))
            rows.append({"Support setting (%)": sp, "Policy support intensity (%)": extra["support_intensity_pct"], "Risk-day reduction (%)": extra["risk_reduction_pct"], "Shortfall reduction (%)": extra["shortfall_reduction_pct"], "Mitigation efficiency": extra["mitigation_efficiency"]})

    support_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.plot(support_df["Support setting (%)"], support_df["Risk-day reduction (%)"], marker="o", label="Risk-day reduction")
    ax.plot(support_df["Support setting (%)"], support_df["Shortfall reduction (%)"], marker="s", label="Shortfall reduction")
    ax.set_xlabel("Additional routing-capacity support (%)")
    ax.set_ylabel("Reduction relative to no support (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(0, max(support_grid))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.dataframe(support_df.round(2), use_container_width=True)
    support_sensitivity_png = fig_to_png_bytes(fig)
    support_sensitivity_table_png = dataframe_to_png_bytes(support_df.round(2), title="Support sensitivity")

# Network-size efficiency
with network_tab:
    st.subheader("Mitigation efficiency over network size")
    st.caption("Uses current EWI timing and support settings across a compact network-size grid.")
    network_grid = list(range(20, 101, 10))
    support_levels = [10, 20, 30, 40]
    rows = []

    for n in network_grid:
        sim_n = simulate_base_paths(int(n), cfg.scenarios, cfg.trading_days, cfg.investment, cfg.buffer_normal_pct, cfg.liquidity_risk_q, cfg.seed, cfg.lognorm_sigma, cfg.lognorm_loc, cfg.lognorm_scale)
        bm_n = evaluate_liquidity(sim_n["direct_liquidity"], sim_n["risk_threshold"])
        design_n = build_nested_ewi_design(sim_n["event_day"], cfg.ewi_lead_time, cfg.seed)
        ewi_n, _ = create_nested_ewi(sim_n["event_day"], design_n, cfg.target_recall, cfg.target_precision, cfg.ewi_lead_time)
        active_n = forward_active_mask(ewi_n, cfg.support_days, cfg.support_start_delay)

        for sp in support_levels:
            _, _, extra = run_support_policy(sim_n["direct_lm"], bm_n, sim_n["risk_threshold"], active_n, cfg.investment, cfg.buffer_normal_pct, float(sp))
            rows.append({"Network size": n, "Support setting (%)": sp, "Policy support intensity (%)": extra["support_intensity_pct"], "Shortfall reduction (%)": extra["shortfall_reduction_pct"], "Mitigation efficiency": extra["mitigation_efficiency"]})

    network_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for sp in support_levels:
        sub = network_df[network_df["Support setting (%)"] == sp]
        ax.plot(sub["Network size"], sub["Mitigation efficiency"], marker="o", label=f"{sp}% support")
    ax.set_xlabel("Network size")
    ax.set_ylabel("Routing shortfall reduction per unit support")
    ax.set_xlim(min(network_grid), max(network_grid))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.dataframe(network_df.round(2), use_container_width=True)
    network_efficiency_png = fig_to_png_bytes(fig)
    network_efficiency_table_png = dataframe_to_png_bytes(network_df.round(2), title="Network-size efficiency")

# EWI targeting value-added
with ewi_tab:
    st.subheader("EWI targeting value-added")
    st.caption(
        "This figure isolates the informational value of the EWI. For each EWI-quality setting, "
        "the EWI-triggered policy is compared with an untargeted benchmark that activates the same "
        "number of support days on randomly selected scenario-days. Positive values mean that EWI timing "
        "improves risk-day reduction beyond what would be achieved by the same support volume without targeting."
    )

    recall_grid = [0.60, 0.70, 0.80]
    precision_grid = np.arange(0.15, 0.81, 0.05)
    value_rows = []

    for rec in recall_grid:
        for prec in precision_grid:
            ewi_h, diag_h = create_nested_ewi(sim["event_day"], design, rec, prec, cfg.ewi_lead_time)
            active_h = forward_active_mask(ewi_h, cfg.support_days, cfg.support_start_delay)
            _, _, extra_h = run_support_policy(sim["direct_lm"], base_metrics, sim["risk_threshold"], active_h, cfg.investment, cfg.buffer_normal_pct, 10.0)

            random_active_h = random_active_mask_same_intensity(
                active_h,
                seed=cfg.seed + int(rec * 1000) + int(prec * 10000),
            )
            _, _, random_extra_h = run_support_policy(sim["direct_lm"], base_metrics, sim["risk_threshold"], random_active_h, cfg.investment, cfg.buffer_normal_pct, 10.0)

            value_rows.append(
                {
                    "Target recall (%)": rec * 100,
                    "Target precision (%)": prec * 100,
                    "EWI support intensity (%)": extra_h["support_intensity_pct"],
                    "Random support intensity (%)": random_extra_h["support_intensity_pct"],
                    "EWI risk-day reduction (%)": round(extra_h["risk_reduction_pct"], 2),
                    "Random risk-day reduction (%)": round(random_extra_h["risk_reduction_pct"], 2),
                    "EWI value-added risk-day reduction (pp)": round(extra_h["risk_reduction_pct"] - random_extra_h["risk_reduction_pct"], 2),
                    "EWI shortfall reduction (%)": round(extra_h["shortfall_reduction_pct"], 2),
                    "Random shortfall reduction (%)": round(random_extra_h["shortfall_reduction_pct"], 2),
                    "EWI value-added shortfall reduction (pp)": round(extra_h["shortfall_reduction_pct"] - random_extra_h["shortfall_reduction_pct"], 2),
                    "EWI mitigation efficiency": round(extra_h["mitigation_efficiency"], 2),
                    "Random mitigation efficiency": round(random_extra_h["mitigation_efficiency"], 2),
                    "Achieved recall (%)": round(diag_h["achieved_recall"], 2),
                    "Achieved precision (%)": round(diag_h["achieved_precision"], 2),
                    "Signal days": round(diag_h["signal_days"], 0),
                    "False-positive signal days": round(diag_h["false_positive_signal_days"], 0),
                }
            )

    ewi_quality_df = pd.DataFrame(value_rows)

    fig_ewi, ax_ewi = plt.subplots(figsize=(8.8, 5.4))
    for rec in recall_grid:
        sub = ewi_quality_df[np.isclose(ewi_quality_df["Target recall (%)"], rec * 100, atol=1e-10)].sort_values("Target precision (%)")
        ax_ewi.plot(sub["Target precision (%)"], sub["EWI value-added risk-day reduction (pp)"], marker="o", linewidth=2, label=f"Recall {rec * 100:.0f}%")

    ax_ewi.axhline(0, color="black", linestyle="--", linewidth=1)
    ax_ewi.set_title("EWI targeting value-added under fixed 10% support")
    ax_ewi.set_xlabel("Target precision (%)")
    ax_ewi.set_ylabel("Value-added risk-day reduction (percentage points)")
    ax_ewi.set_xlim(precision_grid.min() * 100, precision_grid.max() * 100)

    y_min = ewi_quality_df["EWI value-added risk-day reduction (pp)"].min()
    y_max = ewi_quality_df["EWI value-added risk-day reduction (pp)"].max()
    y_padding = max((y_max - y_min) * 0.15, 1.0)
    ax_ewi.set_ylim(y_min - y_padding, y_max + y_padding)
    ax_ewi.grid(True, linestyle="--", alpha=0.35)
    ax_ewi.legend(title="Target recall", fontsize=8, title_fontsize=9)
    fig_ewi.tight_layout()

    st.pyplot(fig_ewi, use_container_width=True)
    st.caption(
        "The plotted value is the difference between EWI-triggered risk-day reduction and the risk-day reduction "
        "from an untargeted benchmark with the same realized support intensity. Values above zero indicate that "
        "the EWI improves policy targeting relative to untargeted liquidity support."
    )
    st.dataframe(ewi_quality_df.round(3), use_container_width=True)

    ewi_quality_map_png = fig_to_png_bytes(fig_ewi)
    ewi_quality_table_png = dataframe_to_png_bytes(ewi_quality_df.round(3), title="EWI targeting value-added")

    st.subheader("Current EWI diagnostics")
    st.dataframe(ewi_summary.round(2), use_container_width=True)
    ewi_summary_png = dataframe_to_png_bytes(ewi_summary.round(2), title="Current EWI diagnostics")

# Definitions
with definitions_tab:
    st.subheader("Model definitions")
    st.markdown("This glossary defines the main terms used in the simulation outputs.")
    glossary_df = glossary_dataframe()
    render_glossary_cards(glossary_df)
    glossary_png = dataframe_to_png_bytes(glossary_df, title="Model definitions", font_size=7)

# Downloads
with downloads_tab:
    st.subheader("Download current outputs")
    all_outputs = {
        "Simulation summary": simulation_summary,
        "EWI summary": ewi_summary,
        "Policy summary": policy_summary,
        "Support sensitivity": support_df,
        "Network efficiency": network_df,
        "EWI targeting value-added": ewi_quality_df,
        "Definitions": glossary_dataframe(),
    }

    excel_bytes = to_excel_download(all_outputs)
    st.download_button(
        "Download summary workbook (.xlsx)",
        data=excel_bytes,
        file_name="streamlit_simulation_engine_summary.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    csv_outputs = {
        "policy_summary.csv": policy_summary,
        "simulation_summary.csv": simulation_summary,
        "support_sensitivity.csv": support_df,
        "network_size_efficiency.csv": network_df,
        "ewi_targeting_value_added.csv": ewi_quality_df,
        "model_definitions.csv": glossary_dataframe(),
    }

    for file_name, df in csv_outputs.items():
        st.download_button(f"Download {file_name}", data=df.to_csv(index=False).encode("utf-8"), file_name=file_name, mime="text/csv")

    st.divider()
    st.subheader("Download figures and tables as PNG")

    png_files = {
        "liquidity_paths.png": liquidity_paths_png,
        "policy_comparison.png": policy_comparison_png,
        "policy_summary_table.png": policy_summary_png,
        "support_sensitivity.png": support_sensitivity_png,
        "support_sensitivity_table.png": support_sensitivity_table_png,
        "network_size_efficiency.png": network_efficiency_png,
        "network_size_efficiency_table.png": network_efficiency_table_png,
        "ewi_targeting_value_added.png": ewi_quality_map_png,
        "ewi_targeting_value_added_table.png": ewi_quality_table_png,
        "ewi_summary_table.png": ewi_summary_png,
        "model_definitions_table.png": glossary_png,
    }

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.download_button("Download liquidity paths PNG", data=liquidity_paths_png, file_name="liquidity_paths.png", mime="image/png")
        st.download_button("Download policy comparison PNG", data=policy_comparison_png, file_name="policy_comparison.png", mime="image/png")
        st.download_button("Download support sensitivity PNG", data=support_sensitivity_png, file_name="support_sensitivity.png", mime="image/png")
    with col_b:
        st.download_button("Download network efficiency PNG", data=network_efficiency_png, file_name="network_size_efficiency.png", mime="image/png")
        st.download_button("Download policy summary table PNG", data=policy_summary_png, file_name="policy_summary_table.png", mime="image/png")
        st.download_button("Download support table PNG", data=support_sensitivity_table_png, file_name="support_sensitivity_table.png", mime="image/png")
    with col_c:
        st.download_button("Download network table PNG", data=network_efficiency_table_png, file_name="network_size_efficiency_table.png", mime="image/png")
        st.download_button("Download EWI targeting value-added PNG", data=ewi_quality_map_png, file_name="ewi_targeting_value_added.png", mime="image/png")
        st.download_button("Download EWI targeting value-added table PNG", data=ewi_quality_table_png, file_name="ewi_targeting_value_added_table.png", mime="image/png")
        st.download_button("Download definitions table PNG", data=glossary_png, file_name="model_definitions_table.png", mime="image/png")

    all_png_zip = png_zip_download(png_files)
    st.download_button(
        "Download all PNG outputs as ZIP",
        data=all_png_zip,
        file_name="streamlit_simulation_engine_png_outputs.zip",
        mime="application/zip",
    )

st.info(
    "Interpretation note: the EWI signal is implemented as a fixed-lead signal "
    f"{cfg.ewi_lead_time} trading day(s) before a direct-capacity liquidity-risk event. "
    "Policy support starts "
    f"{cfg.support_start_delay} trading day(s) after the signal and remains active "
    f"for {cfg.support_days} trading day(s). The EWI value-added graph compares EWI-triggered "
    "support with untargeted support of the same realized volume."
)
