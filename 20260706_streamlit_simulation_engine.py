"""
Streamlit simulation engine for Chapter 8 routing liquidity-risk simulations
====================================================================

This app consolidates the logic from the Chapter 8 simulation scripts:
- lognormal fallback parameters for topology-exponent paths;
- direct and indirect network-liquidity routing-capacity simulation;
- estimated-EWI emulator with target recall and precision;
- routing-capacity policy support after EWI signals;
- support-sensitivity, network-size efficiency, and EWI-quality tables.

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
# Defaults calibrated to the Chapter 8 scripts
# -----------------------------------------------------------------------------

FALLBACK_LOGNORM_PARAMS = (0.3191, 0.0, 1.0899)
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
# Liquidity-multiplier and simulation functions
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def make_liquidity_multiplier_grids(n_nodes: int, grid_size: int = 5000):
    """
    Create interpolation grids for direct and indirect liquidity multipliers.

    Direct multiplier:
        E[k]

    Indirect multiplier:
        E[k^2] / E[k] - 1

    The variable `gamma` is the topology-exponent parameter used to weight
    the degree distribution.
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
    Simulate lognormal topology-exponent paths and direct/indirect
    network-liquidity routing-capacity paths.

    Liquidity-risk events are defined using direct routing capacity only.
    The indirect series is shown for comparison but does not trigger the
    risk indicator in this version.
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


# -----------------------------------------------------------------------------
# EWI design and diagnostics
# -----------------------------------------------------------------------------

def future_event_mask(event_day: np.ndarray, lead_time: int) -> np.ndarray:
    """
    Mark days that are exactly `lead_time` trading days before a liquidity-risk day.

    If event_day[s, t] is True and lead_time = 5, then the pre-event mask is
    True at day t - 5, provided that t - 5 is within the sample.

    This implements the Chapter 7 fixed-lead-time interpretation.
    """
    out = np.zeros_like(event_day, dtype=bool)

    if 0 < lead_time < event_day.shape[1]:
        out[:, :-lead_time] = event_day[:, lead_time:]

    return out


def detected_event_mask(
    event_day: np.ndarray,
    ewi: np.ndarray,
    lead_time: int,
) -> np.ndarray:
    """
    Mark event days that were preceded by an EWI signal exactly `lead_time`
    trading days earlier.

    If an EWI signal occurs on day t and lead_time = X, then day t + X is
    counted as detected if it is a liquidity-risk day.

    The detection is stored on the event day, not on the signal day.
    """
    detected = np.zeros_like(event_day, dtype=bool)

    if 0 < lead_time < event_day.shape[1]:
        detected[:, lead_time:] = (
            event_day[:, lead_time:] & ewi[:, :-lead_time]
        )

    return detected


def build_nested_ewi_design(
    event_day: np.ndarray,
    lead_time: int = 5,
    seed: int = 42,
):
    """
    Create EWI scenario-design positions.

    True EWI signals are placed exactly `lead_time` trading days before
    evaluable liquidity-risk events. False-positive candidates are days that
    are not liquidity-risk days and are not exactly lead_time days before a
    liquidity-risk event.

    True EWI signals are only allowed on non-risk days.
    """
    rng = np.random.default_rng(seed)

    evaluable = event_day.copy()
    evaluable[:, :lead_time] = False

    event_positions = np.argwhere(evaluable)

    if len(event_positions) > 0:
        event_positions = event_positions[rng.permutation(len(event_positions))]

    exact_pre_event = future_event_mask(event_day, lead_time)

    false_positive_candidates = np.argwhere(~event_day & ~exact_pre_event)

    if len(false_positive_candidates) > 0:
        false_positive_candidates = false_positive_candidates[
            rng.permutation(len(false_positive_candidates))
        ]

    true_signal_positions = []
    signalable_event_count = 0

    for s, t in event_positions:
        signal_t = int(t - lead_time)

        if signal_t >= 0 and not event_day[s, signal_t]:
            signalable_event_count += 1
            true_signal_positions.append((int(s), signal_t))

    # Deduplicate true-signal positions while preserving order.
    # This matters when clustered events map to the same signal day.
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
        "false_positive_positions": [
            (int(s), int(t)) for s, t in false_positive_candidates
        ],
    }


def create_nested_ewi(
    event_day: np.ndarray,
    design: dict,
    target_recall: float,
    target_precision: float,
    lead_time: int = 5,
):
    """
    Create an estimated-EWI signal process matching target recall and precision
    as closely as possible.

    Under the fixed-lead-time interpretation, a signal is a true positive only
    if a liquidity-risk event occurs exactly `lead_time` trading days later.
    """
    ewi = np.zeros_like(event_day, dtype=bool)

    n_events = design["n_evaluable_events"]
    n_signalable_events = design.get("n_signalable_events", n_events)

    true_positions = design["true_signal_positions"]
    fp_positions = design["false_positive_positions"]

    n_true_target = int(round(target_recall * n_events)) if n_events else 0
    n_true = min(n_true_target, len(true_positions))

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

    # Signal-level precision:
    # a signal is true positive only if a liquidity-risk event occurs exactly
    # lead_time trading days later.
    tp_signal = np.zeros_like(event_day, dtype=bool)

    if 0 < lead_time < event_day.shape[1]:
        tp_signal[:, :-lead_time] = (
            ewi[:, :-lead_time] & event_day[:, lead_time:]
        )

    signal_days = int(ewi.sum())
    true_positive_signal_days = int(tp_signal.sum())
    false_positive_signal_days = signal_days - true_positive_signal_days

    # Event-level recall:
    # count actual event days detected by a signal exactly lead_time days earlier.
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
        "achieved_recall": (
            detected_event_days / n_events * 100
            if n_events
            else np.nan
        ),
        "achieved_recall_signalable_events": (
            detected_event_days / n_signalable_events * 100
            if n_signalable_events
            else np.nan
        ),
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


# -----------------------------------------------------------------------------
# Policy support and liquidity evaluation
# -----------------------------------------------------------------------------

def forward_active_mask(
    flags: np.ndarray,
    duration: int,
    start_delay: int,
):
    """
    Activate policy support after an EWI signal for a fixed duration.

    If an EWI signal occurs on day t, support starts on day:

        t + start_delay

    and remains active for `duration` trading days.

    Examples:
    - start_delay = 1 means support starts the next trading day.
    - start_delay = 5 means support starts five trading days after the signal.
    """
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
    """
    Apply additional routing-capacity support and calculate policy metrics.

    The policy is evaluated on direct routing capacity, consistent with the
    risk-event definition.
    """
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


# -----------------------------------------------------------------------------
# Download helpers
# -----------------------------------------------------------------------------

def to_excel_download(sheets: dict) -> bytes:
    """Convert a dictionary of DataFrames to an Excel workbook in memory."""
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            safe_name = sheet_name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)

    return output.getvalue()


def fig_to_png_bytes(fig, dpi: int = 300) -> bytes:
    """Convert a Matplotlib figure to PNG bytes."""
    output = BytesIO()

    fig.savefig(
        output,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )

    output.seek(0)
    return output.getvalue()


def format_value_for_table_png(value):
    """Format table values for PNG export."""
    if pd.isna(value):
        return ""

    if isinstance(value, (int, np.integer)):
        return str(value)

    if isinstance(value, (float, np.floating)):
        return f"{value:.3f}"

    return str(value)


def dataframe_to_png_bytes(
    df: pd.DataFrame,
    title: str,
    dpi: int = 300,
    font_size: int = 8,
) -> bytes:
    """Convert a DataFrame to a PNG table image."""
    table_df = df.copy()

    formatted_df = table_df.astype(object).apply(
        lambda col: col.map(format_value_for_table_png)
    )

    n_rows, n_cols = formatted_df.shape

    fig_width = max(8.0, min(26.0, n_cols * 1.8))
    fig_height = max(2.5, min(34.0, n_rows * 0.38 + 1.4))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    ax.set_title(
        title,
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

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

    fig.savefig(
        output,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    output.seek(0)
    return output.getvalue()


def png_zip_download(png_files: dict) -> bytes:
    """Bundle PNG files into a single ZIP archive."""
    output = BytesIO()

    with ZipFile(output, "w") as zip_file:
        for file_name, png_bytes in png_files.items():
            zip_file.writestr(file_name, png_bytes)

    output.seek(0)
    return output.getvalue()


# -----------------------------------------------------------------------------
# Model definitions and interpretation
# -----------------------------------------------------------------------------

GLOSSARY_ROWS = [
    {
        "Term": "Investment base",
        "Definition": "Reference amount used to scale available routing capacity and policy support.",
        "Formula / implementation": "investment",
    },
    {
        "Term": "Baseline available liquidity",
        "Definition": "Percentage of the investment base that is (un)available under normal buffer conditions.",
        "Formula / implementation": "baseline_available = investment * (1 - buffer_normal_pct / 100)",
    },
    {
        "Term": "Direct routing capacity",
        "Definition": "Network-liquidity capacity based on the expected degree multiplier E[k]. This measure defines liquidity-risk events.",
        "Formula / implementation": "direct_liquidity = baseline_available * direct_lm",
    },
    {
        "Term": "Indirect routing capacity",
        "Definition": (
            "Network-liquidity capacity based on the indirect multiplier E[k^2] / E[k] - 1. "
            "It is shown for comparison but does not trigger liquidity-risk events in this version."
        ),
        "Formula / implementation": "indirect_liquidity = baseline_available * indirect_lm",
    },
    {
        "Term": "Liquidity-risk threshold",
        "Definition": "Lower-tail threshold used to identify direct-capacity liquidity-risk days.",
        "Formula / implementation": "nanquantile(direct_liquidity, liquidity_risk_q)",
    },
    {
        "Term": "Liquidity-risk day",
        "Definition": "A scenario-day on which direct routing capacity falls below the liquidity-risk threshold.",
        "Formula / implementation": "direct_liquidity < risk_threshold",
    },
    {
        "Term": "Liquidity-risk scenario",
        "Definition": "A simulated path with at least one liquidity-risk day.",
        "Formula / implementation": "risk.any(axis=1)",
    },
    {
        "Term": "Total routing shortfall",
        "Definition": "Cumulative amount by which direct routing capacity falls below the risk threshold.",
        "Formula / implementation": "sum(max(0, risk_threshold - direct_liquidity))",
    },
    {
        "Term": "Fixed EWI lead time",
        "Definition": (
            "Number of trading days before a liquidity-risk event at which a true EWI signal is placed. "
            "For example, with a lead time of 5, an event on day t has a signal on day t - 5."
        ),
        "Formula / implementation": "signal_day = event_day - ewi_lead_time",
    },
    {
        "Term": "Target recall",
        "Definition": "Intended percentage of evaluable liquidity-risk event days that receive a true EWI signal.",
        "Formula / implementation": "round(target_recall * evaluable_event_days)",
    },
    {
        "Term": "Achieved recall",
        "Definition": (
            "Actual percentage of evaluable event days detected by an EWI signal exactly "
            "the fixed lead time earlier."
        ),
        "Formula / implementation": "detected_event_days / evaluable_event_days",
    },
    {
        "Term": "Signalable event days",
        "Definition": (
            "Event days for which a valid non-risk signal day exists exactly the fixed lead time "
            "before the event."
        ),
        "Formula / implementation": "event day t is signalable if t - lead_time exists and is not already a risk day",
    },
    {
        "Term": "Achieved recall on signalable events",
        "Definition": (
            "Detected event days divided by the number of event days that could receive a clean "
            "fixed-lead signal."
        ),
        "Formula / implementation": "detected_event_days / signalable_event_days",
    },
    {
        "Term": "Target precision",
        "Definition": "Intended percentage of EWI signal days that should be true positives.",
        "Formula / implementation": "true_positive_signals / total_signals",
    },
    {
        "Term": "Achieved precision",
        "Definition": (
            "Actual percentage of EWI signals that are followed by a liquidity-risk event exactly "
            "the fixed lead time later."
        ),
        "Formula / implementation": "true_positive_signal_days / signal_days",
    },
    {
        "Term": "Signal-day rate",
        "Definition": "Share of all scenario-days on which an EWI signal is active.",
        "Formula / implementation": "signal_days / total_scenario_days",
    },
    {
        "Term": "False-positive rate",
        "Definition": (
            "Share of all scenario-days with an EWI signal that is not followed by a liquidity-risk "
            "event exactly the fixed lead time later."
        ),
        "Formula / implementation": "false_positive_signal_days / total_scenario_days",
    },
    {
        "Term": "Support setting",
        "Definition": "Additional routing-capacity support expressed as a percentage of the investment base.",
        "Formula / implementation": "support = investment * support_pct / 100",
    },
    {
        "Term": "Support-start delay",
        "Definition": "Number of trading days between an EWI signal and the activation of policy support.",
        "Formula / implementation": "support starts on signal_day + support_start_delay",
    },
    {
        "Term": "Support duration",
        "Definition": "Number of trading days for which support remains active after activation.",
        "Formula / implementation": "support remains active for support_days",
    },
    {
        "Term": "Policy support intensity",
        "Definition": (
            "Realized policy support relative to baseline available capacity across all scenario-days. "
            "It accounts for both the support size and how often support is active."
        ),
        "Formula / implementation": "sum of support / (baseline_available * number_of_scenario_days)",
    },
    {
{
    "Term": "Risk-day reduction",
    "Definition": (
        "Percentage reduction in the number of liquidity-risk days relative to the "
        "baseline no-support case. A positive value means that policy support reduces "
        "the number of risk days; a negative value means that risk days increase."
    ),
    "Formula / implementation": (
        "(baseline_risk_days - policy_risk_days) / baseline_risk_days * 100"
    ),
},
{
    "Term": "Shortfall reduction",
    "Definition": (
        "Percentage reduction in cumulative routing shortfall relative to the baseline "
        "no-support case. This captures how much the policy reduces the depth of "
        "liquidity stress, even when some risk days remain."
    ),
    "Formula / implementation": (
        "(baseline_shortfall - policy_shortfall) / baseline_shortfall * 100"
    ),
},
{
    "Term": "Mitigation efficiency",
    "Definition": (
        "Shortfall reduction achieved per percentage point of realized policy support "
        "intensity. Higher values indicate that a given amount of realized support "
        "produces a larger reduction in cumulative routing shortfall."
    ),
    "Formula / implementation": (
        "shortfall_reduction_pct / support_intensity_pct"
    ),
},
{
    "Term": "Evaluable event days",
    "Definition": (
        "Liquidity-risk event days that occur late enough in the simulated path "
        "to evaluate whether an EWI signal occurred exactly the fixed lead time earlier."
    ),
    "Formula / implementation": (
        "n/a"
    ),
},
{
    "Term": "Signalable event days",
    "Definition": (
        "Evaluable event days for which the fixed-lead signal day exists and is not "
        "itself already a liquidity-risk day. These are the events for which a clean "
        "early-warning signal can be placed."
    ),
    "Formula / implementation": (
        "n/a"
    ),
},
    },
]


def glossary_dataframe() -> pd.DataFrame:
    """Return a glossary dataframe for display and download."""
    return pd.DataFrame(GLOSSARY_ROWS)


def render_wrapped_glossary_table(df: pd.DataFrame, max_height: int = 560):
    """
    Render the glossary as a wrapped, screen-friendly HTML table.

    This is more readable than st.dataframe for long text columns because it
    wraps definitions and formulas instead of forcing horizontal scrolling.
    """
    display_df = df[["Term", "Definition", "Formula / implementation"]].fillna("")

    rows_html = []

    for _, row in display_df.iterrows():
        term = html.escape(str(row["Term"]))
        definition = html.escape(str(row["Definition"]))
        formula = html.escape(str(row["Formula / implementation"]))

        rows_html.append(
            f"""
            <tr>
                <td class="term-col">{term}</td>
                <td class="definition-col">{definition}</td>
                <td class="formula-col"><code>{formula}</code></td>
            </tr>
            """
        )

    table_html = "\n".join(rows_html)

    st.markdown(
        f"""
        <style>
            .glossary-wrapper {{
                max-height: {max_height}px;
                overflow-y: auto;
                border: 1px solid #e6e6e6;
                border-radius: 8px;
                margin-top: 0.5rem;
            }}

            .glossary-table {{
                width: 100%;
                border-collapse: collapse;
                table-layout: fixed;
                font-size: 0.88rem;
                line-height: 1.35;
            }}

            .glossary-table th {{
                position: sticky;
                top: 0;
                background-color: #f7f7f7;
                z-index: 1;
                text-align: left;
                border-bottom: 1px solid #d9d9d9;
                padding: 0.55rem;
            }}

            .glossary-table td {{
                vertical-align: top;
                border-bottom: 1px solid #eeeeee;
                padding: 0.55rem;
                white-space: normal;
                word-wrap: break-word;
                overflow-wrap: break-word;
            }}

            .term-col {{
                width: 22%;
                font-weight: 600;
            }}

            .definition-col {{
                width: 48%;
            }}

            .formula-col {{
                width: 30%;
                font-family: monospace;
                font-size: 0.82rem;
                background-color: #fbfbfb;
            }}

            .formula-col code {{
                white-space: normal;
                word-break: break-word;
            }}
        </style>

        <div class="glossary-wrapper">
            <table class="glossary-table">
                <thead>
                    <tr>
                        <th class="term-col">Term</th>
                        <th class="definition-col">Definition</th>
                        <th class="formula-col">Formula / implementation</th>
                    </tr>
                </thead>
                <tbody>
                    {table_html}
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_glossary_cards(df: pd.DataFrame):
    """
    Render definitions as searchable expanders.

    This is useful when the table becomes too dense on smaller screens.
    """
    st.markdown("#### Search definitions")

    search = st.text_input(
        "Search by term or keyword",
        value="",
        placeholder="For example: recall, precision, shortfall, support",
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

st.set_page_config(
    page_title="Liquidity-risk routing capacity simulation engine",
    layout="wide",
)

st.title("Liquidity-risk routing capacity simulation engine")

st.caption(
    "Interactive version of the Chapter 8 simulation: topology-exponent paths, "
    "network-liquidity routing capacity, fixed-lead EWI signals, and "
    "routing-capacity policy support."
)

with st.expander("Model definitions and timing interpretation", expanded=False):
    st.markdown(
        """
        This app separates the **warning system** from the **policy response**.

        - The **fixed EWI lead time** determines when a valid warning signal is placed before a liquidity-risk event.  
          For example, with a lead time of 5 trading days, an event on day *t* has a true signal on day *t - 5*.
        - The **support-start delay** determines when policy support becomes active after an EWI signal.  
          If the support-start delay is 3 trading days, support starts on day *t + 3*.
        - A **liquidity-risk day** is defined using **direct network-liquidity routing capacity**.  
          Indirect routing capacity is shown for comparison but does not trigger the risk indicator in this version.
        - **Recall** is event-based: it measures how many event days are detected.
        - **Precision** is signal-based: it measures how many EWI signals are true positives.
        - **Mitigation efficiency** measures how much cumulative shortfall reduction is achieved per unit of realized support intensity.
        """
    )

    st.dataframe(
        glossary_dataframe(),
        use_container_width=True,
        hide_index=True,
    )


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

with st.sidebar:
    st.header("1. Network simulation settings")

    n_nodes = st.slider(
        "Network size",
        min_value=10,
        max_value=150,
        value=24,
        step=1,
        help="Number of nodes in the simulated financial network.",
    )

    scenarios = st.number_input(
        "Simulation scenarios",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        help="Number of simulated paths.",
    )

    trading_days = st.number_input(
        "Trading days",
        min_value=50,
        max_value=1000,
        value=200,
        step=10,
        help="Number of trading days per simulated path.",
    )

    seed = st.number_input(
        "Random seed",
        min_value=1,
        max_value=999999,
        value=42,
        step=1,
        help="Seed used for reproducible random simulation draws.",
    )

    investment = st.number_input(
        "Investment base",
        min_value=1.0,
        max_value=10000.0,
        value=INVESTMENT_DEFAULT,
        step=10.0,
        help="Reference amount used to scale routing capacity and support.",
    )

    buffer_normal_pct = st.slider(
        "Normal buffer / unavailable liquidity (%)",
        min_value=0.0,
        max_value=90.0,
        value=BUFFER_NORMAL_DEFAULT,
        step=10.0,
        help=(
            "Percentage of the investment base that is unavailable under normal conditions. "
            "The remaining share defines baseline available liquidity."
        ),
    )

    liquidity_risk_q = st.slider(
        "Liquidity-risk threshold quantile",
        min_value=0.001,
        max_value=0.100,
        value=0.025,
        step=0.001,
        format="%.3f",
        help=(
            "Lower-tail quantile of direct routing capacity used to define liquidity-risk days."
        ),
    )

    st.header("2. EWI and policy settings")

    target_recall = st.slider(
        "Target EWI recall",
        min_value=0.5,
        max_value=1.0,
        value=0.70,
        step=0.05,
        help=(
            "Intended share of evaluable liquidity-risk event days that receive a true EWI signal "
            "exactly the fixed lead time before the event."
        ),
    )

    target_precision = st.slider(
        "Target EWI precision",
        min_value=0.05,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help=(
            "Intended share of EWI signals that are true positives. A signal is true positive "
            "only if a liquidity-risk event occurs exactly the fixed lead time later."
        ),
    )

    ewi_lead_time = st.slider(
        "Fixed EWI lead time before event, trading days",
        min_value=1,
        max_value=30,
        value=5,
        step=1,
        help=(
            "Number of trading days between a true EWI signal and the liquidity-risk event. "
            "A value of X means the signal is placed X trading days before the event."
        ),
    )

    support_days = st.slider(
        "Support duration after activation",
        min_value=1,
        max_value=60,
        value=10,
        step=1,
        help="Number of trading days for which policy support remains active after it starts.",
    )

    support_start_delay = st.slider(
        "Support start delay after EWI, trading days",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help=(
            "Number of trading days between an EWI signal and the start of policy support. "
            "If this equals the fixed EWI lead time, support starts on the expected event day."
        ),
    )

    support_pct = st.slider(
        "Additional routing-capacity support (%)",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=1.0,
        help=(
            "Additional routing capacity expressed as a percentage of the investment base. "
            "The realized support intensity also depends on how often support is active."
        ),
    )

    st.header("3. Distribution settings")

    st.caption(
        "Default lognormal distribution parameters used to draw the topology-exponent paths."
    )

    lognorm_sigma = st.number_input(
        "Lognormal sigma",
        min_value=0.01,
        max_value=5.0,
        value=FALLBACK_LOGNORM_PARAMS[0],
        step=0.01,
        help="Volatility parameter of the fallback lognormal topology-exponent distribution.",
    )

    lognorm_loc = st.number_input(
        "Location shift",
        min_value=0.0,
        max_value=10.0,
        value=FALLBACK_LOGNORM_PARAMS[1],
        step=0.01,
        help="Location shift added to the simulated topology-exponent paths.",
    )

    lognorm_scale = st.number_input(
        "Lognormal scale",
        min_value=0.01,
        max_value=10.0,
        value=FALLBACK_LOGNORM_PARAMS[2],
        step=0.01,
        help="Scale parameter of the fallback lognormal topology-exponent distribution.",
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
    ewi_lead_time=int(ewi_lead_time),
    support_days=int(support_days),
    support_start_delay=int(support_start_delay),
    support_pct=float(support_pct),
    lognorm_sigma=float(lognorm_sigma),
    lognorm_loc=float(lognorm_loc),
    lognorm_scale=float(lognorm_scale),
)


# -----------------------------------------------------------------------------
# Timing interpretation
# -----------------------------------------------------------------------------

if cfg.support_start_delay > cfg.ewi_lead_time:
    st.warning(
        "Timing note: the support-start delay is longer than the fixed EWI lead time. "
        "Policy support starts after the expected liquidity-risk event."
    )

elif cfg.support_start_delay == cfg.ewi_lead_time:
    st.caption(
        "Timing note: support starts on the expected liquidity-risk event day because "
        "the support-start delay equals the fixed EWI lead time."
    )

else:
    st.caption(
        "Timing note: support starts before the expected liquidity-risk event day because "
        "the support-start delay is shorter than the fixed EWI lead time."
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
        cfg.ewi_lead_time,
        cfg.seed,
    )

    ewi_flags, ewi_diag = create_nested_ewi(
        sim["event_day"],
        design,
        cfg.target_recall,
        cfg.target_precision,
        cfg.ewi_lead_time,
    )

    active_mask = forward_active_mask(
        ewi_flags,
        cfg.support_days,
        cfg.support_start_delay,
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
    "Baseline routing risk-day rate",
    f"{base_metrics['risk_day_rate']:.2f}%",
)

col3.metric(
    "Policy risk-day rate",
    f"{policy_metrics['risk_day_rate']:.2f}%",
)

col4.metric(
    "Routing shortfall reduction",
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

ewi_summary = pd.DataFrame(
    [[k.replace("_", " ").title(), v] for k, v in ewi_diag.items()],
    columns=["Metric", "Value"],
)

policy_summary = pd.DataFrame(
    [
        {
            "Policy": "Baseline no support",
            "Policy support intensity (%)": 0.0,
            "Liquidity-risk scenario rate (%)": base_metrics["risk_scenario_rate"],
            "Liquidity-risk day rate (%)": base_metrics["risk_day_rate"],
            "Total routing shortfall": base_metrics["total_shortfall"],
            "Risk-day reduction (%)": 0.0,
            "Shortfall reduction (%)": 0.0,
            "Mitigation efficiency": np.nan,
        },
        {
            "Policy": "Routing-capacity support",
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

fig_tab, policy_tab, sensitivity_tab, network_tab, ewi_tab, definitions_tab, downloads_tab = st.tabs(
    [
        "Liquidity routing paths",
        "Policy result",
        "Routing support efficacy",
        "Impact of network size",
        "EWI quality",
        "Definitions",
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
            "A. Direct network-liquidity routing capacity",
            "darkblue",
        ),
        (
            axes[1],
            sim["indirect_liquidity"],
            "B. Indirect network-liquidity routing capacity",
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
            label="Direct-capacity risk threshold",
        )

        y_max = max(
            np.nanmax(sim["direct_liquidity"]),
            np.nanmax(sim["indirect_liquidity"]),
        )

        y_upper = math.ceil((y_max * 1.05) / 100) * 100
        min_y_upper = math.ceil((cfg.investment * 1.05) / 100) * 100
        y_upper = max(y_upper, min_y_upper)

        ax.set_ylim(cfg.investment, y_upper)

        ax.set_title(title)
        ax.set_xlabel("Trading day")
        ax.grid(True, linestyle="--", alpha=0.35)

    axes[0].set_ylabel("Direct routing capacity")
    axes[0].legend(fontsize=8, loc="upper right")

    axes[1].set_ylabel("Indirect routing capacity")
    axes[1].legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    st.pyplot(fig)

    st.caption(
        "Note: the y-axis starts at the selected investment base.  \n"
        "Average direct and indirect network-liquidity routing capacities are calculated "
        "over all simulated paths. Lower-tail values and the direct-capacity risk threshold "
        "may fall below the visible range."
    )

    liquidity_paths_png = fig_to_png_bytes(fig)


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
    axes[0].set_title("A. Routing risk scenario rate")
    axes[0].set_ylabel("% of simulated paths")

    axes[1].bar(
        plot_df["Policy"],
        plot_df["Liquidity-risk day rate (%)"],
        color=colors,
    )
    axes[1].set_title("B. Routing risk-day rate")
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
    axes[2].set_ylabel("% of baseline no support")

    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()

    st.pyplot(fig)
    st.dataframe(policy_summary.round(3), use_container_width=True)

    st.caption(
        "The liquidity-risk threshold is set at the selected lower percentile of "
        "direct network-liquidity routing capacity over all simulated paths.  \n"
    )

    policy_comparison_png = fig_to_png_bytes(fig)

    policy_summary_png = dataframe_to_png_bytes(
        policy_summary.round(3),
        title="Policy summary",
    )


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
                    "Support start delay": cfg.support_start_delay,
                    "Support duration": cfg.support_days,
                    "Policy support intensity (%)": 0.0,
                    "Risk-day reduction (%)": 0.0,
                    "Shortfall reduction (%)": 0.0,
                    "Mitigation efficiency": np.nan,
                }
            )
        else:
            _, _, extra = run_support_policy(
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
                    "Support start delay": cfg.support_start_delay,
                    "Support duration": cfg.support_days,
                    "Policy support intensity (%)": extra[
                        "support_intensity_pct"
                    ],
                    "Risk-day reduction (%)": extra["risk_reduction_pct"],
                    "Shortfall reduction (%)": extra[
                        "shortfall_reduction_pct"
                    ],
                    "Mitigation efficiency": extra["mitigation_efficiency"],
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
    ax.set_ylabel("Reduction relative to baseline no support (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(0, max(support_grid))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    fig.tight_layout()

    st.pyplot(fig)
    st.dataframe(support_df.round(3), use_container_width=True)

    st.caption(
        "This panel keeps the EWI timing, support-start delay, and support duration fixed, "
        "while varying the size of routing-capacity support."
    )

    support_sensitivity_png = fig_to_png_bytes(fig)

    support_sensitivity_table_png = dataframe_to_png_bytes(
        support_df.round(3),
        title="Support sensitivity",
    )


# -----------------------------------------------------------------------------
# Network-size efficiency tab
# -----------------------------------------------------------------------------

with network_tab:
    st.subheader("Mitigation efficiency over network size")

    st.caption(
        "Uses the current EWI settings, support-duration setting, and "
        "support-start-delay setting across a compact default grid."
    )

    st.caption(
        "Network-size comparisons use the same random seed across sizes to improve comparability."
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
            cfg.ewi_lead_time,
            cfg.seed,
        )

        ewi_n, _ = create_nested_ewi(
            sim_n["event_day"],
            design_n,
            cfg.target_recall,
            cfg.target_precision,
            cfg.ewi_lead_time,
        )

        active_n = forward_active_mask(
            ewi_n,
            cfg.support_days,
            cfg.support_start_delay,
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
                    "Support start delay": cfg.support_start_delay,
                    "Support duration": cfg.support_days,
                    "Policy support intensity (%)": extra[
                        "support_intensity_pct"
                    ],
                    "Shortfall reduction (%)": extra[
                        "shortfall_reduction_pct"
                    ],
                    "Mitigation efficiency": extra["mitigation_efficiency"],
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
    ax.set_xlim(min(network_grid), max(network_grid))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    fig.tight_layout()

    st.pyplot(fig)
    st.dataframe(network_df.round(3), use_container_width=True)

    network_efficiency_png = fig_to_png_bytes(fig)

    network_efficiency_table_png = dataframe_to_png_bytes(
        network_df.round(3),
        title="Network-size efficiency",
    )


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
            cfg.ewi_lead_time,
        )

        active_q = forward_active_mask(
            ewi_q,
            cfg.support_days,
            cfg.support_start_delay,
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
                "Achieved recall (%)": diag_q["achieved_recall"],
                "Target precision (%)": prec * 100,
                "Achieved precision (%)": diag_q["achieved_precision"],
                "Policy support intensity (%)": extra_q["support_intensity_pct"],
                "Risk-day reduction (%)": extra_q["risk_reduction_pct"],
                "Shortfall reduction (%)": extra_q["shortfall_reduction_pct"],
                "Mitigation efficiency": extra_q["mitigation_efficiency"],
            }
        )

    ewi_quality_df = pd.DataFrame(rows)

    st.dataframe(ewi_quality_df.round(3), use_container_width=True)

    st.caption(
        "Recall is event-based: an event is detected only if an EWI signal occurs"
        "the fixed lead time before the event. Precision is signal-based: a signal is true "
        "positive only if a risk event occurs exactly the fixed lead time later."
    )

    st.subheader("Current EWI diagnostics")
    st.dataframe(ewi_summary.round(3), use_container_width=True)

    ewi_quality_table_png = dataframe_to_png_bytes(
        ewi_quality_df.round(3),
        title="EWI-quality impact under fixed 10% support",
    )

    ewi_summary_png = dataframe_to_png_bytes(
        ewi_summary.round(3),
        title="Current EWI diagnostics",
    )


# -----------------------------------------------------------------------------
# Definitions tab
# -----------------------------------------------------------------------------

with definitions_tab:
    st.subheader("Model definitions")

    st.markdown(
        """
        This glossary defines the main terms used in the simulation outputs.
        The definitions are aligned with the implementation in this Streamlit app.
        """
    )

    glossary_df = glossary_dataframe()

    view_mode = st.radio(
        "Definition view",
        options=["Wrapped table", "Searchable cards"],
        horizontal=True,
        help=(
            "Use the wrapped table for an overview. Use searchable cards when you want "
            "to inspect one definition at a time."
        ),
    )

    if view_mode == "Wrapped table":
        render_wrapped_glossary_table(
            glossary_df,
            max_height=620,
        )
    else:
        render_glossary_cards(glossary_df)

    st.caption(
        "Note: liquidity-risk events, risk-day reductions, and routing shortfalls are "
        "defined using direct network-liquidity routing capacity in this version of the app."
    )

    glossary_png = dataframe_to_png_bytes(
        glossary_df,
        title="Model definitions",
        font_size=7,
    )


# -----------------------------------------------------------------------------
# Downloads tab
# -----------------------------------------------------------------------------

with downloads_tab:
    st.subheader("Download current outputs")

    all_outputs = {
        "Simulation summary": simulation_summary,
        "EWI summary": ewi_summary,
        "Policy summary": policy_summary,
        "Support sensitivity": support_df,
        "Network efficiency": network_df,
        "EWI quality": ewi_quality_df,
        "Definitions": glossary_dataframe(),
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

    st.download_button(
        "Download support sensitivity (.csv)",
        data=support_df.to_csv(index=False).encode("utf-8"),
        file_name="support_sensitivity.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download network efficiency (.csv)",
        data=network_df.to_csv(index=False).encode("utf-8"),
        file_name="network_size_efficiency.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download EWI quality (.csv)",
        data=ewi_quality_df.to_csv(index=False).encode("utf-8"),
        file_name="ewi_quality.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download definitions (.csv)",
        data=glossary_dataframe().to_csv(index=False).encode("utf-8"),
        file_name="model_definitions.csv",
        mime="text/csv",
    )

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
        "ewi_quality_table.png": ewi_quality_table_png,
        "ewi_summary_table.png": ewi_summary_png,
        "model_definitions_table.png": glossary_png,
    }

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.download_button(
            "Download liquidity paths PNG",
            data=liquidity_paths_png,
            file_name="liquidity_paths.png",
            mime="image/png",
        )

        st.download_button(
            "Download policy comparison PNG",
            data=policy_comparison_png,
            file_name="policy_comparison.png",
            mime="image/png",
        )

        st.download_button(
            "Download support sensitivity PNG",
            data=support_sensitivity_png,
            file_name="support_sensitivity.png",
            mime="image/png",
        )

    with col_b:
        st.download_button(
            "Download network efficiency PNG",
            data=network_efficiency_png,
            file_name="network_size_efficiency.png",
            mime="image/png",
        )

        st.download_button(
            "Download policy summary table PNG",
            data=policy_summary_png,
            file_name="policy_summary_table.png",
            mime="image/png",
        )

        st.download_button(
            "Download support table PNG",
            data=support_sensitivity_table_png,
            file_name="support_sensitivity_table.png",
            mime="image/png",
        )

    with col_c:
        st.download_button(
            "Download network table PNG",
            data=network_efficiency_table_png,
            file_name="network_size_efficiency_table.png",
            mime="image/png",
        )

        st.download_button(
            "Download EWI-quality table PNG",
            data=ewi_quality_table_png,
            file_name="ewi_quality_table.png",
            mime="image/png",
        )

        st.download_button(
            "Download definitions table PNG",
            data=glossary_png,
            file_name="model_definitions_table.png",
            mime="image/png",
        )

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
    f"for {cfg.support_days} trading day(s). Liquidity-risk events and policy effects "
    "are evaluated using direct network-liquidity routing capacity. The topology "
    "exponent paths are drawn from fallback lognormal parameters."
)