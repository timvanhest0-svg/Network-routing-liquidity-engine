"""
S=1 default demonstration for the Network Liquidity Routing Engine
=================================================================

This script runs one reproducible single-path demonstration of the engine
mechanics:

- s1_default_parameters.csv
- s1_default_path.csv
- s1_default_summary.csv
- s1_default_data_dictionary.csv
- s1_default_overview_outcome.png

S=1 is a documentation and smoke-test run. It is not intended for statistical
inference or policy calibration. The multi-scenario Streamlit app should be used
for policy comparison, sensitivity analysis, and distributional outcomes.

Main alignment choices
----------------------
- Network size N=24, trading days T=200, investment base=100.
- Degree support uses K=N-1, consistent with the thesis notation.
- Liquidity-risk days are defined from direct routing capacity below the
  lower-tail threshold.
- The EWI uses a fixed lead-time interpretation: a true EWI signal for an
  event on day t is placed exactly lead_time days earlier.
- Policy support is a single combined routing-capacity support variable.
  The script deliberately does not separate dynamic-buffer release and
  central-bank injection because both act through the same mechanical channel
  in this reduced-form S=1 demonstration.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Default parameters
# -----------------------------------------------------------------------------

N_NODES = 24
S = 1
T = 200
INVESTMENT = 100.0
SEED = 42

# 2010-2018 lognormal parameters for gamma draws.
GAMMA_SHAPE = 0.32
GAMMA_LOC = 0.0
GAMMA_SCALE = 1.09

LIQUIDITY_RISK_Q = 0.025
BUFFER_NORMAL = 40.0
TARGET_RECALL = 0.70
TARGET_PRECISION = 0.25
EWI_LEAD_TIME = 5
SUPPORT_START_DELAY = 5
SUPPORT_PCT = 10.0
SUPPORT_DAYS = 10
GRID_SIZE = 5000

OUTPUT_DIR = Path(__file__).resolve().parent


# -----------------------------------------------------------------------------
# Liquidity-multiplier and simulation functions
# -----------------------------------------------------------------------------

def make_liquidity_multiplier_grids(n_nodes: int, grid_size: int = 5000):
    """
    Create interpolation grids for direct and indirect liquidity multipliers.

    Degree support follows the thesis notation K=N-1:
        k = 1, ..., N-1

    Direct multiplier:
        E[k]

    Indirect multiplier:
        E[k^2] / E[k] - 1
    """
    degree = np.arange(1, n_nodes, dtype=float)
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


def simulate_base_path():
    """Simulate one gamma path and the corresponding routing-capacity path."""
    rng = np.random.default_rng(SEED)
    gamma_grid, direct_lm_grid, indirect_lm_grid = make_liquidity_multiplier_grids(
        N_NODES,
        GRID_SIZE,
    )

    gamma_paths = GAMMA_LOC + rng.lognormal(
        mean=np.log(GAMMA_SCALE),
        sigma=GAMMA_SHAPE,
        size=(S, T),
    )

    direct_lm = np.empty_like(gamma_paths)
    indirect_lm = np.empty_like(gamma_paths)

    for s in range(S):
        direct_lm[s] = np.interp(gamma_paths[s], gamma_grid, direct_lm_grid)
        indirect_lm[s] = np.interp(gamma_paths[s], gamma_grid, indirect_lm_grid)

    baseline_available_liquidity = INVESTMENT * (1.0 - BUFFER_NORMAL / 100.0)
    direct_liquidity = baseline_available_liquidity * direct_lm
    indirect_liquidity = baseline_available_liquidity * indirect_lm

    risk_threshold = float(np.nanquantile(direct_liquidity, LIQUIDITY_RISK_Q))
    event_day = direct_liquidity < risk_threshold

    return {
        "gamma_paths": gamma_paths,
        "direct_lm": direct_lm,
        "indirect_lm": indirect_lm,
        "direct_liquidity": direct_liquidity,
        "indirect_liquidity": indirect_liquidity,
        "risk_threshold": risk_threshold,
        "event_day": event_day,
        "baseline_available_liquidity": baseline_available_liquidity,
    }


# -----------------------------------------------------------------------------
# EWI design and diagnostics
# -----------------------------------------------------------------------------

def future_event_mask(event_day: np.ndarray, lead_time: int) -> np.ndarray:
    """Mark days exactly lead_time trading days before a liquidity-risk event."""
    out = np.zeros_like(event_day, dtype=bool)

    if 0 < lead_time < event_day.shape[1]:
        out[:, :-lead_time] = event_day[:, lead_time:]

    return out


def detected_event_mask(event_day: np.ndarray, ewi: np.ndarray, lead_time: int) -> np.ndarray:
    """Mark event days preceded by an EWI signal exactly lead_time days earlier."""
    detected = np.zeros_like(event_day, dtype=bool)

    if 0 < lead_time < event_day.shape[1]:
        detected[:, lead_time:] = event_day[:, lead_time:] & ewi[:, :-lead_time]

    return detected


def build_nested_ewi_design(event_day: np.ndarray, lead_time: int, seed: int):
    """
    Create EWI candidate positions using the fixed-lead-time interpretation.

    True signals are placed exactly lead_time trading days before evaluable
    liquidity-risk days. False positives are chosen from days that are neither
    risk days nor exact pre-event days.
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

    # Deduplicate while preserving order. This can matter for clustered events.
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
    lead_time: int,
):
    """Create an estimated EWI signal process matching recall and precision."""
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
        "target_recall_pct": target_recall * 100,
        "target_precision_pct": target_precision * 100,
        "fixed_lead_time_days": lead_time,
        "evaluable_event_days": n_events,
        "signalable_event_days": n_signalable_events,
        "selected_true_signal_days": n_true,
        "detected_event_days": detected_event_days,
        "achieved_recall_pct": detected_event_days / n_events * 100 if n_events else np.nan,
        "achieved_recall_signalable_events_pct": (
            detected_event_days / n_signalable_events * 100
            if n_signalable_events
            else np.nan
        ),
        "signal_days": signal_days,
        "true_positive_signal_days": true_positive_signal_days,
        "false_positive_signal_days": false_positive_signal_days,
        "achieved_precision_pct": (
            true_positive_signal_days / signal_days * 100
            if signal_days
            else np.nan
        ),
        "signal_day_rate_pct": signal_days / event_day.size * 100,
        "false_positive_rate_pct": false_positive_signal_days / event_day.size * 100,
    }

    return ewi, diagnostics


# -----------------------------------------------------------------------------
# Policy support and liquidity evaluation
# -----------------------------------------------------------------------------

def forward_active_mask(flags: np.ndarray, duration: int, start_delay: int):
    """Activate support after an EWI signal for a fixed duration."""
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


def evaluate_liquidity(arr: np.ndarray, threshold: float):
    """Evaluate liquidity-risk days, scenarios, and cumulative shortfall."""
    risk = arr < threshold
    shortfall = np.maximum(0.0, threshold - arr)

    return {
        "risk_days": int(risk.sum()),
        "risk_day_rate_pct": risk.sum() / arr.size * 100,
        "risk_scenarios": int(risk.any(axis=1).sum()),
        "risk_scenario_rate_pct": risk.any(axis=1).sum() / arr.shape[0] * 100,
        "total_shortfall": float(shortfall.sum()),
    }


def run_support_policy(
    direct_lm: np.ndarray,
    baseline_metrics: dict,
    risk_threshold: float,
    active_mask: np.ndarray,
    baseline_available_liquidity: float,
):
    """Apply combined routing-capacity support and calculate policy metrics."""
    support_amount = INVESTMENT * (SUPPORT_PCT / 100.0) * active_mask
    policy_direct_liquidity = (baseline_available_liquidity + support_amount) * direct_lm
    policy_metrics = evaluate_liquidity(policy_direct_liquidity, risk_threshold)

    support_intensity_pct = (
        support_amount.sum() / (baseline_available_liquidity * direct_lm.size) * 100
        if baseline_available_liquidity > 0
        else np.nan
    )

    risk_day_reduction_pct = (
        (baseline_metrics["risk_days"] - policy_metrics["risk_days"])
        / baseline_metrics["risk_days"]
        * 100
        if baseline_metrics["risk_days"]
        else np.nan
    )

    shortfall_reduction_pct = (
        (baseline_metrics["total_shortfall"] - policy_metrics["total_shortfall"])
        / baseline_metrics["total_shortfall"]
        * 100
        if baseline_metrics["total_shortfall"]
        else np.nan
    )

    mitigation_efficiency = (
        shortfall_reduction_pct / support_intensity_pct
        if support_intensity_pct > 0
        else np.nan
    )

    extra = {
        "support_intensity_pct": support_intensity_pct,
        "risk_day_reduction_pct": risk_day_reduction_pct,
        "shortfall_reduction_pct": shortfall_reduction_pct,
        "mitigation_efficiency": mitigation_efficiency,
    }

    return policy_direct_liquidity, policy_metrics, extra, support_amount


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------

def make_overview_plot(path_df: pd.DataFrame, summary_df: pd.DataFrame, output_path: Path):
    """Create a compact S=1 overview plot."""
    x = path_df["day"].to_numpy()
    risk_threshold = float(path_df["liquidity_risk_threshold"].iloc[0])

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(11, 14),
        sharex=False,
    )

    axes[0].plot(x, path_df["gamma"], color="black", linewidth=1.4)
    axes[0].set_title("1. Simulated topology exponent path")
    axes[0].set_ylabel("Gamma")
    axes[0].grid(True, linestyle="--", alpha=0.35)

    axes[1].plot(x, path_df["baseline_direct_liquidity"], color="tab:blue", label="Baseline direct routing capacity")
    axes[1].plot(x, path_df["policy_direct_liquidity"], color="tab:green", label="Supported direct routing capacity")
    axes[1].axhline(risk_threshold, color="red", linestyle="--", linewidth=1.0, label="Liquidity-risk threshold")
    axes[1].set_title("2. Direct routing capacity before and after support")
    axes[1].set_ylabel("Routing capacity")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, linestyle="--", alpha=0.35)

    axes[2].plot(x, path_df["baseline_indirect_liquidity"], color="tab:red", label="Baseline indirect routing capacity")
    axes[2].set_title("3. Indirect routing capacity")
    axes[2].set_ylabel("Routing capacity")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, linestyle="--", alpha=0.35)

    axes[3].step(x, path_df["ewi_signal"].astype(int), where="post", color="tab:orange", label="EWI signal")
    axes[3].step(x, path_df["support_active"].astype(int), where="post", color="tab:green", label="Support active")
    axes[3].scatter(
        path_df.loc[path_df["liquidity_risk_day"], "day"],
        np.repeat(1.08, int(path_df["liquidity_risk_day"].sum())),
        color="red",
        s=22,
        label="Liquidity-risk day",
        zorder=3,
    )
    axes[3].set_ylim(-0.05, 1.20)
    axes[3].set_title("4. Fixed-lead EWI signals and combined support activation")
    axes[3].set_ylabel("Indicator")
    axes[3].legend(fontsize=8, loc="upper right")
    axes[3].grid(True, linestyle="--", alpha=0.35)

    axes[4].bar(x, path_df["baseline_shortfall"], color="tab:blue", alpha=0.45, label="Baseline shortfall")
    axes[4].bar(x, path_df["policy_shortfall"], color="tab:green", alpha=0.55, label="Policy shortfall")
    axes[4].set_title("5. Routing-capacity shortfall below the threshold")
    axes[4].set_xlabel("Trading day")
    axes[4].set_ylabel("Shortfall")
    axes[4].legend(fontsize=8)
    axes[4].grid(True, axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    sim = simulate_base_path()

    baseline_metrics = evaluate_liquidity(
        sim["direct_liquidity"],
        sim["risk_threshold"],
    )

    design = build_nested_ewi_design(
        sim["event_day"],
        EWI_LEAD_TIME,
        SEED,
    )

    ewi_signal, ewi_diagnostics = create_nested_ewi(
        sim["event_day"],
        design,
        TARGET_RECALL,
        TARGET_PRECISION,
        EWI_LEAD_TIME,
    )

    support_active = forward_active_mask(
        ewi_signal,
        SUPPORT_DAYS,
        SUPPORT_START_DELAY,
    )

    policy_direct_liquidity, policy_metrics, policy_extra, support_amount = run_support_policy(
        sim["direct_lm"],
        baseline_metrics,
        sim["risk_threshold"],
        support_active,
        sim["baseline_available_liquidity"],
    )

    baseline_shortfall = np.maximum(0.0, sim["risk_threshold"] - sim["direct_liquidity"])
    policy_shortfall = np.maximum(0.0, sim["risk_threshold"] - policy_direct_liquidity)

    parameters = pd.DataFrame(
        [
            ["N_NODES", N_NODES, "Number of nodes in the stylised financial network."],
            ["S", S, "Number of simulated scenarios. S=1 is a documentation and smoke-test run."],
            ["T", T, "Number of trading days in the simulated path."],
            ["INVESTMENT", INVESTMENT, "Fixed investment base used to scale routing capacity."],
            ["SEED", SEED, "Random seed for reproducibility."],
            ["GAMMA_SHAPE", GAMMA_SHAPE, "Shape parameter of the fallback lognormal gamma distribution."],
            ["GAMMA_LOC", GAMMA_LOC, "Location parameter of the fallback lognormal gamma distribution."],
            ["GAMMA_SCALE", GAMMA_SCALE, "Scale parameter of the fallback lognormal gamma distribution."],
            ["LIQUIDITY_RISK_Q", LIQUIDITY_RISK_Q, "Quantile used to define the lower-tail liquidity-risk threshold."],
            ["BUFFER_NORMAL", BUFFER_NORMAL, "Percentage of investment unavailable in the no-mitigation baseline."],
            ["TARGET_RECALL", TARGET_RECALL, "Target share of evaluable liquidity-risk days detected by the EWI."],
            ["TARGET_PRECISION", TARGET_PRECISION, "Target share of EWI signals that are true-positive signals."],
            ["EWI_LEAD_TIME", EWI_LEAD_TIME, "Fixed number of trading days between a true EWI signal and an event."],
            ["SUPPORT_START_DELAY", SUPPORT_START_DELAY, "Number of trading days between an EWI signal and support activation."],
            ["SUPPORT_PCT", SUPPORT_PCT, "Combined additional routing-capacity support as a percentage of the investment base."],
            ["SUPPORT_DAYS", SUPPORT_DAYS, "Number of trading days for which support remains active after activation."],
            ["GRID_SIZE", GRID_SIZE, "Number of gamma-grid points used for interpolation."],
        ],
        columns=["parameter", "default_value", "explanation"],
    )

    path = pd.DataFrame(
        {
            "day": np.arange(1, T + 1),
            "gamma": sim["gamma_paths"].reshape(-1),
            "direct_liquidity_multiplier_E_k": sim["direct_lm"].reshape(-1),
            "indirect_liquidity_multiplier_E_k2_over_E_k_minus_1": sim["indirect_lm"].reshape(-1),
            "baseline_direct_liquidity": sim["direct_liquidity"].reshape(-1),
            "baseline_indirect_liquidity": sim["indirect_liquidity"].reshape(-1),
            "liquidity_risk_threshold": np.repeat(sim["risk_threshold"], S * T),
            "liquidity_risk_day": sim["event_day"].reshape(-1),
            "ewi_signal": ewi_signal.reshape(-1),
            "support_active": support_active.reshape(-1),
            "support_amount": support_amount.reshape(-1),
            "policy_direct_liquidity": policy_direct_liquidity.reshape(-1),
            "baseline_shortfall": baseline_shortfall.reshape(-1),
            "policy_shortfall": policy_shortfall.reshape(-1),
        }
    )

    summary = pd.DataFrame(
        [
            ["baseline", "liquidity_risk_threshold", sim["risk_threshold"]],
            ["baseline", "risk_days", baseline_metrics["risk_days"]],
            ["baseline", "risk_day_rate_pct", baseline_metrics["risk_day_rate_pct"]],
            ["baseline", "risk_scenarios", baseline_metrics["risk_scenarios"]],
            ["baseline", "risk_scenario_rate_pct", baseline_metrics["risk_scenario_rate_pct"]],
            ["baseline", "total_shortfall", baseline_metrics["total_shortfall"]],
            ["policy", "risk_days", policy_metrics["risk_days"]],
            ["policy", "risk_day_rate_pct", policy_metrics["risk_day_rate_pct"]],
            ["policy", "risk_scenarios", policy_metrics["risk_scenarios"]],
            ["policy", "risk_scenario_rate_pct", policy_metrics["risk_scenario_rate_pct"]],
            ["policy", "total_shortfall", policy_metrics["total_shortfall"]],
            ["policy_effect", "support_intensity_pct", policy_extra["support_intensity_pct"]],
            ["policy_effect", "risk_day_reduction_pct", policy_extra["risk_day_reduction_pct"]],
            ["policy_effect", "shortfall_reduction_pct", policy_extra["shortfall_reduction_pct"]],
            ["policy_effect", "mitigation_efficiency", policy_extra["mitigation_efficiency"]],
            ["ewi", "target_recall_pct", ewi_diagnostics["target_recall_pct"]],
            ["ewi", "target_precision_pct", ewi_diagnostics["target_precision_pct"]],
            ["ewi", "fixed_lead_time_days", ewi_diagnostics["fixed_lead_time_days"]],
            ["ewi", "support_start_delay_days", SUPPORT_START_DELAY],
            ["ewi", "evaluable_event_days", ewi_diagnostics["evaluable_event_days"]],
            ["ewi", "signalable_event_days", ewi_diagnostics["signalable_event_days"]],
            ["ewi", "selected_true_signal_days", ewi_diagnostics["selected_true_signal_days"]],
            ["ewi", "detected_event_days", ewi_diagnostics["detected_event_days"]],
            ["ewi", "achieved_recall_pct", ewi_diagnostics["achieved_recall_pct"]],
            ["ewi", "achieved_precision_pct", ewi_diagnostics["achieved_precision_pct"]],
            ["ewi", "signal_days", ewi_diagnostics["signal_days"]],
            ["ewi", "false_positive_signal_days", ewi_diagnostics["false_positive_signal_days"]],
            ["ewi", "signal_day_rate_pct", ewi_diagnostics["signal_day_rate_pct"]],
            ["ewi", "false_positive_rate_pct", ewi_diagnostics["false_positive_rate_pct"]],
        ],
        columns=["section", "metric", "value"],
    )

    data_dictionary = pd.DataFrame(
        [
            ["day", "Trading day index from 1 to T."],
            ["gamma", "Drawn topology tail-exponent value controlling the degree distribution."],
            ["direct_liquidity_multiplier_E_k", "Direct liquidity multiplier, equal to expected degree E[k]."],
            ["indirect_liquidity_multiplier_E_k2_over_E_k_minus_1", "Indirect multiplier, equal to E[k^2] / E[k] - 1."],
            ["baseline_direct_liquidity", "Direct routing capacity without EWI-triggered policy support."],
            ["baseline_indirect_liquidity", "Indirect routing capacity without EWI-triggered policy support."],
            ["liquidity_risk_threshold", "Lower-tail threshold used to identify liquidity-risk days."],
            ["liquidity_risk_day", "True when baseline direct routing capacity falls below the threshold."],
            ["ewi_signal", "True when the estimated EWI emits a fixed-lead signal."],
            ["support_active", "True when combined routing-capacity support is active."],
            ["support_amount", "Additional routing-capacity support while support is active."],
            ["policy_direct_liquidity", "Direct routing capacity after combined EWI-triggered support."],
            ["baseline_shortfall", "Shortfall of baseline direct routing capacity below the risk threshold."],
            ["policy_shortfall", "Shortfall of policy direct routing capacity below the risk threshold."],
        ],
        columns=["column", "explanation"],
    )

    parameters.to_csv(OUTPUT_DIR / "s1_default_parameters.csv", index=False)
    path.to_csv(OUTPUT_DIR / "s1_default_path.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "s1_default_summary.csv", index=False)
    data_dictionary.to_csv(OUTPUT_DIR / "s1_default_data_dictionary.csv", index=False)
    make_overview_plot(path, summary, OUTPUT_DIR / "s1_default_overview_outcome.png")

    print("S=1 default run complete")
    print("Wrote:")
    print("- s1_default_parameters.csv")
    print("- s1_default_path.csv")
    print("- s1_default_summary.csv")
    print("- s1_default_data_dictionary.csv")
    print("- s1_default_overview_outcome.png")


if __name__ == "__main__":
    main()
