"""
S=1 default demonstration for the Network Liquidity Routing Engine.

This script runs one scenario with default settings and writes GitHub-friendly
outputs that explain the engine mechanics:

- s1_default_parameters.csv
- s1_default_path.csv
- s1_default_summary.csv
- s1_default_data_dictionary.csv
- s1_default_overview_outcome.png

S=1 is a documentation and smoke-test run. It is not intended for statistical
inference or policy calibration.
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

# Fallback 2010-2018 lognormal parameters for gamma draws.
GAMMA_SHAPE = 0.3190514240176254
GAMMA_LOC = 0.0
GAMMA_SCALE = 1.0899155671153902

LIQUIDITY_RISK_Q = 0.025
BUFFER_NORMAL = 40.0
TARGET_RECALL = 0.70
TARGET_PRECISION = 0.25
EWI_LEAD_WINDOW = 5
SUPPORT_PCT = 10.0
SUPPORT_DAYS = 10
GRID_SIZE = 5000

OUTPUT_DIR = Path(__file__).resolve().parent


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
        indirect_grid[i] = expected_k2 / expected_k - 1.0 if expected_k > 0 else np.nan

    return gamma_grid, direct_grid, indirect_grid


def build_ewi_design(event_day: np.ndarray, lead_window: int, seed: int):
    rng = np.random.default_rng(seed)
    evaluable = event_day.copy()
    evaluable[:, :lead_window] = False

    event_positions = np.argwhere(evaluable)
    if len(event_positions) > 0:
        event_positions = event_positions[rng.permutation(len(event_positions))]

    non_event_positions = np.argwhere(~event_day)
    if len(non_event_positions) > 0:
        non_event_positions = non_event_positions[rng.permutation(len(non_event_positions))]

    true_signal_positions = []
    for s, t in event_positions:
        candidate_days = np.arange(max(0, t - lead_window), t)
        if len(candidate_days) > 0:
            true_signal_positions.append((int(s), int(rng.choice(candidate_days))))

    return {
        "n_evaluable_events": int(len(event_positions)),
        "true_signal_positions": true_signal_positions,
        "false_positive_positions": [(int(s), int(t)) for s, t in non_event_positions],
    }


def create_estimated_ewi(event_day: np.ndarray, design: dict, recall: float, precision: float, lead_window: int):
    ewi = np.zeros_like(event_day, dtype=bool)
    n_events = design["n_evaluable_events"]

    n_true = min(int(round(recall * n_events)), len(design["true_signal_positions"]))
    for s, t in design["true_signal_positions"][:n_true]:
        ewi[s, t] = True

    target_total_signals = int(round(n_true / precision)) if precision > 0 and n_true > 0 else n_true
    n_false_positive = min(max(0, target_total_signals - n_true), len(design["false_positive_positions"]))
    for s, t in design["false_positive_positions"][:n_false_positive]:
        ewi[s, t] = True

    true_positive_signal = np.zeros_like(event_day, dtype=bool)
    for s, t in np.argwhere(ewi):
        true_positive_signal[s, t] = bool(event_day[s, t + 1 : t + 1 + lead_window].any())

    signal_days = int(ewi.sum())
    true_positive_signal_days = int(true_positive_signal.sum())
    diagnostics = {
        "target_recall_pct": recall * 100,
        "target_precision_pct": precision * 100,
        "lead_window_days": lead_window,
        "evaluable_event_days": n_events,
        "detected_event_days": n_true,
        "achieved_recall_pct": n_true / n_events * 100 if n_events else np.nan,
        "signal_days": signal_days,
        "true_positive_signal_days": true_positive_signal_days,
        "false_positive_signal_days": signal_days - true_positive_signal_days,
        "achieved_precision_pct": true_positive_signal_days / signal_days * 100 if signal_days else np.nan,
        "signal_day_rate_pct": signal_days / event_day.size * 100,
    }
    return ewi, diagnostics


def forward_active_mask(flags: np.ndarray, duration: int):
    out = np.zeros_like(flags, dtype=bool)
    if duration <= 0:
        return out
    kernel = np.ones(duration, dtype=int)
    for s in range(flags.shape[0]):
        out[s] = np.convolve(flags[s].astype(int), kernel, mode="full")[: flags.shape[1]] > 0
    return out


def evaluate_liquidity(arr: np.ndarray, threshold: float):
    risk = arr < threshold
    shortfall = np.maximum(0.0, threshold - arr)
    return {
        "risk_days": int(risk.sum()),
        "risk_day_rate_pct": float(risk.sum() / arr.size * 100),
        "risk_scenarios": int(risk.any(axis=1).sum()),
        "risk_scenario_rate_pct": float(risk.any(axis=1).sum() / arr.shape[0] * 100),
        "total_shortfall": float(shortfall.sum()),
    }


def make_overview_plot(path_df: pd.DataFrame, summary_df: pd.DataFrame, output_path: Path):
    """
    Create a five-panel overview outcome figure for the Network Liquidity
    Routing Engine.

    Panels:
    1. Direct liquidity multiplier, E[k]
    2. Indirect liquidity multiplier, E[k^2] / E[k] - 1
    3. Liquidity routing capacity = direct multiplier * investment * (1 - normal buffer)
    4. Dynamic buffer and central-bank injections
    5. Routing capacity including dynamic buffer and central-bank injections
    """
    x = path_df["trading_day"].to_numpy()
    direct_lm = path_df["direct_liquidity_multiplier_E_k"].to_numpy()
    indirect_lm = path_df["indirect_liquidity_multiplier_E_k2_over_E_k_minus_1"].to_numpy()
    support_active = path_df["support_active"].astype(bool).to_numpy()

    # Baseline routing capacity follows the requested definition:
    # direct liquidity multiplier * investment amount * (1 - normal buffer)
    available_without_policy = INVESTMENT * (1.0 - BUFFER_NORMAL / 100.0)
    liquidity_routing_capacity = direct_lm * available_without_policy

    # Dynamic buffer is represented as a release of the normal buffer during support days.
    # Central-bank injection is represented as the additional support fraction during support days.
    normal_buffer_fraction = BUFFER_NORMAL / 100.0
    dynamic_buffer_release = np.where(support_active, SUPPORT_PCT / 100.0, 0.0)
    effective_dynamic_buffer = np.maximum(0.0, normal_buffer_fraction - dynamic_buffer_release)
    central_bank_injection = np.where(support_active, SUPPORT_PCT / 100.0, 0.0)

    # Routing capacity including dynamic buffer release and central-bank injections.
    available_with_policy = INVESTMENT * (1.0 - effective_dynamic_buffer + central_bank_injection)
    routing_capacity_with_policy = direct_lm * available_with_policy

    direct_mean = float(np.nanmean(direct_lm))
    indirect_mean = float(np.nanmean(indirect_lm))
    base_mean = float(np.nanmean(liquidity_routing_capacity))
    policy_mean = float(np.nanmean(routing_capacity_with_policy))

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.ravel()

    axes[0].plot(x, direct_lm, color="tab:blue", label="Direct liquidity, E[k]")
    axes[0].axhline(direct_mean, color="green", linestyle="--", linewidth=1.0, label=f"Average direct liquidity ({direct_mean:.2f})")
    axes[0].set_title("1. Direct Liquidity Over Time")
    axes[0].set_xlabel("Trading day")
    axes[0].set_ylabel("Direct liquidity multiplier")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, linestyle="--", alpha=0.35)

    axes[1].plot(x, indirect_lm, color="tab:purple", label="Indirect liquidity, E[k^2] / E[k] - 1")
    axes[1].axhline(indirect_mean, color="green", linestyle="--", linewidth=1.0, label=f"Average indirect liquidity ({indirect_mean:.2f})")
    axes[1].set_title("2. Indirect Liquidity Over Time")
    axes[1].set_xlabel("Trading day")
    axes[1].set_ylabel("Indirect liquidity multiplier")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, linestyle="--", alpha=0.35)

    axes[2].plot(x, liquidity_routing_capacity, color="tab:blue", label="Liquidity routing capacity")
    axes[2].axhline(base_mean, color="green", linestyle="--", linewidth=1.0, label=f"Average routing capacity ({base_mean:.1f})")
    axes[2].set_title("3. Liquidity Routing Capacity")
    axes[2].set_xlabel("Trading day")
    axes[2].set_ylabel("Routing capacity")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, linestyle="--", alpha=0.35)

    axes[3].plot(x, effective_dynamic_buffer, color="tab:blue", label="Effective dynamic buffer")
    axes[3].plot(x, central_bank_injection, color="tab:orange", label="Central-bank injection")
    axes[3].set_title("4. Dynamic Buffer and Central-Bank Injection")
    axes[3].set_xlabel("Trading day")
    axes[3].set_ylabel("Fraction of investment")
    axes[3].set_ylim(0.0, max(0.65, float(max(effective_dynamic_buffer.max(), central_bank_injection.max())) + 0.05))
    axes[3].legend(fontsize=8)
    axes[3].grid(True, linestyle="--", alpha=0.35)

    axes[4].plot(x, routing_capacity_with_policy, color="green", label="Routing capacity incl. dynamic buffer and CB injections")
    axes[4].plot(x, liquidity_routing_capacity, color="tab:blue", linestyle="--", alpha=0.75, label="Baseline routing capacity")
    axes[4].axhline(policy_mean, color="green", linestyle=":", linewidth=1.0, label=f"Average supported capacity ({policy_mean:.1f})")
    axes[4].set_title("5. Routing Capacity Including Dynamic Buffer and CB Injections")
    axes[4].set_xlabel("Trading day")
    axes[4].set_ylabel("Routing capacity")
    axes[4].legend(fontsize=8)
    axes[4].grid(True, linestyle="--", alpha=0.35)

    axes[5].axis("off")
    axes[5].text(
        0.02,
        0.92,
        "S=1 default smoke-test overview\n"
        "Baseline capacity = E[k] * investment * (1 - normal buffer)\n"
        "Supported capacity = E[k] * investment * (1 - dynamic buffer + CB injection)",
        va="top",
        ha="left",
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

def main():
    rng = np.random.default_rng(SEED)
    gamma_grid, direct_grid, indirect_grid = make_liquidity_multiplier_grids(N_NODES, GRID_SIZE)

    gamma_paths = GAMMA_LOC + rng.lognormal(mean=np.log(GAMMA_SCALE), sigma=GAMMA_SHAPE, size=(S, T))
    direct_lm = np.empty_like(gamma_paths)
    indirect_lm = np.empty_like(gamma_paths)

    for s in range(S):
        direct_lm[s] = np.interp(gamma_paths[s], gamma_grid, direct_grid)
        indirect_lm[s] = np.interp(gamma_paths[s], gamma_grid, indirect_grid)

    baseline_available_liquidity = INVESTMENT * (1.0 - BUFFER_NORMAL / 100.0)
    direct_liquidity = baseline_available_liquidity * direct_lm
    indirect_liquidity = baseline_available_liquidity * indirect_lm
    risk_threshold = float(np.nanquantile(direct_liquidity, LIQUIDITY_RISK_Q))
    liquidity_risk_day = direct_liquidity < risk_threshold

    design = build_ewi_design(liquidity_risk_day, EWI_LEAD_WINDOW, SEED)
    ewi_signal, ewi_diagnostics = create_estimated_ewi(liquidity_risk_day, design, TARGET_RECALL, TARGET_PRECISION, EWI_LEAD_WINDOW)
    support_active = forward_active_mask(ewi_signal, SUPPORT_DAYS)
    support_amount = INVESTMENT * SUPPORT_PCT / 100.0 * support_active
    policy_direct_liquidity = (baseline_available_liquidity + support_amount) * direct_lm

    baseline_metrics = evaluate_liquidity(direct_liquidity, risk_threshold)
    policy_metrics = evaluate_liquidity(policy_direct_liquidity, risk_threshold)
    shortfall_baseline = np.maximum(0.0, risk_threshold - direct_liquidity)
    shortfall_policy = np.maximum(0.0, risk_threshold - policy_direct_liquidity)

    support_intensity_pct = support_amount.sum() / (baseline_available_liquidity * S * T) * 100 if baseline_available_liquidity > 0 else np.nan
    risk_day_reduction_pct = (baseline_metrics["risk_days"] - policy_metrics["risk_days"]) / baseline_metrics["risk_days"] * 100 if baseline_metrics["risk_days"] else np.nan
    shortfall_reduction_pct = (baseline_metrics["total_shortfall"] - policy_metrics["total_shortfall"]) / baseline_metrics["total_shortfall"] * 100 if baseline_metrics["total_shortfall"] else np.nan
    mitigation_efficiency = shortfall_reduction_pct / support_intensity_pct if support_intensity_pct > 0 else np.nan

    parameters = pd.DataFrame([
        ["N_NODES", N_NODES, "Number of nodes in the stylised financial network."],
        ["S", S, "Number of simulated scenarios. S=1 is a documentation and smoke-test run."],
        ["T", T, "Number of trading days in the simulated path."],
        ["INVESTMENT", INVESTMENT, "Fixed investment base used to scale liquidity capacity."],
        ["SEED", SEED, "Random seed for reproducibility."],
        ["GAMMA_SHAPE", GAMMA_SHAPE, "Shape parameter of the fallback lognormal gamma distribution."],
        ["GAMMA_LOC", GAMMA_LOC, "Location parameter of the fallback lognormal gamma distribution."],
        ["GAMMA_SCALE", GAMMA_SCALE, "Scale parameter of the fallback lognormal gamma distribution."],
        ["LIQUIDITY_RISK_Q", LIQUIDITY_RISK_Q, "Quantile used to define the lower-tail liquidity-risk threshold."],
        ["BUFFER_NORMAL", BUFFER_NORMAL, "Percentage of investment treated as unavailable in the no-mitigation baseline."],
        ["TARGET_RECALL", TARGET_RECALL, "Target share of evaluable liquidity-risk days detected by the estimated EWI."],
        ["TARGET_PRECISION", TARGET_PRECISION, "Target share of EWI signals that are true-positive signals."],
        ["EWI_LEAD_WINDOW", EWI_LEAD_WINDOW, "Number of days before a liquidity-risk day in which an EWI signal can be placed."],
        ["SUPPORT_PCT", SUPPORT_PCT, "Additional routing-capacity support as a percentage of the investment base."],
        ["SUPPORT_DAYS", SUPPORT_DAYS, "Number of days for which policy support remains active after an EWI signal."],
        ["GRID_SIZE", GRID_SIZE, "Number of gamma-grid points used for interpolation."],
    ], columns=["parameter", "default_value", "explanation"])

    path = pd.DataFrame({
        "scenario": np.repeat(np.arange(1, S + 1), T),
        "trading_day": np.tile(np.arange(1, T + 1), S),
        "gamma": gamma_paths.reshape(-1),
        "direct_liquidity_multiplier_E_k": direct_lm.reshape(-1),
        "indirect_liquidity_multiplier_E_k2_over_E_k_minus_1": indirect_lm.reshape(-1),
        "baseline_direct_liquidity": direct_liquidity.reshape(-1),
        "baseline_indirect_liquidity": indirect_liquidity.reshape(-1),
        "liquidity_risk_threshold": np.repeat(risk_threshold, S * T),
        "liquidity_risk_day": liquidity_risk_day.reshape(-1),
        "ewi_signal": ewi_signal.reshape(-1),
        "support_active": support_active.reshape(-1),
        "support_amount": support_amount.reshape(-1),
        "policy_direct_liquidity": policy_direct_liquidity.reshape(-1),
        "baseline_shortfall": shortfall_baseline.reshape(-1),
        "policy_shortfall": shortfall_policy.reshape(-1),
    })

    summary_rows = []
    for key, value in baseline_metrics.items():
        summary_rows.append(["baseline", key, value])
    for key, value in policy_metrics.items():
        summary_rows.append(["policy", key, value])
    for key, value in ewi_diagnostics.items():
        summary_rows.append(["ewi", key, value])
    summary_rows.extend([
        ["policy_effect", "support_intensity_pct", support_intensity_pct],
        ["policy_effect", "risk_day_reduction_pct", risk_day_reduction_pct],
        ["policy_effect", "shortfall_reduction_pct", shortfall_reduction_pct],
        ["policy_effect", "mitigation_efficiency", mitigation_efficiency],
    ])
    summary = pd.DataFrame(summary_rows, columns=["section", "metric", "value"])

    data_dictionary = pd.DataFrame([
        ["scenario", "Scenario number. S=1 means this is one simulated path."],
        ["trading_day", "Trading-day index from 1 to T."],
        ["gamma", "Drawn tail-exponent value that controls the degree distribution."],
        ["direct_liquidity_multiplier_E_k", "Direct liquidity multiplier, equal to expected degree E[k]."],
        ["indirect_liquidity_multiplier_E_k2_over_E_k_minus_1", "Indirect multiplier, equal to E[k^2] / E[k] - 1."],
        ["baseline_direct_liquidity", "Direct network liquidity without EWI-triggered policy support."],
        ["baseline_indirect_liquidity", "Indirect network liquidity without EWI-triggered policy support."],
        ["liquidity_risk_threshold", "Lower-tail threshold used to identify liquidity-risk days."],
        ["liquidity_risk_day", "True when baseline direct liquidity falls below the threshold."],
        ["ewi_signal", "True when the estimated early-warning indicator emits a signal."],
        ["support_active", "True when policy support is active after an EWI signal."],
        ["support_amount", "Additional routing-capacity support added while support is active."],
        ["policy_direct_liquidity", "Direct liquidity after EWI-triggered support."],
        ["baseline_shortfall", "Shortfall of baseline direct liquidity below the risk threshold."],
        ["policy_shortfall", "Shortfall of policy direct liquidity below the risk threshold."],
    ], columns=["column", "explanation"])

    parameters.to_csv(OUTPUT_DIR / "s1_default_parameters.csv", index=False)
    path.to_csv(OUTPUT_DIR / "s1_default_path.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "s1_default_summary.csv", index=False)
    data_dictionary.to_csv(OUTPUT_DIR / "s1_default_data_dictionary.csv", index=False)
    make_overview_plot(path, summary, OUTPUT_DIR / "s1_default_overview_outcome.png")

    print("S=1 default run complete")
    print("Wrote s1_default_overview_outcome.png")


if __name__ == "__main__":
    main()
