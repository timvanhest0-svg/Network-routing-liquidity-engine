# S=1 default demonstration with overview outcome figure

This folder provides a compact, demonstration of the **Network Liquidity Routing Engine** using one simulated scenario (`S=1`) and the default parameter settings.

The purpose is transparency and reproducibility. The example shows how the engine maps a stochastic topology tail-exponent path into liquidity multipliers, baseline routing capacity, fixed-lead EWI signals, combined EWI-triggered support, and mitigated liquidity outcomes.

`S=1` is a documentation and smoke-test run. It is not intended for statistical inference or policy calibration. For thesis-level figures, sensitivity analysis, or policy comparisons, use the multi-scenario Streamlit app or a larger simulation count such as `S=1000`.

## Main alignment choices

The S=1 script follows the same core mechanics as the Streamlit simulation engine, but keeps the output deliberately simple:

- **Network size:** `N=24`.
- **Simulation length:** `T=200` trading days.
- **Scenario count:** `S=1`, for documentation only.
- **Investment base:** `100`.
- **Normal unavailable buffer:** `40%`, so baseline available routing capacity is `investment * (1 - buffer_normal / 100)`.
- **Degree support:** `k = 1, ..., N-1`, consistent with the thesis notation.
- **Liquidity-risk definition:** a liquidity-risk day occurs when baseline direct routing capacity falls below the selected lower-tail threshold.
- **EWI timing:** a true EWI signal for an event on day `t` is placed exactly `EWI_LEAD_TIME` trading days earlier.
- **Support timing:** support starts after `SUPPORT_START_DELAY` trading days and remains active for `SUPPORT_DAYS` trading days.
- **Policy support:** support is modelled as one combined routing-capacity support variable. The S=1 script deliberately does **not** separate dynamic-buffer release and central-bank injection, because both affect routing capacity through the same reduced-form mechanical channel in this demonstration.

## Files

| File | Purpose |
|---|---|
| `test_s1_default.py` | Reproducible script that runs the S=1 default example and regenerates all outputs. If using the downloaded fixed script, rename `test_s1_default_fixed.py` to `test_s1_default.py` before committing to the repository. |
| `s1_default_overview_outcome.png` | Five-panel overview figure showing the topology path, direct routing capacity before and after support, indirect routing capacity, EWI/support timing, and shortfalls. |
| `s1_default_parameters.csv` | Default parameter values with plain-language explanations. |
| `s1_default_path.csv` | Day-by-day output for the single simulated path. |
| `s1_default_summary.csv` | Compact summary of baseline, EWI, and policy-effect metrics. |
| `s1_default_data_dictionary.csv` | Explanation of all path-output columns. |
| `requirements.txt` | Minimal dependencies for running the script locally. |

## Overview figure panels

The figure `s1_default_overview_outcome.png` contains five panels:

1. **Simulated topology exponent path** — the stochastic path of the topology tail exponent `gamma`.
2. **Direct routing capacity before and after support** — baseline direct routing capacity, supported direct routing capacity, and the liquidity-risk threshold.
3. **Indirect routing capacity** — indirect routing capacity based on `E[k^2] / E[k] - 1`.
4. **Fixed-lead EWI signals and combined support activation** — EWI signal days, support-active days, and baseline liquidity-risk days.
5. **Routing-capacity shortfall below the threshold** — baseline shortfall and policy shortfall relative to the liquidity-risk threshold.

## Run locally

```bash
pip install -r requirements.txt
python test_s1_default.py
```

If you keep the downloaded filename, run:

```bash
python test_s1_default_fixed.py
```

Running the script regenerates:

```text
s1_default_parameters.csv
s1_default_path.csv
s1_default_summary.csv
s1_default_data_dictionary.csv
s1_default_overview_outcome.png
```

## Interpretation note

This S=1 example is a **mechanics demonstration**, not a statistical result. Because it contains only one simulated path, the output is sensitive to the random draw and should not be interpreted as a stable estimate of mitigation effectiveness.

Use this example to understand and verify the engine mechanics:

1. draw a topology path;
2. map topology into direct and indirect liquidity multipliers;
3. compute baseline direct routing capacity;
4. identify lower-tail liquidity-risk days;
5. generate fixed-lead EWI signals with target recall and precision;
6. activate combined routing-capacity support after the configured start delay;
7. compare baseline and policy shortfalls.

For policy interpretation, use a larger scenario count such as `S=1000` in the Streamlit app or in a batch simulation script.

## Modelling simplification in this S=1 script

The S=1 script uses a single combined support variable:

```python
support_amount = INVESTMENT * (SUPPORT_PCT / 100.0) * support_active
policy_direct_liquidity = (baseline_available_liquidity + support_amount) * direct_lm
```

This is intentional. In the reduced-form simulation, both buffer release and central-bank support increase available routing capacity through the same mechanical channel. Keeping one support variable makes the S=1 example clearer and avoids suggesting that the smoke-test script identifies separate institutional channels.
