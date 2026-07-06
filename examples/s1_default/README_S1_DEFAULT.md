# S=1 default demonstration with overview outcome figure

This folder provides a compact, GitHub-ready demonstration of the **Network Liquidity Routing Engine** using one simulated scenario (`S=1`) and the default parameter settings.

The purpose is transparency. The example shows how the engine maps a stochastic tail-exponent path into liquidity multipliers, baseline liquidity, EWI-triggered support, and mitigated liquidity outcomes.

## Files

| File | Purpose |
|---|---|
| `test_s1_default.py` | Reproducible script that runs the S=1 default example and regenerates all outputs. |
| `s1_default_overview_outcome.png` | Five-panel overview figure with direct liquidity, indirect liquidity, baseline routing capacity, dynamic buffers/central-bank injections, and supported routing capacity. |
| `s1_default_parameters.csv` | Default parameter values with plain-language explanations. |
| `s1_default_path.csv` | Day-by-day output for the single simulated path. |
| `s1_default_summary.csv` | Compact summary of baseline, EWI and policy-effect metrics. |
| `s1_default_data_dictionary.csv` | Explanation of all path-output columns. |
| `requirements.txt` | Minimal dependencies. |

## Overview figure panels

The figure `s1_default_overview_outcome.png` contains five panels:

1. **Direct Liquidity Over Time** — direct liquidity multiplier, equal to `E[k]`.
2. **Indirect Liquidity Over Time** — indirect liquidity multiplier, equal to `E[k^2] / E[k] - 1`.
3. **Liquidity Routing Capacity** — direct liquidity multiplier times investment amount times `(1 - normal buffer)`.
4. **Dynamic Buffer and Central-Bank Injection** — effective dynamic buffer and EWI-triggered central-bank injection fractions.
5. **Routing Capacity Including Dynamic Buffer and CB Injections** — routing capacity after dynamic buffer release and central-bank injections.

## Run locally

```bash
pip install -r requirements.txt
python test_s1_default.py
```

## Interpretation note

`S=1` is a smoke test and documentation example. The output should not be interpreted as a stable policy estimate. For thesis-level figures or policy comparisons, use a larger scenario count such as `S=1000`.
