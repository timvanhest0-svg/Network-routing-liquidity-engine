# Network Routing Liquidity Engine

A Streamlit-based simulation engine for analysing network-based liquidity-risk dynamics, early-warning indicators (EWIs), and routing-capacity support policies.

Repository: [Network Routing Liquidity Engine](https://github.com/timvanhest0-svg/Network-routing-liquidity-engine/tree/main)

## Overview

The Network Routing Liquidity Engine is an interactive research tool for simulating how changes in financial-network topology affect aggregate liquidity-routing capacity. The engine translates simulated network states into direct and indirect liquidity-routing capacity measures, identifies liquidity-risk days, and evaluates the effect of EWI-triggered policy support.

The app is designed as a transparent policy-laboratory environment. It allows users to compare a baseline system without support with a system in which routing-capacity support is activated after early-warning signals.

The current Streamlit app is used for Chapter 8 routing liquidity-risk simulations. It consolidates the simulation logic for:

- topology-exponent paths;
-- direct and indirect network-liquidity routing-capacity simulation;
- estimated EWI signals with target recall and target precision;
- routing-capacity support after EWI signals;
- support-sensitivity analysis;
- network-size efficiency analysis;
- EWI-quality and targeting-value analysis;
- downloadable figures, tables, and model definitions.

## Main application

The main application file is:

```text
streamlit_simulation_engine_app.py
```

Run the app from the repository root with:

```bash
streamlit run streamlit_simulation_engine_app.py
```

The app opens a browser-based interface titled:

Streamlit: https://network-routing-liquidity-engine-xvl9lfdpbnneamxqyxpmtb.streamlit.app/


## Repository structure

```text
Network-routing-liquidity-engine/

│
├── examples/
│   └── S1/
│       ├── .gitignore
|       ├── README_S1_DEFAULT
|       ├── requirements
|       ├── S1_DEFAULT_data_dictionary
|       ├── S1_DEFAULT_data_outcome
|       ├── S1_DEFAULT_parameters
|       ├── S1_DEFAULT_path
|       ├── S1_DEFAULT_summary
│       └── test_s1_default.py
│
├── .gitignore
├── model_definitions.csv
├── README.md
├── requirements.txt
├── LICENSE
├── CITATION.cff
├── streamlit_simulation_engine_app.py
├── docs/
│   └── thesis

```

The `examples/S1` folder contains the minimal smoke test for a single-scenario reproducibility check.

## Installation

Create and activate a Python environment.

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Requirements

The app requires the following main Python packages:

```text
streamlit
numpy
pandas
matplotlib
openpyxl
```

A minimal `requirements.txt` is:

```text
streamlit
numpy
pandas
matplotlib
openpyxl
```

## Quick start

After installation, run:

```bash
streamlit run streamlit_simulation_engine_app.py
```

Then use the sidebar to adjust:

- network size;
- number of simulation scenarios;
- number of trading days;
- random seed;
- investment base;
- normal buffer / unavailable liquidity;
- liquidity-risk threshold quantile;
- target EWI recall;
- target EWI precision;
- fixed EWI lead time;
- support duration;
- support start delay;
- support intensity;
- lognormal topology parameters.

The default random seed is:

```text
42
```

Using the same parameter settings and the same random seed should reproduce the same simulation outputs.

## Model interpretation

The app separates three concepts:

1. **Liquidity-risk event**  
   A scenario-day on which direct routing capacity falls below the liquidity-risk threshold.

2. **EWI signal day**  
   A warning signal placed a fixed number of trading days before a liquidity-risk event.

3. **Policy support day**  
   A day on which routing-capacity support is active after the configured support-start delay.

This timing distinction is important. A correct EWI does not automatically imply immediate support. Support starts only after the configured implementation delay.

## Timing logic

The EWI lead time is implemented as a fixed number of trading days before a liquidity-risk event.

If a liquidity-risk event occurs on day `t`, and the fixed EWI lead time is `X`, then a true EWI signal is placed on:

```text
t - X
```

Policy support is implemented separately. If an EWI signal occurs on day `s`, support starts on:

```text
s + support_start_delay
```

and remains active for:

```text
support_days
```

This means that the support-start delay determines whether support starts before, on, or after the expected liquidity-risk event.

## Default configuration

The default simulation configuration is:

| Parameter | Default |
|---|---:|
| Network size | 24 |
| Simulation scenarios | 1000 |
| Trading days | 200 |
| Investment base | 100 |
| Normal buffer / unavailable liquidity | 40% |
| Liquidity-risk threshold quantile | 0.025 |
| Random seed | 42 |
| Target recall | 0.70 |
| Target precision | 0.25 |
| Fixed EWI lead time | 5 trading days |
| Support duration | 10 trading days |
| Support start delay | 5 trading days |
| Support intensity | 10% |
| Lognormal sigma | 0.3191 |
| Lognormal location shift | 0.0 |
| Lognormal scale | 1.0899 |

## Key definitions

| Term | Definition |
|---|---|
| Direct routing capacity | Network-liquidity capacity based on `E[k]`. This measure defines liquidity-risk events. |
| Indirect routing capacity | Network-liquidity capacity based on `E[k^2] / E[k] - 1`. This measure is shown for comparison only. |
| Liquidity-risk day | A scenario-day on which direct routing capacity falls below the liquidity-risk threshold. |
| Liquidity-risk scenario | A simulated path with at least one liquidity-risk day. |
| Fixed EWI lead time | Number of trading days before a liquidity-risk event at which a true EWI signal is placed. |
| Target recall | Intended percentage of evaluable liquidity-risk event days that receive a true EWI signal. |
| Target precision | Intended percentage of EWI signal days that should be true positives. |
| Policy support intensity | Realized support relative to baseline available capacity across all scenario-days. |
| Risk-day reduction | Percentage reduction in liquidity-risk days relative to baseline no support. |
| Shortfall reduction | Percentage reduction in cumulative routing shortfall relative to baseline no support. |
| Mitigation efficiency | Shortfall reduction per percentage point of realized support intensity. |
| EWI targeting value-added | Difference between EWI-triggered risk-day reduction and untargeted risk-day reduction at the same support intensity. |

## Streamlit interface

The app contains the following tabs:

### 1. Liquidity routing paths

Shows simulated direct and indirect network-liquidity routing capacity over trading days.
Direct routing capacity is used in the app to define risk days and shortfall based on surpassing the 2.5% quantile threshold.

### 2. Mitigation result

Compares baseline no-support outcomes with routing-capacity support outcomes.

### 3. Routing support impact

Shows how mitigation outcomes change across different support levels.

### 4. Impact of network size

Evaluates mitigation efficiency across a compact network-size grid.

### 5. EWI quality

Compares EWI-triggered support with an untargeted benchmark that activates the same number of support days randomly across scenario-days.

Positive values indicate that EWI timing improves risk-day reduction beyond what would be achieved by untargeted support of the same realized volume.

### 6. Definitions

Provides searchable definitions for the model terms used in the app.

### 7. Downloads

Allows users to download current simulation outputs, including tables and PNG files.

## Outputs

The app generates the following output tables:

- simulation summary;
- EWI summary;
- policy summary;
- support sensitivity;
- network efficiency;
- EWI targeting value-added;
- definitions.

The app also supports downloads of figures and tables as PNG files and tabular outputs as Excel files.

## S=1 test

A S=1 test is provided in:

```text
examples S1
```

The purpose of the S=1 test is to verify that the app runs successfully under a minimal example configuration. 

The test uses a single-scenario example, `S = 1`, to explain and confirm that the main simulation workflow runs without errors and that the interface elements are generated correctly.

## examples S1

This folder contains the minimal S=1 test for the Network Routing Liquidity Engine to explain the simulation workflow.

## Purpose

The purpose of this S=1 test is to explain the simulation outcome with an S=1 example. 
It is not a validation of the model assumptions.

## Test configuration

Minimal S=1-test setting:

| Parameter | Value |
|---|---:|
| Simulation scenarios | 1 |
| Network size | 24 |
| Trading days | 200 |
| Investment amount | 100 |
| Buffer | 40 |
| Liquidity risk threshold | 2.5 |
| Random seed | 42 |
| Target recall | 70 |
| Target precision | 25 |
| Fixed EWI lead time | 5 |
| Support start delay | 5 |
| Support duration | 10 |
| Support intensity | 10% |
| Lognorm sigma | 0.32 |
| Lognorm location | 0.0 |
| Lognorm scale | 1.09 |

## Run command

From the repository root:

```bash
Python: run test_s1_default.py
```
Then verify that:

1. the S=1 simulation script completes;
2. the s1_default_data_dictionary.csv is produced and similar to the example S1 file
3. the s1_default_overview_outcome.png is produced and similar to the example S1 file
4. the s1_default_parameters.csv is produced and similar to the example S1 file
5. the s1_default_path.csv is produced and similar to the example S1 file
6. the s1_default_summary.csv is produced and similar to the example S1 file
7. the example S1 files can be used in combination with the model_definitions.csv to understand the simulation workflow


## Streamlit-app test

A streamlit-app test is provided in:

```text
examples/streamlit app
```

The purpose of the streamlit-app test is to verify that the app runs successfully under a minimal example configuration. 

### Test procedure app

From the repository root, run:

```bash
streamlit run streamlit_simulation_engine_app.py
```

## Test configuration

Default streamlit app-test setting:

| Parameter | Value |
|---|---:|
| Simulation scenarios | 1000 |
| Network size | 24 |
| Trading days | 200 |
| Investment amount | 100 |
| Buffer | 40 |
| Liquidity risk threshold | 2.5 |
| Random seed | 42 |
| Target recall | 70 |
| Target precision | 25 |
| Fixed EWI lead time | 5 |
| Support start delay | 5 |
| Support duration | 10 |
| Support intensity | 10% |
| Lognorm sigma | 0.32 |
| Lognorm location | 0.0 |
| Lognorm scale | 1.09 |


Then verify that:

1. the Streamlit app starts successfully;
2. the page title appears;
3. the default simulation completes;
4. the KPI metrics are displayed;
5. the main tabs can be opened;
6. the definitions tab is visible;
7. the downloads tab is available;
8. the app can generate downloadable output files;
9. the `chapter 8 figures/` materials can be used to reproduce the streamlit-app configuration with outcomes.

The streamlit-app test is not intended to validate every scientific assumption in the model. It is a minimal execution check to confirm that the computational workflow is operational. The app supports reproducibility by making the random seed and model parameters explicit. The main simulation assumptions are visible through the interface and through the definitions tab.


## Citation

If you use this software in academic work, cite the archived release if a DOI is available. If no DOI is available yet, cite the GitHub repository:

```text
van Hest, T. Network Routing Liquidity Engine. GitHub repository:
https://github.com/timvanhest0-svg/Network-routing-liquidity-engine/tree/main
```

If a `CITATION.cff` file is available, use the citation metadata provided there.

## License

See the `LICENSE` file for license terms.

## Maintainer

Tim van Hest.
