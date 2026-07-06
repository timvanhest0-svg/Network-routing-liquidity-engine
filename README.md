# Liquidity-risk simulation engine

Interactive Streamlit app for the Chapter 8 liquidity-risk simulation engine.

The app combines:

- gamma-driven network-liquidity simulation;
- direct and indirect liquidity multipliers;
- estimated early-warning-indicator (EWI) recall and precision settings;
- EWI-triggered routing-capacity support;
- policy comparison, support sensitivity and EWI-quality diagnostics;
- downloadable CSV/XLSX summary tables.

## Repository contents

```text
streamlit_simulation_engine.py   # Main Streamlit app
requirements.txt                 # Python dependencies
.gitignore                       # Files excluded from Git
README.md                        # Project documentation
LICENSE                          # Apache 2.0 license
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate       # Windows PowerShell
pip install -r requirements.txt
streamlit run streamlit_simulation_engine.py
```

## Optional input file

The app can use an uploaded gamma-fit summary CSV with these columns:

```text
distribution, window, params
```

If no file is uploaded, the app uses embedded fallback 2010-2018 lognormal parameters as described in the thesis.

## Interpretation note

The policy channel is implemented as additional routing-capacity support after an estimated-EWI signal. In the current setup, false-positive signals are not directly penalized. Lower precision can therefore improve raw liquidity outcomes because it activates support more often. Efficiency metrics should be read together with raw risk-reduction outcomes.

## Set up Git locally

```bash
git init
git add .
git commit -m "Initial Streamlit liquidity-risk simulation engine"
git branch -M main
```

## Push to GitHub

Create an empty GitHub repository first, then run:

```bash
git remote add origin https://github.com/YOUR-USERNAME/liquidity-risk-simulation-engine.git
git push -u origin main
```

If you use the GitHub CLI, you can instead run:

```bash
gh repo create liquidity-risk-simulation-engine --private --source=. --remote=origin --push
```
