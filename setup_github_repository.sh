#!/usr/bin/env bash
set -e

git init
git add .
git commit -m "Initial Streamlit liquidity-risk simulation engine"
git branch -M main

# Option A: after creating an empty repository on GitHub, uncomment and edit:
# git remote add origin https://github.com/YOUR-USERNAME/liquidity-risk-simulation-engine.git
# git push -u origin main

# Option B: with GitHub CLI installed and authenticated, uncomment:
# gh repo create liquidity-risk-simulation-engine --private --source=. --remote=origin --push
