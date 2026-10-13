#!/usr/bin/env bash
#
# Optuna (TPE) baseline HPO for nca-1% @ 1e18, 30 trials. Every trial
# self-logs to W&B project optuna_nca_1pct_1e18.
#
#   bash experiments/optuna_baseline/run_optuna_nca_1pct_1e18.sh
#   N_TRIALS=50 bash experiments/optuna_baseline/run_optuna_nca_1pct_1e18.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/experiments/training/_env_setup.sh"

uv run --extra optuna python -m experiments.optuna_baseline.optuna_search \
  --config experiments.tasks.nca_1pct_multiscale \
  --scale 1e18 \
  --n-trials "${N_TRIALS:-30}" \
  --wandb-project optuna_nca_1pct_1e18
