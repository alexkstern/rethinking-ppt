#!/usr/bin/env bash
#
# Hyperparameter search for NCA pre-pretraining at a 1% dose, over the FLOP
# scales 1e18 / 3e18 / 9e18. Resumes from its ledger.
#
#   bash experiments/tasks/run_nca_1pct_multiscale.sh
#
# Needs OPENROUTER_API_KEY (LLM) + HF_TOKEN (data) in .env.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source "$REPO_ROOT/experiments/training/_env_setup.sh"

OUT_DIR="outputs/llmtuna_nca_1pct_multiscale"
resume_args=()
if [ -f "$OUT_DIR/scale_results.json" ] || ls "$OUT_DIR"/metrics/trials_*.jsonl >/dev/null 2>&1; then
  resume_args=(--llmtuna-resume-from "$OUT_DIR")
fi

uv run python -m experiments.tuner.llmtuna_driver \
  --config experiments.tasks.nca_1pct_multiscale ${resume_args[@]+"${resume_args[@]}"}
