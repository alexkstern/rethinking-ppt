#!/usr/bin/env bash
#
# Score one checkpoint on every measure the paper reports — bits-per-byte on
# the held-out window, BLiMP, and DCLM CORE. With no arguments it scores a
# published 1e18 no-PPT baseline downloaded from the Hub. Scoring is
# single-device.
#
# Training removes its weights when it finishes, so ask for a checkpoint with
#
#   uv run python -m reproduce.run_cell <cell_id> --seeds 0 --keep-checkpoints
#
# which prints `[checkpoint] kept: <dir>` — that directory, or its `pt/`
# subdirectory for a two-phase cell, is what CHECKPOINT takes.
#
#   bash reproduce/run_eval.sh
#   (your own weights: CHECKPOINT=<dir> bash reproduce/run_eval.sh)
#   (another published run: HF_REPO=<org/name> bash reproduce/run_eval.sh)
#   (one suite: SUITES="--suite bpb" bash reproduce/run_eval.sh)
#   (log to W&B: WANDB_PROJECT=eval_suites bash reproduce/run_eval.sh)
#   (quick check of the path, not a result: SMOKE=1 bash reproduce/run_eval.sh)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/experiments/training/_env_setup.sh"

# The no-PPT baseline at the smallest reported budget.
HF_REPO="${HF_REPO:-alexkstern/vanilla_1e18_seed0_2026-07-04_01-23-35_659615-pt}"
SUITES="${SUITES:---suite bpb --suite blimp --suite core}"

ARGS=()
if [ -n "${CHECKPOINT:-}" ]; then
    ARGS+=(--checkpoint "$CHECKPOINT")
else
    ARGS+=(--hf-repo "$HF_REPO")
fi
if [ "${SMOKE:-0}" = "1" ]; then
    echo "=== SMOKE: a few examples per task; the numbers are not results ==="
    # Above the largest few-shot count, which is drawn from the same capped set.
    ARGS+=(--max-per-task 16 --max-per-uid 16)
fi
if [ -n "${WANDB_PROJECT:-}" ]; then
    ARGS+=(--wandb-project "$WANDB_PROJECT")
    [ -n "${WANDB_ENTITY:-}" ] && ARGS+=(--wandb-entity "$WANDB_ENTITY")
fi

# shellcheck disable=SC2086  # SUITES is a deliberate multi-word argument
uv run python -m reproduce.evaluate "${ARGS[@]}" $SUITES
