#!/usr/bin/env bash
#
# Train the reported cells at one seed each, up to the 9e18 compute budget.
# Validation builds every cell's configuration first, before any GPU time.
#
#   bash reproduce/run_all_cells.sh
#   (continue an interrupted sweep: RESUME=1 bash reproduce/run_all_cells.sh)
#   (three seeds: SEEDS="0 1 2" bash reproduce/run_all_cells.sh)
#   (every budget, no cap: MAX_FLOPS=none bash reproduce/run_all_cells.sh)
#   (other GPU count: NPROC=8 bash reproduce/run_all_cells.sh)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/experiments/training/_env_setup.sh"

SEEDS="${SEEDS:-0}"
MAX_FLOPS="${MAX_FLOPS:-9e18}"
NPROC="${NPROC:-4}"
DEST="${DEST:-confirm_final_runs_ppt}"

# `none` drops the flag rather than standing in for no cap with a huge number.
CAP=(--max-target-flops "$MAX_FLOPS")
[ "$MAX_FLOPS" = "none" ] && CAP=()

# A finished cell drops a marker and is skipped next time. Markers describe the
# manifest as it stood, so a fresh sweep clears them; RESUME=1 keeps them.
if [ "${RESUME:-0}" != "1" ]; then
    rm -rf "outputs/$DEST/done"
fi

uv run python -m reproduce.run_cell --all --validate \
  ${CAP[@]+"${CAP[@]}"} \
  --nproc "$NPROC" --dest "$DEST"

# shellcheck disable=SC2086  # SEEDS is a deliberate multi-value argument
uv run python -m reproduce.run_cell --all \
  ${CAP[@]+"${CAP[@]}"} \
  --seeds $SEEDS \
  --nproc "$NPROC" --dest "$DEST"

# The interleave cells sit above the budget cap; one of them keeps
# `experiments.training.interleave` covered here, run_interleave_cells.sh has
# the rest.
# shellcheck disable=SC2086
uv run python -m reproduce.run_cell interleave__code_interleaved \
  --seeds $SEEDS \
  --nproc "$NPROC" --dest "$DEST"
