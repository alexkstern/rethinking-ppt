#!/usr/bin/env bash
#
# Train the additive-compute cells, one seed each: twelve runs, 6.7B tokens, all
# at depth 16, and the only cells training a weight initialization other than
# the spectral default (three N(0,1) embeddings, two N(0,0.02) throughout).
# Validation builds every selected cell's configuration first.
#
#   bash reproduce/run_additive_cells.sh
#   (continue an interrupted sweep: RESUME=1 bash reproduce/run_additive_cells.sh)
#   (three seeds: SEEDS="0 1 2" bash reproduce/run_additive_cells.sh)
#   (other GPU count: NPROC=4 bash reproduce/run_additive_cells.sh)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/experiments/training/_env_setup.sh"

SEEDS="${SEEDS:-0}"
NPROC="${NPROC:-2}"
DEST="${DEST:-confirm_final_runs_ppt}"

# A finished cell drops a marker and is skipped next time. Markers describe the
# manifest as it stood, so a fresh sweep clears them; RESUME=1 keeps them.
if [ "${RESUME:-0}" != "1" ]; then
    rm -f "outputs/$DEST/done/additive_"*
fi

uv run python -m reproduce.run_cell --setting additive --validate \
  --nproc "$NPROC" --dest "$DEST"

# shellcheck disable=SC2086  # SEEDS is a deliberate multi-value argument
uv run python -m reproduce.run_cell --setting additive \
  --seeds $SEEDS \
  --nproc "$NPROC" --dest "$DEST"
