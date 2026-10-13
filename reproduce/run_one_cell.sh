#!/usr/bin/env bash
#
# Train one reported cell, at one seed; the default is the cheapest one. Writes
# to the same W&B project and done-marker directory as the full sweep, so a cell
# finished here is skipped when the sweep runs.
#
#   bash reproduce/run_one_cell.sh
#   (another cell: CELL=constrained__nca_1pct bash reproduce/run_one_cell.sh)
#   (other GPU count: NPROC=8 bash reproduce/run_one_cell.sh)
#   (train it again: RESUME=0 bash reproduce/run_one_cell.sh)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/experiments/training/_env_setup.sh"

CELL="${CELL:-additive__50m_spectral_vanilla_0}"
DEST="${DEST:-confirm_final_runs_ppt}"

# A finished cell drops a marker and is skipped next time, here and in the
# sweep. RESUME=1 (the default) keeps it; RESUME=0 clears this cell's marker
# and trains it again.
if [ "${RESUME:-1}" != "1" ]; then
    rm -f "outputs/$DEST/done/${CELL//__/_}_s0"
fi

uv run python -m reproduce.run_cell "$CELL" \
  --seeds 0 \
  --dest "$DEST" \
  --nproc "${NPROC:-4}"
