#!/usr/bin/env bash
#
# Train the five interleave cells `run_all_cells.sh` does not, one seed each:
# the NCA and k-Dyck mixing arms and three phase-separated references, all at
# 2.7e19. Only `code_phase` is the mixing arms' program at `pure_code_frac=1`;
# `nca_phase` and `kdyck_phase` are the two-phase runs those arms are compared
# against in the paper, and differ from them in LR and transition policy too.
#
#   bash reproduce/run_interleave_cells.sh
#   (continue an interrupted sweep: RESUME=1 bash reproduce/run_interleave_cells.sh)
#   (three seeds: SEEDS="0 1 2" bash reproduce/run_interleave_cells.sh)
#   (other GPU count: NPROC=8 bash reproduce/run_interleave_cells.sh)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/experiments/training/_env_setup.sh"

SEEDS="${SEEDS:-0}"
NPROC="${NPROC:-4}"
DEST="${DEST:-confirm_final_runs_ppt}"

CELLS=(
  interleave__nca_interleaved
  interleave__kdyck_interleaved
  interleave__nca_phase
  interleave__kdyck_phase
  interleave__code_phase
)

# A finished cell drops a marker and is skipped next time. Markers describe the
# manifest as it stood, so a fresh sweep clears them; RESUME=1 keeps them.
if [ "${RESUME:-0}" != "1" ]; then
    for cell in "${CELLS[@]}"; do
        rm -f "outputs/$DEST/done/${cell//__/_}_s"*
    done
fi

for cell in "${CELLS[@]}"; do
    uv run python -m reproduce.run_cell "$cell" --validate --nproc "$NPROC" --dest "$DEST"
done

for cell in "${CELLS[@]}"; do
    # shellcheck disable=SC2086  # SEEDS is a deliberate multi-value argument
    uv run python -m reproduce.run_cell "$cell" \
      --seeds $SEEDS \
      --nproc "$NPROC" --dest "$DEST"
done
