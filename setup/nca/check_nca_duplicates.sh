#!/usr/bin/env bash
#
# Report exact-duplicate trajectories in the generated NCA bins, reshaping each
# bin to one fixed-length trajectory per row. A small duplicate fraction is
# expected: the per-rule gzip filter scores one init, so an init-sensitive rule
# can collapse to the same degenerate fixed point across several sims.
#
# CPU-only (no jax). Reads the local paper bins by default.
#   bash setup/nca/check_nca_duplicates.sh
#   bash setup/nca/check_nca_duplicates.sh /path/train.bin /path/val.bin
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

pip install -q numpy

BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
BINS="$BASE_DIR/bins"
export NCA_TRAIN_BIN="${1:-$BINS/nca_paper_share20_seq2048_657M/nca_paper_share20_seq2048_657M.bin}"
export NCA_VAL_BIN="${2:-$BINS/nca_paper_share20_seq2048_657M_val/nca_paper_share20_seq2048_657M_val.bin}"

python - <<'PY'
import os, json
import numpy as np


def check(path):
    meta = json.load(open(os.path.join(os.path.dirname(path), "meta.json")))
    tpt = meta["gen_params"]["tokens_per_trajectory"]
    toks = np.fromfile(path, dtype=np.uint16)
    n = len(toks) // tpt
    rows = toks[: n * tpt].reshape(n, tpt)
    uniq = np.unique(rows, axis=0)
    dup = n - len(uniq)
    print(f"\n=== {meta['split']} : {os.path.basename(path)} ({n:,} trajectories) ===")
    print(f"    unique trajectories : {len(uniq):,}")
    print(f"    duplicates          : {dup:,} ({100 * dup / n:.4f}%)")
    if dup:
        _, counts = np.unique(rows, axis=0, return_counts=True)
        top = int(counts.max())
        print(f"    max multiplicity    : {top}x (one trajectory repeated {top} times)")


for p in (os.environ["NCA_TRAIN_BIN"], os.environ["NCA_VAL_BIN"]):
    check(p)
PY
