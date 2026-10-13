#!/usr/bin/env bash
#
# Validate NCA bins against their meta.json (CPU-only, no jax), any seq_len:
# trajectory shape and count, the START/END delimiter lattice around every
# grid, and ids below the meta's vocab_size. Geometry and expected counts come
# from the gen_params pretokenize.py records next to each bin.
#
#   bash setup/nca/check_nca_bin.sh [train.bin] [val.bin]
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
import json
import os
import numpy as np

CHUNK_ROWS = 8192

def check(path, is_train):
    with open(os.path.join(os.path.dirname(path), "meta.json")) as f:
        meta = json.load(f)
    g = meta["gen_params"]
    word, tpg = g["tokens_per_trajectory"], g["tokens_per_grid"]
    grids = g["grids_per_trajectory"]
    start, end, vocab = g["start_token"], g["end_token"], meta["vocab_size"]

    toks = np.memmap(path, dtype=np.uint16, mode="r")
    assert toks.size % word == 0, f"{path}: {toks.size} tokens is not whole {word}-token trajectories"
    rows = toks.reshape(-1, word)
    print(f"{path}: {rows.shape[0]:,} trajectories x {word}")
    if is_train:
        expect = g["num_epochs"] * g["num_sims"]
        assert rows.shape[0] == expect, f"expected {expect:,} trajectories (num_epochs x num_sims)"
    for lo in range(0, rows.shape[0], CHUNK_ROWS):
        chunk = np.array(rows[lo:lo + CHUNK_ROWS]).reshape(-1, grids, tpg)
        assert int(chunk.max()) < vocab, f"token id >= vocab {vocab} in rows {lo}+"
        assert (chunk[:, :, 0] == start).all(), f"grid not opening with START in rows {lo}+"
        assert (chunk[:, :, -1] == end).all(), f"grid not closing with END in rows {lo}+"
        assert (chunk[:, :, 1:-1] < start).all(), f"delimiter id inside grid body in rows {lo}+"
    print("  delimiter lattice: OK")

check(os.environ["NCA_TRAIN_BIN"], is_train=True)
check(os.environ["NCA_VAL_BIN"], is_train=False)

print("ALL CHECKS PASSED")
PY
