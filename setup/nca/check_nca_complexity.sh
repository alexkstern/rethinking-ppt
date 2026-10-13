#!/usr/bin/env bash
#
# Report the gzip-complexity of the generated NCA trajectories at two scales,
# using the same gzip as the generation filter (per-grid delimiters stripped,
# int32 bytes, level 9). Filter scale (the first FILTER_STEPS grids) is the
# meaningful check: ~95% should clear the band, not 100%, since the bin's
# trajectories use different random inits than the filter's single per-rule
# rollout. Full-trajectory numbers are informational — longer rollouts compress
# lower and the filter never constrained that length.
#
# CPU-only (no jax). Reads the local bins by default.
#   bash setup/nca/check_nca_complexity.sh
#   bash setup/nca/check_nca_complexity.sh /path/train.bin /path/val.bin
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

pip install -q numpy joblib

BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
BINS="$BASE_DIR/bins"
export NCA_TRAIN_BIN="${1:-$BINS/nca_paper_share20_seq2048_657M/nca_paper_share20_seq2048_657M.bin}"
export NCA_VAL_BIN="${2:-$BINS/nca_paper_share20_seq2048_657M_val/nca_paper_share20_seq2048_657M_val.bin}"

python - <<'PY'
import os, io, gzip, json
import numpy as np
from joblib import Parallel, delayed


def gzip_ratio(byte_data):
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=9) as f:
        f.write(byte_data)
    return len(buf.getvalue()) / len(byte_data)


def _chunk(chunk):  # (m, patches) uint16 -> per-row gzip ratio over int32 bytes
    return [gzip_ratio(chunk[i].astype(np.int32).tobytes()) for i in range(len(chunk))]


def _ratios(patches):
    n_jobs = os.cpu_count() or 1
    parts = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_chunk)(c) for c in np.array_split(patches, n_jobs * 4))
    return np.array([x for p in parts for x in p])


def _report(label, r, lo, hi, n):
    ib = (r > lo) & (r < hi)
    print(f"  {label}")
    print(f"    ratio min/med/mean/max : {r.min():.3f} / {np.median(r):.3f} / {r.mean():.3f} / {r.max():.3f}")
    print(f"    in band ({lo},{hi})    : {int(ib.sum()):,}/{n:,} ({100*ib.mean():.2f}%)   below {lo}: {100*(r <= lo).mean():.2f}%")


def check(path):
    meta = json.load(open(os.path.join(os.path.dirname(path), "meta.json")))
    g = meta["gen_params"]
    tpt, tpg = g["tokens_per_trajectory"], g["tokens_per_grid"]
    lo, hi = g["gzip_threshold"], g["gzip_upper_bound"]
    fs = g.get("filter_steps")
    fs_note = "" if fs is not None else " (not in meta; assuming paper default 10)"
    fs = fs or 10
    toks = np.fromfile(path, dtype=np.uint16)
    n = len(toks) // tpt
    grids = toks.reshape(n, tpt // tpg, tpg)             # [:, :, 1:-1] drops START/END
    print(f"\n=== {meta['split']} : {os.path.basename(path)} ({n:,} trajectories) ===")
    _report(f"FILTER SCALE (first {fs} grids){fs_note}  <- the meaningful check",
            _ratios(grids[:, :fs, :][:, :, 1:-1].reshape(n, -1)), lo, hi, n)
    _report("FULL TRAJECTORY (informational; many below-band is expected)",
            _ratios(grids[:, :, 1:-1].reshape(n, -1)), lo, hi, n)


for p in (os.environ["NCA_TRAIN_BIN"], os.environ["NCA_VAL_BIN"]):
    check(p)
PY
