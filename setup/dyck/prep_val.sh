#!/usr/bin/env bash
#
# Build dyck_k128_seq_len_2048_val.bin — held-out k-Dyck VAL bin (k=128). Same
# grammar as the train bin at a disjoint generator seed (1 vs 0), ~10M tokens,
# a whole number of seq_length-token words. Idempotent.
#
#   bash setup/dyck/prep_val.sh
#   Env overrides: NANOCHAT_BASE_DIR
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"

K=128
SEQ_LEN=2048
MAX_DEPTH=16
P_OPEN=0.5
VOCAB=$((2 * K))
WORDS=4883                         # 4883 * 2048 = 10,000,384 tokens (~10M), whole words
TARGET=$((WORDS * SEQ_LEN))

OUT_DIR="$BASE_DIR/bins/dyck_k128_seq_len_2048_val"
BIN="$OUT_DIR/dyck_k128_seq_len_2048_val.bin"

if [[ -f "$OUT_DIR/meta.json" ]]; then
  echo "[dyck prep_val] already built: $OUT_DIR/meta.json — skipping."
  exit 0
fi

echo "[dyck prep_val] generating $TARGET k-Dyck tokens (k=$K, seq_len=$SEQ_LEN, seed 1) -> $BIN"
uv run python -m nanochat.data.pretokenize \
  --source dyck \
  --vocab-size "$VOCAB" \
  --k "$K" \
  --seq-length "$SEQ_LEN" \
  --max-depth "$MAX_DEPTH" \
  --p-open "$P_OPEN" \
  --seed 1 \
  --split val \
  --target-tokens "$TARGET" \
  --output "$BIN"

echo "[dyck prep_val] done -> $BIN"
