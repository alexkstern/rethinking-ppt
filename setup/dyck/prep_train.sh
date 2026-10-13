#!/usr/bin/env bash
#
# Build dyck_k128_seq_len_2048_1B.bin — 1B-token k-shuffle Dyck TRAIN bin
# (k=128), procedurally generated at seed 0; val uses a disjoint seed. Each word
# is exactly seq_length tokens and the bin holds a whole number of words, so it
# reshapes into one word per row. Output (~2 GB) under the base dir. Idempotent.
#
#   bash setup/dyck/prep_train.sh
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
VOCAB=$((2 * K))                   # 0-based ids span [0, 2k)
WORDS=488281                       # 488281 * 2048 = 999,999,488 tokens (~1B), whole words
TARGET=$((WORDS * SEQ_LEN))

OUT_DIR="$BASE_DIR/bins/dyck_k128_seq_len_2048_1B"
BIN="$OUT_DIR/dyck_k128_seq_len_2048_1B.bin"

if [[ -f "$OUT_DIR/meta.json" ]]; then
  echo "[prep_dyck_train] already built: $OUT_DIR/meta.json — skipping."
  exit 0
fi

echo "[prep_dyck_train] generating $TARGET k-Dyck tokens (k=$K, seq_len=$SEQ_LEN, seed 0) -> $BIN"
uv run python -m nanochat.data.pretokenize \
  --source dyck \
  --vocab-size "$VOCAB" \
  --k "$K" \
  --seq-length "$SEQ_LEN" \
  --max-depth "$MAX_DEPTH" \
  --p-open "$P_OPEN" \
  --seed 0 \
  --split train \
  --target-tokens "$TARGET" \
  --output "$BIN"

echo "[prep_dyck_train] done -> $BIN"
