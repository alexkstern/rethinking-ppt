#!/usr/bin/env bash
#
# Build declref_01_seq_len_2048_val.bin — held-out decl-ref-01 VAL bin. Same
# grammar as the train bin at a disjoint generator seed (1 vs 0), ~10M tokens,
# a whole number of seq_length-token documents. Idempotent.
#
#   bash setup/declref_01/prep_val.sh
#   Env overrides: NANOCHAT_BASE_DIR, NUM_WORKERS
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"

SEQ_LEN=2048
VOCAB=1028
SHARDS=256                         # the bin's contents are fixed by (seed, SHARDS)
WORDS=4864                         # 4864 * 2048 = 9,961,472 tokens (~10M), 256 * 19 whole documents
TARGET=$((WORDS * SEQ_LEN))
NUM_WORKERS="${NUM_WORKERS:-$(getconf _NPROCESSORS_ONLN)}"

OUT_DIR="$BASE_DIR/bins/declref_01_seq_len_2048_val"
BIN="$OUT_DIR/declref_01_seq_len_2048_val.bin"

if [[ -f "$OUT_DIR/meta.json" ]]; then
  echo "[prep_declref_01_val] already built: $OUT_DIR/meta.json — skipping."
  exit 0
fi

echo "[prep_declref_01_val] generating $TARGET decl-ref-01 tokens (seq_len=$SEQ_LEN, seed 1) -> $BIN"
uv run python -m nanochat.data.pretokenize \
  --source declref-01 \
  --vocab-size "$VOCAB" \
  --seq-length "$SEQ_LEN" \
  --seed 1 \
  --num-shards "$SHARDS" \
  --num-workers "$NUM_WORKERS" \
  --split val \
  --target-tokens "$TARGET" \
  --output "$BIN"

echo "[prep_declref_01_val] done -> $BIN"
