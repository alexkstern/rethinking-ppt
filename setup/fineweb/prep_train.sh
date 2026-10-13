#!/usr/bin/env bash
#
# Build fineweb_rustbpe_20B.bin — 20B-token FineWeb-Edu TRAIN bin (rustbpe),
# downloading any missing shards first. Output under the base dir. Idempotent.
#
#   bash setup/fineweb/prep_train.sh
#   Env overrides: NANOCHAT_BASE_DIR, TOKENIZER_THREADS, DOWNLOAD_WORKERS
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
THREADS="${TOKENIZER_THREADS:-18}"
DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-8}"
# 389 train shards (+1 reserved).
NUM_SHARDS=390

OUT_DIR="$BASE_DIR/bins/fineweb_rustbpe_20B"
BIN="$OUT_DIR/fineweb_rustbpe_20B.bin"

if [[ -f "$OUT_DIR/meta.json" ]]; then
  echo "[fineweb prep_train] already built: $OUT_DIR/meta.json — skipping."
  exit 0
fi

echo "[fineweb prep_train] ensuring $NUM_SHARDS FineWeb-Edu shards under $BASE_DIR/base_data ..."
uv run python -m nanochat.data.sources.fineweb.source -n "$NUM_SHARDS" -w "$DOWNLOAD_WORKERS"

echo "[fineweb prep_train] tokenizing 20B tokens with rustbpe ($THREADS threads) -> $BIN"
uv run python -m nanochat.data.pretokenize \
  --source fineweb \
  --tokenizer rustbpe \
  --data-path-local "$BASE_DIR/base_data" \
  --split train \
  --target-tokens 20000000000 \
  --tokenizer-threads "$THREADS" \
  --output "$BIN"

echo "[fineweb prep_train] done -> $BIN"
