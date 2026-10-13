#!/usr/bin/env bash
#
# Build c4_rustbpe_10B.bin — 10B-token C4 TRAIN bin (nanochat/rustbpe BPE),
# streamed from the Hub. Output (~20 GB) under the base dir. Idempotent.
#
#   bash setup/c4/prep_train.sh
#   Env overrides: NANOCHAT_BASE_DIR, TOKENIZER_THREADS, C4_CONFIG
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
THREADS="${TOKENIZER_THREADS:-8}"             # streaming is network-bound
C4_CONFIG="${C4_CONFIG:-en}"

OUT_DIR="$BASE_DIR/bins/c4_rustbpe_10B"
BIN="$OUT_DIR/c4_rustbpe_10B.bin"

if [[ -f "$OUT_DIR/meta.json" ]]; then
  echo "[c4 prep_train] already built: $OUT_DIR/meta.json — skipping."
  exit 0
fi

echo "[c4 prep_train] streaming + tokenizing 10B tokens with rustbpe ($THREADS threads) -> $BIN"
uv run python -m nanochat.data.pretokenize \
  --source c4 \
  --tokenizer rustbpe \
  --c4-config "$C4_CONFIG" \
  --split train \
  --target-tokens 10000000000 \
  --tokenizer-threads "$THREADS" \
  --output "$BIN"

echo "[c4 prep_train] done -> $BIN"
