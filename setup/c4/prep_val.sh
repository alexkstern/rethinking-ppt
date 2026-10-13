#!/usr/bin/env bash
#
# Build c4_rustbpe_val.bin — C4's official held-out `validation` split (rustbpe),
# drained once (--target-tokens -1), ~165M tokens. Idempotent.
#
#   bash setup/c4/prep_val.sh
#   Env overrides: NANOCHAT_BASE_DIR, TOKENIZER_THREADS, C4_CONFIG
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
THREADS="${TOKENIZER_THREADS:-8}"
C4_CONFIG="${C4_CONFIG:-en}"

OUT_DIR="$BASE_DIR/bins/c4_rustbpe_val"
BIN="$OUT_DIR/c4_rustbpe_val.bin"

if [[ -f "$OUT_DIR/meta.json" ]]; then
  echo "[c4 prep_val] already built: $OUT_DIR/meta.json — skipping."
  exit 0
fi

echo "[c4 prep_val] streaming + tokenizing C4's full validation split with rustbpe -> $BIN"
uv run python -m nanochat.data.pretokenize \
  --source c4 \
  --tokenizer rustbpe \
  --c4-config "$C4_CONFIG" \
  --split val \
  --target-tokens -1 \
  --tokenizer-threads "$THREADS" \
  --output "$BIN"

echo "[c4 prep_val] done -> $BIN"
