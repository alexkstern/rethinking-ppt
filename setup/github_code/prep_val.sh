#!/usr/bin/env bash
#
# Build github_code_rustbpe_val.bin — 10M-token GitHub-code VAL bin (rustbpe).
# Upstream codeparrot/github-code has only a train split, so val is the FIRST
# 20,000 documents of the parquet-branch stream, truncated to 10M
# tokens; the train prep skips those same documents. Idempotent.
#
#   bash setup/github_code/prep_val.sh
#   Env overrides: NANOCHAT_BASE_DIR, TOKENIZER_THREADS
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
THREADS="${TOKENIZER_THREADS:-8}"

OUT_DIR="$BASE_DIR/bins/github_code_rustbpe_val"
BIN="$OUT_DIR/github_code_rustbpe_val.bin"

if [[ -f "$OUT_DIR/meta.json" ]]; then
  echo "[github_code prep_val] already built: $OUT_DIR/meta.json — skipping."
  exit 0
fi

echo "[github_code prep_val] streaming + tokenizing 10M val tokens with rustbpe -> $BIN"
uv run python -m nanochat.data.pretokenize \
  --source github-code \
  --tokenizer rustbpe \
  --split val \
  --target-tokens 10000000 \
  --tokenizer-threads "$THREADS" \
  --output "$BIN"

echo "[github_code prep_val] done -> $BIN"
