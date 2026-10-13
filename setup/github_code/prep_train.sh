#!/usr/bin/env bash
#
# Build github_code_rustbpe_1B.bin — 1B-token GitHub-code TRAIN bin (rustbpe),
# streamed from the parquet branch of codeparrot/github-code, config all-all.
# Upstream has only a train split; this stream starts after the first 20,000
# documents, which the val prep holds out, so the splits are disjoint.
# Output (~2 GB) under the base dir. Idempotent.
#
#   bash setup/github_code/prep_train.sh
#   Env overrides: NANOCHAT_BASE_DIR, TOKENIZER_THREADS
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
THREADS="${TOKENIZER_THREADS:-8}"             # streaming is network-bound

OUT_DIR="$BASE_DIR/bins/github_code_rustbpe_1B"
BIN="$OUT_DIR/github_code_rustbpe_1B.bin"

if [[ -f "$OUT_DIR/meta.json" ]]; then
  echo "[github_code prep_train] already built: $OUT_DIR/meta.json — skipping."
  exit 0
fi

echo "[github_code prep_train] streaming + tokenizing 1B tokens with rustbpe ($THREADS threads) -> $BIN"
uv run python -m nanochat.data.pretokenize \
  --source github-code \
  --tokenizer rustbpe \
  --split train \
  --target-tokens 1000000000 \
  --tokenizer-threads "$THREADS" \
  --output "$BIN"

echo "[github_code prep_train] done -> $BIN"
