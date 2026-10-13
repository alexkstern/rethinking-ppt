#!/usr/bin/env bash
#
# Build fineweb_rustbpe_val.bin from a named held-out shard (rustbpe).
# FineWeb-Edu has no official val split, so the repo's LAST shard
# (shard_01822.parquet of 0000..1822) is held out. Idempotent.
#
#   bash setup/fineweb/prep_val.sh
#   Env overrides: NANOCHAT_BASE_DIR, TOKENIZER_THREADS
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
THREADS="${TOKENIZER_THREADS:-16}"
REPO_ID="karpathy/fineweb-edu-100b-shuffle"
VAL_SHARD="shard_01822.parquet"                 # the repo's last shard (0000..1822)
VAL_DIR="$BASE_DIR/base_data_fineweb_val"        # holds ONLY the val shard

OUT_DIR="$BASE_DIR/bins/fineweb_rustbpe_val"
BIN="$OUT_DIR/fineweb_rustbpe_val.bin"

if [[ -f "$OUT_DIR/meta.json" ]]; then
  echo "[fineweb prep_val] already built: $OUT_DIR/meta.json — skipping."
  exit 0
fi

echo "[fineweb prep_val] downloading $VAL_SHARD from $REPO_ID ..."
uv run python -c "from huggingface_hub import hf_hub_download; hf_hub_download('$REPO_ID', '$VAL_SHARD', repo_type='dataset', local_dir='$VAL_DIR')"

echo "[fineweb prep_val] tokenizing the held-out shard with rustbpe -> $BIN"
uv run python -m nanochat.data.pretokenize \
  --source fineweb \
  --tokenizer rustbpe \
  --data-path-local "$VAL_DIR" \
  --split val \
  --target-tokens -1 \
  --tokenizer-threads "$THREADS" \
  --output "$BIN"

echo "[fineweb prep_val] done -> $BIN"
