#!/usr/bin/env bash
#
# Publish the decl-ref-01 bins (train + val + metas) as the PUBLIC HF dataset
# alexkstern/declref-01-declref_01_symbols-2B, backing registry key
# declref-01-seq_len_2048-2B. Procedural source, so no tokenizer files (0-based
# symbol ids, vocab 1028). Run the two prep scripts first; HF_TOKEN comes from
# the repo-root .env, which hf_register loads itself.
#
#   bash setup/declref_01/hf_register.sh
#   HF_ORG=<account> bash setup/declref_01/hf_register.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
ORG="${HF_ORG:-alexkstern}"
BINS="$BASE_DIR/bins"

uv run python -m nanochat.data.hf_register \
  --train-bin "$BINS/declref_01_seq_len_2048_2B/declref_01_seq_len_2048_2B.bin" \
  --val-bin   "$BINS/declref_01_seq_len_2048_val/declref_01_seq_len_2048_val.bin" \
  --org "$ORG"
