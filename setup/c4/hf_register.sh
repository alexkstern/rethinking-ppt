#!/usr/bin/env bash
#
# Publish the C4 rustbpe bins (train + val + metas + tokenizer) as the PUBLIC
# HF dataset alexkstern/c4-nanochatbpe-10B, backing registry key
# c4-nanochatbpe-10B. Run the two prep scripts first.
#
#   bash setup/c4/hf_register.sh
#   HF_ORG=<account> bash setup/c4/hf_register.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
ORG="${HF_ORG:-alexkstern}"
BINS="$BASE_DIR/bins"

uv run python -m nanochat.data.hf_register \
  --train-bin "$BINS/c4_rustbpe_10B/c4_rustbpe_10B.bin" \
  --val-bin   "$BINS/c4_rustbpe_val/c4_rustbpe_val.bin" \
  --tokenizer-dir "$BASE_DIR/tokenizer" \
  --org "$ORG"
