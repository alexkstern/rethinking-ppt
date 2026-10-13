#!/usr/bin/env bash
#
# Publish the FineWeb rustbpe bins (train + val + metas + tokenizer) as the
# PUBLIC HF dataset alexkstern/fineweb-nanochatbpe-20B, backing registry key
# fineweb-nanochatbpe-20B. Run the two prep scripts first.
#
#   bash setup/fineweb/hf_register.sh
#   HF_ORG=<account> bash setup/fineweb/hf_register.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
ORG="${HF_ORG:-alexkstern}"
BINS="$BASE_DIR/bins"

uv run python -m nanochat.data.hf_register \
  --train-bin "$BINS/fineweb_rustbpe_20B/fineweb_rustbpe_20B.bin" \
  --val-bin   "$BINS/fineweb_rustbpe_val/fineweb_rustbpe_val.bin" \
  --tokenizer-dir "$BASE_DIR/tokenizer" \
  --org "$ORG"
