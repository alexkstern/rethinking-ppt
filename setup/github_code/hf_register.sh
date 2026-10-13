#!/usr/bin/env bash
#
# Publish the GitHub-code rustbpe bins (train + val + metas + tokenizer) as the
# PUBLIC HF dataset alexkstern/github-code-nanochatbpe-1B, backing registry key
# github-code-nanochatbpe-1B. Run the two prep scripts first.
#
#   bash setup/github_code/hf_register.sh
#   HF_ORG=<account> bash setup/github_code/hf_register.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
ORG="${HF_ORG:-alexkstern}"
BINS="$BASE_DIR/bins"

uv run python -m nanochat.data.hf_register \
  --train-bin "$BINS/github_code_rustbpe_1B/github_code_rustbpe_1B.bin" \
  --val-bin   "$BINS/github_code_rustbpe_val/github_code_rustbpe_val.bin" \
  --tokenizer-dir "$BASE_DIR/tokenizer" \
  --org "$ORG"
