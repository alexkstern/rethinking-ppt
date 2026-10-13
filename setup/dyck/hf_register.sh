#!/usr/bin/env bash
#
# Publish the k-Dyck bins (train + val + metas) as the PUBLIC HF dataset
# alexkstern/dyck-k128-seq_len_2048-1B, backing registry key
# dyck-k128-seq_len_2048-1B. Procedural source, so no tokenizer files (0-based
# bracket ids, vocab 2k=256). Run the two prep scripts first.
#
#   bash setup/dyck/hf_register.sh
#   HF_ORG=<account> bash setup/dyck/hf_register.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
ORG="${HF_ORG:-alexkstern}"
BINS="$BASE_DIR/bins"

uv run python -m nanochat.data.hf_register \
  --train-bin "$BINS/dyck_k128_seq_len_2048_1B/dyck_k128_seq_len_2048_1B.bin" \
  --val-bin   "$BINS/dyck_k128_seq_len_2048_val/dyck_k128_seq_len_2048_val.bin" \
  --org "$ORG"
