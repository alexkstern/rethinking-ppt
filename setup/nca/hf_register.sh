#!/usr/bin/env bash
#
# Publish the share20 NCA bins (train + val + metas) as the PUBLIC HF dataset
# alexkstern/nca-paper-share20-seq_len_2048-657M, backing registry key
# nca-paper-share20-2048; the repo name derives from the train bin's meta.
# Procedural source, so no tokenizer files (2x2-patch ids + 2 delimiters,
# vocab 10002). Run after the two prep scripts. Needs HF_TOKEN in the
# environment or a .env file.
#
#   bash setup/nca/hf_register.sh
#   HF_ORG=<account> bash setup/nca/hf_register.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Upload deps only; registration needs no jax.
pip install -q huggingface_hub hf_xet python-dotenv

BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
ORG="${HF_ORG:-alexkstern}"
BINS="$BASE_DIR/bins"

python -m nanochat.data.hf_register \
  --train-bin "$BINS/nca_paper_share20_seq2048_657M/nca_paper_share20_seq2048_657M.bin" \
  --val-bin   "$BINS/nca_paper_share20_seq2048_657M_val/nca_paper_share20_seq2048_657M_val.bin" \
  --org "$ORG"
