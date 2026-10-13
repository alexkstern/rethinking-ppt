#!/usr/bin/env bash
#
# Train the nanochat BPE tokenizer that every `*-nanochatbpe-*` corpus in the
# data registry was built with, writing `tokenizer.pkl` and `token_bytes.pt`
# into `{base_dir}/tokenizer`. Runs on CPU. The arguments below are the
# script's defaults.
#
# Raw FineWeb-Edu shards must already be on disk:
#   python -m nanochat.data.sources.fineweb.source -n <num_shards>
# Fetch at least ~41 (the last shard is held out for validation).
#
#   bash setup/tokenizer/train_tokenizer.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

uv run python -m setup.tokenizer.train_tokenizer \
  --max_chars=10000000000 \
  --doc_cap=10000 \
  --vocab_size=65536
