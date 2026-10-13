#!/usr/bin/env bash
#
# Train the nanochat BPE tokenizer that every `*-nanochatbpe-*` corpus in the
# data registry was built with, writing `tokenizer.pkl` and `token_bytes.pt`
# into `{base_dir}/tokenizer`. Runs on CPU. The arguments below are the
# script's defaults, spelled out so the run is self-describing; vocab_size
# 65536 matches the `--pad-vocab=65536` every experiment launcher passes.
#
# Raw FineWeb-Edu shards must already be on disk:
#   python -m nanochat.data.sources.fineweb.source -n <num_shards>
# The tokenizer reads the train split (every shard but the last, held out for
# validation) in sorted order, stopping at `max_chars`. At ~250M characters per
# shard the 10B default consumes about 40, so fetch at least ~41; more shards
# do not change the result, since the same leading shards are read either way.
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
