#!/usr/bin/env bash
#
# Train reported cells from reproduce/cells.json. Sets up the environment, then
# hands every argument to `reproduce.launch`:
#
#   bash scripts/train.sh scaling__c4_1_1e18                  # one cell, three seeds
#   bash scripts/train.sh scaling__c4_1_1e18 --seeds 0        # one seed
#   bash scripts/train.sh --all                               # every cell
#   bash scripts/train.sh --setting interleave                # one setting
#   bash scripts/train.sh --all --validate                    # build every config, train nothing
#
# Each run keeps its final weights for scripts/eval.sh and prints where they
# are. `--nproc N` overrides the GPU count, `--fresh` retrains runs that already
# finished, `--no-wandb` trains without logging.
# `uv run python -m reproduce.launch --help` lists the rest.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source experiments/training/_env_setup.sh

uv run python -m reproduce.launch --keep-checkpoints "$@"
