#!/usr/bin/env bash
#
# Score a checkpoint on every measure the paper reports: bits-per-byte on the
# held-out window, BLiMP and DCLM CORE. Sets up the environment, then hands
# every argument to `reproduce.evaluate`:
#
#   bash scripts/eval.sh --cell scaling__nca_1_1e18 --seed 0   # a reported model, from the Hub
#   bash scripts/eval.sh --checkpoint <dir>                    # weights you trained
#   bash scripts/eval.sh --hf-repo <org/name>                  # any published checkpoint
#   bash scripts/eval.sh --cell scaling__nca_1_1e18 --suite bpb   # one suite
#   bash scripts/eval.sh --cell scaling__nca_1_1e18 --smoke    # a quick check of the path, not a result
#
# `uv run python -m reproduce.evaluate --help` lists the rest.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source experiments/training/_env_setup.sh

uv run python -m reproduce.evaluate "$@"
