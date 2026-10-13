#!/usr/bin/env bash
#
# Fetch the published tokenizer into `{base_dir}/tokenizer`, which the
# `setup/*/hf_register.sh` scripts bundle into each text corpus they publish.
# Only publishing needs it: corpus prep, training and eval read the tokenizer
# from the Hub.
#
#   bash setup/tokenizer/install_tokenizer.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

uv run python - <<'PY'
import os
import shutil

from huggingface_hub import hf_hub_download

from nanochat.common import get_base_dir

REPO_ID = "alexkstern/fineweb-nanochatbpe-100M"

dest = os.path.join(get_base_dir(), "tokenizer")
os.makedirs(dest, exist_ok=True)
for name in ("tokenizer.pkl", "token_bytes.pt"):
    src = hf_hub_download(REPO_ID, f"tokenizer/{name}", repo_type="dataset")
    shutil.copyfile(src, os.path.join(dest, name))
    print(f"installed {os.path.join(dest, name)}")
PY
