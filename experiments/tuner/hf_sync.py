"""Push a study's outputs directory to a HuggingFace model repo."""

from __future__ import annotations

import os
from pathlib import Path

IGNORE_PATTERNS = ["wandb/*", "*.pyc", "__pycache__/*"]


def sync_to_hf(*, folder: Path, repo_name: str, commit_message: str, api=None) -> str:
    """Upload ``folder`` to ``<hf_user>/<repo_name>``; return the repo URL.

    Creates the repo if missing. ``repo_name`` is typically the wandb project,
    so artifacts colocate. The default ``api`` is an ``HfApi`` built with the
    ``HF_TOKEN`` env var.
    """
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN"))
    user = api.whoami()["name"]
    repo_id = f"{user}/{repo_name}"
    api.create_repo(repo_id, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        commit_message=commit_message,
        ignore_patterns=IGNORE_PATTERNS,
    )
    return f"https://huggingface.co/{repo_id}"
