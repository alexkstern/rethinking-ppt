"""Shared iterator type + path resolver for nanochat/data/."""

from pathlib import Path
from typing import Any, Iterator, TypeAlias

import numpy as np
import torch
from huggingface_hub import hf_hub_download

IteratorYield: TypeAlias = tuple[torch.Tensor, torch.Tensor, dict | None]
DataIterator: TypeAlias = Iterator[IteratorYield]

# A source yields token ids, not text or tensors: each item is a 1-D uint16
# array of ids for one document / rollout / chunk — not padded, not packed.
TokenStream: TypeAlias = Iterator[np.ndarray]


def resolve_data_path(path: dict[str, Any]) -> Path:
    """Resolve a single-source path dict to a local filesystem path.

    `path` must have exactly one key:
      * `{"local": "/filesystem/path.bin"}` — value is the local path.
      * `{"hf": {"repo_id": "<repo>", "filename": "<file>"}}` — value is the
        kwargs dict for `huggingface_hub.hf_hub_download` (with `repo_type=
        "dataset"` injected). Both `repo_id` and `filename` are required.
    """
    if len(path) != 1:
        raise ValueError(f"path needs exactly one key, got {sorted(path)}")
    if "local" in path:
        return Path(path["local"])
    if "hf" in path:
        return Path(hf_hub_download(repo_type="dataset", **path["hf"]))
    raise ValueError(f"unknown path source {sorted(path)!r}; expected 'local' or 'hf'")
