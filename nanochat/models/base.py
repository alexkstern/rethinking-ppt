"""Base class every arch in `nanochat/models/` inherits from.

Subclasses must call `super().__init__()`, set `self.config` to their config
dataclass instance, and implement
`forward(idx, targets=None, kv_cache=None, loss_reduction="mean")`, returning
a scalar loss when `targets` is given and logits otherwise. Weight init stays
per-arch.

`loss_reduction="none"` is required and must return a per-position `(B, T)`
loss, which the bits-per-byte eval uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelSpec:
    """How the training factory talks to one model arch.

    `embed_key` is the state-dict key of the token-embedding weight; its
    `shape[0]` is read at load time to decide between a strict load and vocab
    surgery. `surgery_keys` are the embed/head keys dropped (and re-inited) for
    a body-only load across a vocab change.
    """

    build_model: Callable
    build_param_groups: Callable
    embed_key: str
    surgery_keys: frozenset[str]


class BaseModel(nn.Module):

    config: object  # populated by subclass __init__

    def get_device(self) -> torch.device:
        return next(self.parameters()).device

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        """Total parameter count; `exclude_embeddings=True` skips `nn.Embedding`
        params, matching the `6 * (N - N_embed)` formula in `nanochat/flops.py`."""
        if not exclude_embeddings:
            return sum(p.numel() for p in self.parameters())
        embedding_params: set[int] = set()
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                embedding_params.update(id(p) for p in module.parameters())
        return sum(
            p.numel() for p in self.parameters() if id(p) not in embedding_params
        )
