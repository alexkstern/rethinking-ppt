"""Measure FLOPs-per-token for nanochat_gpt configs, cached on
``(depth, vocab_size)`` so repeated trials at the same depth skip the probe.

Separate from the per-trial measurement inside ``training_run.build_phase``:
this is for the driver, which needs estimates without launching training.
"""

from __future__ import annotations

import torch

from nanochat.flops import flops_per_token as measure_fpt
from nanochat.models.nanochat_gpt.model_utils import nanochat_dim_heads
from nanochat.models.nanochat_gpt.vanilla_3a5e0bc import NanochatGPTConfig, build_model


class FlopsEstimator:
    """Memoized flops-per-token measurement for nanochat_gpt."""

    def __init__(
        self,
        *,
        seq_len: int = 2048,
        vocab_size: int = 65536,
    ):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self._cache: dict[tuple[int, int], float] = {}

    def flops_per_token(self, depth: int, *, vocab_size: int | None = None) -> float:
        """Return measured FLOPs-per-token, cached on ``(depth, vocab_size)``."""
        v = vocab_size if vocab_size is not None else self.vocab_size
        key = (depth, v)
        if key not in self._cache:
            self._cache[key] = self._measure(depth, v)
        return self._cache[key]

    def _measure(self, depth: int, vocab_size: int) -> float:
        model_dim, num_heads = nanochat_dim_heads(depth)
        cfg = NanochatGPTConfig(
            sequence_len=self.seq_len,
            vocab_size=vocab_size,
            n_layer=depth,
            n_head=num_heads,
            n_kv_head=num_heads,
            n_embd=model_dim,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(cfg)
        model.to_empty(device=device)
        model.init_weights()
        try:
            return measure_fpt(
                model, self.seq_len, vocab_size,
                use_measured=True, device=device,
            )
        finally:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
