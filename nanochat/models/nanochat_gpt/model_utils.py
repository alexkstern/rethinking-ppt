"""Sizing helpers for the nanochat_gpt family."""
from __future__ import annotations


def nanochat_dim_heads(depth: int) -> tuple[int, int]:
    """Map depth -> (model_dim, num_heads): depth * 64 rounded up to a multiple
    of 128, at a fixed head_dim of 128."""
    aspect_ratio = 64
    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + 128 - 1) // 128) * 128
    num_heads = model_dim // 128
    return model_dim, num_heads
