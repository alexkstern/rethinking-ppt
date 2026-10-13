"""Shared eval-layer scaffolding: the in-loop eval-fn protocol (`nn.Module ->
dict`, merged into the W&B log payload) and the builder that turns named
validation sets into one bits-per-byte eval fn each.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Callable, TypeAlias

import torch.nn as nn

from nanochat.config import EvalConfig
from nanochat.eval.bpb import build_bpb_eval

EvalFn: TypeAlias = Callable[[nn.Module], dict[str, Any]]


def build_eval(
    cfg: EvalConfig,
    ctx: Any,
    *,
    val_sets: dict[str, tuple],
) -> list[EvalFn]:
    """Build one BPB eval fn per named validation set.

    `val_sets` maps a W&B metric key to `(loader, token_bytes)`, optionally
    followed by `eval_tokens` — a per-set token budget overriding
    `cfg.eval_tokens` — and `batches_per_pass`, the loader's one-pass batch
    count. `token_bytes` is per set, so datasets on different tokenizers are
    handled. Batches are read once, here, into a fixed list that every eval
    call re-scores.

    With a token budget, the batch count is `ceil(budget /
    positions_per_batch_global)` and the tail mask limits scoring to the first
    `budget` positions (see `_mask_tail_for_token_budget`). With no budget, the
    first `cfg.eval_steps` batches are scored in full.

    A window wider than one pass over the val set raises here, at setup.
    """
    eval_fns = []
    for metric_key, entry in val_sets.items():
        loader, token_bytes = entry[0], entry[1]
        budget = entry[2] if len(entry) > 2 else cfg.eval_tokens
        available = entry[3] if len(entry) > 3 else None
        it = iter(loader)
        first = next(it)
        if budget is None:
            steps = cfg.eval_steps
        else:
            steps = max(1, math.ceil(budget / (first[1].numel() * ctx.world_size)))
        if available is not None and steps > available:
            raise ValueError(
                f"val set {metric_key!r} needs {steps} batches but one pass over it "
                f"yields {available} at eval_device_batch_size="
                f"{cfg.eval_device_batch_size} x world_size={ctx.world_size}; the "
                f"window would wrap and re-score tokens. Use a smaller "
                f"eval_device_batch_size or a smaller eval window."
            )
        batches = [first] + list(itertools.islice(it, steps - 1))
        if budget is not None:
            batches[-1] = _mask_tail_for_token_budget(
                batches[-1], steps, budget, ctx.rank, ctx.world_size
            )
        eval_fns.append(
            build_bpb_eval(batches, steps, token_bytes, metric_key=metric_key)
        )
    return eval_fns


def _mask_tail_for_token_budget(
    last_batch: tuple, eval_steps: int, eval_tokens: int, rank: int, world_size: int,
) -> tuple:
    """Mask target positions past the token budget, by canonical stream index.

    A target position at (step `s`, rank `r`, flat in-batch index `i`) sits at
    canonical stream index ``g = s*(B*T*W) + r*(B*T) + i``, mirroring the val
    loader's slab layout. Positions with ``g >= eval_tokens`` become `-1`, which
    `evaluate_bpb` drops from both the loss and byte sums. Only the last batch
    can cross the boundary, so only it is masked; a rank whose whole last batch
    lies past the boundary is masked in full. Returns a new tuple.
    """
    x, y, state = last_batch
    per_rank = y.numel()
    start_g = (eval_steps - 1) * per_rank * world_size + rank * per_rank
    keep = min(max(eval_tokens - start_g, 0), per_rank)
    if keep == per_rank:
        return last_batch
    y_masked = y.clone()
    y_masked.view(-1)[keep:] = -1
    return (x, y_masked, state)
