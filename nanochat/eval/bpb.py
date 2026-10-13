"""In-distribution bits-per-byte (BPB) eval for nanochat training loops."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    # A runtime import would be circular: base.py imports this module.
    from nanochat.eval.base import EvalFn


@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes) -> float:
    """Summed loss over summed target bytes: a loss comparable across vocab
    sizes.

    `token_bytes` is a `(vocab_size,)` tensor of bytes per token id, 0 for token
    ids that should not count (special tokens); targets at `ignore_index` (-1)
    are dropped too.
    """
    nats, byte_count = _accumulate(model, batches, steps, token_bytes)
    if byte_count == 0:
        return float("inf")
    return nats / (math.log(2) * byte_count)


def _accumulate(model, batches, steps, token_bytes) -> tuple[float, int]:
    """Sum loss and bytes over `steps` batches, all-reduced.

    A position counts only when its target's byte length is nonzero, so both
    sums describe the same position set.
    """
    device = model.get_device()
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=device)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=device)
    batch_iter = iter(batches)
    for _ in range(steps):
        x, y, _ = next(batch_iter)
        loss2d = model(x, y, loss_reduction='none').view(-1)
        y = y.view(-1)
        if (y.int() < 0).any(): # mps has no int64 kernel for < 0, only int32
            # Targets below 0 are ignore_index: never index token_bytes with them.
            valid = y >= 0
            num_bytes = torch.where(
                valid,
                token_bytes[torch.where(valid, y, torch.zeros_like(y))],
                torch.zeros_like(y, dtype=token_bytes.dtype),
            )
        else:
            num_bytes = token_bytes[y]
        total_nats += (loss2d * (num_bytes > 0)).sum()
        total_bytes += num_bytes.sum()
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
    return total_nats.item(), total_bytes.item()


def build_bpb_eval(
    val_loader,
    steps: int,
    token_bytes: torch.Tensor,
    *,
    metric_key: str = "val/bpb",
) -> EvalFn:
    """Wrap `evaluate_bpb` as an `EvalFn` closing over `val_loader`, an iterator
    of `(x, y, state)` batches."""
    def eval_fn(model) -> dict[str, float]:
        return {metric_key: evaluate_bpb(model, val_loader, steps, token_bytes)}
    return eval_fn
