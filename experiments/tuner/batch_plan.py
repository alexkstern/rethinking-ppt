"""Depth-aware micro-batch planning for OOM-safe llmtuna trials.

Caps the micro-batch by depth while holding the effective batch
(``device_batch_size * grad_accum_steps``) constant.
"""

from __future__ import annotations

# (min_depth, max_device_batch_size), ascending: a depth takes the last entry
# whose min_depth <= depth.
DEVICE_BATCH_SCHEDULE: list[tuple[int, int]] = [
    (0, 64),
    (12, 32),
    (23, 16),
    (27, 8),
    (32, 4),
    (37, 2),
]


def device_batch_cap(
    depth: int, schedule: list[tuple[int, int]] = DEVICE_BATCH_SCHEDULE
) -> int:
    """Largest allowed micro-batch (device_batch_size) for ``depth``."""
    cap = schedule[0][1]
    for min_depth, dbs in schedule:
        if depth >= min_depth:
            cap = dbs
        else:
            break
    return cap


def plan_microbatch(
    depth: int,
    seqs_per_step: int,
    schedule: list[tuple[int, int]] = DEVICE_BATCH_SCHEDULE,
) -> tuple[int, int]:
    """Split ``seqs_per_step`` into ``(device_batch_size, grad_accum_steps)``.

    Picks the largest micro-batch that is both within the depth cap and an
    exact divisor of ``seqs_per_step``, so their product is preserved.
    """
    if seqs_per_step < 1:
        raise ValueError(f"seqs_per_step must be >= 1, got {seqs_per_step}")
    dbs = min(device_batch_cap(depth, schedule), seqs_per_step)
    while dbs > 1 and seqs_per_step % dbs != 0:
        dbs -= 1
    return dbs, seqs_per_step // dbs


def plan_batch(
    total_batch_tokens: int,
    *,
    ngpus: int,
    seq_len: int,
    depth: int,
    schedule: list[tuple[int, int]] = DEVICE_BATCH_SCHEDULE,
) -> tuple[int, int]:
    """Split a run-wide effective batch of ``total_batch_tokens`` per optimizer
    step into a per-GPU ``(device_batch_size, grad_accum_steps)``.

    ``total_batch_tokens`` must factor exactly as
    ``ngpus * seq_len * device_batch_size * grad_accum_steps``; anything else
    raises rather than silently changing the effective batch.
    """
    per_gpu_tokens, rem = divmod(total_batch_tokens, ngpus)
    if rem:
        raise ValueError(
            f"total_batch_tokens ({total_batch_tokens}) not divisible by ngpus ({ngpus})"
        )
    seqs_per_gpu, rem = divmod(per_gpu_tokens, seq_len)
    if rem:
        raise ValueError(
            f"per-GPU token batch ({per_gpu_tokens}) not divisible by seq_len ({seq_len})"
        )
    return plan_microbatch(depth, seqs_per_gpu, schedule)
