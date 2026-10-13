"""Pure dict-builders for training-loop W&B logging.

Three blocks — validation metrics, the CORE metric, periodic training metrics
— each emitted on its own cadence. All three carry `step` and
`total_training_flops` so every chart can share either x-axis, and metric names
are namespaced by block (`val/`, `train/`, `flops/`, `data/`). The caller holds
the `wandb.Run | None` and does the logging; nothing here imports wandb.
"""

from __future__ import annotations

from typing import Any


def build_val_metrics(
    step: int,
    total_training_flops: float,
    total_training_time: float,
    metrics: dict[str, float],
) -> dict[str, Any]:
    """Validation-eval block.

    `metrics` is the merged output of the run's eval fns, one entry per
    validation set keyed by its W&B metric name (e.g. ``{"val/bpb": ...}``, or
    ``{"val/fineweb_bpb": ..., "val/dyck_bpb": ...}``). `nanochat/eval/base.py`
    owns those names; this only adds the shared step/flops/time envelope.
    """
    return {
        "step": step,
        "total_training_flops": total_training_flops,
        "total_training_time": total_training_time,
        **metrics,
    }


def build_train_metrics(
    step: int,
    total_training_flops: float,
    total_training_time: float,
    train_loss: float,
    lrm: float,
    dt: float,
    tok_per_sec: int,
    mfu: float,
    grad_norm: float | None = None,
    flops_budget_remaining: float | None = None,
) -> dict[str, Any]:
    """Periodic training-step metrics.

    `train/grad_norm`, `flops/budget_remaining` and `data/epochs_consumed` are
    dropped from the dict entirely when the run has no gradient clipping, no
    FLOP-budget horizon, or no finite corpus to count epochs against, rather
    than charted as a meaningless zero.
    """
    d: dict[str, Any] = {
        "step": step,
        "total_training_flops": total_training_flops,
        "total_training_time": total_training_time,
        "train/loss": train_loss,
        "train/lrm": lrm,
        "train/dt": dt,
        "train/tok_per_sec": tok_per_sec,
        "train/mfu": mfu,
    }
    if grad_norm is not None:
        d["train/grad_norm"] = grad_norm
    if flops_budget_remaining is not None:
        d["flops/budget_remaining"] = flops_budget_remaining
    return d
