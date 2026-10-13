"""LR-multiplier schedules, one `(params dataclass, builder)` pair per shape.

A builder returns a function `step -> multiplier` in `[0, 1]`. The train loop
multiplies that against `param_group["initial_lr"]` (set by
`optim/factory.py`) and writes the result into `param_group["lr"]` each step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class TrapezoidScheduleParams:
    """Linear warmup -> flat plateau at 1.0 -> linear warmdown to `final_lr_frac`.

    `warmup_ratio + warmdown_ratio == 1.0` collapses the plateau into a pure
    warmup-then-warmdown shape.
    """

    warmup_ratio: float = 0.0
    warmdown_ratio: float = 0.0
    final_lr_frac: float = 0.0


@dataclass
class CosineScheduleParams:
    """Linear warmup -> cosine decay across all remaining steps to `final_lr_frac`.

    No plateau. The decay spans `num_iterations - warmup_iters` steps and
    reaches `final_lr_frac` at `step == num_iterations`, one past the last
    training step.
    """

    warmup_ratio: float = 0.0
    final_lr_frac: float = 0.0


def _validate_ratio(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def trapezoid_schedule(num_iterations: int, params: TrapezoidScheduleParams):
    """Trapezoid LR-multiplier function for a run of `num_iterations` steps."""
    if num_iterations <= 0:
        raise ValueError(f"num_iterations must be positive, got {num_iterations}")
    _validate_ratio("warmup_ratio", params.warmup_ratio)
    _validate_ratio("warmdown_ratio", params.warmdown_ratio)
    _validate_ratio("final_lr_frac", params.final_lr_frac)
    total_ratio = params.warmup_ratio + params.warmdown_ratio
    if total_ratio > 1.0:
        raise ValueError(
            f"warmup_ratio + warmdown_ratio must be <= 1, got "
            f"{params.warmup_ratio} + {params.warmdown_ratio} = {total_ratio}"
        )

    warmup_iters = round(params.warmup_ratio * num_iterations)
    plateau_iters = round((1.0 - params.warmup_ratio - params.warmdown_ratio) * num_iterations)
    warmdown_iters = num_iterations - warmup_iters - plateau_iters
    plateau_end = warmup_iters + plateau_iters
    final = params.final_lr_frac

    def schedule(step: int) -> float:
        if step < warmup_iters:
            return step / max(1, warmup_iters)
        if step <= plateau_end:
            return 1.0
        x = (step - plateau_end) / warmdown_iters
        decay = 1.0 - x
        return final + (1.0 - final) * decay

    return schedule


def cosine_schedule(num_iterations: int, params: CosineScheduleParams):
    """Cosine LR-multiplier function for a run of `num_iterations` steps."""
    if num_iterations <= 0:
        raise ValueError(f"num_iterations must be positive, got {num_iterations}")
    _validate_ratio("warmup_ratio", params.warmup_ratio)
    _validate_ratio("final_lr_frac", params.final_lr_frac)

    warmup_iters = round(params.warmup_ratio * num_iterations)
    decay_iters = num_iterations - warmup_iters
    final = params.final_lr_frac

    def schedule(step: int) -> float:
        if step < warmup_iters:
            return step / max(1, warmup_iters)
        if decay_iters <= 0:
            return 1.0
        progress = (step - warmup_iters) / decay_iters
        coeff = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return final + (1.0 - final) * coeff

    return schedule
