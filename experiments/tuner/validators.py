"""Joint constraints coupling several hparams; per-param bounds are already
enforced by the llmtuna search space. ``ScaleRunner`` feeds the returned error
back to the LLM as a rejection note, so the next ``suggest()`` sees it.
"""

from __future__ import annotations


def validate_constraints(cfg: dict) -> str | None:
    """Return an error message if ``cfg`` violates a constraint, else None.
    Only fields that are present are checked."""
    model_dim = cfg.get("model_dim")
    if model_dim is not None and model_dim % 128 != 0:
        return (
            f"model_dim ({model_dim}) must be a multiple of 128 (head_dim is "
            f"fixed at 128, so n_head = model_dim / 128 must be an integer)."
        )
    for prefix in ("", "ppt_"):
        warmup = cfg.get(f"{prefix}warmup_ratio")
        warmdown = cfg.get(f"{prefix}warmdown_ratio")
        if warmup is None or warmdown is None:
            continue
        total = warmup + warmdown
        if total > 1.0:
            return (
                f"{prefix}warmup_ratio ({warmup}) + {prefix}warmdown_ratio "
                f"({warmdown}) = {total} > 1.0. They must sum to at most 1.0."
            )
    return None
