"""W&B init.

`init_wandb_run` returns `None` on non-rank-0 or when W&B is disabled, and
callers branch on `is not None` at every log / finish site; there is no no-op
shim. `run_name` is the run's own name; the local checkpoint dir and the HF
repo name add a timestamp to it (see `paths.resolve_checkpoint_dir`).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import wandb

from nanochat.config import RunConfig


def init_wandb_run(cfg: RunConfig, ctx: Any, *, run_name: str) -> wandb.Run | None:
    """Initialize a W&B run on rank 0 when enabled, else return `None`. `ctx`
    must expose `rank: int`."""
    if ctx.rank != 0:
        return None
    if not cfg.wandb.enabled:
        return None

    if not cfg.model_project:
        raise ValueError("cfg.model_project must be a non-empty string when wandb is enabled")
    if not run_name:
        raise ValueError("run_name must be a non-empty string")

    return wandb.init(
        project=cfg.model_project,
        name=run_name,
        group=cfg.wandb.group,
        tags=cfg.wandb.tags,
        entity=cfg.wandb.entity,
        notes=cfg.wandb.notes,
        config=dataclasses.asdict(cfg),
    )
