"""Path composition for run artifacts under `get_base_dir()`.

Pure string composition — no I/O, no directory creation. Callers mkdir on
first use.
"""

from __future__ import annotations

import os
from pathlib import Path

from nanochat.common import download_file_with_lock, get_base_dir
from nanochat.config import RunConfig
from nanochat.runtime import RuntimeCtx

# Re-exported: this module is the supported home for base-dir path/download
# helpers, so consumers need not reach into nanochat.common.
__all__ = ["download_file_with_lock", "get_base_dir", "resolve_checkpoint_dir"]


def resolve_checkpoint_dir(cfg: RunConfig, ctx: RuntimeCtx) -> Path:
    """Compose the per-run checkpoint dir:

        {base_dir}/checkpoints/{wandb|no_wandb}/{model_project}/{run_name}_{ts}/

    The leaf `{run_name}_{ts}` is the canonical per-run identity: pass its
    `.name` to `init_wandb_run` and to the HF push so every artifact for one
    run shares one handle. `ts` is sampled once at process start by
    `runtime.setup` and broadcast across DDP ranks, so repeated calls in the
    same process agree.
    """
    if not cfg.model_project:
        raise ValueError("cfg.model_project must be a non-empty string")
    if not cfg.run_name:
        raise ValueError("cfg.run_name must be a non-empty string")

    tracker_dir = "wandb" if cfg.wandb.enabled else "no_wandb"
    # Overrides just the checkpoint root, leaving data bins in the cache.
    ckpt_root = os.environ.get("NANOCHAT_CHECKPOINT_DIR")
    root = Path(ckpt_root) if ckpt_root else Path(get_base_dir()) / "checkpoints"
    return root / tracker_dir / cfg.model_project / f"{cfg.run_name}_{ctx.run_timestamp}"
