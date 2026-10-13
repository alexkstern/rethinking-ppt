"""Checkpoint IO primitives.

Five files per save step `N`:

    {checkpoint_dir}/
    ├── model_{N:06d}.pt           # rank-0, torch.save(model.state_dict())
    ├── optim_{N:06d}_rank{R}.pt   # per-rank, dict keyed by joined group names
    ├── meta_{N:06d}.json          # rank-0, dataclasses.asdict(CheckpointMeta)
    ├── rng_{N:06d}.pt             # rank-0, torch.save(get_rng_state())
    └── config_{N:06d}.json        # rank-0, dataclasses.asdict(experiment cfg)

Per-rank optimizer state is keyed by the comma-joined param-group names each
optimizer handles (e.g. `"matrix"` for the body Muon, `"embedding,unembedding"`
for the head AdamW), so the factory may add / reorder / split optimizers without
breaking resume; mismatched names raise at load time.

This module performs no model reconstruction and no checkpoint auto-discovery:
experiment scripts inline whichever load they need (full state for crash
recovery, weights only for a new phase at unchanged shapes, or a filtered
`strict=False` load when a vocabulary change makes embedding and head weights
unloadable).

Because per-rank optimizer files are written independently, resume requires the
same world size that saved them.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class CheckpointMeta:
    """Per-step metadata for resume and bookkeeping.

    `seed` is denormalized from `RunConfig.seed` so `jq .seed` over the meta
    files works without parsing the config snapshot.

    `dataloader_state` is opaque here — the data loader produces and consumes
    it on its own contract — and must be JSON-serializable.

    `min_objective`, `smooth_train_loss` and `total_training_time` are the loop
    state a seamless resume needs: the EMA and the wall-clock accumulator
    continue, and the best objective stays a running minimum across the whole
    run. `min_objective` is `None` before the first eval (rather than an
    infinity that would not round-trip through JSON); a resumer maps it to
    `+inf`.
    """

    step: int
    phase: str | None
    seed: int
    flops_used: float
    flops_per_token: float
    iterations_planned: int
    dataloader_state: dict
    wandb_run_id: str | None
    min_objective: float | None
    smooth_train_loss: float
    total_training_time: float


def _uncompiled_model(model: torch.nn.Module) -> torch.nn.Module:
    """The underlying module if `model` is a `torch.compile` wrapper, else `model`.

    A compiled model's `state_dict` keys are prefixed `_orig_mod.`; unwrapping
    keeps the saved file loadable by an eager model, as the PPT->PT handoff needs.
    """
    return getattr(model, "_orig_mod", model)


def save_checkpoint(
    checkpoint_dir: str,
    step: int,
    *,
    model: torch.nn.Module,
    optimizers: list[torch.optim.Optimizer],
    meta: CheckpointMeta,
    rng_state: dict,
    config: object,
    rank: int,
) -> None:
    """Write all five files for step `step`.

    Every rank writes its own optim file; model/meta/rng/config are rank-0 only.
    The mkdir runs on every rank so a per-rank optim write cannot race it.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    optim_dict = {
        ",".join(g["name"] for g in opt.param_groups): opt.state_dict()
        for opt in optimizers
    }
    optim_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
    torch.save(optim_dict, optim_path)
    logger.info(f"Saved optimizer state to: {optim_path}")

    if rank != 0:
        return

    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    torch.save(_uncompiled_model(model).state_dict(), model_path)
    logger.info(f"Saved model parameters to: {model_path}")

    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(meta), f, indent=2)
    logger.info(f"Saved metadata to: {meta_path}")

    rng_path = os.path.join(checkpoint_dir, f"rng_{step:06d}.pt")
    torch.save(rng_state, rng_path)
    logger.info(f"Saved RNG state to: {rng_path}")

    config_path = os.path.join(checkpoint_dir, f"config_{step:06d}.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(config), f, indent=2)
    logger.info(f"Saved config snapshot to: {config_path}")


def load_model_state(checkpoint_dir: str, step: int, device: torch.device) -> dict:
    path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    return torch.load(path, map_location=device)


def load_optim_state(
    checkpoint_dir: str,
    step: int,
    rank: int,
    device: torch.device,
) -> dict[str, dict]:
    """The saved name-keyed dict. The caller reassembles by looking up
    `",".join(g["name"] for g in opt.param_groups)` per optimizer."""
    path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
    return torch.load(path, map_location=device)


def load_meta(checkpoint_dir: str, step: int) -> CheckpointMeta:
    path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(path, "r", encoding="utf-8") as f:
        return CheckpointMeta(**json.load(f))


def load_rng_state(checkpoint_dir: str, step: int) -> dict:
    path = os.path.join(checkpoint_dir, f"rng_{step:06d}.pt")
    # weights_only=False: the file holds numpy + python random tuples that the
    # weights-only allowlist rejects.
    return torch.load(path, map_location="cpu", weights_only=False)


def load_config_snapshot(checkpoint_dir: str, step: int) -> dict:
    """A plain dict; `RunConfig` is per-experiment and unknown here, so
    reconstruction is the caller's job."""
    path = os.path.join(checkpoint_dir, f"config_{step:06d}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_rng_state() -> dict:
    """Capture python `random`, numpy, torch CPU and torch CUDA RNG state. A
    single rank-0 view; no per-rank RNG state is kept."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    """Restore RNG state. The CUDA branch is skipped when CUDA is unavailable,
    so a checkpoint captured on GPU loads on CPU."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
