"""Vanilla FineWeb PT (no PPT) — multi-scale sweep 1e18 / 3e18 / 9e18.

Hand to the tuner driver as ``--config experiments.tasks.vanilla``; anything
not stated here is a TaskSpec default. Each scale warm-starts from the best
config of the one below.
"""

from __future__ import annotations

from experiments.tuner.task import TaskSpec

TASK = TaskSpec(
    name="vanilla",
    wandb_project="llmtuna_vanilla_1e18_3e18_9e18",
    alpha_ppt=0.0,
    ppt_vocab_size=None,
    data_pt="fineweb-nanochatbpe-20B",
    description="Vanilla FineWeb PT (no PPT), multi-scale",
    hparams=["depth", "matrix_lr", "warmup_ratio", "warmdown_ratio"],
    # Pinned to the same values every method uses.
    pinned_params={"embedding_lr": 0.3, "grad_clip": 1.0},
    nproc=4,
    max_iters_per_scale=30,
    patience=5,
    tol=0.001,
    # Select trials on a val prefix disjoint from the reported
    # [EVAL_OFFSET_TOKENS, +EVAL_TOKENS) window.
    eval_tokens=2_016_000,
    eval_offset_tokens=0,
    # First-scale warm-start baseline; must cover every tuned hparam.
    initial_params={
        "depth": 9, "matrix_lr": 0.03,
        "warmup_ratio": 0.0, "warmdown_ratio": 0.4,
    },
)
