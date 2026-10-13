"""NCA pre-pretraining at a 1% dose into FineWeb, tuned at 1e18 / 3e18 / 9e18.

The search covers the PPT learning-rate overlay and the PPT effective batch
alongside the usual model and schedule knobs; the PPT phase trains a narrow
NCA vocabulary and hands over with the embeddings reinitialized and the
optimizer reset. Each scale warm-starts from the best config of the one below.
"""

from __future__ import annotations

from experiments.training.common import PPT_VOCAB_NCA
from experiments.training.task import TaskSpec

TASK = TaskSpec(
    name="nca_1pct",
    wandb_project="llmtuna_nca_1pct_1e18_3e18_9e18",
    alpha_ppt=0.01,
    ppt_vocab_size=PPT_VOCAB_NCA,
    data_ppt="nca-paper-share20-2048",
    data_pt="fineweb-nanochatbpe-20B",
    description=(
        "NCA PPT (1%) -> FineWeb PT; scales 1e18 -> 3e18 -> 9e18 with "
        "warm-started transfer. Separate PPT LR overlay, embed reinit + "
        "optimizer reset. Tunable PPT batch multiple. All warmups start at 0."
    ),
    hparams=[
        "depth", "matrix_lr", "warmup_ratio", "warmdown_ratio",
        "ppt_lr", "ppt_warmup_ratio", "ppt_warmdown_ratio", "ppt_batch_mult",
    ],
    pinned_params={"embedding_lr": 0.3, "grad_clip": 1.0},
    scales=["1e18", "3e18", "9e18"],
    max_iters_per_scale=30,
    patience=5,
    tol=0.001,
    # Select trials on a val prefix disjoint from the reported
    # [EVAL_OFFSET_TOKENS, +EVAL_TOKENS) window.
    eval_tokens=2_016_000,
    eval_offset_tokens=0,
    nproc=4,
    out_dir="outputs/llmtuna_nca_1pct_multiscale",
    # Trial 0 of the first scale; later scales warm-start from their predecessor.
    initial_params={
        "depth": 13, "matrix_lr": 0.03,
        "warmup_ratio": 0.0, "warmdown_ratio": 0.4,
        "ppt_lr": 0.03, "ppt_warmup_ratio": 0.0, "ppt_warmdown_ratio": 0.4,
        "ppt_batch_mult": 2,
    },
)
