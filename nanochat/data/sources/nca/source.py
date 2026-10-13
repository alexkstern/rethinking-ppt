"""NCA pre-pretraining data source. Wraps the lifted generation core
(nca_sim / nca_tokenizer) as a token-stream generator. Requires the optional
`nca` extra (JAX/Flax).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np

from .default_generate_params import (
    FILTER_STEPS,
    GRID_SIZE,
    GZIP_THRESHOLD,
    GZIP_UPPER_BOUND,
    NUM_COLORS,
    PAPER_NUM_EPOCHS,
    PAPER_NUM_RULES,
    PAPER_NUM_SIMS,
    PATCH_SIZE,
    SEQ_LEN,
    trajectory_shape,
)
from .nca_sim import generate_nca_dataset, generate_rules_batch
from .nca_tokenizer import NCA_Tokenizer

if TYPE_CHECKING:
    from nanochat.data.base import TokenStream


def nca_paper_source(
    *,
    seed: int = 0,
    num_epochs: int = PAPER_NUM_EPOCHS,
    num_rules: int = PAPER_NUM_RULES,
    num_sims: int = PAPER_NUM_SIMS,
    seq_len: int = SEQ_LEN,
) -> TokenStream:
    """Yield the paper's NCA pre-pretraining corpus — one epoch per yield, then
    stop (a finite source of `num_epochs` arrays).

    ``seq_len`` sets the model context each trajectory must cover; it only
    changes the per-trajectory rollout length. Rule generation and the
    complexity filter — which scores a separate fixed FILTER_STEPS rollout —
    are seq_len-independent, so at a fixed `(seed, num_epochs)` the accepted
    rule set is identical at every seq_len and a longer corpus is exactly the
    shorter one's trajectories, rolled out further.

    Reproduces the data the paper trained on (`src/nca_ppt.py:main`): the
    gzip-filtered rule batch is generated ONCE from `epoch_seeds[0]` and reused
    across all epochs; each epoch's sim trajectories use a distinct rng
    `epoch_seeds[epoch] = split(PRNGKey(seed), num_epochs)[epoch]`. The epochs
    concatenate in order, so the bin is
    `num_epochs * num_sims * TRAIN_STEPS * TOKENS_PER_GRID` tokens
    (e.g. 10 * 16000 * 27 * 38 = 164,160,000), giving 16,000 unique rules with
    `num_epochs` different sim trajectories each.

    Reusing a single rule batch mirrors the reference implementation, whose
    per-epoch regeneration gate `epoch+1 % args.generate_rules == 0` parses as
    `epoch + (1 % args.generate_rules) == 0` and so fires only at `epoch==0`.

    `generate_nca_dataset` / `generate_rules_batch` / `NCA_Tokenizer` are
    byte-identical to the reference implementation's `utils/nca.py` and
    `utils/tokenizers.py`; the filter band, grid, patch, colors and
    `start_step=10` all match its pre-pretraining launcher.
    """
    tokenizer = NCA_Tokenizer(patch=PATCH_SIZE, num_colors=NUM_COLORS)
    train_steps, _ = trajectory_shape(seq_len)
    epoch_seeds = jax.random.split(jax.random.PRNGKey(seed), num_epochs)

    rule_seeds = generate_rules_batch(
        seed=epoch_seeds[0],
        num_rules=num_rules,
        tokenizer=tokenizer,
        threshold=GZIP_THRESHOLD,
        upper_bound=GZIP_UPPER_BOUND,
        dT=1,
        n_steps=FILTER_STEPS,
        mode="gzip",
        start_step=10,
        grid=GRID_SIZE,
        d_state=NUM_COLORS,
        identity_bias=0.0,
        temperature=1e-4,
    )

    for epoch in range(num_epochs):
        sims = generate_nca_dataset(
            epoch_seeds[epoch],
            num_sims=num_sims,
            grid=GRID_SIZE,
            d_state=NUM_COLORS,
            n_groups=1,
            identity_bias=0.0,
            temperature=1e-4,
            num_examples=train_steps,
            dT=1,
            rule_seeds=rule_seeds,
            num_rules=rule_seeds.shape[0],
            start_step=10,
        )
        seq, _targets = tokenizer.encode_task(sims)
        yield np.asarray(seq, dtype=np.uint16).reshape(-1)
