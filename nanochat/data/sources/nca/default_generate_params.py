"""Default NCA generation parameters (Lee et al. 2026)."""

import numpy as np

PATCH_SIZE = 2
NUM_COLORS = 10
GRID_SIZE = 12
PATCHES_PER_GRID = (GRID_SIZE // PATCH_SIZE) ** 2  # 36
TOKENS_PER_GRID = PATCHES_PER_GRID + 2  # 38 (36 patches + start + end)
START_TK = NUM_COLORS ** (PATCH_SIZE ** 2)  # 10000
END_TK = START_TK + 1  # 10001

def trajectory_shape(seq_len: int) -> tuple[int, int]:
    """``(grids_per_trajectory, tokens_per_trajectory)`` for a model context of
    ``seq_len`` tokens.

    A trajectory is a whole number of delimiter-wrapped grids (TOKENS_PER_GRID
    each), rounded UP to cover ``seq_len``, so the stored trajectory is slightly
    longer than the context and the loader truncates the tail (1024 -> 27 grids
    / 1026 tokens; 2048 -> 54 grids / 2052 tokens).
    """
    grids = int(np.ceil(seq_len / TOKENS_PER_GRID))
    return grids, grids * TOKENS_PER_GRID


SEQ_LEN = 1024  # NCA trajectory length (paper default)
TRAIN_STEPS = trajectory_shape(SEQ_LEN)[0]  # 27 grids rolled out per sim
# Complexity-filter rollout length (paper default 10), scored on a separate
# short per-rule rollout.
FILTER_STEPS = 10

# gzip-complexity band: keep a rule iff its gzip ratio is in (THRESHOLD, UPPER)
GZIP_THRESHOLD = 0.5
GZIP_UPPER_BOUND = 1.0

# Paper training-run config (Lee et al.). Per epoch, a distinct seed split from
# PRNGKey(seed) rolls out PAPER_NUM_SIMS trajectories over PAPER_NUM_RULES
# gzip-filtered rules. The epochs concatenate in order into the bin.
PAPER_NUM_RULES = 16000
PAPER_NUM_SIMS = 16000
PAPER_NUM_EPOCHS = 10
