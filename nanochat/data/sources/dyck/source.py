"""k-shuffle Dyck procedural data source.

A callable returning a `TokenStream` — a generator of 1-D `uint16` token-id
arrays (see the source-function contract in `nanochat/data/sources/`).
`dyck_source` generates k-shuffle Dyck bracket sequences on the fly and emits
their token ids directly; there is no tokenizer and no disk I/O.

The grammar generator follows Hu et al., "Between Circuits and Chomsky:
Pre-pretraining on Formal Languages Imparts Linguistic Biases" (2025), appendix
Figure 10 (arXiv:2502.19249), including its 0-based token ids: an opening
bracket of type i is the id i and its matching close is i + k, so ids span
[0, 2 * k) and a pretokenized dyck bin has vocab_size = 2 * k.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

from .default_generate_params import K, MAX_DEPTH, P_OPEN, SEQ_LENGTH

if TYPE_CHECKING:
    from nanochat.data.base import TokenStream


def generate_shuff_dyck(k, max_length=2048, p_open=0.5, max_depth=16):
    """Generate a k-shuffle Dyck sequence, truncated at `max_length`.

    Returns a list where `i` is an opening bracket of type i and `i + k` its
    matching close. When max depth is reached, one bracket is closed and
    generation continues. The final Dyck word may be invalid due to truncation.
    """
    sequence = []
    counts = [0] * k  # open brackets of each type
    while len(sequence) < max_length:
        depth = sum(counts)

        # Must open if all brackets are closed
        if depth == 0:
            bracket_idx = random.randint(0, k - 1)
            sequence.append(bracket_idx)  # opening token
            counts[bracket_idx] += 1
            continue

        if depth >= max_depth:  # force a close
            open_brackets = [i for i, count in enumerate(counts) if count > 0]
            bracket_idx = random.choice(open_brackets)
            sequence.append(k + bracket_idx)  # closing token
            counts[bracket_idx] -= 1
            continue

        if random.random() < p_open:
            bracket_idx = random.randint(0, k - 1)
            sequence.append(bracket_idx)  # opening token
            counts[bracket_idx] += 1
        else:
            open_brackets = [i for i, count in enumerate(counts) if count > 0]
            bracket_idx = random.choice(open_brackets)
            sequence.append(k + bracket_idx)  # closing token
            counts[bracket_idx] -= 1
    
    return sequence


def dyck_source(
    *,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 0,
    k: int = K,
    max_depth: int = MAX_DEPTH,
    p_open: float = P_OPEN,
    seq_length: int = SEQ_LENGTH,
) -> TokenStream:
    """Yield 1-D uint16 arrays of k-shuffle Dyck token ids, one sequence per
    yield, infinite. Rank-disjoint via per-(seed, rank) RNG seeding.

    Token ids are 0-based: an opening bracket of type i is the id i and its
    matching close is i + k, so ids span [0, 2 * k). Each yielded array is
    exactly `seq_length` tokens long, so packing into `(B, S)` is left to the
    consumer. `world_size` is accepted for source-contract conformance but
    unused: rank-disjointness comes from seeding on `rank` alone.
    """
    # generate_shuff_dyck draws from the process-global `random` module, so
    # another global-random user in the process would perturb this stream.
    random.seed(seed * 1_000_003 + rank)
    while True:
        seq = generate_shuff_dyck(k, max_length=seq_length, p_open=p_open, max_depth=max_depth)
        yield np.array(seq, dtype=np.uint16)
