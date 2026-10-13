"""decl-ref-01 procedural data source.

A callable returning a `TokenStream` — a generator of 1-D `uint16` token-id
arrays (see the source-function contract in `nanochat/data/sources/`).
`declref_01_source` generates documents on the fly and emits their token ids
directly; there is no tokenizer and no disk I/O.

The language
------------
A random "program" of scopes, `DECL name value` (bind a fresh symbol) and
`REF name value` (repeat a visible symbol and its bound value, retrievable only
by attention to the earlier DECL). Names and values are 1-3 subtoken "parts".
On top of that skeleton, four mechanisms shape the per-position entropy dynamics
so that a model trained on this data matches natural language / code:

- noisy references (`NOISE_P`): a REF sometimes re-binds a fresh value instead of
  copying;
- a sparse part-transition chain (`CHAIN_MIX`, `CHAIN_BRANCH`): each part has a
  few plausible successors, so content is partially predictable;
- interleaved openings (`STAY_LO` -> `STAY_HI` ramp): keyword runs start short so
  the document opens with mixed DECL/REF;
- topic shifts (`SHIFT_PERIOD`, `SHIFT_FRAC`): the name distribution is
  periodically re-permuted.

The `(part -> successors)` chain table is fixed by `CHAIN_SEED`.

Example (ids mapped to letters for readability; `let` = DECL, `use` = REF,
lowercase = name-parts, uppercase = value-parts):

    let ra QRS  use ra QRS  { let nbc RS  use nbc RS  use ra QRS } use ra Q

Token ids are 0-based: 0=OPEN 1=CLOSE 2=DECL 3=REF, name-parts occupy
[4, 4 + N_PARTS) and value-parts [4 + N_PARTS, 4 + 2 * N_PARTS), so ids span
[0, 4 + 2 * N_PARTS) and a pretokenized bin has vocab_size = 1028.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from .default_generate_params import (
    ALPHA_RANGE,
    CHAIN_BRANCH,
    CHAIN_MIX,
    CHAIN_SEED,
    CHAIN_SUCC_POOL,
    FIRST_SHIFT,
    LEN_PROBS,
    MAX_DEPTH,
    N_PARTS,
    NOISE_P,
    P_BRACKET,
    REF_RECENCY_ALPHA,
    REF_SHARE_HI,
    REF_SHARE_LO,
    SEQ_LENGTH,
    SHIFT_FRAC,
    SHIFT_PERIOD,
    STAY_HI,
    STAY_LO,
)

if TYPE_CHECKING:
    from nanochat.data.base import TokenStream

OPEN, CLOSE, DECL, REF = 0, 1, 2, 3
NAME0 = 4
VALUE0 = 4 + N_PARTS
VOCAB_SIZE = 4 + 2 * N_PARTS

# Fixed successor table: row p lists CHAIN_SUCC_POOL plausible next parts for
# part p, built once from CHAIN_SEED.
_CHAIN_SUCC = np.stack([
    np.random.default_rng(CHAIN_SEED + p).choice(N_PARTS, size=CHAIN_SUCC_POOL, replace=False)
    for p in range(N_PARTS)
])
_LEN_PROBS = np.array(LEN_PROBS)


def _zipf(n: int, alpha: float) -> np.ndarray:
    """Zipf probabilities over ranks 1..n with exponent `alpha`."""
    p = 1.0 / np.arange(1, n + 1) ** alpha
    return p / p.sum()


@lru_cache(maxsize=None)
def _recency_probs(n: int) -> np.ndarray:
    """REF recency weights over `n` visible names (most recent first). Cached,
    so callers must treat the returned array as read-only."""
    return _zipf(n, REF_RECENCY_ALPHA)[::-1]


def generate_declref_01_doc(rng: np.random.Generator, seq_length: int = SEQ_LENGTH) -> np.ndarray:
    """Generate one decl-ref-01 document of exactly `seq_length` tokens.

    Per-document latents are drawn first — a Zipf exponent from `ALPHA_RANGE`
    and random permutations of the part vocabulary — so symbol statistics differ
    per document and must be inferred in-context. The final construct may be
    truncated by the `seq_length` cut.
    """
    alpha = rng.uniform(*ALPHA_RANGE)
    zdoc = _zipf(N_PARTS, alpha)
    name_probs = zdoc[np.argsort(rng.permutation(N_PARTS))]
    value_probs = zdoc[np.argsort(rng.permutation(N_PARTS))]
    chain_probs = _zipf(CHAIN_BRANCH, 1.2)
    last_part = None  # threaded across symbols so the chain spans the document

    def sample_part(part_probs: np.ndarray) -> int:
        nonlocal last_part
        if last_part is not None and rng.random() < CHAIN_MIX:
            part = int(_CHAIN_SUCC[last_part, rng.choice(CHAIN_BRANCH, p=chain_probs)])
        else:
            part = int(rng.choice(N_PARTS, p=part_probs))
        last_part = part
        return part

    def sample_symbol(part_probs: np.ndarray) -> tuple[int, ...]:
        length = 1 + rng.choice(_LEN_PROBS.size, p=_LEN_PROBS)
        return tuple(sample_part(part_probs) for _ in range(length))

    toks: list[int] = []
    scopes: list[dict[tuple[int, ...], tuple[int, ...]]] = [{}]
    prev_action = DECL
    next_shift = FIRST_SHIFT

    while len(toks) < seq_length:
        if len(toks) >= next_shift:  # topic shift
            n_shift = int(N_PARTS * SHIFT_FRAC)
            sel = rng.choice(N_PARTS, size=n_shift, replace=False)
            name_probs[sel] = name_probs[rng.permutation(sel)]
            name_probs /= name_probs.sum()
            next_shift += SHIFT_PERIOD

        depth = len(scopes) - 1
        visible: dict[tuple[int, ...], tuple[int, ...]] = {}
        for scope in scopes:
            visible.update(scope)

        frac = len(toks) / seq_length
        stay = STAY_LO + (STAY_HI - STAY_LO) * min(1.0, frac * 3)
        ref_share = REF_SHARE_LO + (REF_SHARE_HI - REF_SHARE_LO) * frac
        p_open = P_BRACKET if depth < MAX_DEPTH else 0.0
        p_close = P_BRACKET if depth > 0 else 0.0
        rest = 1.0 - p_open - p_close
        if visible:
            p_ref = rest * (min(0.92, stay) if prev_action == REF
                            else ref_share * (1.0 - stay) * 2)
            p_ref = min(p_ref, rest * 0.92)
        else:
            p_ref = 0.0
        acts = np.array([p_open, p_close, rest - p_ref, p_ref])
        action = int(rng.choice(4, p=acts / acts.sum()))

        if action == OPEN:
            toks.append(OPEN)
            scopes.append({})
        elif action == CLOSE:
            toks.append(CLOSE)
            scopes.pop()
        elif action == DECL:
            name, value = sample_symbol(name_probs), sample_symbol(value_probs)
            scopes[-1][name] = value
            toks += [DECL, *(NAME0 + p for p in name), *(VALUE0 + p for p in value)]
            prev_action = DECL
        else:
            names = list(visible.keys())
            recency = _recency_probs(len(names))
            name = names[int(rng.choice(len(names), p=recency))]
            if rng.random() < NOISE_P:
                value = sample_symbol(value_probs)  # noisy reference: fresh value ...
                scopes[-1][name] = value            # ... which becomes the new binding
            else:
                value = visible[name]
            toks += [REF, *(NAME0 + p for p in name), *(VALUE0 + p for p in value)]
            prev_action = REF

    return np.array(toks[:seq_length], dtype=np.uint16)


def declref_01_source(
    *,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 0,
    seq_length: int = SEQ_LENGTH,
) -> TokenStream:
    """Yield 1-D uint16 arrays of decl-ref-01 token ids, one document per
    yield, infinite. Rank-disjoint via per-(seed, rank) seeding.

    Each yielded array is exactly `seq_length` tokens (one whole document), so
    a pretokenized bin reshapes into one document per row for the word-aligned
    loader. `world_size` is accepted for source-contract conformance but unused:
    rank-disjointness comes from seeding on `rank` alone.
    """
    rng = np.random.default_rng(seed * 1_000_003 + rank)
    while True:
        yield generate_declref_01_doc(rng, seq_length=seq_length)
