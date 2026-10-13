"""Pre-tokenized uint16 bin loaders.

Three read modes over on-disk `uint16` `.bin`s, all yielding
`(inputs, targets, state)` 3-tuples — the shared iterator contract for the
whole data submodule — with `state` always `None`:

  * `build_bin_loader` — a continuous sliding window over the token stream;
    sequences may cross document boundaries. The default path for natural text.
  * `build_interleaved_bin_loader` — two continuous bins served in one batch,
    with the per-step split set by a caller-supplied schedule.
  * `build_word_aligned_loader` — one fixed-length row per sequence, never
    crossing rows, for procedural corpora where each row is a self-contained
    sample (Dyck words, NCA trajectories). Optionally truncates each row from
    the bin's on-disk ``stored_word_length`` down to the model's ``seq_length``,
    dropping per-sample delimiter wrapping.

Per-batch target transforms live in `nanochat/data/transforms.py`.
"""

import json
from functools import cache
from pathlib import Path

import numpy as np
import torch

from nanochat.config import DataConfig

from .base import DataIterator, resolve_data_path


@cache
def _load_bin(data_path: str) -> np.ndarray:
    """Load a uint16 .bin into memory, memoized per resolved path."""
    return np.fromfile(data_path, dtype=np.uint16)


def bin_token_count(cfg: DataConfig) -> int:
    """Tokens in the bin `cfg` names (memoized with the load itself)."""
    return len(_load_bin(str(resolve_data_path(cfg.path))))


def expected_epochs(expected_tokens: int, total_tokens: int) -> float:
    """Projected epochs over the bin for a run that will consume `expected_tokens`.

    `expected_tokens` is the full token count the run will read: the caller
    pre-multiplies across optimizer steps, grad-accum steps, ranks, batch and
    sequence length (in nanochat's convention, `total_batch_size *
    num_iterations`).
    """
    return expected_tokens / total_tokens


def _cycle_permutation(num_steps: int, cycle: int, global_seed: int) -> np.ndarray:
    """Index permutation of `[0, num_steps)` for one cycle. Cycle 0 is the
    identity, so a single-pass run reads in natural order; cycles 1+ are
    seed-derived shuffles."""
    pi = np.arange(num_steps)
    if cycle == 0:
        return pi
    np.random.default_rng(seed=(global_seed, cycle)).shuffle(pi)
    return pi


def _read_batch(data, pos, total_tokens, device_batch_size, seq_length, device):
    """Read one (B, S+1)-token slab starting at `pos % total_tokens` with modular wrap."""
    tokens_needed = device_batch_size * (seq_length + 1)
    p = pos % total_tokens

    if p + tokens_needed <= total_tokens:
        slab_np = data[p:p + tokens_needed]
    else:
        slab_np = np.empty(tokens_needed, dtype=np.uint16)
        remaining = tokens_needed
        dst = 0
        while remaining > 0:
            chunk = min(remaining, total_tokens - p)
            slab_np[dst:dst + chunk] = data[p:p + chunk]
            dst += chunk
            remaining -= chunk
            p = (p + chunk) % total_tokens

    # Separate tensors rather than slices of one slab: `slab[:, 1:]` would be a
    # non-contiguous view, and consumers flatten targets with `.view(-1)`.
    slab_2d = slab_np.reshape(device_batch_size, seq_length + 1)
    inputs = torch.tensor(slab_2d[:, :-1], dtype=torch.long, device=device)
    targets = torch.tensor(slab_2d[:, 1:], dtype=torch.long, device=device)
    return inputs, targets, None


def build_bin_loader(
    cfg: DataConfig,
    device_batch_size: int,
    seq_length: int,
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
    *,
    global_seed: int = 0,
    expected_tokens: int | None = None,
    start_token: int = 0,
) -> DataIterator:
    """Generate infinite `(inputs, targets, None)` batches over a pre-tokenized bin.

    The bin is conceptually sliced into
    `N = (total_tokens - start_token) // (B*S*world_size)` step-slabs
    (`B` = `device_batch_size`, `S` = `seq_length`). One *cycle* visits all N
    slabs; order within a cycle is a permutation derived from
    `(global_seed, cycle)`, with cycle 0 the identity (a plain linear read).
    `global_seed` must match across ranks so they agree on visit order.
    The floor division leaves a tail of up to `tokens_per_step - 1` tokens
    unread per cycle.

    `expected_tokens` — the total token count the run will consume — is used
    only for the rank-0 startup log; the loader yields indefinitely regardless.
    `start_token` shifts the whole slab layout to start at that token.

    Yields `(inputs, targets, None)`, both `torch.long` of shape
    `(device_batch_size, seq_length)` on `device`, with
    `targets[:, :-1] == inputs[:, 1:]`. Raises `ValueError` if the bin cannot
    hold one full gradient step.
    """
    data = _load_bin(str(resolve_data_path(cfg.path)))
    total_tokens = len(data)
    if not 0 <= start_token < total_tokens:
        raise ValueError(f"start_token={start_token} outside bin of {total_tokens:,} tokens")
    tokens_per_batch = device_batch_size * seq_length
    tokens_per_step = tokens_per_batch * world_size
    num_steps_per_cycle = (total_tokens - start_token) // tokens_per_step

    if num_steps_per_cycle == 0:
        raise ValueError(
            f"Bin too small: {total_tokens - start_token:,} tokens readable "
            f"(bin {total_tokens:,} - start_token {start_token:,}) < "
            f"tokens_per_step={tokens_per_step:,} (device_batch_size * seq_length * world_size). "
            f"Either grow the bin or shrink (device_batch_size × seq_length × world_size)."
        )

    if rank == 0:
        readable = total_tokens - start_token
        tail = readable - num_steps_per_cycle * tokens_per_step
        tail_pct = (tail / readable) * 100
        print(
            f"[bin_loader] {cfg.path}: {total_tokens:,} tokens in bin, "
            f"{num_steps_per_cycle:,} steps per cycle "
            f"(tail {tail:,} = {tail_pct:.4f}% unread per cycle)"
        )
        if expected_tokens is None:
            print(
                "[bin_loader] no expected_tokens supplied; "
                "loader yields indefinitely — caller controls run length"
            )
        else:
            ratio = expected_epochs(expected_tokens, total_tokens)
            if ratio > 1.0:
                print(
                    f"[bin_loader] WARNING: {ratio:.2f} epochs expected "
                    f"({expected_tokens:,} tokens consumed / {total_tokens:,} tokens in bin)"
                )

    cycle = 0
    while True:
        for step in _cycle_permutation(num_steps_per_cycle, cycle, global_seed):
            pos = start_token + step * tokens_per_step + tokens_per_batch * rank
            yield _read_batch(data, pos, total_tokens, device_batch_size, seq_length, device)
        cycle += 1


def build_interleaved_bin_loader(
    code_cfg: DataConfig,
    text_cfg: DataConfig,
    device_batch_size: int,
    seq_length: int,
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
    *,
    code_seqs_before,
    code_start_seq: int = 0,
    code_stride: int | None = None,
    code_id_offset: int = 0,
) -> DataIterator:
    """Generate infinite `(inputs, targets, None)` batches mixing two bins.

    Each step's global batch of `B * world_size` sequences is split between a
    "code" stream (`code_cfg`) and a "text" stream (`text_cfg`) by
    `code_seqs_before(j)` — the cumulative count of code sequences consumed
    before (0-based) step `j`. Step `j` serves
    `c_j = code_seqs_before(j+1) - code_seqs_before(j)` code sequences in
    global slots `0..c_j-1` and text in the rest; rank `r` owns slots
    `[r*B, (r+1)*B)`. The `k`-th sequence of a stream is read where
    `build_bin_loader` (cycle 0, start_token=0) would place sequence `k`.
    `code_start_seq` shifts the code stream's `k` by the sequences a
    preceding pure phase (standard loader) already consumed.

    `code_stride` selects how the code stream maps sequence index -> bin
    position. `None` (default) uses the `build_bin_loader` layout. An int
    reads sequence `k` at `pos = k * code_stride` instead, for word-aligned
    synthetic corpora:

      * stride == stored row length (>= S+1): every sequence is one whole
        sample's first S+1 tokens, never crossing rows (NCA's 2052-token rows);
      * stride == S+1: dense sequential chunks, crossing row boundaries the
        way text packing crosses documents (dyck's 2048-token rows, which are
        one token too short to hold a whole S+1 sequence).

    `code_id_offset` is added to code-stream tokens (`DataSource.id_offset`
    mapping — the corpus occupies a disjoint high slice of an extended vocab).
    Mask-id handling stays the CALLER's job: wrap the returned iterator with
    `with_token_id_masking` using the source's `effective_mask_ids()`.

    Both streams must stay within one pass of their bins; a read past either
    bin raises rather than wrapping.

    Yields the `build_bin_loader` contract: `(inputs, targets, None)`, both
    `torch.long` of shape `(device_batch_size, seq_length)` on `device`.
    """
    if code_stride is not None and code_stride < seq_length + 1:
        raise ValueError(f"{code_stride=} < seq_length+1={seq_length + 1}: sequences would overlap")
    code = _load_bin(str(resolve_data_path(code_cfg.path)))
    text = _load_bin(str(resolve_data_path(text_cfg.path)))
    B, S, W = device_batch_size, seq_length, world_size
    seqs_per_step = B * W

    def seq_pos(k: int) -> int:
        step, within = divmod(k, seqs_per_step)
        r, slot = divmod(within, B)
        return step * seqs_per_step * S + r * B * S + slot * (S + 1)

    def code_pos(k: int) -> int:
        return k * code_stride if code_stride is not None else seq_pos(k)

    def read_seq(data, k: int, label: str):
        p = code_pos(k) if label == "code" else seq_pos(k)
        if p + S + 1 > len(data):
            raise ValueError(
                f"interleaved loader: {label} sequence {k} needs tokens "
                f"[{p}, {p + S + 1}) but the bin has {len(data):,} — the "
                f"schedule plans more {label} than the bin holds"
            )
        return data[p : p + S + 1]

    def generate() -> DataIterator:
        j = 0
        while True:
            before = code_seqs_before(j)
            c = code_seqs_before(j + 1) - before
            if not 0 <= c <= seqs_per_step:
                raise ValueError(f"code_seqs_before gives {c} code sequences at step {j}, batch holds {seqs_per_step}")
            text_before = j * seqs_per_step - before
            # int64, not the bins' uint16: an id_offset corpus occupies ids past 65535.
            rows = np.empty((B, S + 1), dtype=np.int64)
            for b in range(B):
                slot = rank * B + b
                if slot < c:
                    rows[b] = read_seq(code, code_start_seq + before + slot, "code")
                    rows[b] += code_id_offset
                else:
                    rows[b] = read_seq(text, text_before + (slot - c), "text")
            # Contiguous copies, not slices of one array (see _read_batch).
            inputs = torch.tensor(rows[:, :-1], dtype=torch.long, device=device)
            targets = torch.tensor(rows[:, 1:], dtype=torch.long, device=device)
            yield inputs, targets, None
            j += 1

    # Inner generator, so validation and bin loading happen at build time
    # rather than at first next().
    return generate()


def _bin_row_length(bin_path: Path) -> int:
    """Stored row length, from the bin folder's `meta.json` sidecar.

    Each generator records it under the name that fits what a row holds: NCA
    stores a whole trajectory, longer than the context it is truncated to,
    while dyck and decl-ref store one document per row.
    """
    meta_path = bin_path.parent / "meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        gen = json.load(f)["gen_params"]
    for key in ("tokens_per_trajectory", "seq_length"):
        if key in gen:
            return gen[key]
    raise KeyError(
        f"{meta_path}: gen_params records no row length "
        f"(looked for tokens_per_trajectory, seq_length)")


def build_word_aligned_loader(
    cfg: DataConfig,
    device_batch_size: int,
    seq_length: int,
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
    *,
    global_seed: int = 0,
    stored_word_length: int | None = None,
) -> DataIterator:
    """Generate infinite `(inputs, targets, None)` batches of whole rows.

    The flat bin is reshaped to one row per sequence —
    `(num_words, stored_word_length)` — optionally trimmed to `seq_length`, and
    each example is the next-token shift *within a row* (`inputs = row[:-1]`,
    `targets = row[1:]`), never crossing rows.

    Use for procedural corpora whose rows are self-contained samples (Dyck
    words, NCA trajectories). `stored_word_length > seq_length` covers rows
    wrapped with begin/end delimiters on disk; the wrapping is dropped at read
    time.

    Cycle 0 visits rows in natural order, cycles 1+ in a `global_seed`-derived
    shuffle of the row index — over whole rows, never within a row. Each step
    consumes `device_batch_size * world_size` rows; rank `r` takes the `r`-th
    `device_batch_size` slice of that block, so ranks read disjoint rows.

    Yields `(inputs, targets, None)`, both `torch.long` of shape
    `(device_batch_size, seq_length - 1)` on `device`.
    """
    stored = stored_word_length if stored_word_length is not None else seq_length
    if stored < seq_length:
        raise ValueError(
            f"stored_word_length ({stored}) < seq_length ({seq_length}); cannot "
            f"pad rows up — grow the bin or shrink seq_length."
        )

    bin_path = resolve_data_path(cfg.path)
    data = _load_bin(str(bin_path))
    total_tokens = len(data)

    # Only the local pretokenize layout has a plain `meta.json` sibling; an
    # HF-downloaded bin names its sidecar `<split>.meta.json`, so there the
    # caller-provided stored_word_length is trusted.
    meta_path = bin_path.parent / "meta.json"
    if meta_path.exists():
        bin_row_length = _bin_row_length(bin_path)
        if bin_row_length != stored:
            raise ValueError(
                f"stored_word_length {stored} != the bin's recorded row length "
                f"{bin_row_length} ({cfg.path}); word-aligned reads require they match."
            )
    if total_tokens % stored != 0:
        raise ValueError(
            f"Word-aligned bin {cfg.path} has {total_tokens:,} tokens, not a whole number of "
            f"{stored}-token rows; cannot reshape."
        )

    num_words = total_tokens // stored
    rows = data.reshape(num_words, stored)
    if stored > seq_length:
        rows = rows[:, :seq_length]
    words_per_step = device_batch_size * world_size
    num_steps_per_cycle = num_words // words_per_step

    if num_steps_per_cycle == 0:
        raise ValueError(
            f"Word-aligned bin too small: {num_words:,} rows < rows_per_step={words_per_step:,} "
            f"(device_batch_size * world_size). Either grow the bin or shrink the batch."
        )

    if rank == 0:
        tail = num_words - num_steps_per_cycle * words_per_step
        size_msg = (
            f"{seq_length} tokens (truncated from {stored})" if stored > seq_length
            else f"{seq_length} tokens"
        )
        print(
            f"[word_aligned_loader] {cfg.path}: {num_words:,} rows x {size_msg}, "
            f"{num_steps_per_cycle:,} steps per cycle (tail {tail:,} rows unread per cycle)"
        )

    cycle = 0
    while True:
        order = _cycle_permutation(num_words, cycle, global_seed)
        for step in range(num_steps_per_cycle):
            base = step * words_per_step + rank * device_batch_size
            row_idx = order[base:base + device_batch_size]
            # Contiguous copies, not slices of one array (see _read_batch).
            words = rows[row_idx]
            inputs = torch.tensor(words[:, :-1], dtype=torch.long, device=device)
            targets = torch.tensor(words[:, 1:], dtype=torch.long, device=device)
            yield inputs, targets, None
        cycle += 1
