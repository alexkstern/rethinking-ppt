"""Named data sources for experiments.

Each corpus is registered once with everything needed to consume it: its HF repo,
its W&B bits-per-byte metric name, how to read it, and how to score bpb.
Experiments refer to a source by KEY — e.g. ``--data-pt fineweb-nanochatbpe-20B``
— so an ``.sh`` names exactly which corpora a run uses.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Literal

import torch

from nanochat.config import DataConfig
from nanochat.data.base import resolve_data_path
from nanochat.data.bin import (
    bin_token_count,
    build_bin_loader,
    build_word_aligned_loader,
)


@dataclass(frozen=True)
class DataSource:
    """How to consume one corpus.

    `metric` is the W&B bpb key used when this source is an eval set. `loader`
    picks the reader: ``bin`` (continuous next-token windows) or ``word_aligned``
    (one fixed-length row per sequence). `token_bytes` picks the bpb
    normalization: ``rustbpe`` (real per-token byte lengths, read from the
    ``tokenizer/token_bytes.pt`` the repo bundles) or ``ones`` (1 byte/symbol,
    for synthetic data with no natural byte length). `mask_ids` are token ids
    excluded from bpb (set to 0 bytes) — e.g. NCA's grid delimiters, which the
    paper also excludes from the loss. `stored_word_length` is the on-disk row
    size when it exceeds the presented context, so the loader truncates (NCA
    stores 1026-token rows wrapped with start/end delimiters; the model trains
    on the first 1024); ``None`` when row size equals the context (e.g. Dyck).
    """

    repo_id: str
    metric: str
    loader: Literal["bin", "word_aligned"] = "bin"
    token_bytes: Literal["rustbpe", "ones"] = "rustbpe"
    mask_ids: tuple[int, ...] = ()
    stored_word_length: int | None = None
    # Shift every token id up by this much at load time: the corpus occupies a
    # disjoint high slice of a larger model vocab, so a synthetic phase can
    # share the text model's embedding table (no vocab surgery). The model must
    # be padded to at least ``source vocab + id_offset``.
    id_offset: int = 0

    def effective_mask_ids(self) -> tuple[int, ...]:
        """``mask_ids`` as they appear in the loaded (offset) id stream."""
        return tuple(m + self.id_offset for m in self.mask_ids)


DATA_SOURCES: dict[str, DataSource] = {
    # rustbpe / nanochatbpe corpora (vocab 65536)
    "fineweb-nanochatbpe-20B": DataSource("alexkstern/fineweb-nanochatbpe-20B", "val/fineweb_bpb", token_bytes="rustbpe"),
    # 50M / 100M: train.bin is the byte-exact prefix of the 20B train.bin.
    "fineweb-nanochatbpe-50M": DataSource("alexkstern/fineweb-nanochatbpe-50M", "val/fineweb_bpb", token_bytes="rustbpe"),
    "fineweb-nanochatbpe-100M": DataSource("alexkstern/fineweb-nanochatbpe-100M", "val/fineweb_bpb", token_bytes="rustbpe"),
    # 55B extension: its first 20,000,000,000 train tokens are byte-identical to
    # the 20B train.bin. All four share the same val.bin (held-out shard_01822,
    # 52,336,096 tokens), so val/fineweb_bpb is comparable across them.
    "fineweb-nanochatbpe-55B": DataSource("alexkstern/fineweb-nanochatbpe-55B", "val/fineweb_bpb", token_bytes="rustbpe"),
    "c4-nanochatbpe-10B":      DataSource("alexkstern/c4-nanochatbpe-10B",      "val/c4_bpb",      token_bytes="rustbpe"),
    "github-code-nanochatbpe-1B": DataSource("alexkstern/github-code-nanochatbpe-1B", "val/code_bpb", token_bytes="rustbpe"),
    # synthetic
    "dyck-k128-seq_len_2048-1B": DataSource(
        "alexkstern/dyck-k128-seq_len_2048-1B", "val/dyck_bpb",
        loader="word_aligned", token_bytes="ones",
    ),
    # Same dyck bin mapped into ids 65536..65791: a full-vocab (65536+256)
    # model trains dyck and text through ONE embedding table, so the PPT->PT
    # transition can keep the optimizer state and a single unified LR schedule.
    "dyck-k128-seq_len_2048-1B-offset65536": DataSource(
        "alexkstern/dyck-k128-seq_len_2048-1B", "val/dyck_bpb",
        loader="word_aligned", token_bytes="ones", id_offset=65536,
    ),
    # decl-ref-01: scoped declare/reference formal language (vocab 1028), one
    # 2048-token document per row — word alignment is essential, since a
    # window starting mid-document orphans its references.
    "declref-01-seq_len_2048-2B": DataSource(
        "alexkstern/declref-01-declref_01_symbols-2B", "val/declref_01_bpb",
        loader="word_aligned", token_bytes="ones",
    ),
    # NCA @ seq_len 2048: 54 delimiter-wrapped grids x 38 tokens = 2052 stored
    # per trajectory, truncated to the model context by the loader. share20 =
    # 20 epochs over the one 16k-rule batch (the paper's is 10) — a size knob,
    # sized so PPT stays single-pass through alpha 5% at 9e18.
    "nca-paper-share20-2048": DataSource(
        "alexkstern/nca-paper-share20-seq_len_2048-657M", "val/nca_bpb",
        loader="word_aligned", token_bytes="ones", mask_ids=(10000, 10001),
        stored_word_length=2052,
    ),
    # Same share20 bin mapped into ids 65536..75537: a full-vocab (65536+10004,
    # padded so the sharded AdamW divides by world size) model trains NCA and
    # text through ONE embedding table.
    "nca-paper-share20-2048-offset65536": DataSource(
        "alexkstern/nca-paper-share20-seq_len_2048-657M", "val/nca_bpb",
        loader="word_aligned", token_bytes="ones", mask_ids=(10000, 10001),
        stored_word_length=2052, id_offset=65536,
    ),
    # share200 = 200 epochs over the one 16k-rule batch (~6.57B tokens =
    # 200 x 16000 sims x 2052), sized so even a 27e18 point (~6.4B tokens at
    # d24) stays single-pass.
    "nca-paper-share200-2048": DataSource(
        "alexkstern/nca-paper-share200-seq_len_2048-6.5B", "val/nca_bpb",
        loader="word_aligned", token_bytes="ones", mask_ids=(10000, 10001),
        stored_word_length=2052,
    ),
}


def get_source(key: str) -> DataSource:
    try:
        return DATA_SOURCES[key]
    except KeyError:
        raise ValueError(f"unknown data source {key!r}; known: {sorted(DATA_SOURCES)}") from None


def _bin_path(source: DataSource, split: str) -> dict[str, Any]:
    return {"hf": {"repo_id": source.repo_id, "filename": f"{split}.bin"}}


def read_meta(source: DataSource, split: str) -> dict[str, Any]:
    """The bin's sidecar meta (HF repos name it ``<split>.meta.json``)."""
    hf = {"repo_id": source.repo_id, "filename": f"{split}.meta.json"}
    with open(resolve_data_path({"hf": hf}), "r", encoding="utf-8") as f:
        return json.load(f)


def source_vocab(source: DataSource, split: str = "val") -> int:
    return read_meta(source, split)["vocab_size"]


def source_word_length(source: DataSource, split: str = "val") -> int | None:
    """On-disk row length for a ``word_aligned`` source, else ``None`` — the
    same length the loader reshapes against, before any truncation."""
    if source.loader != "word_aligned":
        return None
    if source.stored_word_length is not None:
        return source.stored_word_length
    return read_meta(source, split)["gen_params"]["seq_length"]


def build_token_bytes(source: DataSource, vocab: int, device) -> torch.Tensor:
    """(vocab,) byte-length-per-token tensor for the source's tokenizer, for BPB."""
    if source.token_bytes == "rustbpe":
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(source.repo_id, "tokenizer/token_bytes.pt", repo_type="dataset")
        tb = torch.load(path, map_location=device, weights_only=True)
        # Exactly (vocab,): every token id the model can emit must index
        # in-bounds during bpb eval. Pad rows above the tokenizer's range are
        # zeroed — this source's val stream never emits them.
        assert tb.ndim == 1, f"{source.repo_id} token_bytes has shape {tuple(tb.shape)}"
        tb = tb.to(torch.int32)
        if tb.shape[0] < vocab:
            tb = torch.cat([tb, torch.zeros(vocab - tb.shape[0], dtype=torch.int32, device=device)])
        return tb[:vocab]
    tb = torch.ones(vocab, dtype=torch.int32, device=device)  # 1 byte / synthetic symbol
    if source.mask_ids:
        tb[list(source.effective_mask_ids())] = 0  # excluded from bpb
    return tb


def build_tokenizer(kind: str, *, repo_id: str | None = None):
    """Tokenizer instance for a text tokenizer kind, keyed like ``token_bytes``.

    ``"rustbpe"`` loads from the ``tokenizer/tokenizer.pkl`` the ``repo_id`` HF
    dataset repo bundles, else from the local ``{base_dir}/tokenizer``.
    Synthetic kinds (``"ones"``) have no text tokenizer and raise.
    """
    from nanochat.tokenizer import RustBPETokenizer

    if kind == "rustbpe":
        if repo_id is not None:
            from huggingface_hub import hf_hub_download
            pkl = hf_hub_download(repo_id, "tokenizer/tokenizer.pkl", repo_type="dataset")
            return RustBPETokenizer.from_directory(os.path.dirname(pkl))
        from nanochat.common import get_base_dir
        return RustBPETokenizer.from_directory(os.path.join(get_base_dir(), "tokenizer"))
    raise ValueError(f"no text tokenizer for kind {kind!r}; expected 'rustbpe'")


def batches_per_pass(
    source: DataSource, split: str, *, device_batch_size, seq_len, world_size,
    start_token=0,
) -> int:
    """Batches one full pass over `split` yields at this batch geometry.

    Every loader here is infinite: past one pass it wraps into a reshuffled
    cycle and re-serves tokens it has already served. A fixed eval window must
    fit inside one pass or it scores some positions twice.
    """
    total_tokens = bin_token_count(DataConfig(path=_bin_path(source, split)))
    if source.loader == "word_aligned":
        stored = source.stored_word_length
        stored = seq_len if stored is None else stored
        return (total_tokens // stored) // (device_batch_size * world_size)
    return (total_tokens - start_token) // (device_batch_size * seq_len * world_size)


def build_loader(
    source: DataSource, split: str, *, device_batch_size, seq_len, ctx, global_seed,
    expected_tokens=None, start_token=0,
):
    """Build the source's loader for `split`, dispatching on its reader kind.
    `expected_tokens` and `start_token` apply to the continuous ``bin`` reader
    only (a word-aligned source has no continuous stream to offset into)."""
    cfg = DataConfig(path=_bin_path(source, split))
    if source.loader == "word_aligned":
        if start_token:
            raise ValueError(
                f"start_token={start_token} requested for word-aligned source "
                f"{source.repo_id}; only continuous bin sources support an offset"
            )
        loader = build_word_aligned_loader(
            cfg, device_batch_size, seq_len, ctx.device, ctx.rank, ctx.world_size,
            global_seed=global_seed, stored_word_length=source.stored_word_length,
        )
    else:
        loader = build_bin_loader(
            cfg, device_batch_size, seq_len, ctx.device, ctx.rank, ctx.world_size,
            global_seed=global_seed, expected_tokens=expected_tokens,
            start_token=start_token,
        )
    if source.id_offset:
        from nanochat.data.transforms import with_id_offset
        loader = with_id_offset(loader, source.id_offset)
    return loader
