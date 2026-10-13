"""C4 token source: stream raw text from the Hub, tokenize, yield token-id arrays.

A callable returning a `TokenStream` — a generator of 1-D `uint16` token-id
arrays, one per document (see the source-function contract in
`nanochat/data/sources/`).

Ingestion type: **HF datasets-streaming** via
`datasets.load_dataset(..., streaming=True)`, so a bounded slice can be
tokenized without downloading the whole corpus. Point `repo_id`/`config` at
another dataset to reuse it.

Work is partitioned across `(rank, world_size)` with
`datasets.distributed.split_dataset_by_node`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from datasets import load_dataset
from datasets.distributed import split_dataset_by_node

from nanochat.data.sources.text import batched, encode_documents

if TYPE_CHECKING:
    from nanochat.data.base import TokenStream


def c4_source(
    *,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 0,
    split: str,
    tokenizer,
    repo_id: str = "allenai/c4",
    config: str = "en",
    text_column: str = "text",
    cycle: bool = True,
    tokenizer_threads: int = 4,
    tokenizer_batch_size: int = 128,
) -> TokenStream:
    """Yield 1-D `uint16` token-id arrays from a streamed HF text dataset, one per document.

    `cycle=True` (the default) re-opens the stream forever, for the live
    training loader; `cycle=False` drains it exactly once. Each rank
    reads a disjoint slice of documents via `split_dataset_by_node`.
    `tokenizer_batch_size` documents are encoded per multithreaded
    `tokenizer.encode(..., num_threads=tokenizer_threads)` call.

    `split` is `"train"` or `"val"` (mapped to the dataset's `"validation"`).
    `seed` is part of the source-function contract and unused here. Arrays are
    not padded or packed; the consumer packs them into (B, S).
    """
    assert split in ("train", "val"), "split must be 'train' or 'val'"
    hf_split = "validation" if split == "val" else split

    def _one_pass():
        ds = load_dataset(repo_id, config, split=hf_split, streaming=True)
        if world_size > 1:  # no per-node split for the CLI's single stream
            ds = split_dataset_by_node(ds, rank=rank, world_size=world_size)
        documents = (example[text_column] for example in ds)
        for chunk in batched(documents, tokenizer_batch_size):
            yield from encode_documents(tokenizer, chunk, num_threads=tokenizer_threads)

    if not cycle:
        yield from _one_pass()
        return
    while True:
        yield from _one_pass()
