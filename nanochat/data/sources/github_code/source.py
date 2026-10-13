"""GitHub-code token source: stream raw source code from the Hub, tokenize, yield token-id arrays.

A callable returning a `TokenStream` — a generator of 1-D `uint16` token-id
arrays, one per document (see the source-function contract in
`nanochat/data/sources/`).

Ingestion type: **HF datasets-streaming**, same shape as the sibling
`sources/c4` source. Upstream `codeparrot/github-code`'s canonical revision is a
dataset *loading script*, which modern `datasets` refuses to load, so this
streams the Hub's auto-converted **parquet branch** (`refs/convert/parquet`,
config `all-all`, ~605k documents) via the plain `parquet` builder instead.

Upstream has only a `train` split, so this source carves its own held-out
partition deterministically: the stream is unshuffled and its order is fixed by
the parquet files, so **val = the first `val_docs` (20,000) documents** and
**train = everything after** (`skip(val_docs)`). The two splits are disjoint by
construction as long as both sides use the same `val_docs`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from datasets import load_dataset
from datasets.distributed import split_dataset_by_node

from nanochat.data.sources.text import batched, encode_documents

if TYPE_CHECKING:
    from nanochat.data.base import TokenStream

# Number of leading documents reserved for the val split. Baked in (not a config
# knob) so train/val disjointness can never drift between two prep runs.
VAL_DOCS = 20_000


def github_code_source(
    *,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 0,
    split: str,
    tokenizer,
    repo_id: str = "codeparrot/github-code",
    config: str = "all-all",
    text_column: str = "code",
    cycle: bool = True,
    tokenizer_threads: int = 4,
    tokenizer_batch_size: int = 128,
) -> TokenStream:
    """Yield 1-D `uint16` token-id arrays from streamed GitHub source files, one per document.

    `cycle=True` (the default) re-opens the stream forever, for the live
    training loader; `cycle=False` drains it exactly once (bin creation). Each
    rank reads a disjoint slice of documents via `split_dataset_by_node`, and
    tokenization is batched exactly as in the c4 source.

    `split` is `"train"` or `"val"` — upstream has only a train split, so
    `"val"` is the first `VAL_DOCS` documents of the deterministic stream and
    `"train"` skips them. `seed` is part of the source-function contract and
    unused here. Arrays are not padded or packed; the consumer packs them into (B, S).
    """
    assert split in ("train", "val"), "split must be 'train' or 'val'"
    data_files = f"hf://datasets/{repo_id}@refs/convert/parquet/{config}/partial-train/*.parquet"

    def _one_pass():
        ds = load_dataset("parquet", data_files=data_files, split="train", streaming=True)
        # Carve the held-out partition before any per-node split, so the val/train
        # boundary is identical regardless of world_size.
        ds = ds.take(VAL_DOCS) if split == "val" else ds.skip(VAL_DOCS)
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
