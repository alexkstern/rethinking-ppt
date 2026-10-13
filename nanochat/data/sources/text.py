"""Shared helpers for text token sources (parquet, c4).

A source iterates documents however it likes (sliced row groups, a streamed
dataset, ...) and writes `for chunk in batched(docs, n): yield from
encode_documents(tok, chunk, num_threads=t)`. The whole chunk must reach
`encode_documents` in one piece: the parallelism lives inside the single
multithreaded `tokenizer.encode` call.
"""

from __future__ import annotations

from itertools import islice
from typing import TYPE_CHECKING, Iterable, Iterator

import numpy as np

if TYPE_CHECKING:
    from nanochat.data.base import TokenStream


def batched(documents: Iterable[str], size: int) -> Iterator[list[str]]:
    """Group `documents` into lists of at most `size` (the final chunk may be shorter)."""
    iterator = iter(documents)
    while chunk := list(islice(iterator, size)):
        yield chunk


def encode_documents(tokenizer, documents: list[str], *, num_threads: int) -> TokenStream:
    """Encode a chunk of documents, yielding one BOS-prefixed 1-D `uint16` array
    per document. The whole chunk goes through one multithreaded
    `tokenizer.encode` call; only the yield is per-document."""
    bos_token = tokenizer.get_bos_token_id()
    for tokens in tokenizer.encode(documents, prepend=bos_token, num_threads=num_threads):
        yield np.asarray(tokens, dtype=np.uint16)
