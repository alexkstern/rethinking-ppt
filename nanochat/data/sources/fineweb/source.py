"""FineWeb token source: read local parquet shards, tokenize, yield token-id arrays.

A callable returning a `TokenStream` — a generator of 1-D `uint16` token-id
arrays, one per document (see the source-function contract in
`nanochat/data/sources/`).

Ingestion type: **local parquet**, read with pyarrow — the path for FineWeb-Edu
and for any other local dir of parquet shards with a `text` column. (The sibling
`sources/c4` source is the **HF datasets-streaming** path for text that lives on
the Hub.) Work is partitioned disjointly across `(rank, world_size)` at row-group
granularity (`rg_idx = rank; rg_idx += world_size`). Iteration is forward-only —
there is no resume cursor.

Run `python -m nanochat.data.sources.fineweb.source` to bulk-fetch the hosted
FineWeb-Edu shards via `BASE_URL`; data already on disk is read by passing
`data_path={"local": ...}`.
"""

import argparse
import os
import time
from multiprocessing import Pool

import pyarrow.parquet as pq
import requests

from nanochat.common import get_base_dir
from nanochat.data.base import TokenStream, resolve_data_path
from nanochat.data.sources.text import batched, encode_documents

BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
MAX_SHARD = 1822  # the last datashard is shard_01822.parquet
index_to_filename = lambda index: f"shard_{index:05d}.parquet"
DATA_DIR = os.path.join(get_base_dir(), "base_data")
os.makedirs(DATA_DIR, exist_ok=True)


def list_parquet_files(data_dir):
    """Full paths to every parquet file in `data_dir`."""
    parquet_files = sorted(
        f for f in os.listdir(data_dir)
        if f.endswith(".parquet") and not f.endswith(".tmp")
    )
    return [os.path.join(data_dir, f) for f in parquet_files]


def _shard_paths(data_path, split):
    """Resolve `data_path` to the list of parquet shard paths for `split`.

    `data_path` is the `DataConfig.path` single-source dict: `{"local": dir}`
    gives all parquet files in that dir, while `{"hf": {...}}` downloads a
    *single* file, so it yields a one-shard dataset — fine for a self-contained
    val bin, not for a multi-shard train corpus.

    The last shard is the val split; `split="train"` takes all but the last.
    """
    resolved = resolve_data_path(data_path)
    paths = list_parquet_files(resolved) if resolved.is_dir() else [str(resolved)]
    if not paths:
        raise ValueError(f"no parquet shards found for data_path={data_path!r}")
    shards = paths[:-1] if split == "train" else paths[-1:]
    if not shards:
        raise ValueError(
            f"data_path={data_path!r} resolved to only {len(paths)} parquet shard(s); "
            f"the last shard is reserved for the val split, so split='train' needs >= 2."
        )
    return shards


def fineweb_source(
    *,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 0,
    split: str,
    tokenizer,
    data_path: dict,
    cycle: bool = True,
    tokenizer_threads: int = 4,
    tokenizer_batch_size: int = 128,
) -> TokenStream:
    """Yield 1-D `uint16` token-id arrays from local parquet shards, one per document.

    `cycle=True` (the default) re-reads the shards forever, for the live training
    loader; `cycle=False` reads them exactly once, which is what the pretokenizer
    passes so a bin never contains duplicated documents. Each rank reads a
    disjoint set of row groups (`rg_idx = rank`, striding by `world_size` within
    every file). A row group is tokenized in sub-batches of
    `tokenizer_batch_size` documents via a single multithreaded
    `tokenizer.encode(..., num_threads=tokenizer_threads)` call; only the *yield*
    is per-document.

    `split` is `"train"` (all but the last shard) or `"val"` (the last shard);
    pass `data_path={"local": DATA_DIR}` to read the FineWeb shards. `seed` is
    part of the source-function contract and unused here — parquet iteration
    order is fixed by shard and row-group index. Arrays are not padded or
    packed; the consumer packs them into (B, S).
    """
    assert split in ("train", "val"), "split must be 'train' or 'val'"
    shard_paths = _shard_paths(data_path, split)

    def _one_pass():
        for filepath in shard_paths:
            pf = pq.ParquetFile(filepath)
            rg_idx = rank  # each rank starts at its own row group ...
            while rg_idx < pf.num_row_groups:
                texts = pf.read_row_group(rg_idx).column("text").to_pylist()
                for chunk in batched(texts, tokenizer_batch_size):
                    yield from encode_documents(tokenizer, chunk, num_threads=tokenizer_threads)
                rg_idx += world_size  # ... and strides by world_size (DDP-disjoint)

    if not cycle:
        yield from _one_pass()
        return
    while True:
        yield from _one_pass()


def download_single_file(index):
    """Download one FineWeb-Edu shard by index, with backoff."""
    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            temp_path = filepath + ".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            if attempt < max_attempts:  # exponential backoff: 2^attempt seconds
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FineWeb-Edu 100BT dataset shards")
    parser.add_argument("-n", "--num-files", type=int, default=-1, help="Number of shards to download (default: -1), -1 = disable")
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of parallel download workers (default: 4)")
    args = parser.parse_args()

    num = MAX_SHARD + 1 if args.num_files == -1 else min(args.num_files, MAX_SHARD + 1)
    ids_to_download = list(range(num))
    print(f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers...")
    print(f"Target directory: {DATA_DIR}")
    print()
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(download_single_file, ids_to_download)

    successful = sum(1 for success in results if success)
    print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {DATA_DIR}")
