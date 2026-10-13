"""Carve a flat uint16 .bin into a train or val part by holding out a tail.

For corpora with no natural held-out split: tokenize the whole corpus into one
.bin once, then carve `--part train` (everything except the last `--val-tokens`)
and `--part val` (the last `--val-tokens`) into their own bins.

    python -m nanochat.data.carve --input full.bin --part val   --val-tokens 20000000 --output val.bin
    python -m nanochat.data.carve --input full.bin --part train --val-tokens 20000000 --output train.bin
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def _meta_path(bin_path: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(bin_path)), "meta.json")


def carve(input_bin: str, output: str, *, part: str, val_tokens: int, chunk_tokens: int = 64_000_000) -> dict:
    """Write the train (head) or val (tail) slice of `input_bin` to `output` + meta.json.

    `part="val"` writes the last `val_tokens` tokens; `part="train"` writes
    everything before them. The source `meta.json` is copied with `num_tokens`
    overwritten to the slice length.
    """
    if val_tokens <= 0:
        raise ValueError(f"val_tokens must be positive, got {val_tokens}")
    in_dir = os.path.dirname(os.path.abspath(input_bin))
    if os.path.dirname(os.path.abspath(output)) == in_dir:
        raise ValueError(
            f"output must be in a different folder than the input ({in_dir}); "
            f"carving in place would clobber the source meta.json"
        )
    src = np.memmap(input_bin, dtype=np.uint16, mode="r")
    total = len(src)
    if val_tokens >= total:
        raise ValueError(f"val_tokens ({val_tokens}) must be < total tokens ({total})")
    if part == "train":
        lo, hi = 0, total - val_tokens
    elif part == "val":
        lo, hi = total - val_tokens, total
    else:
        raise ValueError(f"part must be 'train' or 'val', got {part!r}")

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "wb") as f:
        for i in range(lo, hi, chunk_tokens):
            src[i : min(i + chunk_tokens, hi)].tofile(f)

    with open(_meta_path(input_bin)) as f:
        meta = json.load(f)
    meta["num_tokens"] = hi - lo
    meta["split"] = part
    with open(_meta_path(output), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Carved {part}: {hi - lo:,} tokens ({(hi - lo) * 2 / 1e6:.1f} MB) -> {output}")
    return meta


def main(argv=None):
    p = argparse.ArgumentParser(description="Carve a flat uint16 .bin into a train/val part by holding out a tail.")
    p.add_argument("--input", required=True, help="source .bin (its meta.json sits alongside)")
    p.add_argument("--output", required=True, help="output .bin path (meta.json written alongside)")
    p.add_argument("--part", required=True, choices=["train", "val"])
    p.add_argument("--val-tokens", type=int, required=True, help="size of the held-out tail")
    a = p.parse_args(argv)
    carve(a.input, a.output, part=a.part, val_tokens=a.val_tokens)


if __name__ == "__main__":
    main()
