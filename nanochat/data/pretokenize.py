"""Offline pre-tokenizer: run one generating source to a token budget and write a `.bin`.

A *source* (see `nanochat/data/sources/`) is a callable returning a `TokenStream`
— a generator of 1-D `uint16` token-id arrays. This tool drains one source to
`target_tokens`, writes the tokens to a `.bin`, and writes a `meta.json` sidecar
next to it. `--upload` additionally pushes both to a HuggingFace dataset repo.

Layout convention: give each bin its own folder so the plain `meta.json` sidecar
never collides:

    /data/fineweb_rustbpe_1B/
        fineweb_rustbpe_1B.bin
        meta.json

Each source's module is imported only when that source is selected, so
pre-tokenizing fineweb never imports the `nca` extra's JAX dependency.

Examples
--------
    python -m nanochat.data.pretokenize --source fineweb --target-tokens 1_000_000_000 \
        --output /data/fineweb_rustbpe_1B/fineweb_rustbpe_1B.bin \
        --data-path-local /data/base_data --split train --tokenizer rustbpe

    python -m nanochat.data.pretokenize --source c4 --target-tokens 1_000_000_000 \
        --output /data/c4_rustbpe_1B/c4_rustbpe_1B.bin --tokenizer rustbpe --c4-config en

    python -m nanochat.data.pretokenize --source nca --target-tokens 164_160_000 \
        --output /data/nca_paper_164M/nca_paper_164M.bin --vocab-size 10002 \
        --num-rules 16000 --num-sims 16000 --num-epochs 10    # requires .[nca]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from typing import TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    from .base import TokenStream


def _build_tokenizer(name: str):
    """Build the tokenizer a text source encodes with. Text sources only."""
    from .registry import build_tokenizer
    if name == "rustbpe":
        return build_tokenizer("rustbpe")
    raise ValueError(f"unknown tokenizer {name!r}; expected 'rustbpe'")


def build_source(args) -> tuple[TokenStream, int, str, dict]:
    """Dispatch `args.source` to a constructed source plus its `(vocab_size, tokenizer, gen_params)` metadata.

    A text source reports its own vocab size via its tokenizer; a procedural
    source's is supplied by the caller (`--vocab-size`), since any value at or
    above the source's id range works (e.g. >= 2*k for dyck).

    `gen_params` is the procedural source's resolved generation config (every
    knob as a concrete value, CLI override or default) — written into
    `meta.json` so a procedural bin is self-describing. Empty for text sources.
    """
    if args.source == "nca":
        if args.vocab_size is None:
            raise ValueError("--source nca requires --vocab-size (>= 10002 for the paper patch=2 / 10-color config; may be padded higher)")
        from .sources.nca import default_generate_params as P
        from .sources.nca import nca_paper_source
        # The full generation config goes in meta.json so the bin is
        # self-describing: a consumer can reshape(-1, tokens_per_trajectory),
        # pack K trajectories per row, and mask the delimiter ids from meta
        # alone, with no source import.
        num_rules = args.num_rules if args.num_rules is not None else P.PAPER_NUM_RULES
        num_sims = args.num_sims if args.num_sims is not None else P.PAPER_NUM_SIMS
        num_epochs = args.num_epochs if args.num_epochs is not None else P.PAPER_NUM_EPOCHS
        seq_len = args.seq_length if args.seq_length is not None else P.SEQ_LEN
        grids_per_trajectory, tokens_per_trajectory = P.trajectory_shape(seq_len)
        gen_params = {
            # Recipe marker: the rule batch is generated once (epoch 0) and
            # SHARED across all epochs. Repo naming renders this as
            # share<num_epochs>.
            "variant": "share",
            "num_rules": num_rules,
            "num_sims": num_sims,
            "num_epochs": num_epochs,
            "filter_steps": P.FILTER_STEPS,
            "grid_size": P.GRID_SIZE,
            "num_colors": P.NUM_COLORS,
            "patch_size": P.PATCH_SIZE,
            "seq_len": seq_len,
            "tokens_per_grid": P.TOKENS_PER_GRID,
            "grids_per_trajectory": grids_per_trajectory,
            "tokens_per_trajectory": tokens_per_trajectory,
            "start_token": P.START_TK,
            "end_token": P.END_TK,
            "gzip_threshold": P.GZIP_THRESHOLD,
            "gzip_upper_bound": P.GZIP_UPPER_BOUND,
        }
        source = nca_paper_source(seed=args.seed, num_epochs=num_epochs, num_rules=num_rules, num_sims=num_sims, seq_len=seq_len)
        return source, args.vocab_size, "nca_patch", gen_params
    if args.source == "declref-01":
        if args.vocab_size is None:
            raise ValueError("--source declref-01 requires --vocab-size (>= 1028, its full id range; may be padded higher)")
        from .sources.declref_01 import declref_01_source
        from .sources.declref_01.default_generate_params import SEQ_LENGTH
        # Every knob resolved to a concrete value, so meta.json records what was
        # generated rather than only the overrides.
        gen_params = {
            "seq_length": args.seq_length if args.seq_length is not None else SEQ_LENGTH,
        }
        source = declref_01_source(seed=args.seed, **gen_params)
        return source, args.vocab_size, "declref_01_symbols", gen_params
    if args.source == "dyck":
        if args.vocab_size is None:
            raise ValueError("--source dyck requires --vocab-size (>= 2 * k for k bracket types; may be padded higher)")
        from .sources.dyck import dyck_source
        from .sources.dyck.default_generate_params import K, MAX_DEPTH, P_OPEN, SEQ_LENGTH
        # Every knob resolved to a concrete value, so meta.json records what was
        # generated rather than only the overrides.
        gen_params = {
            "k": args.k if args.k is not None else K,
            "max_depth": args.max_depth if args.max_depth is not None else MAX_DEPTH,
            "p_open": args.p_open if args.p_open is not None else P_OPEN,
            "seq_length": args.seq_length if args.seq_length is not None else SEQ_LENGTH,
        }
        source = dyck_source(seed=args.seed, **gen_params)
        return source, args.vocab_size, "dyck_brackets", gen_params
    if args.source == "fineweb":
        from .sources.fineweb import fineweb_source
        if not args.data_path_local:
            raise ValueError("--source fineweb requires --data-path-local")
        tokenizer = _build_tokenizer(args.tokenizer)
        source = fineweb_source(
            split=args.split,
            tokenizer=tokenizer,
            data_path={"local": args.data_path_local},
            seed=args.seed,
            cycle=False,  # single pass: never duplicate documents in a bin
            tokenizer_threads=args.tokenizer_threads,
            tokenizer_batch_size=args.tokenizer_batch_size,
        )
        return source, tokenizer.get_vocab_size(), args.tokenizer, {}
    if args.source == "c4":
        from .sources.c4 import c4_source
        tokenizer = _build_tokenizer(args.tokenizer)
        source = c4_source(
            tokenizer=tokenizer,
            config=args.c4_config,
            split=args.split,
            seed=args.seed,
            cycle=False,  # single pass: never duplicate documents in a bin
            tokenizer_threads=args.tokenizer_threads,
            tokenizer_batch_size=args.tokenizer_batch_size,
        )
        return source, tokenizer.get_vocab_size(), args.tokenizer, {}
    if args.source == "github-code":
        from .sources.github_code import github_code_source
        tokenizer = _build_tokenizer(args.tokenizer)
        source = github_code_source(
            tokenizer=tokenizer,
            split=args.split,
            seed=args.seed,
            cycle=False,  # single pass: never duplicate documents in a bin
            tokenizer_threads=args.tokenizer_threads,
            tokenizer_batch_size=args.tokenizer_batch_size,
        )
        return source, tokenizer.get_vocab_size(), args.tokenizer, {}
    raise ValueError(f"unknown source {args.source!r}; expected 'fineweb', 'c4', 'github-code', 'nca', 'dyck', or 'declref-01'")


def pretokenize(
    source: TokenStream,
    *,
    output: str,
    target_tokens: int,
    tokenizer: str,
    vocab_size: int,
    source_name: str,
    gen_params: dict | None = None,
    split: str = "train",
    do_upload: bool = False,
    hf_repo: str | None = None,
    repo_dir: str | None = None,
) -> dict:
    """Drain `source` to `target_tokens`, write the `.bin` + `meta.json`, optionally upload.

    Tokens are written incrementally, so peak memory is one chunk and the file
    on disk is exactly `num_tokens * 2` bytes. `target_tokens=-1` drains the
    entire source (bounded sources only; an infinite one would never stop);
    otherwise the last chunk is truncated to land exactly on the budget.
    `vocab_size` must exceed every token id the source emits, and may be padded
    above the source's id range. Returns the `meta.json` contents.
    """
    unbounded = target_tokens == -1
    if not unbounded and target_tokens <= 0:
        raise ValueError(f"target_tokens must be positive, or -1 to drain the full source; got {target_tokens}")
    if do_upload:
        # Checked up front: a multi-hour tokenization must not fail at the end
        # because the repo or token is missing.
        if not hf_repo:
            raise ValueError("do_upload=True requires hf_repo")
        if not os.environ.get("HF_TOKEN"):
            raise RuntimeError("do_upload=True requires HF_TOKEN in the environment (.env is auto-loaded)")

    out_dir = os.path.dirname(os.path.abspath(output))
    os.makedirs(out_dir, exist_ok=True)

    written = 0
    last_print = time.time()
    with open(output, "wb") as f:
        for arr in source:
            if not unbounded and written + len(arr) >= target_tokens:
                arr = arr[: target_tokens - written]
            if arr.size and int(arr.max()) >= vocab_size:
                raise ValueError(
                    f"source emitted token id {int(arr.max())} >= vocab_size {vocab_size}; "
                    f"raise vocab_size to at least cover the source's id range"
                )
            f.write(arr.tobytes())
            written += len(arr)
            now = time.time()
            if now - last_print > 5.0:
                if unbounded:
                    print(f"  {written:>14,} tokens")
                else:
                    print(f"  {written:>14,}/{target_tokens:,} tokens ({100.0 * written / target_tokens:.1f}%)")
                sys.stdout.flush()
                last_print = now
            if not unbounded and written >= target_tokens:
                break

    # The source ran dry before the budget: remove the partial bin rather than
    # leave one that isn't the size it claims. (`-1` means "take all of it", so
    # a short result is the expected full drain.)
    if not unbounded and written < target_tokens:
        os.remove(output)
        raise ValueError(
            f"source produced only {written:,} tokens, short of --target-tokens {target_tokens:,}: "
            f"the dataset is smaller than requested. Lower --target-tokens, or use -1 to take all of it."
        )
    if written == 0:
        os.remove(output)
        raise ValueError("source produced zero tokens; nothing to write (check the source / repo / split)")

    meta = {
        "tokenizer": tokenizer,
        "vocab_size": vocab_size,
        "num_tokens": written,
        "source": source_name,
        "split": split,
        "dtype": "uint16",
    }
    if gen_params:
        meta["gen_params"] = gen_params
    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {written:,} tokens ({written * 2 / 1e6:.1f} MB) -> {output}")
    print(f"Wrote meta -> {meta_path}")

    if do_upload:
        _upload(output, meta_path, hf_repo=hf_repo, repo_dir=repo_dir or os.path.basename(out_dir))

    return meta


def _upload(bin_path: str, meta_path: str, *, hf_repo: str, repo_dir: str) -> None:
    """Upload the bin + its meta sidecar into the `repo_dir` subfolder of an HF dataset repo.

    For small, expensive-to-regenerate bins shared across machines (NCA, C4);
    huge regeneratable bins (FineWeb slices) stay local. The repo is a plain
    dataset repo of `.bin` files, downloaded back by filename via the bin
    loader's `{"hf": {...}}` path.
    """
    from huggingface_hub import HfApi

    from nanochat.hf_upload import retry_on_timeout as _retry_on_timeout  # exponential backoff on transient network errors
    api = HfApi(token=os.environ["HF_TOKEN"])
    # Public: data bins are shared artifacts.
    _retry_on_timeout(api.create_repo, repo_id=hf_repo, repo_type="dataset",
                      exist_ok=True, private=False)
    print(f"Uploading to {hf_repo} ...")
    for local_path in (bin_path, meta_path):
        dest = f"{repo_dir}/{os.path.basename(local_path)}" if repo_dir else os.path.basename(local_path)
        _retry_on_timeout(
            api.upload_file,
            path_or_fileobj=local_path,
            path_in_repo=dest,
            repo_id=hf_repo,
            repo_type="dataset",
        )
        print(f"  -> {hf_repo}/{dest}")


def _shard_worker(params: dict) -> int:
    """Drain one rank-disjoint shard of a procedural source to its shard file.

    Runs in a pool subprocess; `params` carries `source`, `seed`, `rank`,
    `gen_params`, `vocab_size`, `shard_tokens` and `path`.
    """
    if params["source"] == "declref-01":
        from .sources.declref_01 import declref_01_source as source_fn
    elif params["source"] == "dyck":
        from .sources.dyck import dyck_source as source_fn
    else:
        raise ValueError(f"sharded generation does not support source {params['source']!r}")
    source = source_fn(seed=params["seed"], rank=params["rank"], **params["gen_params"])
    written = 0
    with open(params["path"], "wb") as f:
        for arr in source:
            if written + len(arr) > params["shard_tokens"]:
                arr = arr[: params["shard_tokens"] - written]
            if arr.size and int(arr.max()) >= params["vocab_size"]:
                raise ValueError(
                    f"source emitted token id {int(arr.max())} >= vocab_size {params['vocab_size']}"
                )
            f.write(arr.tobytes())
            written += len(arr)
            if written >= params["shard_tokens"]:
                break
    return written


def run_sharded(args) -> dict:
    """Generate `args.num_shards` rank-disjoint shards in parallel and concatenate.

    The bin is the concatenation of shards `rank = 0 .. num_shards - 1` in rank
    order, so its content is fixed by `(seed, num_shards)` alone — `num_workers`
    only sets the pool size. Pin `num_shards` in the prep script that defines a
    corpus; vary `num_workers` freely per machine. Procedural rank-capable
    sources only (declref-01, dyck).
    """
    from multiprocessing import Pool

    if args.source not in ("declref-01", "dyck"):
        raise ValueError(f"--num-shards > 1 supports procedural sources 'declref-01' and 'dyck'; got {args.source!r}")
    if args.target_tokens <= 0:
        raise ValueError("--num-shards > 1 requires a positive --target-tokens")
    _, vocab_size, tokenizer, gen_params = build_source(args)  # lazy: nothing is generated here
    seq_length = gen_params["seq_length"]
    if args.upload:
        _check_upload_ready(args.hf_repo)

    per_shard = args.target_tokens // args.num_shards
    if args.target_tokens % args.num_shards or per_shard % seq_length:
        raise ValueError(
            f"--target-tokens {args.target_tokens:,} must split into num_shards={args.num_shards} "
            f"whole-document shards (shard size a multiple of seq_length={seq_length})"
        )

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    shard_paths = [f"{args.output}.shard{i:03d}" for i in range(args.num_shards)]
    jobs = [
        {"source": args.source, "seed": args.seed, "rank": i, "gen_params": gen_params,
         "vocab_size": vocab_size, "shard_tokens": per_shard, "path": shard_paths[i]}
        for i in range(args.num_shards)
    ]
    workers = args.num_workers or os.cpu_count()
    print(f"Generating {args.target_tokens:,} tokens as {args.num_shards} shards on {workers} workers ...")
    try:
        from tqdm import tqdm
        progress = tqdm(total=args.num_shards, unit="shard", desc="generating")
    except ImportError:
        progress = None
    with Pool(processes=workers) as pool:
        done = 0
        for _ in pool.imap_unordered(_shard_worker, jobs):
            done += 1
            if progress is not None:
                progress.update(1)
            else:
                print(f"  shard {done}/{args.num_shards} done")
                sys.stdout.flush()
    if progress is not None:
        progress.close()

    with open(args.output, "wb") as out:
        for path in shard_paths:
            with open(path, "rb") as f:
                shutil.copyfileobj(f, out)
            os.remove(path)

    meta = {
        "tokenizer": tokenizer,
        "vocab_size": vocab_size,
        "num_tokens": args.target_tokens,
        "source": args.source,
        "split": args.split,
        "dtype": "uint16",
        "gen_params": {**gen_params, "num_shards": args.num_shards},
    }
    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {args.target_tokens:,} tokens ({args.target_tokens * 2 / 1e6:.1f} MB) -> {args.output}")
    print(f"Wrote meta -> {meta_path}")
    if args.upload:
        _upload(args.output, meta_path, hf_repo=args.hf_repo,
                repo_dir=args.repo_dir or os.path.basename(out_dir))
    return meta


def _check_upload_ready(hf_repo: str | None) -> None:
    """A long generation must not fail at the end for a missing repo or token."""
    if not hf_repo:
        raise ValueError("--upload requires --hf-repo")
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("--upload requires HF_TOKEN in the environment (.env is auto-loaded)")


def run(args) -> dict:
    """Build the selected source and pre-tokenize it to a bin per `args`."""
    if args.num_shards > 1:
        return run_sharded(args)
    source, vocab_size, tokenizer, gen_params = build_source(args)
    return pretokenize(
        source,
        output=args.output,
        target_tokens=args.target_tokens,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        source_name=args.source,
        gen_params=gen_params,
        split=args.split,
        do_upload=args.upload,
        hf_repo=args.hf_repo,
        repo_dir=args.repo_dir,
    )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Pre-tokenize a generating source to a uint16 .bin.")
    parser.add_argument("--source", required=True, choices=["fineweb", "c4", "github-code", "nca", "dyck", "declref-01"], help="generating source")
    parser.add_argument("--target-tokens", type=int, required=True, help="token budget for the bin (-1 = drain the full source; bounded sources only)")
    parser.add_argument("--output", required=True, help="output .bin path (meta.json is written alongside)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1, help="split generation into this many rank-disjoint shards, concatenated in rank order; the bin is fixed by (seed, num_shards), so pin it per corpus (procedural sources only)")
    parser.add_argument("--num-workers", type=int, default=None, help="process-pool size for sharded generation (default: all cores); never affects bin content")
    parser.add_argument("--upload", action="store_true", help="upload the bin + meta to an HF dataset repo")
    parser.add_argument("--hf-repo", default=None, help="target HF dataset repo id (required with --upload)")
    parser.add_argument("--repo-dir", default=None, help="repo subfolder for the upload (defaults to the bin's folder name)")
    # text sources (fineweb, c4)
    parser.add_argument("--tokenizer", default="rustbpe", choices=["rustbpe"], help="text-source tokenizer")
    parser.add_argument("--data-path-local", default=None, help="local dir of parquet shards (fineweb source)")
    parser.add_argument("--c4-config", default="en", help="HF dataset config/subset (c4 source)")
    parser.add_argument("--split", default="train", choices=["train", "val"], help="dataset split (fineweb, c4, github-code)")
    parser.add_argument("--tokenizer-threads", type=int, default=4, help="encode threads for text sources (4 is near-optimal for streaming; raise for big local jobs)")
    parser.add_argument("--tokenizer-batch-size", type=int, default=128, help="documents per encode call for text sources")
    # procedural sources (nca, dyck); omitted knobs fall back to source defaults
    parser.add_argument("--vocab-size", type=int, default=None, help="vocab size for procedural sources (>= the source's id range, e.g. >= 2*k for dyck). text sources derive it from the tokenizer")
    parser.add_argument("--num-rules", type=int, default=None, help="gzip-filtered rules per epoch (nca source)")
    parser.add_argument("--num-sims", type=int, default=None, help="total simulations per epoch (nca source)")
    parser.add_argument("--num-epochs", type=int, default=None, help="distinct-seed epochs to concatenate (nca source)")
    parser.add_argument("--k", type=int, default=None, help="bracket types (dyck source)")
    parser.add_argument("--max-depth", type=int, default=None, help="max nesting depth (dyck source)")
    parser.add_argument("--p-open", type=float, default=None, help="open-bracket probability (dyck source)")
    parser.add_argument("--seq-length", type=int, default=None, help="model context the generated sequences must cover (dyck/declref-01: exact tokens per sequence; nca: trajectory rounds UP to whole delimiter-wrapped grids, e.g. 2048 -> 2052 stored)")
    return parser.parse_args(argv)


def main(argv=None):
    load_dotenv()
    run(_parse_args(argv))


if __name__ == "__main__":
    main()
