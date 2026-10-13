"""Publish a train+val bin pair (and the tokenizer) as a public HF dataset repo.

One repo per (dataset, tokenizer). It holds `train.bin` + `val.bin` — pulled by
filename via the bin loader's `{"hf": {"repo_id": ..., "filename": "train.bin"}}`
path — each bin's `meta.json`, the tokenizer files, and a small README.

Repo id is derived from the TRAIN bin's `meta.json`:
`{org}/{source}-{tokenizer}-{trainN}`, where the rustbpe tokenizer is published
under the public name **nanochatbpe** and `trainN` is the rounded train token
count (e.g. `20B`).

    python -m nanochat.data.hf_register \
        --train-bin ~/.cache/nanochat/bins/fineweb_rustbpe_20B/fineweb_rustbpe_20B.bin \
        --val-bin   ~/.cache/nanochat/bins/fineweb_rustbpe_val/fineweb_rustbpe_val.bin \
        --tokenizer-dir ~/.cache/nanochat/tokenizer
"""

from __future__ import annotations

import argparse
import json
import os

# Impl name -> public name.
_TOKENIZER_PUBLIC = {"rustbpe": "nanochatbpe"}

# Human-facing source name + upstream HF dataset, used in the README.
_SOURCE_INFO = {
    "fineweb": ("FineWeb-Edu", "karpathy/fineweb-edu-100b-shuffle"),
    "c4": ("C4 (en)", "allenai/c4"),
    "github-code": ("GitHub Code (all-all)", "codeparrot/github-code"),
}


def _token_label(n: int) -> str:
    """Round token count to a compact label, e.g. 20_000_000_000 -> '20B'.

    The B/M thresholds are set at the rounding boundary (>= 0.9995 of a unit) so a
    count like 999_999_999 labels as '1B', never '1000M'.
    """
    if n >= 999_500_000:        # rounds to >= 1B
        return f"{round(n / 1e9)}B"
    if n >= 999_500:            # rounds to >= 1M
        return f"{round(n / 1e6)}M"
    return str(n)


def _size_category(n: int) -> str:
    """HF dataset-card size bucket for `n` examples (here, tokens)."""
    for hi, label in (
        (1_000, "n<1K"), (10_000, "1K<n<10K"), (100_000, "10K<n<100K"),
        (1_000_000, "100K<n<1M"), (10_000_000, "1M<n<10M"), (100_000_000, "10M<n<100M"),
        (1_000_000_000, "100M<n<1B"), (10_000_000_000, "1B<n<10B"),
        (100_000_000_000, "10B<n<100B"), (1_000_000_000_000, "100B<n<1T"),
    ):
        if n < hi:
            return label
    return "n>1T"


def _meta(bin_path: str) -> dict:
    with open(os.path.join(os.path.dirname(os.path.abspath(bin_path)), "meta.json")) as f:
        return json.load(f)


def _descriptor(meta: dict) -> str:
    """The middle name-segment distinguishing bins of the same source: the
    tokenizer for text, generation params for a procedural source (dyck ->
    `k{k}-seq_len_{seq_length}`), which has none."""
    if meta["source"] == "dyck":
        g = meta["gen_params"]
        return f"k{g['k']}-seq_len_{g['seq_length']}"
    if meta["source"] == "nca":
        g = meta["gen_params"]
        # "paper-share<N>-" reuses one batch of rules across all N epochs, so
        # each rule gets N simulation trajectories. Some metas spell that same
        # recipe "share10" rather than "share".
        variant = g.get("variant")
        if variant == "share":
            prefix = f"paper-share{g['num_epochs']}-"
        elif variant == "share10":
            prefix = "paper-share10-"
        elif variant == "paper":
            prefix = "paper-"
        else:
            prefix = ""
        return f"{prefix}seq_len_{g['seq_len']}"
    return _TOKENIZER_PUBLIC.get(meta["tokenizer"], meta["tokenizer"])


# gen_params knobs that affect generation size/partitioning only, not the data
# distribution — train and val may legitimately differ on them (e.g. a smaller,
# single-stream val vs a sharded train).
_BATCH_ONLY_GEN_PARAMS = {"num_rules", "num_sims", "num_epochs", "num_shards"}


def _distributional_gen_params(meta: dict):
    """gen_params minus batch-only knobs — the part that actually defines the data."""
    g = meta.get("gen_params")
    if not g:
        return g
    out = {k: v for k, v in g.items() if k not in _BATCH_ONLY_GEN_PARAMS}
    # "share10" and "share" name the same rules-generated-once recipe.
    if out.get("variant") == "share10":
        out["variant"] = "share"
    return out


def repo_id_for(train_bin: str, org: str) -> str:
    """Derive `{org}/{source}-{descriptor}-{trainN}` from the train bin's meta."""
    m = _meta(train_bin)
    return f"{org}/{m['source']}-{_descriptor(m)}-{_token_label(m['num_tokens'])}"


def _front_matter(repo_id: str, train_meta: dict) -> str:
    """Minimal HF dataset-card YAML header: tags plus a size bucket. Per-split
    token counts are deliberately left out — declaring split `num_examples`
    would clash with the dataset viewer's auto-detected view."""
    head = "---\n" + f"pretty_name: {repo_id.split('/')[-1]}\n"
    src = train_meta["source"]
    if src == "dyck":
        # Synthetic bracket sequences — not natural language, so no `language` tag.
        body = "tags:\n- pretokenized\n- formal-language\n- synthetic\n- dyck\n"
    elif src == "nca":
        body = "tags:\n- pretokenized\n- synthetic\n- cellular-automata\n- nca\n"
    elif src == "declref-01":
        body = "tags:\n- pretokenized\n- formal-language\n- synthetic\n- declref\n"
    else:
        body = "language:\n- en\n" + "tags:\n- pretokenized\n- language-modeling\n"
    return head + body + f"size_categories:\n- {_size_category(train_meta['num_tokens'])}\n" + "---\n\n"


def _declref_doc_notes(g: dict) -> str:
    """Shared README `notes` for the declare/reference sources, which store one
    fixed-length document per row."""
    return (
        f"The bin is a concatenation of fixed-length **{g['seq_length']}-token documents** "
        f"(`num_tokens` is a whole multiple of it): read via `tokens.reshape(-1, {g['seq_length']})` "
        "to train one document per row. Document alignment matters — a window starting "
        "mid-document orphans its references. `train` (seed 0) and `val` (a disjoint seed) are "
        "independent streams of the same grammar; token count = `filesize / 2`; the metas carry "
        "the full config, including the shard layout that fixes the bin's exact content."
    )


def _readme(repo_id: str, train_meta: dict, val_meta: dict) -> str:
    name = repo_id.split("/")[-1]
    table = (
        "| file | split | tokens |\n| --- | --- | --- |\n"
        f"| `train.bin` | train | {train_meta['num_tokens']:,} |\n"
        f"| `val.bin` | val | {val_meta['num_tokens']:,} |\n"
    )
    load = (
        "**Load a bin** with the standard Hugging Face downloader:\n\n"
        "```python\n"
        "from huggingface_hub import hf_hub_download\n"
        "import numpy as np\n\n"
        f'path = hf_hub_download(repo_id="{repo_id}", filename="train.bin", repo_type="dataset")\n'
        'tokens = np.memmap(path, dtype="uint16", mode="r")\n'
        "```\n"
    )
    if train_meta["source"] == "dyck":
        g = train_meta["gen_params"]
        desc = (
            "Procedurally generated **k-shuffle Dyck** bracket sequences "
            "(Hu et al. 2025, [arXiv:2502.19249](https://arxiv.org/abs/2502.19249)), as flat "
            "`uint16` token-id `.bin` files. Token ids are 0-based: opening bracket type *i* is "
            f"id *i* and its matching close is *i + k*, so ids span `[0, 2k)` and the vocabulary is "
            f"`2k = {train_meta['vocab_size']}`.\n\n"
            "**Grammar parameters**\n\n"
            "| param | value |\n| --- | --- |\n"
            f"| k (bracket types) | {g['k']} |\n"
            f"| max_depth | {g['max_depth']} |\n"
            f"| p_open | {g['p_open']} |\n"
            f"| seq_length | {g['seq_length']} |\n"
        )
        notes = (
            f"The bin is a concatenation of fixed-length **{g['seq_length']}-token words**, each a "
            "complete depth-0-starting Dyck word (so `num_tokens` is a whole multiple of "
            "`seq_length`). Read it in `seq_length`-token chunks — `tokens.reshape(-1, "
            f"{g['seq_length']})` — to train one word per row, exactly as generated. `train` "
            "(seed 0) and `val` (a disjoint generator seed) are independent streams of the same "
            "grammar. Token count = `filesize / 2`; `train.meta.json` / `val.meta.json` carry the full config."
        )
        return _front_matter(repo_id, train_meta) + f"# {name}\n\n{desc}\n{table}\n{notes}\n\n{load}"
    if train_meta["source"] == "declref-01":
        g = train_meta["gen_params"]
        desc = (
            "Procedurally generated **decl-ref-01** documents — a scoped declare/reference "
            "formal language with noisy references, sparse part-transition chains, "
            "interleaved openings, and periodic topic shifts tuned so that a model trained on it "
            "matches natural-language / code entropy dynamics (positional entropy profile, its "
            "fluctuation texture, and the entropy-quantile distribution), and holds that match as "
            "training doubles. Each document is a random program of `{` `}` scopes, "
            "`DECL name value` statements and `REF name value` statements whose value tokens repeat "
            "an earlier declaration — predictable only by attending back to it. Names and values "
            "are 1–3 subtoken symbols. Token ids: 0=OPEN 1=CLOSE 2=DECL 3=REF, then name-parts and "
            f"value-parts in disjoint blocks; vocab = {train_meta['vocab_size']:,}.\n\n"
            "**Grammar parameters**\n\n"
            "| param | value |\n| --- | --- |\n"
            f"| seq_length (document) | {g['seq_length']} |\n"
            f"| shards (canonical layout) | {g.get('num_shards', 1)} |\n"
        )
        notes = _declref_doc_notes(g)
        return _front_matter(repo_id, train_meta) + f"# {name}\n\n{desc}\n{table}\n{notes}\n\n{load}"
    if train_meta["source"] == "nca":
        g = train_meta["gen_params"]
        desc = (
            "Procedurally generated **Neural Cellular Automata** trajectories (Lee et al. 2026), as flat "
            f"`uint16` token-id `.bin` files. Random NCA rules are rolled out on a {g['grid_size']}×{g['grid_size']} "
            f"grid of {g['num_colors']} cell states and tokenized by {g['patch_size']}×{g['patch_size']} patches "
            f"(base-{g['num_colors']}); only high-complexity rules survive a gzip-ratio filter (kept iff in "
            f"({g['gzip_threshold']}, {g['gzip_upper_bound']})). Token ids: {g['start_token']:,} patch ids plus two "
            f"grid delimiters (start={g['start_token']}, end={g['end_token']}); vocab = {train_meta['vocab_size']:,}.\n\n"
            "**Configuration**\n\n"
            "| param | value |\n| --- | --- |\n"
            f"| grid | {g['grid_size']}×{g['grid_size']} |\n"
            f"| colors (states) | {g['num_colors']} |\n"
            f"| patch | {g['patch_size']}×{g['patch_size']} |\n"
            f"| seq_len (target) | {g['seq_len']} |\n"
            f"| tokens / grid | {g['tokens_per_grid']} |\n"
            f"| grids / trajectory | {g['grids_per_trajectory']} |\n"
            f"| tokens / trajectory | {g['tokens_per_trajectory']} |\n"
            f"| gzip band | ({g['gzip_threshold']}, {g['gzip_upper_bound']}) |\n"
        )
        notes = (
            f"The bin is a concatenation of fixed-length **{g['tokens_per_trajectory']}-token trajectory records** "
            f"(`num_tokens` is a whole multiple of it): load via `tokens.reshape(-1, {g['tokens_per_trajectory']})`, "
            "then pack one or more trajectories per row as needed. The delimiter ids "
            f"({g['start_token']}, {g['end_token']}) mark grid boundaries and are masked in the loss. `train` "
            "(seed 0) and `val` (a disjoint seed) are independent streams of the same generator; token count = "
            "`filesize / 2`; the metas carry the full config.\n\n"
            f"**The complexity filter is per-rule.** The gzip band ({g['gzip_threshold']}, {g['gzip_upper_bound']}) "
            f"is applied to each *rule* once, scored on a single random init (a {g['filter_steps']}-grid rollout). "
            "The sims generated per kept rule use fresh inits and are not re-scored, so — because NCA rules can "
            "be init-sensitive (multiple basins) — a tiny fraction (~0.01% of trajectories) "
            "collapse to a degenerate fixed point (e.g. a uniform grid), and such collapses can repeat exactly across "
            "sims. These are low-complexity and harmless; it is inherent to the per-rule filter, not a generation error."
        )
        return _front_matter(repo_id, train_meta) + f"# {name}\n\n{desc}\n{table}\n{notes}\n\n{load}"
    tok_pub = _TOKENIZER_PUBLIC.get(train_meta["tokenizer"], train_meta["tokenizer"])
    pretty, upstream = _SOURCE_INFO[train_meta["source"]]
    src = f"**{pretty}**"
    if upstream:
        src += f" (from [`{upstream}`](https://huggingface.co/datasets/{upstream}))"
    desc = (
        f"{src}, pre-tokenized with the `{tok_pub}` tokenizer (vocab {train_meta['vocab_size']:,}) "
        "and packaged as flat `uint16` token-id `.bin` files for fast memmap training."
    )
    notes = (
        "`train` and `val` are disjoint held-out partitions. Each `.bin` is a raw "
        "little-endian `uint16` stream (no header); token count = `filesize / 2`, and "
        "`train.meta.json` / `val.meta.json` carry the full metadata. The `tokenizer/` "
        "files (when present) are the exact tokenizer used to produce these ids."
    )
    return _front_matter(repo_id, train_meta) + f"# {name}\n\n{desc}\n\n{table}\n{notes}\n\n{load}"


def register(train_bin: str, val_bin: str, *, org: str, tokenizer_dir: str | None = None,
             private: bool = False, repo_id: str | None = None) -> str:
    """Create the (public by default) dataset repo and upload bins + metas + tokenizer."""
    # Xet upload concurrency for the multi-GB bins. Must be set BEFORE
    # huggingface_hub is imported, hence the lazy import below; export
    # HF_XET_HIGH_PERFORMANCE=0 to opt out.
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    from dotenv import load_dotenv
    from huggingface_hub import HfApi

    from nanochat.hf_upload import retry_on_timeout as _retry_on_timeout  # exponential backoff on transient errors

    load_dotenv()
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN required in the environment (.env is auto-loaded)")

    repo_id = repo_id or repo_id_for(train_bin, org)
    train_meta, val_meta = _meta(train_bin), _meta(val_bin)
    # The train + val bins must be the same corpus/tokenizer.
    for key in ("source", "tokenizer", "vocab_size"):
        if train_meta[key] != val_meta[key]:
            raise ValueError(
                f"train and val bins disagree on {key!r} ({train_meta[key]!r} vs {val_meta[key]!r}): "
                f"are --train-bin and --val-bin from the same corpus?"
            )
    # Procedural sources: the data-defining config must match, else the repo
    # name (derived from train) would misdescribe val. Size-only knobs are
    # excluded — train and val are routinely generated at different sizes.
    if _distributional_gen_params(train_meta) != _distributional_gen_params(val_meta):
        raise ValueError(
            f"train and val bins disagree on data-defining gen_params "
            f"({_distributional_gen_params(train_meta)!r} vs {_distributional_gen_params(val_meta)!r})"
        )
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    print(f"Registering {repo_id} ({'private' if private else 'public'})")

    uploads = [
        (train_bin, "train.bin"),
        (os.path.join(os.path.dirname(train_bin), "meta.json"), "train.meta.json"),
        (val_bin, "val.bin"),
        (os.path.join(os.path.dirname(val_bin), "meta.json"), "val.meta.json"),
    ]
    if tokenizer_dir:
        # Never silently publish a tokenizer-less (undecodable) dataset.
        for fn in ("tokenizer.pkl", "token_bytes.pt"):
            p = os.path.join(tokenizer_dir, fn)
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"--tokenizer-dir {tokenizer_dir!r} is missing {fn}; "
                    f"refusing to publish a dataset without its tokenizer"
                )
            uploads.append((p, f"tokenizer/{fn}"))

    for local_path, dest in uploads:
        _retry_on_timeout(api.upload_file, path_or_fileobj=local_path, path_in_repo=dest,
                          repo_id=repo_id, repo_type="dataset")
        print(f"  -> {repo_id}/{dest}")

    _retry_on_timeout(api.upload_file, path_or_fileobj=_readme(repo_id, train_meta, val_meta).encode(),
                      path_in_repo="README.md", repo_id=repo_id, repo_type="dataset")
    print(f"  -> {repo_id}/README.md")
    print(f"Done: https://huggingface.co/datasets/{repo_id}")
    return repo_id


def main(argv=None):
    p = argparse.ArgumentParser(description="Publish a train+val bin pair as a public HF dataset repo.")
    p.add_argument("--train-bin", required=True, help="path to the train .bin (meta.json sits alongside)")
    p.add_argument("--val-bin", required=True, help="path to the val .bin (meta.json sits alongside)")
    p.add_argument("--org", default="alexkstern", help="HF org/user for the repo")
    p.add_argument("--tokenizer-dir", default=None, help="dir with tokenizer.pkl + token_bytes.pt (rustbpe only)")
    p.add_argument("--private", action="store_true", help="create a private repo (default: public)")
    p.add_argument("--repo-id", default=None,
                   help="full repo id override (org/name); default derives from the train meta. "
                        "For variants the meta cannot distinguish, e.g. a re-generation seed")
    a = p.parse_args(argv)
    register(a.train_bin, a.val_bin, org=a.org, tokenizer_dir=a.tokenizer_dir, private=a.private,
             repo_id=a.repo_id)


if __name__ == "__main__":
    main()
