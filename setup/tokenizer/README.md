# Tokenizer

The BPE tokenizer that every `*-nanochatbpe-*` corpus in the data registry was
tokenized with, plus the code that produced it. Two artifacts:

- `tokenizer.pkl` — the tokenizer itself (vocab 65536)
- `token_bytes.pt` — per-token byte lengths, which the bits-per-byte eval needs
  to weight each position by its token's byte length

Both are published under `tokenizer/` in every `*-nanochatbpe-*` dataset repo
on the Hub, byte-identical across all of them, and that is the only copy — the
Hub is the source of truth, nothing is committed here.

## Installing it — required before building any corpus bin

```
bash setup/tokenizer/install_tokenizer.sh
```

Corpus prep resolves the tokenizer **locally, not from the Hub**: `pretokenize
--tokenizer rustbpe` calls `build_tokenizer("rustbpe")` with no repo id, which
loads `{base_dir}/tokenizer/tokenizer.pkl`. Nothing populates that directory on
its own, so this has to run once before any `setup/*/prep_*.sh` that tokenizes
text; it downloads both artifacts into place. Training runs are unaffected —
`nanochat/data/registry.py` fetches the copy bundled with whichever dataset the
run uses.

## Retraining it from scratch

```
bash setup/tokenizer/train_tokenizer.sh
```

Writes the same two artifacts to the same place. Reads raw FineWeb-Edu parquet
shards from the base dir, so fetch those first (see `setup/fineweb/`); it trains
on the train split — every shard except the last, which is reserved for
validation — so at least two shards must be present. Not needed to reproduce any
result.

## Arguments

The published tokenizer was trained with the script's defaults — 65536 vocab,
a 10B-character budget, documents cropped to 10k characters — which the wrapper
passes explicitly so the run is self-describing. The vocab size checks out
against the artifact: its `token_bytes.pt` holds exactly 65536 entries, 9 of
them special tokens with zero byte length, matching the `--pad-vocab=65536`
every experiment launcher passes.

Training is deterministic given the shards: documents are read in sorted shard
and row-group order and the run stops at the character budget, so the shard
count only has to be large enough to reach it (~41 at roughly 250M characters
per shard). Downloading more does not change what the tokenizer sees.

## The two parquet readers

`fineweb_parquet.py` here and `nanochat/data/sources/fineweb/source.py` walk the
same shards, in the same order, with the same split rule and the same row-group
striding. They differ in one respect: the source tokenizes before yielding,
while this one yields raw text, which a tokenizer trainer needs — a BPE
tokenizer cannot be trained through a reader that already requires a tokenizer.
