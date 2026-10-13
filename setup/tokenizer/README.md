# Tokenizer

The BPE tokenizer that every `*-nanochatbpe-*` corpus in the data registry was
tokenized with, plus the code that produced it. Two artifacts:

- `tokenizer.pkl` — the tokenizer itself (vocab 65536)
- `token_bytes.pt` — per-token byte lengths, which the bits-per-byte eval needs
  to weight each position by its token's byte length

Both are published under `tokenizer/` in every `*-nanochatbpe-*` dataset repo
on the Hub. Nothing is committed here.

## Using it

Nothing to install. Corpus prep (`pretokenize --tokenizer rustbpe`) downloads
the tokenizer from the published 20B FineWeb repo, and training and eval read
the copy bundled with whichever dataset a run uses.

Publishing a text corpus bundles `{base_dir}/tokenizer` into its dataset repo
(the `setup/*/hf_register.sh` scripts), so before publishing, put the published
copy there:

```
bash setup/tokenizer/install_tokenizer.sh
```

## Retraining it from scratch

```
bash setup/tokenizer/train_tokenizer.sh
```

Writes the same two artifacts to `{base_dir}/tokenizer`. Reads raw FineWeb-Edu parquet
shards from the base dir, so fetch at least ~41 first (see `setup/fineweb/`);
the last shard is reserved for validation. Not needed to reproduce any result.

## Arguments

The published tokenizer was trained with the script's defaults — 65536 vocab,
a 10B-character budget, documents cropped to 10k characters.
