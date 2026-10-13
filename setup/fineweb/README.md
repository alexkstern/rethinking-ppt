# fineweb_rustbpe_55B — prep notes

Companion to `prep_train_55B.sh`. Goal: a 55B-token FineWeb-Edu
train bin whose **first 20,000,000,000 tokens are byte-identical** to
`fineweb_rustbpe_20B.bin` (HF: `alexkstern/fineweb-nanochatbpe-20B`, registry key
`fineweb-nanochatbpe-20B`), so 50B-token runs consume a strict superset of what
every previous 20B-corpus run saw.

## Determinism analysis (why the prefix is reproducible)

The token stream is a pure function of (shard files, tokenizer, code path):

- **Document order is fixed.** `nanochat/data/sources/fineweb/source.py` lists
  `*.parquet` in the data dir with `sorted(os.listdir(...))`, reserves the last
  file (excluded from `split="train"`), and reads the rest file-by-file, row
  group 0, 1, 2, ... (`rank=0, world_size=1` in the pretokenizer → stride 1).
  No shuffle anywhere; the source's `seed` parameter is documented unused.
- **Tokenization is order-preserving and thread-count-independent.**
  Documents go through tiktoken's `encode_ordinary_batch` (via
  `RustBPETokenizer.encode`); it returns results in input order, and
  `--tokenizer-threads` / `--tokenizer-batch-size` only change parallelism and
  call grouping, never ids or order. Output is little-endian `uint16` on both
  build machines (macOS arm64 / Linux x86_64 are both little-endian).
- **The 20B cut is a plain truncation of the same stream.** `pretokenize`
  drains the stream and slices the final document's array to land exactly on
  `--target-tokens` (the 20B bin ends mid-document). A 55B run replays the
  identical stream and simply continues through that document, so byte-identity
  holds across the old cut point — no boundary special-case.
- **Extending the shard set does not perturb the prefix.** The 20B build used
  `NUM_SHARDS=390`: train = shards 0000..0388 (shard_00389 was the reserved
  last, tokenized into neither split). The 55B build uses `NUM_SHARDS=1150` in
  the same dir: train = 0000..1148. Sorted order keeps 0000..0388 first and
  unchanged; shard_00389 simply enters the stream after 0388, well past the 20B
  cut (~shard 0378 at ~52.9M tokens/shard). Val is untouched: it has always
  been the far-away held-out `shard_01822.parquet` (built by
  `prep_val.sh` from its own dir), never the dir's reserved
  last shard — so there is no train/val contamination and no val rebuild.

The 55B script pins both of its inputs:

- **Upstream revision.** Shards are downloaded from
  `karpathy/fineweb-edu-100b-shuffle` at revision
  `4c8f30d6756da75362432a4d5569e1b229263b71` rather than at `main`, so an
  upstream re-upload cannot silently change them.
- **Tokenizer.** `pretokenize --tokenizer rustbpe` loads
  `$BASE_DIR/tokenizer/{tokenizer.pkl,token_bytes.pt}`. The script
  sha256-compares those against the copies bundled in the published 20B repo
  (installing from there if absent) and refuses to proceed on mismatch.

Stage 4 then byte-compares the new bin's full 40 GB prefix against the 20B bin
and hard-fails, with no upload, on the first differing byte.

## Provenance chain

```
karpathy/fineweb-edu-100b-shuffle @ 4c8f30d6756da75362432a4d5569e1b229263b71
  └─ shards 0000..1149 → $BASE_DIR/base_data/          (1150 files, ~108 GB;
     last shard reserved by the split rule → train = 0000..1148)
       └─ fineweb_source(split=train, cycle=False)      sorted files → row groups → docs
            └─ rustbpe tokenizer (== 20B repo tokenizer/, sha256-verified), BOS-prefixed
                 └─ pretokenize --target-tokens 55e9 → fineweb_rustbpe_55B.bin (110 GB uint16)
                      ├─ VERIFY: cmp first 40 GB == fineweb_rustbpe_20B.bin  (must pass)
                      └─ hf_register → alexkstern/fineweb-nanochatbpe-55B
                           train.bin (new) + val.bin (byte-identical copy of the
                           20B repo's val: shard_01822, 52,336,096 tokens) + tokenizer/
```

Registry key: `fineweb-nanochatbpe-55B` (metric `val/fineweb_bpb`, unchanged —
the val set is the same bytes, so bpb is directly comparable with runs on the 20B corpus).

## Expected wall time and disk

Measured on the 20B build with 18 tokenizer threads: sustained ~11.8M
tokens/s, so 20B in ~30 min. Scaling:

- shard download: ~760 new shards x ~94 MB ≈ 71 GB (fresh machine: 1150 ≈ 108 GB);
  ~20-60 min at 30-90 MB/s with 8 workers
- tokenize 55B: proportionally longer than the 20B build above; budget 1-3 h
  depending on host CPU
- prefix verify: reads 2 x 40 GB; ~5-20 min disk-bound (+40 GB download of the
  reference train.bin if the machine has no local 20B bin)
- upload 110 GB to HF (Xet): ~1-3 h

Disk high-water mark ≈ 260 GB (108 GB parquet + 110 GB new bin + 40 GB reference).

## Risks

- Stray parquet files in `base_data/` (anything outside shard_00000..01149)
  would change the stream; the script counts the files, checks the last
  filename and aborts rather than tokenize.
- Drift in the source-code path (`_shard_paths`, BOS handling, encode batching)
  could perturb the prefix; stage 4's 40 GB `cmp` catches it before any upload.
- The tokens/shard rate (~52.9M) is an average, and `NUM_SHARDS=1150` carries
  ~10% margin. If the source ran dry anyway, `pretokenize` deletes the partial
  bin and exits non-zero — rerun with a higher `NUM_SHARDS`.
