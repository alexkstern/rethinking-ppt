# fineweb_rustbpe_55B

`prep_train_55B.sh` builds a 55B-token FineWeb-Edu train bin whose first
20,000,000,000 tokens are byte-identical to `fineweb_rustbpe_20B.bin` (HF:
`alexkstern/fineweb-nanochatbpe-20B`, registry key `fineweb-nanochatbpe-20B`).

```
bash setup/fineweb/prep_train_55B.sh            # build + verify
bash setup/fineweb/prep_train_55B.sh --upload   # ... then publish (needs HF_TOKEN)
```

It downloads shards 0000..1149 of `karpathy/fineweb-edu-100b-shuffle` at
revision `4c8f30d6756da75362432a4d5569e1b229263b71`, checks the local tokenizer
against the one in the 20B repo, tokenizes, and checks the first 20B tokens
against the 20B bin before any upload. With `--upload` it publishes
`alexkstern/fineweb-nanochatbpe-55B` (registry key `fineweb-nanochatbpe-55B`,
metric `val/fineweb_bpb`) with the 20B repo's val bin.

## Notes

- `base_data/` must hold exactly shard_00000..01149. The script aborts on
  stray parquet files.
- If the source runs out of tokens, `pretokenize` deletes the partial bin and
  exits non-zero. Rerun with a higher `NUM_SHARDS`.
