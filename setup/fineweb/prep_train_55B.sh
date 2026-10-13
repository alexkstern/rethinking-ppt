#!/usr/bin/env bash
#
# Build fineweb_rustbpe_55B.bin — 55B-token FineWeb-Edu TRAIN bin (rustbpe)
# whose FIRST 20,000,000,000 tokens are byte-identical to fineweb_rustbpe_20B.bin,
# the corpus behind the fineweb-nanochatbpe-20B registry key: it is the 20B
# pipeline with only the token target raised. README.md has the full analysis of
# why the prefix property holds; this script verifies it byte-for-byte and
# refuses to publish on any mismatch. With --upload, publishes as
# $ORG/fineweb-nanochatbpe-55B, reusing the 20B repo's val bin.
#
# Idempotent; safe to re-run after an interruption.
# Disk: ~108 GB parquet shards + 110 GB new bin + 40 GB reference 20B bin.
#
#   bash setup/fineweb/prep_train_55B.sh            # build + verify
#   bash setup/fineweb/prep_train_55B.sh --upload   # ... then publish (needs HF_TOKEN)
#   Env overrides: NANOCHAT_BASE_DIR, TOKENIZER_THREADS, DOWNLOAD_WORKERS,
#                  HF_ORG, FINEWEB_20B_REF_BIN
#
set -euo pipefail

DO_UPLOAD=false
for arg in "$@"; do
  case "$arg" in
    --upload) DO_UPLOAD=true ;;
    *) echo "unknown argument: $arg (expected --upload)" >&2; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
THREADS="${TOKENIZER_THREADS:-18}"            # speed only, never content
DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-8}"
ORG="${HF_ORG:-alexkstern}"                   # where the existing data-bin repos live

TARGET_TOKENS=55000000000
PREFIX_TOKENS=20000000000                     # the 20B bin's exact length
PREFIX_BYTES=$((PREFIX_TOKENS * 2))           # uint16 -> 40 GB

# The fineweb source reserves the LAST parquet file in the data dir, so
# NUM_SHARDS=1150 gives 1149 train shards; at the measured ~52.9M rustbpe
# tokens/shard that is ~60.8B available — deliberate margin over 55B, because
# the pretokenizer deletes the bin and fails loudly if the source runs dry. The
# 20B build used NUM_SHARDS=390 from the same dir in the same order, so the
# stream prefix is unchanged. Must stay < 1822: shard_01822.parquet is the
# designated held-out val shard.
NUM_SHARDS=1150

# Upstream corpus, pinned: the 20B build read the then-unpinned main branch,
# whose last commit is this revision, so a future re-upload cannot silently
# change the shards.
UPSTREAM_REPO="karpathy/fineweb-edu-100b-shuffle"
UPSTREAM_REVISION="4c8f30d6756da75362432a4d5569e1b229263b71"

# The published 20B repo: reference train.bin, canonical tokenizer files, and
# the reusable val bin.
REF_REPO="$ORG/fineweb-nanochatbpe-20B"

DATA_DIR="$BASE_DIR/base_data"
OUT_DIR="$BASE_DIR/bins/fineweb_rustbpe_55B"
BIN="$OUT_DIR/fineweb_rustbpe_55B.bin"
VAL_DIR="$BASE_DIR/bins/fineweb_rustbpe_val"

# Stage 1: revision-pinned download of whatever shards are missing.
echo "[prep_fineweb_rustbpe_55B] ensuring $NUM_SHARDS FineWeb-Edu shards under $DATA_DIR (revision $UPSTREAM_REVISION) ..."
mkdir -p "$DATA_DIR"
MISSING=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  f="$(printf 'shard_%05d.parquet' "$i")"
  [[ -f "$DATA_DIR/$f" ]] || MISSING+=("$f")
done
if (( ${#MISSING[@]} )); then
  echo "[prep_fineweb_rustbpe_55B] downloading ${#MISSING[@]} missing shards with $DOWNLOAD_WORKERS workers ..."
  printf '%s\n' "${MISSING[@]}" | xargs -P "$DOWNLOAD_WORKERS" -I {} bash -c '
    set -euo pipefail
    url="https://huggingface.co/datasets/'"$UPSTREAM_REPO"'/resolve/'"$UPSTREAM_REVISION"'/{}"
    out="'"$DATA_DIR"'/{}"
    curl -fsSL --retry 5 --retry-delay 5 -o "$out.tmp" "$url"
    mv "$out.tmp" "$out"
    echo "  downloaded {}"
  '
fi

# The source tokenizes EVERY parquet file in the dir in sorted order, so a stray
# or missing file would change the token stream.
SHARD_COUNT=$(find "$DATA_DIR" -maxdepth 1 -name '*.parquet' ! -name '*.tmp' | wc -l | tr -d ' ')
LAST_SHARD="$(cd "$DATA_DIR" && ls -- *.parquet | sort | tail -1)"
WANT_LAST="$(printf 'shard_%05d.parquet' $((NUM_SHARDS - 1)))"
if [[ "$SHARD_COUNT" -ne "$NUM_SHARDS" || "$LAST_SHARD" != "$WANT_LAST" ]]; then
  echo "[prep_fineweb_rustbpe_55B] FATAL: $DATA_DIR holds $SHARD_COUNT parquet files (last: $LAST_SHARD);" >&2
  echo "  expected exactly $NUM_SHARDS files ending at $WANT_LAST. Remove strays / rerun the download." >&2
  exit 1
fi

# Stage 2: tokenizer identity.
echo "[prep_fineweb_rustbpe_55B] checking the local tokenizer against $REF_REPO ..."
BASE_DIR="$BASE_DIR" REF_REPO="$REF_REPO" uv run python - <<'PY'
import hashlib, os, shutil, sys
from huggingface_hub import hf_hub_download

# The token stream is a function of (documents, tokenizer). Stage 1 pinned the
# documents; this pins the tokenizer against the published 20B dataset.
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

tok_dir = os.path.join(os.environ["BASE_DIR"], "tokenizer")
os.makedirs(tok_dir, exist_ok=True)
for fn in ("tokenizer.pkl", "token_bytes.pt"):
    ref = hf_hub_download(os.environ["REF_REPO"], f"tokenizer/{fn}", repo_type="dataset")
    local = os.path.join(tok_dir, fn)
    if not os.path.exists(local):
        shutil.copyfile(ref, local)
        print(f"  installed {fn} from {os.environ['REF_REPO']}")
    elif sha256(local) != sha256(ref):
        sys.exit(f"FATAL: {local} differs from the {fn} bundled with "
                 f"{os.environ['REF_REPO']}; tokenizing with it would break the "
                 f"20B-prefix guarantee. Reconcile the tokenizer dir first.")
    else:
        print(f"  {fn}: matches {os.environ['REF_REPO']}")
PY

# Stage 3: tokenize.
if [[ -f "$OUT_DIR/meta.json" ]]; then
  echo "[prep_fineweb_rustbpe_55B] already built: $OUT_DIR/meta.json — skipping tokenization."
else
  echo "[prep_fineweb_rustbpe_55B] tokenizing $TARGET_TOKENS tokens with rustbpe ($THREADS threads) -> $BIN"
  uv run python -m nanochat.data.pretokenize \
    --source fineweb \
    --tokenizer rustbpe \
    --data-path-local "$DATA_DIR" \
    --split train \
    --target-tokens "$TARGET_TOKENS" \
    --tokenizer-threads "$THREADS" \
    --output "$BIN"
fi

# Stage 4: prefix verification against the 20B bin.
BIN_BYTES=$(stat -c%s "$BIN" 2>/dev/null || stat -f%z "$BIN")
if [[ "$BIN_BYTES" -ne $((TARGET_TOKENS * 2)) ]]; then
  echo "[prep_fineweb_rustbpe_55B] FATAL: $BIN is $BIN_BYTES bytes, expected $((TARGET_TOKENS * 2))." >&2
  exit 1
fi

# Resolve a real 40 GB reference 20B train.bin: the FINEWEB_20B_REF_BIN
# override, then local caches, then HF with Xet disabled (Xet can return pointer
# stubs). Every candidate is size-gated, so a stub cannot reach the cmp.
ref_ok() { [[ -f "$1" ]] && [[ "$(stat -c%s "$1" 2>/dev/null || stat -f%z "$1")" -ge "$PREFIX_BYTES" ]]; }

REF_BIN="${FINEWEB_20B_REF_BIN:-$BASE_DIR/bins/fineweb_rustbpe_20B/fineweb_rustbpe_20B.bin}"
# The hub cache holds train.bin as a symlink to a hashless blob.
if ! ref_ok "$REF_BIN"; then
  for HFHOME in "${HF_HOME:-}" "$HOME/.cache/huggingface" /root/.cache/huggingface; do
    [[ -n "$HFHOME" ]] || continue
    for link in "$HFHOME"/hub/datasets--"$ORG"--fineweb-nanochatbpe-20B/snapshots/*/train.bin; do
      cand="$(readlink -f "$link" 2>/dev/null || true)"
      if ref_ok "$cand"; then REF_BIN="$cand"; break 2; fi
    done
  done
fi
if ! ref_ok "$REF_BIN"; then
  echo "[prep_fineweb_rustbpe_55B] no local 40 GB bin; downloading from $REF_REPO with Xet disabled ..."
  REF_BIN="$(HF_HUB_DISABLE_XET=1 REF_REPO="$REF_REPO" uv run python -c "
import os
from huggingface_hub import hf_hub_download
print(hf_hub_download(os.environ['REF_REPO'], 'train.bin', repo_type='dataset', force_download=True))")"
fi
if ! ref_ok "$REF_BIN"; then
  REF_BYTES=$(stat -c%s "$REF_BIN" 2>/dev/null || stat -f%z "$REF_BIN" 2>/dev/null || echo 0)
  echo "[prep_fineweb_rustbpe_55B] FATAL: no valid 40 GB reference bin (got '$REF_BIN', $REF_BYTES bytes)." >&2
  echo "  Pass one explicitly: FINEWEB_20B_REF_BIN=/path/to/train.bin bash $0 --upload" >&2
  exit 1
fi
echo "[prep_fineweb_rustbpe_55B] reference bin: $REF_BIN"

echo "[prep_fineweb_rustbpe_55B] verifying the first $PREFIX_TOKENS tokens ($PREFIX_BYTES bytes) match the 20B bin ..."
if ! cmp -n "$PREFIX_BYTES" "$BIN" "$REF_BIN"; then
  echo "[prep_fineweb_rustbpe_55B] FATAL: prefix mismatch — the new bin does NOT reproduce the 20B corpus." >&2
  echo "  Do not train on or upload $BIN. Diagnose (tokenizer? shard contents? code drift?) before retrying." >&2
  exit 1
fi
echo "[prep_fineweb_rustbpe_55B] prefix verified: first 20B tokens are byte-identical to the 20B bin."

# Stage 5: publish (opt-in).
if [[ "$DO_UPLOAD" != true ]]; then
  echo "[prep_fineweb_rustbpe_55B] done (not uploaded; rerun with --upload to publish) -> $BIN"
  exit 0
fi

# Use the canonical val bin (held-out shard_01822) from the 20B repo: extending
# train leaves the val split untouched, so the two repos' val.bin must match. A
# locally built val is byte-compared against the canonical copy, not trusted.
echo "[prep_fineweb_rustbpe_55B] fetching the canonical val bin from $REF_REPO ..."
mkdir -p "$VAL_DIR"
CANON_VAL="$(VAL_REF_ONLY=1 REF_REPO="$REF_REPO" uv run python - <<'PY'
import os
from huggingface_hub import hf_hub_download
print(hf_hub_download(os.environ["REF_REPO"], "val.bin", repo_type="dataset"))
PY
)"
if [[ -f "$VAL_DIR/fineweb_rustbpe_val.bin" ]]; then
  if ! cmp -s "$CANON_VAL" "$VAL_DIR/fineweb_rustbpe_val.bin"; then
    echo "[prep_fineweb_rustbpe_55B] FATAL: local val bin differs from the canonical 20B val.bin" >&2
    echo "  local:     $VAL_DIR/fineweb_rustbpe_val.bin" >&2
    echo "  canonical: $CANON_VAL" >&2
    exit 1
  fi
  echo "[prep_fineweb_rustbpe_55B] local val bin verified byte-identical to canonical"
else
  cp "$CANON_VAL" "$VAL_DIR/fineweb_rustbpe_val.bin"
fi
if [[ ! -f "$VAL_DIR/meta.json" ]]; then
  REF_REPO="$REF_REPO" VAL_DIR="$VAL_DIR" uv run python - <<'PY'
import os, shutil
from huggingface_hub import hf_hub_download
path = hf_hub_download(os.environ["REF_REPO"], "val.meta.json", repo_type="dataset")
shutil.copyfile(path, os.path.join(os.environ["VAL_DIR"], "meta.json"))
PY
fi

echo "[prep_fineweb_rustbpe_55B] publishing $ORG/fineweb-nanochatbpe-55B ..."
uv run python -m nanochat.data.hf_register \
  --train-bin "$BIN" \
  --val-bin   "$VAL_DIR/fineweb_rustbpe_val.bin" \
  --tokenizer-dir "$BASE_DIR/tokenizer" \
  --org "$ORG"

echo "[prep_fineweb_rustbpe_55B] done -> $BIN (published)"
