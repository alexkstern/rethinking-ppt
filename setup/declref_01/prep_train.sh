#!/usr/bin/env bash
#
# Build declref_01_seq_len_2048_2B.bin — 2B-token decl-ref-01 TRAIN bin: a
# declare/reference grammar (4 keywords + 512 name-parts + 512 value-parts,
# vocab 1028) with noisy references, part-chains, interleaved openings and topic
# shifts. Procedurally generated at seed 0; val uses a disjoint seed. Each
# document is exactly seq_length tokens and the bin holds a whole number of
# documents, so it reshapes into one document per row. Output under the base
# dir. Idempotent.
#
#   bash setup/declref_01/prep_train.sh
#   Env overrides: NANOCHAT_BASE_DIR, NUM_WORKERS
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"

SEQ_LEN=2048
VOCAB=1028                         # 4 keywords + 2 * 512 symbol parts
SHARDS=256                         # the bin's contents are fixed by (seed, SHARDS)
WORDS=976896                       # 976896 * 2048 = 2,000,683,008 tokens (~2B), 256 * 3816 whole documents
TARGET=$((WORDS * SEQ_LEN))
NUM_WORKERS="${NUM_WORKERS:-$(getconf _NPROCESSORS_ONLN)}"

OUT_DIR="$BASE_DIR/bins/declref_01_seq_len_2048_2B"
BIN="$OUT_DIR/declref_01_seq_len_2048_2B.bin"

if [[ -f "$OUT_DIR/meta.json" ]]; then
  echo "[prep_declref_01_train] already built: $OUT_DIR/meta.json — skipping."
  exit 0
fi

echo "[prep_declref_01_train] generating $TARGET decl-ref-01 tokens (seq_len=$SEQ_LEN, seed 0) -> $BIN"
uv run python -m nanochat.data.pretokenize \
  --source declref-01 \
  --vocab-size "$VOCAB" \
  --seq-length "$SEQ_LEN" \
  --seed 0 \
  --num-shards "$SHARDS" \
  --num-workers "$NUM_WORKERS" \
  --split train \
  --target-tokens "$TARGET" \
  --output "$BIN"

echo "[prep_declref_01_train] done -> $BIN"
