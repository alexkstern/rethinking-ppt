#!/usr/bin/env bash
#
# Generate the held-out NCA VAL bin at seq_len 2048 on a GPU: same shared-rules
# generator as the train bin at a disjoint seed (1 vs 0) and a single smaller
# epoch, ~10M tokens in whole 2052-token trajectories.
#
# Uses the environment's preinstalled CUDA jax. Do NOT pip install jax/jaxlib.
#   bash setup/nca/prep_val.sh
#   Env overrides: NANOCHAT_BASE_DIR, NCA_NUM_RULES, NCA_NUM_SIMS,
#                  NCA_NUM_EPOCHS, XLA_FLAGS
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=0}"

BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
VOCAB=10002
SEQ_LEN=2048
# 2000 rules is enough for a held-out spot-check and much cheaper to filter
# than the train bin's 16000.
NUM_RULES="${NCA_NUM_RULES:-2000}"
NUM_SIMS="${NCA_NUM_SIMS:-4874}"      # 4874 * 2052 = 10,001,448 tokens (~10M), whole trajectories
NUM_EPOCHS="${NCA_NUM_EPOCHS:-1}"
TOKENS_PER_TRAJECTORY=2052            # trajectory_shape(2048): 54 grids x 38 tokens
TARGET=$((NUM_EPOCHS * NUM_SIMS * TOKENS_PER_TRAJECTORY))

OUT_DIR="$BASE_DIR/bins/nca_paper_share20_seq2048_657M_val"
BIN="$OUT_DIR/nca_paper_share20_seq2048_657M_val.bin"

if [[ -f "$OUT_DIR/meta.json" ]]; then
  echo "[prep_nca_paper_2048_val] already built: $OUT_DIR/meta.json — skipping."
  exit 0
fi

pip install -q flax einops joblib python-dotenv

python - <<'PY'
import jax
devs = jax.devices()
print("[prep_nca_paper_2048_val] jax devices:", devs)
assert any(d.platform == "gpu" for d in devs), (
    "JAX sees no GPU. Use a GPU runtime and do NOT pip install "
    "jax/jaxlib (it replaces the CUDA build with a CPU one)."
)
PY

echo "[prep_nca_paper_2048_val] generating $TARGET NCA tokens on GPU (share20 recipe, seq_len ${SEQ_LEN}, seed 1) -> $BIN"
python -m nanochat.data.pretokenize \
  --source nca \
  --vocab-size "$VOCAB" \
  --num-rules "$NUM_RULES" \
  --num-sims "$NUM_SIMS" \
  --num-epochs "$NUM_EPOCHS" \
  --seq-length "$SEQ_LEN" \
  --seed 1 \
  --split val \
  --target-tokens "$TARGET" \
  --output "$BIN"

echo "[prep_nca_paper_2048_val] done -> $BIN"
