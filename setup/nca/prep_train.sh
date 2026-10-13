#!/usr/bin/env bash
#
# Generate the NCA pre-pretraining TRAIN bin at seq_len 2048 on a GPU, under the
# paper-faithful shared-rules recipe (see the nca_paper_source docstring). Two
# deviations: each trajectory rolls out 54 grids (2052 tokens, covering a 2048
# context) instead of 27, and 20 epochs instead of 10, a size-only knob that
# keeps PPT single-pass through alpha 5% at 9e18.
# Total = 20 x 16000 x 54 x 38 = 656,640,000 tokens (seed 0).
#
# Uses the environment's preinstalled CUDA jax. Do NOT `pip install jax`/`jaxlib`
# — that pulls a CPU build and clobbers the CUDA one, and generation would
# silently run on CPU.
#   bash setup/nca/prep_train.sh
#   Env overrides: NANOCHAT_BASE_DIR, NCA_NUM_RULES, NCA_NUM_SIMS,
#                  NCA_NUM_EPOCHS, XLA_FLAGS
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# The rule filter and sims vmap into grouped convs with thousands of groups, a
# shape cuDNN autotuning chokes on. The conv itself is tiny (12x12, 1x1 kernel),
# so skip the search.
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=0}"

BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
VOCAB=10002
SEQ_LEN=2048
NUM_RULES="${NCA_NUM_RULES:-16000}"
NUM_SIMS="${NCA_NUM_SIMS:-16000}"
NUM_EPOCHS="${NCA_NUM_EPOCHS:-20}"
TOKENS_PER_TRAJECTORY=2052          # trajectory_shape(2048): 54 grids x 38 tokens
# Finite source: num_epochs x num_sims trajectories, so this target drains the
# whole corpus with no truncation.
TARGET=$((NUM_EPOCHS * NUM_SIMS * TOKENS_PER_TRAJECTORY))

OUT_DIR="$BASE_DIR/bins/nca_paper_share20_seq2048_657M"
BIN="$OUT_DIR/nca_paper_share20_seq2048_657M.bin"

if [[ -f "$OUT_DIR/meta.json" ]]; then
  echo "[prep_nca_paper_2048_train] already built: $OUT_DIR/meta.json — skipping."
  exit 0
fi

# Pure-Python NCA deps only, to keep the preinstalled CUDA jax untouched.
pip install -q flax einops joblib python-dotenv

python - <<'PY'
import jax
devs = jax.devices()
print("[prep_nca_paper_2048_train] jax devices:", devs)
assert any(d.platform == "gpu" for d in devs), (
    "JAX sees no GPU. Use a GPU runtime and do NOT pip install "
    "jax/jaxlib (it replaces the CUDA build with a CPU one)."
)
PY

echo "[prep_nca_paper_2048_train] generating $TARGET NCA tokens on GPU (share20 recipe, seq_len ${SEQ_LEN}: ${NUM_EPOCHS} epochs x ${NUM_SIMS} sims x ${NUM_RULES} rules, seed 0) -> $BIN"
python -m nanochat.data.pretokenize \
  --source nca \
  --vocab-size "$VOCAB" \
  --num-rules "$NUM_RULES" \
  --num-sims "$NUM_SIMS" \
  --num-epochs "$NUM_EPOCHS" \
  --seq-length "$SEQ_LEN" \
  --seed 0 \
  --split train \
  --target-tokens "$TARGET" \
  --output "$BIN"

echo "[prep_nca_paper_2048_train] done -> $BIN"
