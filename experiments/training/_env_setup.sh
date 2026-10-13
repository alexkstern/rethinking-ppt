# Sourced by the launchers in this repository, from REPO_ROOT. Bootstraps uv +
# project .venv + deps; idempotent. Corpora and rustbpe token_bytes pull
# from public HF dataset repos on demand.

export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if ! command -v uv &> /dev/null; then
    sudo chown -R "$USER:$USER" "$HOME/.config" "$HOME/.cache" "$HOME/.local" 2>/dev/null || true
    mkdir -p "$HOME/.config" "$HOME/.cache" "$HOME/.local/bin" 2>/dev/null || true
    curl -LsSf https://astral.sh/uv/install.sh | sh || true
    export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null || { echo "ERROR: uv installation failed"; exit 1; }
unset UV_SYSTEM_PYTHON
[ -d ".venv" ] || uv venv
source .venv/bin/activate
uv sync --extra optimize
.venv/bin/python -c "import torch" || { echo "ERROR: torch is not importable in .venv after install"; exit 1; }

if [ -f .env ]; then set -a; source .env; set +a; fi
