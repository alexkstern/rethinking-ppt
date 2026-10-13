"""Re-score every published checkpoint on the held-out FineWeb window.

Walks `reproduce/cells.json`, downloads each record's checkpoint per seed from
the Hugging Face Hub, and scores it with the `bpb` suite of
`reproduce.evaluate`, on the window the paper's numbers were scored at. Each
result is written to `outputs/rescore/<cell_id>__seed<n>.json`; a result that
already exists is skipped, so a relaunch resumes. At the end it prints every
rescored value.

    uv run python -m reproduce.rescore_fineweb

Environment:
    WANDB_PROJECT   also log each score to this W&B project (default: off)
    WANDB_ENTITY    the W&B entity that project belongs to (default: your own)
    SHARD           to score on several GPUs at once, start one copy of this
                    script per GPU, each with its own SHARD. With 4 GPUs:
                    `CUDA_VISIBLE_DEVICES=0 SHARD=0/4 uv run ...` on GPU 0,
                    `CUDA_VISIBLE_DEVICES=1 SHARD=1/4 uv run ...` on GPU 1, and
                    so on. The copies split the checkpoints between them.
    EVAL_ARGS       extra arguments for `reproduce.evaluate`, e.g. `--no-compile`
    RESCORE_DIR     where the results go (default: outputs/rescore), so two
                    sweeps with different EVAL_ARGS do not resume each other
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from huggingface_hub import scan_cache_dir

from reproduce.evaluate import main as evaluate

CELLS = Path(__file__).parent / "cells.json"
OUT = Path(os.environ.get("RESCORE_DIR", "outputs/rescore"))


def checkpoints() -> list[tuple[str, str, dict]]:
    cells = json.loads(CELLS.read_text())["cells"]
    return [(c["cell_id"], seed, entry)
            for c in cells for seed, entry in sorted(c.get("checkpoints", {}).items())]


def fineweb_bpb(path: Path) -> float:
    metrics = json.loads(path.read_text())["metrics"]
    return next(v for k, v in metrics.items() if k.endswith("fineweb_bpb"))


def drop_from_cache(repo_id: str) -> None:
    """Delete one model's download; the validation sets stay cached."""
    cache = scan_cache_dir()
    for repo in cache.repos:
        if repo.repo_type == "model" and repo.repo_id == repo_id:
            cache.delete_revisions(*(r.commit_hash for r in repo.revisions)).execute()


def main() -> None:
    todo = checkpoints()
    index, count = (int(x) for x in os.environ.get("SHARD", "0/1").split("/"))
    project = os.environ.get("WANDB_PROJECT")
    entity = os.environ.get("WANDB_ENTITY")
    extra = os.environ.get("EVAL_ARGS", "").split()

    for i, (cell_id, seed, entry) in enumerate(todo):
        out = OUT / f"{cell_id}__seed{seed}.json"
        if i % count != index or out.exists():
            continue
        print(f"\n=== [{i + 1}/{len(todo)}] {cell_id} seed {seed}: {entry['hf_repo']} ===")
        argv = ["--cell", cell_id, "--seed", seed, "--suite", "bpb", "--out", str(out), *extra]
        if project:
            argv += ["--wandb-project", project, "--wandb-run-name", f"{cell_id}__seed{seed}"]
            if entity:
                argv += ["--wandb-entity", entity]
        try:
            evaluate(argv)
        finally:
            drop_from_cache(entry["hf_repo"])

    print(f"\n{'cell':<44} {'seed':>4} {'fineweb':>10}")
    for cell_id, seed, _ in todo:
        out = OUT / f"{cell_id}__seed{seed}.json"
        if out.exists():
            print(f"{cell_id:<44} {seed:>4} {fineweb_bpb(out):>10.5f}")


if __name__ == "__main__":
    main()
