"""Filesystem persistence for multi-scale optimization artifacts.

Layout under ``root`` (typically ``outputs/<wandb_project>/``):

    <root>/
    ├── tuner_<scale_label>.json   # lt.Tuner.save() output, one per scale
    └── scale_results.json         # aggregated per-scale summary dicts

The per-scale tuner files are the handoff between scales: the next scale's
``ContextBuilder`` injects the prior file so the LLM sees the prior thinking,
proposals and results verbatim.
"""

import json
from pathlib import Path

import llmtuna as lt


class CheckpointStore:
    """Writes per-scale tuner JSON and the combined results file under one root."""

    def __init__(self, root: str | Path):
        """The directory is created lazily on the first write."""
        self.root = Path(root)

    def tuner_path(self, scale_label: str) -> Path:
        return self.root / f"tuner_{scale_label}.json"

    def trials_path(self, scale_label: str) -> Path:
        """Per-scale trial ledger, appended as trials complete so a crashed
        scale resumes without re-running finished work."""
        return self.root / "metrics" / f"trials_{scale_label}.jsonl"

    def append_trial(self, scale_label: str, cfg: dict, value: float) -> None:
        path = self.trials_path(scale_label)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps({"cfg": cfg, "value": value}) + "\n")

    @property
    def results_path(self) -> Path:
        return self.root / "scale_results.json"

    def save_tuner(self, tuner: lt.Tuner, scale_label: str) -> Path:
        """Overwrites any prior file at the same path (expected on resume)."""
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.tuner_path(scale_label)
        tuner.save(path)
        return path

    def save_results(self, results: list[dict]) -> Path:
        """Write ``results`` as JSON to ``scale_results.json``."""
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.results_path
        path.write_text(json.dumps(results, indent=2))
        return path
