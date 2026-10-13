"""Torchrun command assembly + trial-metrics JSON reading.

Study-specific argv is assembled in ``experiments/training/trial_cli.py`` and
passed as ``trial_argv``. The metrics file is the JSON a study module writes
at ``--metrics-output-file``.
"""

from __future__ import annotations

import json
from pathlib import Path


def torchrun_cmd(*, nproc: int, entry_module: str, trial_argv: list[str]) -> list[str]:
    """Assemble ``torchrun ... -m <entry_module> <trial_argv>`` for one trial."""
    return [
        "torchrun", "--standalone", f"--nproc_per_node={nproc}",
        "-m", entry_module,
        *trial_argv,
    ]


def read_metrics(path: str | Path) -> dict | None:
    """Read a trial's metrics JSON; None if the file doesn't exist."""
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def read_val_bpb(path: str | Path) -> float | None:
    """Return the FINAL eval objective from the metrics file: a config is
    judged by what it delivers at the full budget, not its best transient."""
    metrics = read_metrics(path)
    if metrics is None:
        return None
    for key in ("objective", "min_objective"):
        if key in metrics:
            return float(metrics[key])
    return None
