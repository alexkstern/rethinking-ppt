"""``TorchrunObjective`` launches one trial of a ``TaskSpec`` study module via
torchrun and returns its val_bpb. FLOPs-per-token estimates are cached per
depth by ``FlopsEstimator``; argv assembly lives in
``experiments.tuner.trial_cli``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Protocol

from experiments.tuner.trial_cli import build_trial_argv

from .flops import FlopsEstimator
from .training import read_val_bpb, torchrun_cmd


def _default_subprocess_runner(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


class ObjectiveRunner(Protocol):
    """One-trial objective. Returns val_bpb (lower = better)."""

    def run(self, cfg: dict, scale_label: str, trial_id: int) -> float:
        ...


class TorchrunObjective:
    """Launches the task's study module via torchrun and returns val_bpb."""

    def __init__(
        self,
        *,
        task,
        flops_estimator: FlopsEstimator,
        out_dir: str | Path,
        subprocess_runner: Callable[[list[str]], None] = _default_subprocess_runner,
    ):
        self.task = task
        self.flops_estimator = flops_estimator
        self.out_dir = Path(out_dir)
        self.subprocess_runner = subprocess_runner

    def run(self, cfg: dict, scale_label: str, trial_id: int) -> float:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        task = self.task
        run_name = f"{task.name}_{scale_label}_trial{trial_id:03d}"
        metrics_path = self.out_dir / f"trial_{scale_label}_{trial_id:03d}.json"

        depth = cfg["depth"]  # cfg is complete: proposal merged with pinned_params
        # Probe at the task's model vocab, so a wider-than-tokenizer head is
        # priced into the trial's step budget.
        flops_per_token = self.flops_estimator.flops_per_token(
            depth, vocab_size=task.pad_vocab,
        )
        ppt_flops_per_token = None
        if task.data_ppt is not None:  # two-phase study
            # ppt_vocab_size=None -> PT vocab, which hits the same cache entry
            # as above.
            ppt_flops_per_token = self.flops_estimator.flops_per_token(
                depth, vocab_size=task.ppt_vocab_size or task.pad_vocab,
            )

        trial_argv = build_trial_argv(
            cfg=cfg,
            task=task,
            target_flops=scale_label,
            run_name=run_name,
            metrics_path=metrics_path,
            flops_per_token=flops_per_token,
            ppt_flops_per_token=ppt_flops_per_token,
        )
        cmd = torchrun_cmd(
            nproc=task.nproc,
            entry_module=task.entry_module,
            trial_argv=trial_argv,
        )

        self.subprocess_runner(cmd)

        val_bpb = read_val_bpb(metrics_path)
        if val_bpb is None:
            raise RuntimeError(
                f"Training finished but no objective/min_objective in metrics at {metrics_path}"
            )
        return val_bpb

