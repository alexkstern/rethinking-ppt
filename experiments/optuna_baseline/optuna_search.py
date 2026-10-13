"""Optuna (TPE) baseline HPO over a training-study TaskSpec.

For the LLMTuna-vs-classical-tuner sample-efficiency comparison: same study,
only the proposer changes. The task's ``TaskSpec``, ``HPARAM_REGISTRY`` bounds
and ``TorchrunObjective`` are all reused, so each trial runs exactly as an
LLMTuna trial does and self-logs to W&B under ``task.wandb_project``.

``warmup+warmdown <= 1`` is enforced by conditional sampling — the same
feasible region LLMTuna enforces by rejection, with no wasted trials. Learning
rates are log-sampled, since uniform over 1e-6..1 would cripple TPE.

    uv run --extra optuna python -m experiments.optuna_baseline.optuna_search \\
        --config experiments.tasks.nca_1pct_multiscale \\
        --scale 1e18 --n-trials 30 --wandb-project optuna_nca_1pct_1e18
"""

from __future__ import annotations

import dataclasses
import importlib
from pathlib import Path

import tyro

# LR-like knobs are log-sampled.
LOG_PARAMS = {"matrix_lr", "embedding_lr", "unembedding_lr", "ppt_lr"}
# Warmdown ranges depend on the matching warmup (enforces warmup+warmdown <= 1).
CONDITIONAL = {"warmdown_ratio": "warmup_ratio", "ppt_warmdown_ratio": "ppt_warmup_ratio"}


def main(
    config: str,
    scale: str = "1e18",
    n_trials: int = 30,
    wandb_project: str | None = None,
    tpe_seed: int = 0,
    out_dir: str | None = None,
) -> None:
    import optuna
    from experiments.tuner.flops import FlopsEstimator
    from experiments.tuner.objective import TorchrunObjective
    from experiments.tuner.space import HPARAM_REGISTRY

    task = importlib.import_module(config).TASK
    if wandb_project is not None:
        task = dataclasses.replace(task, wandb_project=wandb_project)
    out = Path(out_dir) if out_dir else Path(f"outputs/optuna_{task.name}_{scale}")

    estimator = FlopsEstimator(seq_len=task.max_seq_len)
    runner = TorchrunObjective(task=task, flops_estimator=estimator, out_dir=out)

    def suggest(trial, name: str, hi_override: float | None = None):
        spec = HPARAM_REGISTRY[name]
        if hasattr(spec, "options"):                       # lt.Choice
            return trial.suggest_categorical(name, list(spec.options))
        lo, hi = spec.bounds
        if hi_override is not None:
            hi = hi_override
        if type(spec).__name__ == "Int":
            return trial.suggest_int(name, int(lo), int(hi))
        return trial.suggest_float(name, float(lo), float(hi), log=(name in LOG_PARAMS))

    def objective(trial) -> float:
        cfg: dict = {}
        for name in task.hparams:                          # warmup precedes warmdown
            if name in CONDITIONAL:
                hi = max(1e-6, 1.0 - cfg[CONDITIONAL[name]])
                cfg[name] = suggest(trial, name, hi_override=hi)
            else:
                cfg[name] = suggest(trial, name)
        cfg.update(task.pinned_params)
        return runner.run(cfg, scale, trial.number)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=tpe_seed),
        study_name=f"optuna_{task.name}_{scale}",
    )
    if task.initial_params:                                # cold-start seed == LLMTuna trial 0
        study.enqueue_trial(dict(task.initial_params))

    def _log(st, tr):
        v = tr.value if tr.value is not None else float("nan")
        # `catch` records a crashed trial as FAILED and carries on, so there may
        # be no completed trial yet — and an exception here would escape
        # optimize() and end the study.
        done = [t.value for t in st.trials if t.value is not None]
        best = min(done) if done else float("nan")
        print(f"[optuna] trial {tr.number:2d}  val_bpb={v:.4f}  best={best:.4f}", flush=True)

    study.optimize(objective, n_trials=n_trials, callbacks=[_log], catch=(Exception,))

    print("\n=== Optuna done ===")
    print(f"best val_bpb = {study.best_value:.4f}")
    print(f"best params  = {study.best_params}")
    print(f"trials to best = {study.best_trial.number + 1} / {n_trials}")


if __name__ == "__main__":
    tyro.cli(main)
