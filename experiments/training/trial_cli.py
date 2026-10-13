"""Hyperparameter-search trial argv for the two-phase training entry.

Maps llmtuna cfg dicts to tyro CLI flags on ``ExperimentConfig`` and assembles
the full per-trial argv. Knobs the ``TaskSpec`` doesn't carry fall through to
the ``ExperimentConfig`` defaults in ``common.py``.
"""

from __future__ import annotations

from pathlib import Path

from experiments.training.task import TaskSpec
from experiments.tuner.batch_plan import plan_batch

# llmtuna cfg key -> tyro CLI flag on ``ExperimentConfig`` (or a sub-config).
PT_HPARAM_TO_FLAG: dict[str, str] = {
    "depth": "depth",
    "model_dim": "model-pt.n-embd",
    "matrix_lr": "optim.matrix-lr",        # Muon body LR
    "embedding_lr": "optim.embedding-lr",  # AdamW wte LR
    "grad_clip": "train.grad-clip",
    "warmup_ratio": "lr-warmup-ratio",     # PT phase
    "warmdown_ratio": "lr-warmdown-ratio",
}

# PPT overlay, tuned only by studies with a separate PPT schedule. A tuned
# alpha_ppt replaces the TaskSpec's fixed alpha.
PPT_HPARAM_TO_FLAG: dict[str, str] = {
    "alpha_ppt": "alpha-ppt",
    "ppt_lr": "ppt-lr",
    "ppt_warmup_ratio": "ppt-lr-warmup-ratio",
    "ppt_warmdown_ratio": "ppt-lr-warmdown-ratio",
}


HPARAM_TO_FLAG: dict[str, str] = {**PT_HPARAM_TO_FLAG, **PPT_HPARAM_TO_FLAG}


def hparam_cli_args(cfg: dict) -> list[str]:
    """Tyro flags for every cfg entry (tuned proposals + pinned params)."""
    unknown = sorted(set(cfg) - set(HPARAM_TO_FLAG))
    if unknown:
        raise ValueError(f"cfg has no CLI flag mapping for {unknown}")
    return [f"--{HPARAM_TO_FLAG[name]}={cfg[name]}" for name in cfg]


def _bool_flag(name: str, value: bool) -> str:
    return f"--{name}" if value else f"--no-{name}"


def build_trial_argv(
    *,
    cfg: dict,
    task: TaskSpec,
    target_flops: str | float,
    run_name: str,
    metrics_path: str | Path,
    flops_per_token: float | None = None,
    ppt_flops_per_token: float | None = None,
) -> list[str]:
    """Full tyro argv for one llmtuna trial on ``run.py``.

    ``cfg`` is the complete trial config (proposal merged with the task's
    ``pinned_params`` by ScaleRunner), so ``cfg["depth"]`` always exists.
    """
    # Not ExperimentConfig flags: each selects a per-phase effective batch as a
    # multiple of total_batch_tokens, consumed here rather than mapped.
    cfg = dict(cfg)
    ppt_batch_mult = float(cfg.pop("ppt_batch_mult", 1))
    pt_batch_mult = float(cfg.pop("pt_batch_mult", 1))

    argv = hparam_cli_args(cfg)
    # Depth-aware micro-batch split of the PT effective batch (pt_batch_mult x
    # total_batch_tokens), holding tokens-per-optimizer-step fixed.
    device_batch_size, grad_accum_steps = plan_batch(
        int(task.total_batch_tokens * pt_batch_mult),
        ngpus=task.nproc,
        seq_len=task.max_seq_len,
        depth=int(cfg["depth"]),
    )
    if "alpha_ppt" not in cfg:  # tuned alpha already emitted by hparam_cli_args
        argv.append(f"--alpha-ppt={task.alpha_ppt}")
    # FLOP mode (default): the scale label is the target-FLOP budget. Token
    # mode: the PT phase trains exactly task.pt_tokens tokens and the scale
    # label carries the PPT-token dose, emitted in the PPT block below.
    token_mode = task.pt_tokens is not None
    if token_mode and task.ppt_unified_lr and task.data_ppt is not None:
        raise ValueError(
            "token mode (pt_tokens set) does not support ppt_unified_lr=True: the "
            "unified LR span needs target_flops, which token mode does not emit."
        )
    if token_mode:
        argv.append(f"--pt-tokens={task.pt_tokens}")
    else:
        argv.append(f"--target-flops={target_flops}")
    argv.extend([
        f"--data-pt={task.data_pt}",
        f"--hardware.device-batch-size={device_batch_size}",
        f"--hardware.grad-accum-steps={grad_accum_steps}",
        f"--hardware.peak-tflops={task.peak_tflops}",
        f"--model-pt.sequence-len={task.max_seq_len}",
        f"--run-name={run_name}",
        f"--metrics-output-file={metrics_path}",
        f"--model-project={task.wandb_project}",
        f"--wandb.group={task.wandb_group}",
        "--wandb.enabled",
    ])
    # Only emitted when the study pins them; otherwise the run-level defaults apply.
    if task.pad_vocab is not None:
        argv.append(f"--pad-vocab={task.pad_vocab}")
    if task.extra_eval is not None:
        argv += ["--extra-eval", *task.extra_eval]
    if task.eval_tokens is not None:
        argv.append(f"--eval-tokens={task.eval_tokens}")
    if task.eval_offset_tokens is not None:
        argv.append(f"--eval-offset-tokens={task.eval_offset_tokens}")
    if task.aux_eval_tokens is not None:
        argv.append(f"--aux-eval-tokens={task.aux_eval_tokens}")
    if task.data_ppt is not None:
        # ppt_vocab_size=None means the PPT phase trains the PT vocab
        # directly; a value means a narrow PPT embedding.
        if token_mode:  # the scale label carries the PPT-token dose
            argv.append(f"--ppt-tokens={int(float(target_flops))}")
        argv.extend([
            f"--data-ppt={task.data_ppt}",
            _bool_flag("ppt-same-vocab-as-pt", task.ppt_vocab_size is None),
            _bool_flag("ppt-unified-lr", task.ppt_unified_lr),
            _bool_flag("reinit-embed-at-transition", task.reinit_embed_at_transition),
            _bool_flag("reset-optimizer-at-transition", task.reset_optimizer_at_transition),
            _bool_flag("random-pt-head", task.random_pt_head),
        ])
        if task.ppt_vocab_size is not None:
            argv.append(f"--ppt-vocab-size={task.ppt_vocab_size}")
        if ppt_batch_mult != pt_batch_mult:
            # The PPT phase inherits the PT split unless overridden here, so
            # emit its own whenever the two effective batches differ.
            ppt_dbs, ppt_accum = plan_batch(
                int(task.total_batch_tokens * ppt_batch_mult),
                ngpus=task.nproc,
                seq_len=task.max_seq_len,
                depth=int(cfg["depth"]),
            )
            argv += [
                f"--ppt-device-batch-size={ppt_dbs}",
                f"--ppt-grad-accum-steps={ppt_accum}",
            ]
    if flops_per_token is not None:
        argv.append(f"--flops-per-token={flops_per_token}")
        argv.append("--no-use-measured-flops")
    if ppt_flops_per_token is not None:
        argv.append(f"--ppt-flops-per-token={ppt_flops_per_token}")
    return argv
