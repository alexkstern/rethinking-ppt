"""Shared core for the two-phase training studies: ``ExperimentConfig``, the
phase-config builder, eval sizing, and run setup/teardown.

FLOP budgets are passed inline to ``build_phase``:

- vanilla PT: ``flops_budget=cfg.target_flops``
- PPT: ``flops_budget=cfg.alpha_ppt * cfg.target_flops``
- PT after PPT: ``flops_budget=cfg.target_flops - ppt_metrics["total_training_flops"]``

``PT_VOCAB_SIZE`` is the nanochatbpe width every run's PT phase uses, applied
via ``pad_vocab``. A synthetic PPT corpus trains a narrower embedding instead,
set per corpus via ``ppt_vocab_size``; a natural-language PPT corpus shares
the PT width.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import torch
import tyro

from nanochat.config import (
    EvalConfig,
    HardwareConfig,
    OptimConfig,
    TrainConfig,
    WandbConfig,
)
from nanochat.models.base import ModelSpec
from nanochat.models.nanochat_gpt.vanilla_3a5e0bc import (
    GPT,
    MODEL_SPEC,
    NanochatGPTConfig,
    build_param_groups,
)
from nanochat.paths import resolve_checkpoint_dir
from nanochat.runtime import RuntimeCtx, cleanup, print0, setup
from nanochat.phase import LrKind, resolve_run_config, wandb_group_tags
from nanochat.wandb_init import init_wandb_run

class RandomHeadGPT(GPT):
    def init_weights(self):
        super().init_weights()  # zeroes block c_proj AND lm_head
        self._init_weights(self.lm_head)  # un-zero: plain scaled-normal init


def build_random_head_model(cfg: NanochatGPTConfig) -> RandomHeadGPT:
    with torch.device("meta"):
        return RandomHeadGPT(cfg)


# Selected for the PT phase by ``random_pt_head``: a randomly initialized
# lm_head gives full gradient flow through a transferred body from step 0.
RANDOM_HEAD_SPEC = ModelSpec(
    build_model=build_random_head_model,
    build_param_groups=build_param_groups,
    embed_key=MODEL_SPEC.embed_key,
    surgery_keys=MODEL_SPEC.surgery_keys,
)


# The reported metric (data_pt's val bpb) scores the val-stream window
# [EVAL_OFFSET_TOKENS, EVAL_OFFSET_TOKENS + EVAL_TOKENS), disjoint from the
# [0, AUX_EVAL_TOKENS) prefix the auxiliary metrics score.
EVAL_TOKENS = 10_485_760        # 10 * 2^20
EVAL_OFFSET_TOKENS = 2_097_152  # 2^21
AUX_EVAL_TOKENS = 2_097_152

# Model embedding widths (nanochatbpe PT + per-PPT-source narrow vocabs).
PT_VOCAB_SIZE = 65536
# 10,000 grid symbols + 2 delimiters = 10,002 ids, padded to 10,004 so the
# width divides evenly by the world size. Any new value must stay divisible
# by the world sizes used.
PPT_VOCAB_NCA = 10004

# Scored on every trial on top of the run's own PT and PPT corpora. A study
# that wants more sets `extra_eval` rather than changing this.
DEFAULT_EXTRA_EVAL = [
    "c4-nanochatbpe-10B",
]


@dataclass
class ExperimentConfig:
    """The training configuration one trial or one reported cell runs under.

    ``build_phase`` always reads the unprefixed ``lr_*`` fields, so the base
    ``lr_*``/``optim``/``hardware`` fields are the PT phase's. The ``ppt_*``
    fields are the PPT overlay: ``ppt_phase_cfg`` copies them into ``lr_*`` for
    the PPT call only. A run with a unified schedule skips that copy and uses
    the base fields for both phases.
    """

    model_pt: NanochatGPTConfig = field(
        default_factory=lambda: NanochatGPTConfig(sequence_len=2048),
    )
    model_ppt: Annotated[NanochatGPTConfig | None, tyro.conf.Suppress] = None

    optim: OptimConfig = field(
        default_factory=lambda: OptimConfig(
            matrix_lr=0.032, embedding_lr=0.12, unembedding_lr=0.004, weight_decay=0.0,
        )
    )
    train: TrainConfig = field(
        default_factory=lambda: TrainConfig(
            eval_every_n_steps=None, eval_at_end=True, save_at_end=True,
            log_every_n_steps=100, grad_clip=1.4,
            ema_beta=0.0,  # raw per-step train loss; display only
        )
    )
    eval: EvalConfig = field(default_factory=lambda: EvalConfig(eval_steps=8))
    hardware: HardwareConfig = field(
        default_factory=lambda: HardwareConfig(
            device_batch_size=32, grad_accum_steps=1, peak_tflops=989.0,
        )
    )
    wandb: WandbConfig = field(default_factory=lambda: WandbConfig(enabled=False))

    data_pt: str = "fineweb-nanochatbpe-20B"
    data_ppt: str | None = None
    extra_eval: list[str] = field(default_factory=lambda: list(DEFAULT_EXTRA_EVAL))
    train_split: str = "train"
    pad_vocab: int | None = PT_VOCAB_SIZE
    ppt_vocab_size: int | None = None  # narrow PPT embed width when ppt_same_vocab_as_pt=False
    ppt_same_vocab_as_pt: bool = False
    # PPT shares the base ``lr_*`` schedule instead of the ``ppt_*`` overlay.
    # Only read when ``alpha_ppt > 0``.
    ppt_unified_lr: bool = False

    # PT phase LR shape, read directly by build_phase.
    lr_kind: LrKind = "trapezoid"
    lr_warmup_ratio: float = 0.0
    lr_warmdown_ratio: float = 0.4
    lr_final_frac: float = 0.0

    # PPT overlay.
    ppt_lr_kind: LrKind = "trapezoid"
    ppt_lr_warmup_ratio: float | None = None
    ppt_lr_warmdown_ratio: float | None = None
    ppt_lr_final_frac: float = 0.0
    ppt_lr: float | None = None
    ppt_weight_decay: float = 0.0
    # PPT micro-batch override, both set or neither. None keeps the PPT and PT
    # effective batches equal.
    ppt_device_batch_size: int | None = None
    ppt_grad_accum_steps: int | None = None


    eval_tokens: int = EVAL_TOKENS
    eval_offset_tokens: int = EVAL_OFFSET_TOKENS
    aux_eval_tokens: int = AUX_EVAL_TOKENS

    reinit_embed_at_transition: bool = True
    # Draw each new embedding row as mu + sigma * eps from the per-dimension
    # moments of the displaced PPT table, instead of a fresh random init.
    moment_match_embed_reinit: bool = False
    reset_optimizer_at_transition: bool = True
    # Random PT lm_head instead of zeros, so gradient reaches the transferred
    # body from step 0.
    random_pt_head: bool = False

    depth: int | None = None
    compile_model: bool = True

    model_project: str = "training"
    run_name: str = "run"
    target_flops: float = 0.0
    # An alternative to the FLOP split: that phase trains exactly this many
    # tokens, `target_flops` then only sets the LR span, and `alpha_ppt` no
    # longer applies.
    ppt_tokens: int | None = None
    pt_tokens: int | None = None
    # PPT-phase eval cadence; None uses train.eval_every_n_steps.
    ppt_eval_every_n_steps: int | None = None
    alpha_ppt: float = 0.0
    use_measured_flops: bool = True
    flops_per_token: float | None = None
    ppt_flops_per_token: float | None = None
    seed: int = 42

    metrics_output_file: str | None = None

    push_to_hf: bool = False
    # The HF account published checkpoints go to; read only when push_to_hf.
    hf_repo_org: str = ""
    # Delete the run's local checkpoint dir at teardown, after any HF push.
    cleanup_checkpoints: bool = True
    # Save the PPT model's initialization as model_000000.pt before any step;
    # with train.save_at_steps this captures the phase's full weight trajectory.
    save_ppt_init_checkpoint: bool = False


def ppt_phase_cfg(cfg: ExperimentConfig) -> ExperimentConfig:
    """Build a phase config for PPT: copy ``ppt_*`` overlay into active ``lr_*``."""
    missing = [n for n in ("ppt_lr", "ppt_lr_warmup_ratio", "ppt_lr_warmdown_ratio")
               if getattr(cfg, n) is None]
    if missing:
        raise ValueError(
            f"PPT phase with a separate LR schedule requires {missing} "
            f"(pass the --ppt-lr* flags or use ppt_unified_lr)"
        )
    if (cfg.ppt_device_batch_size is None) != (cfg.ppt_grad_accum_steps is None):
        raise ValueError(
            "set both ppt_device_batch_size and ppt_grad_accum_steps, or neither"
        )
    hardware = cfg.hardware if cfg.ppt_device_batch_size is None else dataclasses.replace(
        cfg.hardware,
        device_batch_size=cfg.ppt_device_batch_size,
        grad_accum_steps=cfg.ppt_grad_accum_steps,
    )
    train = cfg.train if cfg.ppt_eval_every_n_steps is None else dataclasses.replace(
        cfg.train, eval_every_n_steps=cfg.ppt_eval_every_n_steps,
    )
    return dataclasses.replace(
        cfg,
        hardware=hardware,
        train=train,
        optim=dataclasses.replace(
            cfg.optim, matrix_lr=cfg.ppt_lr, embedding_lr=cfg.ppt_lr,
            unembedding_lr=cfg.ppt_lr, weight_decay=cfg.ppt_weight_decay,
        ),
        lr_kind=cfg.ppt_lr_kind,
        lr_warmup_ratio=cfg.ppt_lr_warmup_ratio,
        lr_warmdown_ratio=cfg.ppt_lr_warmdown_ratio,
        lr_final_frac=cfg.ppt_lr_final_frac,
    )


def with_eval_steps(cfg: ExperimentConfig) -> ExperimentConfig:
    """Copy the experiment's eval window into the phase's `eval` config; batch
    counts are derived inside `build_eval` from the val-batch geometry."""
    return dataclasses.replace(
        cfg,
        eval=dataclasses.replace(
            cfg.eval, eval_tokens=cfg.eval_tokens,
            eval_offset_tokens=cfg.eval_offset_tokens,
            aux_eval_tokens=cfg.aux_eval_tokens,
        ),
    )


def setup_run(cfg: ExperimentConfig) -> tuple[ExperimentConfig, RuntimeCtx, Path, Any]:
    """DDP setup, resolve model configs, checkpoint dir, W&B init."""
    ctx = setup(seed=cfg.seed)
    cfg = resolve_run_config(cfg)
    wandb_group_tags(cfg)
    base_ckpt = resolve_checkpoint_dir(cfg, ctx)
    wandb_run = init_wandb_run(cfg, ctx, run_name=cfg.run_name)
    return cfg, ctx, base_ckpt, wandb_run


def teardown_run(
    wandb_run,
    *labeled_phases: tuple[str, dict, Any],
    cfg: ExperimentConfig | None = None,
    ctx: Any = None,
    base_ckpt: Path | None = None,
) -> None:
    """Print summaries, optional HF push, finish W&B, cleanup DDP.

    Only the final phase's checkpoint is published.
    """
    for label, metrics, _ in labeled_phases:
        print0(f"{label}: {metrics}")

    if cfg is not None and (cfg.push_to_hf or cfg.cleanup_checkpoints):
        if ctx is None or base_ckpt is None:
            raise ValueError("push_to_hf/cleanup_checkpoints require ctx + base_ckpt")

    if cfg is not None and cfg.push_to_hf:
        if ctx.rank == 0:
            wandb_run_url = wandb_run.url if wandb_run is not None else None
            for label, metrics, ckpt_dir in labeled_phases[-1:]:
                if ckpt_dir is None:
                    continue
                slug = label.lower().replace("(", "-").replace(")", "").replace(" ", "")
                if not cfg.hf_repo_org:
                    raise ValueError("push_to_hf is set but hf_repo_org is empty")
                repo_name = f"{cfg.hf_repo_org}/{base_ckpt.name}-{slug}"
                print0(f"[hf-push] {label}: pushing to {repo_name} ...")
                from nanochat.hf_upload import push_checkpoint
                url = push_checkpoint(
                    ckpt_dir=ckpt_dir,
                    step=metrics["final_step"],
                    repo_name=repo_name,
                    public=True,
                    repo_type="model",
                    wandb_run_url=wandb_run_url,
                )
                print0(f"[hf-push] {label}: {url}")

    if cfg is not None and ctx is not None:
        if cfg.cleanup_checkpoints:
            if ctx.rank == 0:
                import shutil
                # A run with no save cadence never created it.
                if base_ckpt.exists():
                    shutil.rmtree(base_ckpt)
                    print0(f"[cleanup] removed local checkpoints: {base_ckpt}")
        elif ctx.rank == 0:
            # Name the directory that holds the final weights, so a following
            # eval can be pointed straight at it.
            final = next((d for _, _, d in reversed(labeled_phases) if d), base_ckpt)
            print0(f"[checkpoint] kept: {final}")

    if wandb_run is not None:
        wandb_run.finish()
    cleanup()
