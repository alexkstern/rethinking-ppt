"""Shared cross-experiment config dataclasses.

Per-arch ModelConfig variants live next to each model in
`nanochat/models/<arch>.py`; per-experiment top-level configs are composed in
`experiments/<name>.py`. LR-schedule parameters live in
`nanochat/lr_schedules.py`, one dataclass per schedule shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

# Defined here so `RunConfig` can name `lr_kind`'s type; `nanochat.training_run`
# re-exports it alongside the schedule builders.
LrKind = Literal["trapezoid", "cosine"]


@dataclass
class OptimConfig:
    """Optimizer hyperparameters for the common matrix/embedding/unembedding split.

    Arches with extra parameter categories (LayerNorm weights and biases,
    learned position embeddings) define their own optim config next to the
    model. Which optimizer combination is used is determined by which factory
    the experiment script imports, not by a flag here.
    """

    matrix_lr: float = 1e-2          # body / 2D linears                       -> Muon
    embedding_lr: float = 1e-3       # token embeddings (wte)                   -> AdamW
    unembedding_lr: float = 1e-3     # lm_head (separate from wte when untied)  -> AdamW
    weight_decay: float = 0.1


@dataclass
class TrainConfig:
    """Everything `train()` reads, except the LR schedule, which is passed in as
    a callable."""

    # FLOP/token-budgeted runs override this per phase, so introspect
    # Phase.n_iter rather than this field.
    num_iterations: int = 1000

    # Cadences. None means never.
    eval_every_n_steps: int | None = None
    # Fraction of the phase's step budget (0.1 = 10 evals per phase), resolved
    # to eval_every_n_steps by build_phase once the iteration count is known.
    eval_every_flops_frac: float | None = None
    # Exact step numbers, in addition to any periodic cadence.
    eval_at_steps: tuple[int, ...] | None = None
    save_every_n_steps: int | None = None
    save_at_steps: tuple[int, ...] | None = None
    log_every_n_steps: int = 100

    eval_at_end: bool = False
    save_at_end: bool = False

    grad_clip: float = 1.0
    # Smoothing of the LOGGED train loss only, not the optimization: 0.0 = raw
    # per-step value, 0.9 = debiased EMA.
    ema_beta: float = 0.9

    def __post_init__(self) -> None:
        if self.eval_every_flops_frac is not None:
            if self.eval_every_n_steps is not None:
                raise ValueError(
                    "set eval_every_n_steps or eval_every_flops_frac, not both"
                )
            if not 0.0 < self.eval_every_flops_frac <= 1.0:
                raise ValueError(
                    f"eval_every_flops_frac must be in (0, 1], got "
                    f"{self.eval_every_flops_frac}"
                )


@dataclass
class DataConfig:
    """Where to read training tokens from.

    `path` is a single-source dict — exactly one of `{"local": "/data/foo.bin"}`
    or `{"hf": {"repo_id": ..., "filename": ...}}`, whose kwargs flow into
    `huggingface_hub.hf_hub_download` (`repo_type="dataset"` is injected by
    `nanochat.data.base.resolve_data_path`).
    """

    path: dict[str, Any] = field(default_factory=dict)
    vocab_size: int = 50304


@dataclass
class HardwareConfig:
    """Per-machine knobs; affects throughput and utilization reporting, not the
    training math.

    `peak_tflops` is the MFU denominator: the accelerator's peak BF16 matmul
    throughput in TFLOP/s per device (scaled by world size at use).
    """

    device_batch_size: int = 32
    grad_accum_steps: int = 1
    peak_tflops: float = 989.0


@dataclass
class WandbConfig:
    """W&B run metadata. `group` / `tags` are computed by each experiment script
    and passed through; the project and run labels live on `RunConfig`."""

    enabled: bool = True
    notes: str = ""
    group: str | None = None
    tags: list[str] = field(default_factory=list)
    entity: str | None = None


@dataclass
class EvalConfig:
    """Per-eval-call cost and geometry. Cadence lives in `TrainConfig`."""

    eval_steps: int = 100  # batches per eval call (only used when eval_tokens is None)
    eval_device_batch_size: int = 32  # per-rank VAL batch, decoupled from the training batch
    # If set, score exactly the val-stream window
    # [eval_offset_tokens, eval_offset_tokens + eval_tokens) — identical for
    # every (world_size, device_batch_size). None = score every position in the
    # first eval_steps batches, a window that then varies with batch geometry.
    eval_tokens: int | None = None
    eval_offset_tokens: int = 0
    # Budget for the non-primary val sources (data_ppt / extra_eval), always at
    # offset 0. None = same budget as eval_tokens.
    aux_eval_tokens: int | None = None


@runtime_checkable
class ModelConfigProtocol(Protocol):
    """Structural placeholder for per-arch ModelConfig; each arch defines its own
    concrete fields."""

    pass


@runtime_checkable
class RunConfig(Protocol):
    """Structural contract for an experiment script's top-level config.

    Concrete satisfiers live in `experiments/<name>.py` as `@dataclass`es. This
    is the exact set of attributes `nanochat.training_run` reads off `cfg`
    (`build_phase`, `phase_lr_schedule`, `build_val_sets`, `resolve_run_config`,
    `wandb_group_tags`) plus `paths.resolve_checkpoint_dir` / `wandb_init`; keep
    it in sync with those consumers.

    `@runtime_checkable` allows `isinstance(cfg, RunConfig)` as a smoke check,
    which verifies attribute PRESENCE only, not types.

    Conditionally required, and so deliberately not declared here: a two-phase
    run (`data_ppt` set) additionally needs `ppt_same_vocab_as_pt: bool`, and
    `ppt_vocab_size: int | None` once a phase budget splits the run. Both are
    read by `resolve_run_config` inside its `data_ppt is not None` branch.
    """

    optim: OptimConfig
    train: TrainConfig
    eval: EvalConfig
    hardware: HardwareConfig
    wandb: WandbConfig

    # Templates; `resolve_run_config` fills their vocab/dims from `depth` and
    # the data source metas before training.
    model_pt: ModelConfigProtocol
    model_ppt: ModelConfigProtocol | None

    # Data-registry KEYS (see `nanochat/data/registry.py`), not `DataConfig`
    # objects. `data_ppt=None` => single-phase run.
    data_pt: str
    data_ppt: str | None
    extra_eval: list[str]        # extra eval-only registry keys
    train_split: str             # bin split to train on ("train"; canaries use "val")

    depth: int | None            # nanochat_gpt depth shorthand (None => use model_pt dims)
    pad_vocab: int | None        # pin model vocab to this width (None => snap to source)

    # PT-phase LR shape, read directly by `phase_lr_schedule`.
    lr_kind: LrKind
    lr_warmup_ratio: float
    lr_warmdown_ratio: float
    lr_final_frac: float

    compile_model: bool

    model_project: str
    run_name: str
    target_flops: float
    alpha_ppt: float
    use_measured_flops: bool
    # When set, ``build_phase`` uses this value and skips the FlopCounter probe.
    flops_per_token: float | None
    seed: int
