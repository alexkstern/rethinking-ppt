"""One fully-specified hyperparameter-search run.

A ``TaskSpec`` is the entire definition of one shell run. Each
``experiments/tasks/<name>.py`` builds one and binds it to a module-level
``TASK``; the matching ``.sh`` bootstraps env and points the driver at it via
``--config``. Knobs not listed here come from the ``ExperimentConfig``
defaults in ``common.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TaskSpec:
    """Everything needed to launch one llmtuna study run. One per ``.sh``."""

    name: str                    # short label -> run names + wandb summary run name
    wandb_project: str           # W&B project + default out_dir + HF repo name
    alpha_ppt: float             # PPT fraction of target_flops (0.0 = vanilla)
    ppt_vocab_size: int | None   # narrow PPT embed width (None = same vocab as PT)
    description: str

    # Names resolved against space.HPARAM_REGISTRY.
    hparams: list[str]
    # {name: lt param}, consulted before the registry; scoped to this task.
    hparam_overrides: dict = field(default_factory=dict)

    entry_module: str = "experiments.training.run"
    data_pt: str = "fineweb-nanochatbpe-20B"  # data registry key
    data_ppt: str | None = None    # data registry key (alpha_ppt > 0 only)
    ppt_unified_lr: bool = False   # PPT shares base lr_* (no ppt_* overlay)
    reinit_embed_at_transition: bool = True
    reset_optimizer_at_transition: bool = True
    # PT lm_head randomly initialized instead of zeroed, so gradient flows into
    # the transferred body from step 0.
    random_pt_head: bool = False

    # In FLOP mode (default) each entry is a target-FLOP budget ("1e18"). In
    # token mode (pt_tokens set) each entry is instead a PPT-token dose parsed
    # as --ppt-tokens (e.g. "5e6").
    scales: list[str] = field(default_factory=lambda: ["1e18", "3e18", "9e18"])
    max_iters_per_scale: int = 30
    patience: int = 5
    tol: float = 0.001
    # When set, every trial trains the PT phase for exactly this many tokens
    # and the FLOP budget is ignored.
    pt_tokens: int | None = None

    nproc: int = 2
    # Effective batch per optimizer step, in tokens, identical for PPT and PT.
    # plan_batch splits it into a micro-batch and an accumulation count that fit
    # the depth, holding the tokens per optimizer step fixed.
    total_batch_tokens: int = 262_144
    peak_tflops: float = 2250.0  # MFU denominator only; no training effect
    max_seq_len: int = 2048
    # Extra bpb eval sources, on top of data_pt and data_ppt.
    extra_eval: list[str] | None = None
    # Model vocab width, for a corpus whose ids sit above the text tokenizer's.
    # Also sizes the FLOPs-per-token probe.
    pad_vocab: int | None = None

    # Setting these selects trials on a small prefix window, leaving the
    # run-level held-out window unseen by the search.
    eval_tokens: int | None = None
    eval_offset_tokens: int | None = None
    # Window for every non-primary source, scored from the start of its own
    # val stream — no search selects on these.
    aux_eval_tokens: int | None = None

    # Merged into every trial alongside the proposal; may not overlap `hparams`.
    # `depth` must appear in one or the other, since the batch plan needs it.
    pinned_params: dict[str, float | int] = field(default_factory=dict)

    llm_model: str = "anthropic/claude-sonnet-4-6"
    llm_thinking_budget: int = 300
    # None builds the prompt from the knobs this study tunes.
    system_prompt: str | None = None
    # Appended to the effective system prompt rather than replacing it.
    system_prompt_extra: str | None = None

    # Label shown to the LLM. Descriptive only: the optimizer is fixed by the
    # model's param-group spec, not by any flag.
    optimizer: str = "Muon (transformer body) + AdamW (embeddings / unembedding)"

    # Warm-start baseline for the first scale. Must cover every name in
    # `hparams`; the driver rejects a partial one.
    initial_params: dict[str, float | int] = field(default_factory=dict)

    wandb_group: str = "default"
    out_dir: str | None = None         # None -> outputs/<wandb_project>

    summarize_architecture: bool = True
    sync_to_hf: bool = True
    wandb_summary: bool = True
    keep_thinking: bool = False
    # Feed the LLM only prior trials, dropping its own past prose commentary.
    context_trials_only: bool = True
