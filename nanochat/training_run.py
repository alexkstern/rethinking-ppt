"""Shared construction + orchestration for FLOP-budgeted training studies.

A study (an experiment file under ``experiments/``) supplies a config dataclass
and calls ``train()`` itself — once for a single-phase run, twice for a
pre-pretrain (PPT) -> pretrain (PT) run that carries weights across the
transition. Everything here is duck-typed on that config: any dataclass with
the fields these functions read works, and ``alpha_ppt`` of ``target_flops``
goes to PPT.

The flow per study:

  1. ``resolve_run_config(cfg)`` fills the per-phase model configs from each
     source's meta BEFORE W&B init, so the logged config is exactly what trains.
  2. ``wandb_group_tags(cfg)`` derives the W&B group/tags from that config.
  3. ``build_phase(...)`` constructs everything one ``train()`` call needs and
     packages it, so the study can splat ``train(**phase.train_kwargs)``.

Datasets are named registry keys, never hardcoded paths, and each source
declares its own loader and bpb normalization, so nothing here special-cases a
corpus.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, TypeVar

import torch

from nanochat.checkpoint import load_model_state, load_optim_state
from nanochat.config import LrKind, RunConfig  # noqa: F401  (LrKind re-exported)
from nanochat.data.base import DataIterator
from nanochat.data.registry import (
    batches_per_pass,
    build_loader,
    build_token_bytes,
    get_source,
    source_vocab,
    source_word_length,
)
from nanochat.data.transforms import with_token_id_masking
from nanochat.eval.base import build_eval
from nanochat.flops import (
    flops_per_token,
    iterations_for_budget,
    iterations_for_tokens,
    tokens_per_step,
)
from nanochat.lr_schedules import (
    CosineScheduleParams,
    TrapezoidScheduleParams,
    cosine_schedule,
    trapezoid_schedule,
)
from nanochat.models.base import ModelSpec
from nanochat.models.nanochat_gpt.model_utils import nanochat_dim_heads
from nanochat.optim.factory import build_optimizers
from nanochat.runtime import RuntimeCtx, print0

# Preserves each caller's concrete config type through resolve_run_config's
# dataclasses.replace: a RunConfig satisfier in, the same type out.
RunConfigT = TypeVar("RunConfigT", bound=RunConfig)


def phase_lr_schedule(cfg: RunConfig, *, n_iter: int):
    """Select the phase's LR-multiplier function from `cfg.lr_kind`. A flat
    schedule is `trapezoid` with the ratios left at their 0 defaults."""
    if cfg.lr_kind == "trapezoid":
        return trapezoid_schedule(n_iter, TrapezoidScheduleParams(
            warmup_ratio=cfg.lr_warmup_ratio, warmdown_ratio=cfg.lr_warmdown_ratio,
            final_lr_frac=cfg.lr_final_frac,
        ))
    if cfg.lr_kind == "cosine":
        return cosine_schedule(n_iter, CosineScheduleParams(
            warmup_ratio=cfg.lr_warmup_ratio, final_lr_frac=cfg.lr_final_frac,
        ))
    raise ValueError(f"unknown lr_kind {cfg.lr_kind!r}")


def spanned_lr_schedule(cfg: RunConfig, *, span_steps: int, offset_steps: int):
    """One global LR schedule shared by consecutive phases.

    Builds the schedule over ``span_steps`` (the WHOLE run) and shifts this
    phase's local step ``offset_steps`` into it, so a unified-LR two-phase run
    anneals ONCE at the end rather than warming back up at the boundary. The
    lookup clamps at span-1, because measured-FPT rounding can leave the
    realized total a step or two past the estimate and that must not wrap the
    schedule.
    """
    base = phase_lr_schedule(cfg, n_iter=span_steps)
    return lambda step: base(min(step + offset_steps, span_steps - 1))


def build_val_sets(cfg: RunConfig, ctx: RuntimeCtx, *, model_vocab: int, seq_len: int):
    """Validation sets this phase's model can score, keyed by each source's metric
    so the curves concatenate into one global series across phases. A source is
    included only when the model's embedding table fits its ids, so a
    vocab-surgery run's narrow PPT model scores only the compatible sources."""
    sets = {}
    keys = [cfg.data_pt] + ([cfg.data_ppt] if cfg.data_ppt else []) + list(cfg.extra_eval)
    for key in dict.fromkeys(keys):  # dedup, preserve order
        source = get_source(key)
        if model_vocab < source_vocab(source) + source.id_offset:
            continue
        # Val loaders use the eval batch geometry, not the tuned training batch,
        # so the scored position set never moves with the training config. The
        # primary source (data_pt, the reported metric) scores the window
        # [eval_offset_tokens, +eval_tokens); every other source scores
        # [0, aux_eval_tokens).
        primary = key == cfg.data_pt
        offset = cfg.eval.eval_offset_tokens if primary else 0
        budget = cfg.eval.eval_tokens if primary else (
            cfg.eval.aux_eval_tokens if cfg.eval.aux_eval_tokens is not None
            else cfg.eval.eval_tokens)
        loader = build_loader(
            source, "val", device_batch_size=cfg.eval.eval_device_batch_size,
            seq_len=seq_len, ctx=ctx, global_seed=cfg.seed, start_token=offset,
        )
        # token_bytes is sized to the MODEL vocab, since it indexes the model's
        # outputs: a source on a wider-padded model gets a full-width vector
        # with the pad rows zeroed.
        sets[source.metric] = (
            loader,
            build_token_bytes(source, model_vocab, ctx.device),
            budget,
            batches_per_pass(source, "val", device_batch_size=cfg.eval.eval_device_batch_size,
                             seq_len=seq_len, world_size=ctx.world_size, start_token=offset),
        )
    return sets


def resolve_run_config(cfg: RunConfigT) -> RunConfigT:
    """Fill in the per-phase model configs from the source metas and return the
    config that will actually train. Reads metas only, no model build and no GPU."""
    # A two-phase run seeds PT's body from the PPT phase's end-of-phase
    # checkpoint, so that checkpoint must be written. Checked before the
    # network meta reads below, so a misconfig fails on the cheapest signal.
    if cfg.data_ppt is not None and not cfg.train.save_at_end:
        raise ValueError(
            "a two-phase run (data_ppt set) loads the PT body from the PPT phase's "
            "end-of-phase checkpoint; --train.save-at-end is False, so it would never "
            "be written."
        )
    base_model = cfg.model_pt
    if cfg.depth is not None:
        # Depth shorthand, dispatched on the model-config class name: for
        # NanochatGPTConfig, depth also sets width via nanochat_dim_heads.
        arch = type(base_model).__name__
        if arch != "NanochatGPTConfig":
            raise ValueError(
                f"--depth shorthand supports NanochatGPTConfig, got {arch}: "
                f"set --model-pt.n-layer / .n-head / .n-embd directly."
            )
        model_dim, num_heads = nanochat_dim_heads(cfg.depth)
        base_model = dataclasses.replace(
            base_model, n_layer=cfg.depth, n_embd=model_dim, n_head=num_heads, n_kv_head=num_heads
        )
    seq_len = base_model.sequence_len
    source_pt_vocab = source_vocab(get_source(cfg.data_pt))
    # Pin the model vocab: pad up to `pad_vocab` when set, else snap to the
    # source meta. Padding rows are unused; no data id reaches them.
    if cfg.pad_vocab is not None:
        if cfg.pad_vocab < source_pt_vocab:
            raise ValueError(
                f"pad_vocab {cfg.pad_vocab} < data_pt source vocab {source_pt_vocab}; "
                f"the model must be at least as wide as the data."
            )
        model_vocab = cfg.pad_vocab
    else:
        model_vocab = source_pt_vocab

    # Fail fast, rather than silently skipping, on an extra eval source whose
    # vocab the model cannot embed.
    for key in cfg.extra_eval:
        source = get_source(key)
        # build_val_sets skips a source the model cannot embed, offset included,
        # so the guard has to apply the same offset or it lets through exactly
        # the case it exists to catch.
        needed = source_vocab(source) + source.id_offset
        if needed > model_vocab:
            raise ValueError(
                f"extra_eval source {key!r} needs vocab {needed} > model vocab "
                f"{model_vocab}; the model cannot score it."
            )

    ppt_model_cfg = None
    if cfg.data_ppt is not None:
        # The word-length guard runs for every two-phase config, not only when
        # alpha_ppt > 0, because the PPT phase trains on this source however the
        # budget is split.
        ppt_source = get_source(cfg.data_ppt)
        word_len = source_word_length(ppt_source)  # None unless word-aligned
        # With stored_word_length set, the loader truncates each row to seq_len
        # at read time, so word_len >= seq_len suffices; without it there is no
        # truncation and the two must be equal.
        if word_len is not None:
            if ppt_source.stored_word_length is None and word_len != seq_len:
                raise ValueError(
                    f"{cfg.data_ppt} word length {word_len} != model sequence_len {seq_len}; "
                    f"they must match (set --model-pt.sequence-len {word_len})."
                )
            if ppt_source.stored_word_length is not None and word_len < seq_len:
                raise ValueError(
                    f"{cfg.data_ppt} stored_word_length {word_len} < model sequence_len "
                    f"{seq_len}; the loader can't pad rows up — set --model-pt.sequence-len "
                    f"to at most {word_len}."
                )
        if cfg.ppt_same_vocab_as_pt:
            # One model serves both corpora, so its embedding table has to hold
            # this source's ids. Checked for every two-phase config, not only a
            # budget-split one, because an interleaved run consumes the corpus
            # with no phase budget to gate on.
            needed = source_vocab(ppt_source) + ppt_source.id_offset
            if needed > model_vocab:
                raise ValueError(
                    f"{cfg.data_ppt} needs ids up to {needed} (vocab + id_offset) but the "
                    f"shared model vocab is {model_vocab}; raise --pad-vocab to at least {needed}."
                )
        # Consumed only by the PPT phase, which runs when alpha_ppt splits the
        # FLOP budget or a token-denominated ppt_tokens budget is set.
        if cfg.alpha_ppt > 0.0 or getattr(cfg, "ppt_tokens", None) is not None:
            if cfg.ppt_same_vocab_as_pt:
                ppt_vocab = model_vocab
            elif cfg.ppt_vocab_size is not None:
                ppt_vocab = cfg.ppt_vocab_size
            else:
                ppt_vocab = source_vocab(ppt_source)
            ppt_model_cfg = dataclasses.replace(base_model, vocab_size=ppt_vocab)

    return dataclasses.replace(
        cfg,
        model_pt=dataclasses.replace(base_model, vocab_size=model_vocab),
        model_ppt=ppt_model_cfg,
    )


def wandb_group_tags(cfg: RunConfig) -> None:
    """Derive the W&B group/tags from the config. Tags carry the dataset keys so
    runs are filterable by corpus."""
    # Flops-driven runs label by the flop target; token-driven runs
    # (target_flops == 0) by the token total.
    pt_tokens = getattr(cfg, "pt_tokens", None)
    ppt_tokens = getattr(cfg, "ppt_tokens", None)
    if cfg.target_flops > 0:
        budget = f"{cfg.target_flops:.0e}".replace("e+0", "e").replace("e+", "e")
    elif pt_tokens:
        budget = (f"{(pt_tokens + (ppt_tokens or 0)):.0e}"
                  .replace("e+0", "e").replace("e+", "e")) + "tok"
    else:
        raise ValueError("wandb_group_tags: need target_flops > 0 or pt_tokens set")
    if cfg.alpha_ppt > 0.0 or ppt_tokens:
        frac = cfg.alpha_ppt if cfg.alpha_ppt > 0.0 else ppt_tokens / (ppt_tokens + pt_tokens)
        method = f"ppt{frac:g}"
        case = "case_b" if cfg.ppt_same_vocab_as_pt else "case_c"
        ppt_tag = f"ppt_{cfg.data_ppt}"
    else:
        method, case, ppt_tag = "vanilla", "single", "no_ppt"
    flop = budget
    cfg.wandb.group = f"{flop}_{method}"
    cfg.wandb.tags = [
        "nanochat_gpt", f"pt_{cfg.data_pt}", ppt_tag, f"seed_{cfg.seed}", case,
        f"lr_{cfg.lr_kind}", f"depth_{cfg.model_pt.n_layer}",
    ]


def _phase_train_config(train_cfg, n_iter: int):
    """Pin the phase's iteration count and resolve the FLOPs-fraction eval cadence
    to concrete steps. Every step within a phase costs the same FLOPs, so a
    fraction of steps is a fraction of the phase's budget."""
    eval_every = train_cfg.eval_every_n_steps
    if train_cfg.eval_every_flops_frac is not None:
        eval_every = max(1, round(n_iter * train_cfg.eval_every_flops_frac))
    # The frac is cleared once consumed: dataclasses.replace re-runs
    # __post_init__, which forbids both being set.
    return dataclasses.replace(
        train_cfg, num_iterations=n_iter, eval_every_n_steps=eval_every,
        eval_every_flops_frac=None,
    )


@dataclass
class Phase:
    """One ready-to-run training phase. `train_kwargs` splats straight into
    `train(**phase.train_kwargs)`; `model` / `optimizers` are exposed so the
    study can release them at a phase boundary, and `ckpt_dir` so a following
    phase can load this one's body.

    `n_iter` and `flops_per_token` are what `build_phase` derived for THIS phase
    from its budget. They, not `cfg.train.num_iterations`, are the real numbers:
    budgeted runs never read that config field."""

    model: torch.nn.Module
    optimizers: list[torch.optim.Optimizer]
    ckpt_dir: str
    train_kwargs: dict[str, Any]
    n_iter: int
    flops_per_token: float


_UNSET = object()  # build_phase() kwarg default: fall back to cfg.flops_per_token


def _phase_lr(cfg, *, n_iter, fpt, tps, lr_span_flops, lr_offset_flops,
              lr_span_tokens, lr_offset_tokens):
    """Resolve the phase's LR schedule: spanned across a unified multi-phase run
    (FLOP- or token-denominated) or phase-local. The token form must divide
    exactly; a remainder would silently shift the anneal."""
    if lr_span_flops is not None:
        return spanned_lr_schedule(
            cfg,
            span_steps=max(1, iterations_for_budget(
                flops_budget=lr_span_flops, flops_per_token=fpt, tokens_per_step=tps)),
            offset_steps=iterations_for_budget(
                flops_budget=lr_offset_flops, flops_per_token=fpt, tokens_per_step=tps)
            if lr_offset_flops else 0,
        )
    if lr_span_tokens is not None:
        if lr_span_tokens % tps or lr_offset_tokens % tps:
            raise ValueError(
                f"lr_span_tokens={lr_span_tokens} / lr_offset_tokens={lr_offset_tokens} "
                f"must be exact multiples of tokens_per_step={tps}"
            )
        return spanned_lr_schedule(
            cfg, span_steps=lr_span_tokens // tps, offset_steps=lr_offset_tokens // tps,
        )
    return phase_lr_schedule(cfg, n_iter=n_iter)


def load_prior_phase(
    model,
    *,
    model_spec: ModelSpec,
    model_cfg,
    load_from: tuple[str, int],
    device,
    reinit_embed: bool,
    moment_match_embed_reinit: bool,
) -> None:
    """Seed a freshly-initialized `model` from a prior phase's checkpoint.

    The phase-transition weight contract: the transformer body always loads
    verbatim; the embed/head tables (`model_spec.surgery_keys`) load only when
    the vocab matches and no reinit was asked, otherwise the model's own fresh
    init stands for exactly those keys. The `strict=False` path is guarded by
    exact missing/unexpected assertions, so a body tensor that fails to map
    crashes the run rather than silently keeping its fresh init.
    """
    load_dir, load_step = load_from
    sd = load_model_state(load_dir, load_step, device)
    # Read from the checkpoint's embedding rather than cfg.model_ppt, so the
    # decision holds for any caller.
    prior_vocab = sd[model_spec.embed_key].shape[0]
    if prior_vocab == model_cfg.vocab_size and not reinit_embed:
        model.load_state_dict(sd)  # keep the prior embedding
    else:
        surgery_keys = set(model_spec.surgery_keys)
        prior_surgery = {k: sd[k] for k in surgery_keys if k in sd}
        sd = {k: v for k, v in sd.items() if k not in surgery_keys}
        missing, unexpected = model.load_state_dict(sd, strict=False)  # body only
        # The model only misses the surgery keys it actually has, so intersect
        # with its own keys rather than demand the full surgery set.
        expected_missing = surgery_keys & set(model.state_dict())
        assert set(missing) == expected_missing, f"unexpected missing: {set(missing) ^ expected_missing}"
        assert not unexpected, f"unexpected keys: {unexpected}"
        if moment_match_embed_reinit:
            # Redraw each fresh surgery table as mu + sigma * eps from the prior
            # table's per-dimension moments. eps comes from a CPU generator
            # seeded by the checkpoint step, so every DDP rank builds the
            # identical table.
            eps_gen = torch.Generator().manual_seed(load_step)
            with torch.no_grad():
                for key in sorted(expected_missing & set(prior_surgery)):
                    prior = prior_surgery[key].float()
                    mu = prior.mean(dim=0)
                    sigma = prior.std(dim=0)
                    param = model.state_dict()[key]
                    eps = torch.randn(param.shape, generator=eps_gen)
                    param.copy_((mu + sigma * eps.to(mu.device)).to(param.dtype))


def build_phase(
    cfg: RunConfig,
    ctx: RuntimeCtx,
    *,
    source_key: str,
    model_cfg,
    model_spec: ModelSpec,
    metric_prefix: str | None,
    wandb_run,
    ckpt_dir: str,
    flops_budget: float | None = None,
    token_budget: int | None = None,
    # One continuous LR schedule across phases: it spans ``lr_span_flops`` of
    # compute and this phase enters it ``lr_offset_flops`` in. Converted to
    # steps here with this phase's fpt, which is valid only when the phases
    # share a model, as same-vocab unified runs do. None -> phase-local.
    lr_span_flops: float | None = None,
    lr_offset_flops: float = 0.0,
    # Token-denominated twin of the pair above, for token-native budgets.
    # Exact division by tokens_per_step required; mutually exclusive with it.
    lr_span_tokens: int | None = None,
    lr_offset_tokens: int = 0,
    # Serve training batches from this iterator instead of the source's own
    # loader; the source still supplies vocab/metric/eval wiring. Raises if the
    # source declares mask_ids, since an override must then wrap itself with
    # with_token_id_masking.
    train_loader_override: DataIterator | None = None,
    flops_per_token_override: float | None | object = _UNSET,
    reinit_embed: bool = False,
    moment_match_embed_reinit: bool = False,
    reset_optimizer: bool = True,
    load_from: tuple[str, int] | None = None,
    step_offset: int = 0,
    flops_offset: float = 0.0,
    started_time: float = 0.0,
) -> Phase:
    """Build everything one `train()` call needs for the phase on `source_key`,
    returning a `Phase` whose `train_kwargs` splats into `train()`.

    `model_spec` selects the model family. Pass exactly one iteration source:
    `flops_budget` (steps that fit the FLOP budget) or `token_budget` (steps to
    consume that many tokens; one epoch over a corpus is its token count).

    `load_from` is `(checkpoint_dir, step)` of a prior phase to seed this model
    from. The load is strict iff the prior phase's vocab, read from the loaded
    checkpoint's embedding, matches this model's AND `reinit_embed` is False;
    otherwise only the body loads and `init_weights`'s fresh embed/head stands.
    With `moment_match_embed_reinit`, that fresh embed/head is instead drawn as
    `mu + sigma * eps` per row from the prior table's per-dimension moments, so
    it matches the first two moments of the distribution the transferred body
    was trained against rather than the init distribution.

    `reset_optimizer=True` rebuilds the optimizers fresh, which is
    paper-faithful for the NCA recipe; False carries the prior phase's m/v
    across the boundary.

    `flops_per_token_override` sets the FPT used for iteration budgeting.
    Omitted, it falls back to `cfg.flops_per_token`; `None` forces a measurement
    on this phase's `model_cfg`, so a narrow-vocab PPT phase is not poisoned by
    a PT-vocab cache; a float is used as given.
    """
    if (flops_budget is None) == (token_budget is None):
        raise ValueError("pass exactly one of flops_budget / token_budget")
    if lr_span_flops is not None and lr_span_tokens is not None:
        raise ValueError("pass at most one of lr_span_flops / lr_span_tokens")
    source = get_source(source_key)
    seq_len = model_cfg.sequence_len

    model = model_spec.build_model(model_cfg)
    model.to_empty(device=ctx.device)
    model.init_weights()
    if load_from is not None:
        load_prior_phase(
            model, model_spec=model_spec, model_cfg=model_cfg, load_from=load_from,
            device=ctx.device, reinit_embed=reinit_embed,
            moment_match_embed_reinit=moment_match_embed_reinit,
        )

    # Measured on the eager model; a cached FPT skips the probe entirely, since
    # re-measuring costs GPU time across trials at the same depth.
    if flops_per_token_override is _UNSET:
        cached_fpt = cfg.flops_per_token
    else:
        cached_fpt = flops_per_token_override
    if cached_fpt is not None:
        fpt = cached_fpt
    else:
        fpt = flops_per_token(
            model, seq_len, model_cfg.vocab_size,
            use_measured=cfg.use_measured_flops, device=ctx.device,
        )
    tps = tokens_per_step(cfg.hardware.device_batch_size, seq_len, ctx.world_size, cfg.hardware.grad_accum_steps)
    if token_budget is not None:
        n_iter = max(1, iterations_for_tokens(tokens=token_budget, tokens_per_step=tps))
    else:
        n_iter = max(1, iterations_for_budget(
            flops_budget=flops_budget, flops_per_token=fpt, tokens_per_step=tps,
        ))
    tag = (metric_prefix or "pt").upper()
    print0(f"=== {tag} ({source_key}, vocab={model_cfg.vocab_size}): flops/token={fpt:.3e} -> {n_iter} iterations ===")
    if wandb_run is not None:
        # Single-phase runs have no `pt/*` metric axis, so their derived scalars
        # go in unprefixed; phased runs namespace each phase's keys.
        prefix = f"{metric_prefix}/" if metric_prefix else ""
        wandb_run.summary[f"{prefix}flops_per_token"] = fpt
        wandb_run.summary[f"{prefix}n_iter"] = n_iter
        # cfg.train.num_iterations is the plain-usage default that budgeted runs
        # never read, so replace it in the logged config with the real derived
        # step count. Two-phase runs settle on the PT phase's value; per-phase
        # counts remain in summary as {ppt,pt}/n_iter.
        train_cfg_logged = dict(wandb_run.config["train"])
        train_cfg_logged["num_iterations"] = n_iter
        wandb_run.config.update({"train": train_cfg_logged}, allow_val_change=True)

    # expected_tokens applies to the continuous bin reader; the dyck reader
    # ignores it. train_split lets canaries train on the small "val" bin.
    if train_loader_override is not None:
        if source.mask_ids:
            raise ValueError(
                f"train_loader_override with source {source_key!r}, which declares "
                f"mask_ids={source.mask_ids}: the built-loader path would mask those "
                f"ids from training targets. Wrap the override with "
                f"with_token_id_masking, or use a mask-free source."
            )
        train_loader = train_loader_override
    else:
        train_loader = build_loader(
            source, cfg.train_split, device_batch_size=cfg.hardware.device_batch_size,
            seq_len=seq_len, ctx=ctx, global_seed=cfg.seed, expected_tokens=tps * n_iter,
        )
        # Drop the source's declared mask ids from training targets. The bpb
        # eval masks the same ids via token_bytes; this is its train-side twin.
        if source.mask_ids:
            train_loader = with_token_id_masking(train_loader, ids=source.effective_mask_ids())
    eval_fns = build_eval(
        cfg.eval, ctx,
        val_sets=build_val_sets(cfg, ctx, model_vocab=model_cfg.vocab_size, seq_len=seq_len),
    )

    optimizers = build_optimizers(model_spec.build_param_groups(model, cfg.optim), ddp=ctx.world_size > 1)
    if not reset_optimizer:
        if load_from is None:
            raise ValueError("reset_optimizer=False requires load_from (no prior phase to carry state from)")
        # State is matched by the joined param-group names per optimizer, the
        # key checkpoint.save_checkpoint wrote it under; a mismatch raises.
        optim_dict = load_optim_state(load_from[0], load_from[1], ctx.rank, ctx.device)
        for opt in optimizers:
            key = ",".join(g["name"] for g in opt.param_groups)
            if key not in optim_dict:
                raise ValueError(
                    f"cannot carry optimizer state: no saved state for param groups {key!r} "
                    f"(saved: {sorted(optim_dict)}); the phases use different optimizer layouts"
                )
            # load_state_dict also overwrites every group's lr / initial_lr /
            # weight_decay / betas with the previous phase's; carry only the
            # moment state and keep this phase's hyperparameters.
            this_phase = [{k: v for k, v in g.items() if k != "params"} for g in opt.param_groups]
            opt.load_state_dict(optim_dict[key])
            for group, hparams in zip(opt.param_groups, this_phase):
                group.update(hparams)
    # Compile last: after the eager flops measurement and after building the
    # optimizers on the eager model, for clean param-group names. They hold the
    # same param tensors, so they still drive the compiled model.
    if cfg.compile_model:
        model = torch.compile(model, dynamic=False)

    train_kwargs = dict(
        # `num_iterations` is the build-time override derived from the budget;
        # everything else in cfg.train passes through unchanged.
        train_config=_phase_train_config(cfg.train, n_iter),
        model=model,
        optimizers=optimizers,
        train_loader=train_loader,
        eval_fns=eval_fns,
        objective_key=source.metric,
        lr_schedule=_phase_lr(
            cfg, n_iter=n_iter, fpt=fpt, tps=tps,
            lr_span_flops=lr_span_flops, lr_offset_flops=lr_offset_flops,
            lr_span_tokens=lr_span_tokens, lr_offset_tokens=lr_offset_tokens,
        ),
        ctx=ctx,
        flops_per_token=fpt,
        tokens_per_step=tps,
        grad_accum_steps=cfg.hardware.grad_accum_steps,
        peak_tflops=cfg.hardware.peak_tflops,
        flops_budget=flops_budget,
        wandb_run=wandb_run,
        metric_prefix=metric_prefix,
        step_offset=step_offset,
        flops_offset=flops_offset,
        starting_total_training_time=started_time,
        starting_smooth_train_loss=0.0,  # reset EMA at a phase boundary
        checkpoint_dir=ckpt_dir,
        run_config=cfg,
    )
    return Phase(model=model, optimizers=optimizers, ckpt_dir=ckpt_dir,
                 train_kwargs=train_kwargs, n_iter=n_iter, flops_per_token=fpt)
