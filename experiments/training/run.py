"""The two-phase training entry point, shared by every study.

One FLOP-budgeted run: an optional pre-pretraining (PPT) phase at
``alpha_ppt * target_flops``, then FineWeb pretraining (PT) for the remaining
budget. Every study is a point in this config space:

- **vanilla**: ``alpha_ppt=0`` — single-phase PT, nothing else applies.
- **C4**: ``data_ppt`` set, ``ppt_same_vocab_as_pt=True``, ``ppt_unified_lr=True``,
  no embed reinit / optimizer reset at the transition.
- **NCA / K-Dyck**: ``data_ppt`` set with a narrow ``ppt_vocab_size``
  (``resolve_run_config`` builds ``cfg.model_ppt``), separate PPT LR schedule
  (``ppt_phase_cfg``), embed reinit + optimizer reset at the transition.

Two selectors are not config fields, because they choose the models the phases
build rather than values a run is resolved from; ``resolve_model_specs`` is the
only place the mapping is written down:

``--init-variant spectral | n01_embed | n002_all``
    The weight initialization both phases build under.
``--ppt-opt muon | adamw``
    Whether the PPT phase optimizes every parameter group with stock AdamW
    instead of the pretraining recipe. Absent for a run with no PPT phase.

Manual canary:
    torchrun --standalone --nproc_per_node=1 -m experiments.training.run \\
        --target-flops=1e16 --depth=4 \\
        --alpha-ppt=0.01 --data-ppt=dyck-k128-seq_len_2048-1B --ppt-vocab-size=256 \\
        --ppt-lr=0.02 --ppt-lr-warmup-ratio=0.1 --ppt-lr-warmdown-ratio=0.4 \\
        --init-variant=spectral --ppt-opt=muon
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import tyro

from nanochat.config import OptimConfig
from nanochat.models.base import ModelSpec
from nanochat.models.nanochat_gpt.vanilla_3a5e0bc import (
    ADAMW_KWARGS,
    GPT,
    MODEL_SPEC,
    build_param_groups,
)
from nanochat.optim.base import ParamGroupSpec
from nanochat.runtime import release_phase
from nanochat.train import train
from nanochat.training_run import build_phase

from experiments.training.common import (
    RANDOM_HEAD_SPEC,
    ExperimentConfig,
    ppt_phase_cfg,
    setup_run,
    teardown_run,
    with_eval_steps,
)

# Weight initializations the phases can build under. MODEL_SPEC (the
# ``spectral`` variant) is the pretraining recipe's own: fan-in-scaled Linears
# with the residual projections and the lm_head zeroed, and an N(0, 1) token
# embedding. The two variants below weaken it, and apply to BOTH phases.


class N002AllGPT(GPT):
    """nanochat GPT with every parameter at flat N(0, 0.02), nothing zeroed:
    the init-dilution control, stripped of the spectral muP-style init.

    ``embed_std`` is the token-embedding std, overridden by subclasses while
    the flat-0.02 body is shared.
    """

    embed_std: float = 0.02

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.embed_std)

    def init_weights(self):
        # GPT.init_weights minus the zeroing of lm_head and the blocks'
        # c_proj — that zeroing is the spectral-init dividend under test.
        self.apply(self._init_weights)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)


class N01EmbedGPT(N002AllGPT):
    """Flat-0.02 body (nothing zeroed) with the token embedding kept at N(0,1):
    the weak-body / large-embedding mismatch under which PPT helps in the
    prior NCA work.
    """

    embed_std = 1.0


def build_n002_all_model(cfg):
    with torch.device("meta"):
        return N002AllGPT(cfg)


def build_n01_embed_model(cfg):
    with torch.device("meta"):
        return N01EmbedGPT(cfg)


N002_ALL_SPEC = ModelSpec(
    build_model=build_n002_all_model,
    build_param_groups=build_param_groups,
    embed_key=MODEL_SPEC.embed_key,
    surgery_keys=MODEL_SPEC.surgery_keys,
)

N01_EMBED_SPEC = ModelSpec(
    build_model=build_n01_embed_model,
    build_param_groups=build_param_groups,
    embed_key=MODEL_SPEC.embed_key,
    surgery_keys=MODEL_SPEC.surgery_keys,
)


# The PPT phase's optimizer. PT always runs the standard recipe (Muon body +
# AdamW embeddings); an all-AdamW PPT phase swaps in the builder below for that
# phase alone. The transition resets the optimizer, so only weights cross into
# PT either way.


def build_param_groups_adamw(model: GPT, cfg: OptimConfig) -> list[ParamGroupSpec]:
    """nanochat groups with every group on stock AdamW (PPT-phase use).

    Same three groups and LR fields as the standard builder, so the ppt_lr
    overlay lands identically. The matrix group is the experimental variable;
    the embedding/unembedding groups keep the Muon recipe's AdamW kwargs, so
    the comparison isolates one change: which algorithm trains the body.
    """
    stock_kwargs = {"betas": (0.9, 0.999), "eps": 1e-8}
    nanochat_kwargs = dict(ADAMW_KWARGS)
    lr_scale = (model.config.n_embd / 768) ** -0.5
    return [
        ParamGroupSpec("matrix", list(model.transformer.h.parameters()), "adamw",
                       cfg.matrix_lr, cfg.weight_decay, stock_kwargs),
        ParamGroupSpec("embedding", list(model.transformer.wte.parameters()), "adamw",
                       cfg.embedding_lr * lr_scale, cfg.weight_decay, nanochat_kwargs),
        ParamGroupSpec("unembedding", list(model.lm_head.parameters()), "adamw",
                       cfg.unembedding_lr * lr_scale, cfg.weight_decay, nanochat_kwargs),
    ]


ADAMW_SPEC = dataclasses.replace(MODEL_SPEC, build_param_groups=build_param_groups_adamw)
N002_ALL_ADAMW_PPT_SPEC = dataclasses.replace(
    N002_ALL_SPEC, build_param_groups=build_param_groups_adamw,
)
N01_EMBED_ADAMW_PPT_SPEC = dataclasses.replace(
    N01_EMBED_SPEC, build_param_groups=build_param_groups_adamw,
)

# init_variant -> (the spec both phases build under, its all-AdamW PPT twin).
INIT_VARIANTS: dict[str, tuple[ModelSpec, ModelSpec]] = {
    "spectral": (MODEL_SPEC, ADAMW_SPEC),
    "n01_embed": (N01_EMBED_SPEC, N01_EMBED_ADAMW_PPT_SPEC),
    "n002_all": (N002_ALL_SPEC, N002_ALL_ADAMW_PPT_SPEC),
}

# ppt_opt -> whether the PPT phase swaps in the all-AdamW twin of the spec.
PPT_OPTIMIZERS = {"muon": False, "adamw": True}


def resolve_model_specs(init_variant: str,
                        ppt_opt: str | None) -> tuple[ModelSpec, ModelSpec | None]:
    """The (pretraining, pre-pretraining) specs these two selectors name.

    ``ppt_opt`` is None for a run with no PPT phase. The returned PPT spec is
    None whenever that phase builds its parameter groups the way PT does,
    which ``main`` reads as "use the PT spec for both".
    """
    if init_variant not in INIT_VARIANTS:
        raise ValueError(
            f"unknown init_variant {init_variant!r} "
            f"(expected one of {sorted(INIT_VARIANTS)})"
        )
    if ppt_opt is not None and ppt_opt not in PPT_OPTIMIZERS:
        raise ValueError(
            f"unknown ppt_opt {ppt_opt!r} "
            f"(expected null or one of {sorted(PPT_OPTIMIZERS)})"
        )
    model_spec, adamw_ppt_spec = INIT_VARIANTS[init_variant]
    swap_in_adamw = ppt_opt is not None and PPT_OPTIMIZERS[ppt_opt]
    return model_spec, (adamw_ppt_spec if swap_in_adamw else None)


def run_ppt(cfg: ExperimentConfig, ctx, base_ckpt: Path, wandb_run, model_spec=MODEL_SPEC) -> dict:
    """Train the PPT phase and return its metrics."""
    ppt_model_cfg = cfg.model_pt if cfg.ppt_same_vocab_as_pt else cfg.model_ppt
    assert ppt_model_cfg is not None, "alpha_ppt > 0 requires a PPT model config"
    if cfg.ppt_unified_lr and cfg.ppt_device_batch_size is not None:
        raise ValueError(
            "ppt_unified_lr trains the PPT phase with the PT batch; "
            "ppt_device_batch_size / ppt_grad_accum_steps would be ignored"
        )
    ppt_cfg = cfg if cfg.ppt_unified_lr else ppt_phase_cfg(cfg)
    ppt_cfg = with_eval_steps(ppt_cfg)
    ppt = build_phase(
        ppt_cfg, ctx,
        source_key=cfg.data_ppt,
        model_cfg=ppt_model_cfg,
        model_spec=model_spec,
        flops_budget=None if cfg.ppt_tokens is not None else cfg.alpha_ppt * cfg.target_flops,
        token_budget=cfg.ppt_tokens,
        # Unified LR = one schedule over the whole target; this phase is its
        # first segment (anneal happens only at the end of PT).
        lr_span_flops=cfg.target_flops if cfg.ppt_unified_lr else None,
        flops_per_token_override=cfg.ppt_flops_per_token,
        metric_prefix="ppt",
        wandb_run=wandb_run,
        ckpt_dir=str(base_ckpt / "ppt"),
    )
    if cfg.save_ppt_init_checkpoint and ctx.rank == 0:
        import torch
        from nanochat.checkpoint import _uncompiled_model
        ppt_dir = Path(ppt.ckpt_dir)
        ppt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(_uncompiled_model(ppt.model).state_dict(),
                   ppt_dir / "model_000000.pt")
        print(f"saved PPT init checkpoint: {ppt_dir / 'model_000000.pt'}")
    ppt_metrics = train(**ppt.train_kwargs)
    ppt_metrics["ckpt_dir"] = ppt.ckpt_dir
    release_phase(ppt.model, ppt.optimizers)
    return ppt_metrics


def main(cfg: ExperimentConfig, model_spec=MODEL_SPEC, ppt_model_spec=None) -> None:
    cfg, ctx, base_ckpt, wandb_run = setup_run(cfg)
    if wandb_run is not None:
        # Record which ModelSpec (init variant) ran, so downstream tooling
        # reads it from the config instead of inferring it from project names.
        wandb_run.config.update(
            {"model_spec_module": model_spec.build_model.__module__,
             # PPT-phase spec when it differs: the PT spec module alone
             # cannot distinguish an AdamW-PPT twin.
             "ppt_spec_groups": (ppt_model_spec or model_spec).build_param_groups.__name__},
            allow_val_change=True,
        )

    ppt_metrics = None
    if cfg.alpha_ppt > 0 or cfg.ppt_tokens is not None:
        assert cfg.data_ppt is not None, "a PPT budget (alpha_ppt / ppt_tokens) requires data_ppt"
        assert cfg.train.save_at_end, "two-phase run needs save_at_end (PT loads the PPT checkpoint)"
        ppt_metrics = run_ppt(cfg, ctx, base_ckpt, wandb_run,
                              model_spec=ppt_model_spec or model_spec)

    pt_model_spec = model_spec
    if cfg.random_pt_head:
        assert model_spec is MODEL_SPEC, "random_pt_head requires the nanochat_gpt MODEL_SPEC"
        assert ppt_metrics is not None, "random_pt_head only applies at a PPT->PT transition"
        pt_model_spec = RANDOM_HEAD_SPEC

    pt_cfg = with_eval_steps(cfg)
    transition_kwargs = {} if ppt_metrics is None else dict(
        reinit_embed=cfg.reinit_embed_at_transition,
        moment_match_embed_reinit=cfg.moment_match_embed_reinit,
        reset_optimizer=cfg.reset_optimizer_at_transition,
        load_from=(ppt_metrics["ckpt_dir"], ppt_metrics["final_step"]),
        step_offset=ppt_metrics["final_step"] + 1,
        flops_offset=ppt_metrics["total_training_flops"],
        started_time=ppt_metrics["total_training_time"],
    )
    unified_two_phase = ppt_metrics is not None and cfg.ppt_unified_lr
    pt = build_phase(
        pt_cfg, ctx,
        source_key=cfg.data_pt,
        model_cfg=cfg.model_pt,
        model_spec=pt_model_spec,
        flops_budget=None if cfg.pt_tokens is not None else cfg.target_flops - (
            ppt_metrics["total_training_flops"] if ppt_metrics else 0.0
        ),
        token_budget=cfg.pt_tokens,
        # Unified LR: PT continues the SAME global schedule where PPT left it.
        lr_span_flops=cfg.target_flops if unified_two_phase else None,
        lr_offset_flops=ppt_metrics["total_training_flops"] if unified_two_phase else 0.0,
        metric_prefix="pt",
        wandb_run=wandb_run,
        ckpt_dir=str(base_ckpt / "pt"),
        **transition_kwargs,
    )
    pt_metrics = train(**pt.train_kwargs)
    release_phase(pt.model, pt.optimizers)

    if cfg.metrics_output_file and ctx.rank == 0:
        Path(cfg.metrics_output_file).write_text(json.dumps(pt_metrics))

    phases = [("PT", pt_metrics, pt.ckpt_dir)]
    if ppt_metrics is not None:
        phases.insert(0, ("PPT", ppt_metrics, ppt_metrics["ckpt_dir"]))
    teardown_run(wandb_run, *phases, cfg=cfg, ctx=ctx, base_ckpt=base_ckpt)


def _split_spec_selectors(argv: list[str]) -> tuple[str, str | None, list[str]]:
    """Parsed ahead of the training config rather than added to it, so the
    config keeps the flat flag namespace every study is launched with."""
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--init-variant", default="spectral")
    ap.add_argument("--ppt-opt", default=None)
    selectors, config_argv = ap.parse_known_args(argv)
    return selectors.init_variant, selectors.ppt_opt, config_argv


if __name__ == "__main__":
    init_variant, ppt_opt, config_argv = _split_spec_selectors(sys.argv[1:])
    spec, ppt_spec = resolve_model_specs(init_variant, ppt_opt)
    main(tyro.cli(ExperimentConfig, args=config_argv),
         model_spec=spec, ppt_model_spec=ppt_spec)
