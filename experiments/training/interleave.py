"""Interleaved PPT->PT: soften the abrupt dataset switch, same tokens seen.

Reorders an abrupt two-phase reference run (``pure_code_frac=1``) into:

  1. **pure PPT** — ``pure_code_frac`` of the PPT corpus budget, standard
     loader (phase 1, ``ppt/`` metrics);
  2. **constant-rate mixing** — every batch of the rest of the run draws a
     fixed share of PPT sequences, spread evenly over the whole of phase 2
     (``pt/`` metrics), until the PPT corpus budget is exhausted.

The LR schedule is one continuous trapezoid over the whole run and the phase
transition carries weights + optimizer state with no reinit.

Phase 2 also evals at the reference run's PPT->PT transition step (the same
tokens-consumed point, mid-mix here) and at the mixing region's end.

Example:
    torchrun --standalone --nproc_per_node=1 -m experiments.training.interleave \\
        --depth=4 --pure-code-frac=0.5 \\
        --ppt-corpus-tokens=2621440 --pt-corpus-tokens=26214400 \\
        --data-ppt=github-code-nanochatbpe-1B --ppt-same-vocab-as-pt \\
        --ppt-unified-lr --no-reinit-embed-at-transition \\
        --no-reset-optimizer-at-transition
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import tyro

from nanochat.config import DataConfig
from nanochat.data.bin import build_interleaved_bin_loader
from nanochat.data.registry import _bin_path, get_source
from nanochat.flops import tokens_per_step
from nanochat.runtime import print0, release_phase
from nanochat.loop import train
from nanochat.phase import build_phase

from experiments.training.schedule import plan_interleave
from experiments.training.config import (
    MODEL_SPEC,
    ExperimentConfig,
    setup_run,
    teardown_run,
    with_eval_steps,
)


@dataclass
class InterleaveConfig(ExperimentConfig):
    """Token-native interleave study config.

    The two corpus budgets replace the FLOP split (``target_flops`` /
    ``alpha_ppt`` are not used for budgeting): each corpus contributes exactly
    its budget regardless of the dial.
    """

    ppt_corpus_tokens: int = 235_143_168    # code-5% @ 27e18 d24: 897 steps
    pt_corpus_tokens: int = 4_470_341_632   # 17,053 steps
    # Share of the PPT corpus budget spent in the pure phase-1 prefix; the
    # leftover is spread at a constant rate over the whole of phase 2. 1.0 is
    # the abrupt reference run (nothing left to mix in); 0.0 is single-phase,
    # mixed from the very first batch, with no transition at all.
    pure_code_frac: float = 1.0


def main(cfg: InterleaveConfig) -> None:
    cfg, ctx, base_ckpt, wandb_run = setup_run(cfg)
    assert cfg.data_ppt is not None, "interleave needs a PPT corpus (--data-ppt)"
    assert cfg.ppt_same_vocab_as_pt, "one model serves mixed batches: PPT must share the PT vocab"
    assert cfg.ppt_unified_lr, "one continuous LR schedule across phases (--ppt-unified-lr)"
    assert not cfg.reinit_embed_at_transition and not cfg.reset_optimizer_at_transition, (
        "the transition carries weights + optimizer state unchanged"
    )
    assert cfg.ppt_tokens is None and cfg.pt_tokens is None, (
        "use ppt_corpus_tokens / pt_corpus_tokens; phase budgets are derived from the dial"
    )

    seq_len = cfg.model_pt.sequence_len
    tps = tokens_per_step(
        cfg.hardware.device_batch_size, seq_len, ctx.world_size, cfg.hardware.grad_accum_steps
    )
    # The plan is denominated in loader draws (grad_accum per optimizer step),
    # not optimizer steps.
    accum = cfg.hardware.grad_accum_steps
    draw_tokens = tps // accum
    plan = plan_interleave(
        ppt_tokens=cfg.ppt_corpus_tokens,
        pt_tokens=cfg.pt_corpus_tokens,
        pure_frac=cfg.pure_code_frac,
        tokens_per_step=draw_tokens,
        seq_len=seq_len,
    )
    assert plan.pure_steps % accum == 0, (
        f"pure prefix of {plan.pure_steps} draws is not whole optimizer steps "
        f"(grad_accum={accum}); nudge pure_code_frac"
    )
    print0(
        f"=== interleave plan: pure {plan.pure_steps} draws + mixing "
        f"{plan.pt_steps} = "
        f"{plan.total_steps} draws of {draw_tokens:,} tok ({accum} draws/step, "
        f"dial pure_code_frac={cfg.pure_code_frac}) ==="
    )
    if wandb_run is not None:
        wandb_run.config.update(
            {"interleave": {**dataclasses.asdict(plan), "pt_steps": plan.pt_steps}},
            allow_val_change=True,
        )

    # Phase 1: pure PPT. Skipped at pure_steps=0.
    ppt_metrics = None
    fpt_override = {}
    if plan.pure_steps > 0:
        ppt = build_phase(
            with_eval_steps(cfg), ctx,
            source_key=cfg.data_ppt,
            model_cfg=cfg.model_pt,
            model_spec=MODEL_SPEC,
            token_budget=plan.pure_steps * draw_tokens,
            lr_span_tokens=plan.total_steps * draw_tokens,
            metric_prefix="ppt",
            wandb_run=wandb_run,
            ckpt_dir=str(base_ckpt / "ppt"),
        )
        ppt_metrics = train(**ppt.train_kwargs)
        ppt_metrics["ckpt_dir"] = ppt.ckpt_dir
        fpt_override = {"flops_per_token_override": ppt.flops_per_token}  # same model both phases
        release_phase(ppt.model, ppt.optimizers)
        del ppt  # the phase holds the model, its optimizer state and its eval batches

    # Phase 2: mixing region + pure text, one train() call over the mixed loader.
    # A word-aligned source (NCA, dyck) is read at a fixed stride, whole rows
    # when a row holds a full S+1 sequence, dense S+1 chunks otherwise. Its
    # id_offset maps it into the extended vocab (--pad-vocab must cover source
    # vocab + offset).
    ppt_source = get_source(cfg.data_ppt)
    if ppt_source.loader == "word_aligned":
        assert plan.pure_steps == 0, (
            "word-aligned diet sources are only wired for the single-phase "
            "no-curriculum control (pure_code_frac=0); a pure phase would need "
            "the word reader"
        )
        stored = ppt_source.stored_word_length
        code_stride = stored if stored is not None and stored >= seq_len + 1 else seq_len + 1
    else:
        code_stride = None
    mixed_loader = build_interleaved_bin_loader(
        DataConfig(path=_bin_path(ppt_source, cfg.train_split)),
        DataConfig(path=_bin_path(get_source(cfg.data_pt), cfg.train_split)),
        cfg.hardware.device_batch_size, seq_len, ctx.device, ctx.rank, ctx.world_size,
        code_seqs_before=plan.code_seqs_before,
        code_start_seq=plan.pure_steps * plan.seqs_per_step,
        code_stride=code_stride,
        code_id_offset=ppt_source.id_offset,
    )
    if ppt_source.mask_ids:
        from nanochat.data.transforms import with_token_id_masking
        mixed_loader = with_token_id_masking(mixed_loader, ids=ppt_source.effective_mask_ids())
    # Landmark eval at the reference run's transition point, in phase-2-local
    # optimizer steps. Kept only when strictly inside the phase.
    landmarks = []
    for draws in (plan.ppt_steps_total - plan.pure_steps,):
        if 0 < draws < plan.pt_steps:
            assert draws % accum == 0, (
                f"landmark at {draws} draws is not a whole optimizer step (grad_accum={accum})"
            )
            landmarks.append(draws // accum)
    landmarks = tuple(landmarks)
    pt_cfg = with_eval_steps(cfg)
    pt_cfg = dataclasses.replace(
        pt_cfg, train=dataclasses.replace(pt_cfg.train, eval_at_steps=landmarks or None)
    )
    transition_kwargs = {} if ppt_metrics is None else dict(
        reinit_embed=False,
        reset_optimizer=False,
        load_from=(ppt_metrics["ckpt_dir"], ppt_metrics["final_step"]),
        step_offset=ppt_metrics["final_step"] + 1,
        flops_offset=ppt_metrics["total_training_flops"],
        started_time=ppt_metrics["total_training_time"],
    )
    pt = build_phase(
        pt_cfg, ctx,
        source_key=cfg.data_pt,
        model_cfg=cfg.model_pt,
        model_spec=MODEL_SPEC,
        token_budget=plan.pt_steps * draw_tokens,
        lr_span_tokens=plan.total_steps * draw_tokens,
        lr_offset_tokens=plan.pure_steps * draw_tokens,
        train_loader_override=mixed_loader,
        metric_prefix="pt",
        wandb_run=wandb_run,
        ckpt_dir=str(base_ckpt / "pt"),
        **fpt_override,
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


if __name__ == "__main__":
    main(tyro.cli(InterleaveConfig))
