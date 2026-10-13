"""Score a trained checkpoint on the measures the paper reports.

A checkpoint is a directory holding `model_{step:06d}.pt` and
`config_{step:06d}.json`; the model, its tokenizer and its validation sets are
all rebuilt from that snapshot. The highest step present is the end of training.

Three suites, selected by name:

    bpb    Bits-per-byte on the held-out window, rebuilt from the checkpoint's
           own config. The cheapest check that a checkpoint loaded correctly.
    blimp  67 minimal-pair paradigms of English grammar, as the mean over
           pairs, with a breakdown by paradigm, phenomenon and field.
    core   22 DCLM multiple-choice and completion tasks, as centered accuracy:
           each task rescaled by its random-guessing baseline, so 0 is chance
           and 1 is perfect.

    uv run python -m reproduce.evaluate --checkpoint <dir>
    uv run python -m reproduce.evaluate --hf-repo <org/name> --suite bpb
    uv run python -m reproduce.evaluate --checkpoint <dir> --suite blimp --suite core

Training removes its checkpoints unless told otherwise, so pass
`--keep-checkpoints` to `reproduce.run_cell` for a run you intend to evaluate.
`reproduce/run_eval.sh` does both ends for one cell. Scoring is single-device.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class EvalOptions:
    """How to score: the window, and the knobs that shorten a suite.

    The primary validation source is scored over
    `[eval_offset_tokens, eval_offset_tokens + eval_tokens)` and every other
    source over `[0, aux_eval_tokens)`, pinned by token index so the scored
    position set does not move with batch geometry.

    The window is stated here rather than read off the checkpoint: a run's
    config snapshot records the window that run *trained* under, which for the
    reported checkpoints is not the window their published numbers were scored
    at, so inheriting it would quietly produce a number the paper never reports.
    """

    eval_tokens: int = 10_485_760
    eval_offset_tokens: int = 2_097_152
    aux_eval_tokens: int = 9_437_184
    eval_batch_size: int = 32   # val batch; changes how the window is packed
    max_per_task: int = -1      # CORE examples per task
    max_per_uid: int = -1       # BLiMP pairs per paradigm
    batch_size: int = 64        # BLiMP scoring batch

    @property
    def window(self) -> str:
        return (f"{self.eval_tokens}@{self.eval_offset_tokens}"
                f"+aux{self.aux_eval_tokens}")


@dataclass(frozen=True)
class EvalSuite:
    """One scoring route. `score` returns that suite's flat metric dict;
    `headline` names the key worth printing, and a suite that leaves it unset
    prints every key.

    `autocast` scores the suite inside a bfloat16 autocast region on CUDA. The
    model's own forward already autocasts, so this only changes the arithmetic
    the suite does on the returned logits — enough to flip a choice between two
    close continuations. It is set per suite to match how each reported number
    was produced.
    """

    name: str
    summary: str
    score: Callable[..., dict[str, float]]
    headline: str | None = None
    autocast: bool = False


def config_from_snapshot(snapshot: dict):
    """The training config a checkpoint was written under, as its dataclass.

    The snapshot is `dataclasses.asdict` of that config, so it round-trips field
    for field.
    """
    import dataclasses
    import typing

    from experiments.training.common import ExperimentConfig

    def build(cls, values: dict):
        hints = typing.get_type_hints(cls)
        kwargs = {}
        for field in dataclasses.fields(cls):
            if field.name not in values:
                continue
            value, hint = values[field.name], hints[field.name]
            if dataclasses.is_dataclass(hint) and isinstance(value, dict):
                kwargs[field.name] = build(hint, value)
            elif isinstance(value, list) and field.name == "save_at_steps":
                kwargs[field.name] = tuple(value)
            else:
                kwargs[field.name] = value
        return cls(**kwargs)

    return build(ExperimentConfig, snapshot)


def score_bpb(ckpt, device, options: EvalOptions) -> dict[str, float]:
    from nanochat.eval.base import build_eval
    from nanochat.runtime import RuntimeCtx
    from nanochat.training_run import build_val_sets

    import dataclasses

    cfg = config_from_snapshot(ckpt.config)
    cfg = dataclasses.replace(cfg, eval=dataclasses.replace(
        cfg.eval,
        eval_tokens=options.eval_tokens,
        eval_offset_tokens=options.eval_offset_tokens,
        aux_eval_tokens=options.aux_eval_tokens,
        eval_device_batch_size=options.eval_batch_size,
    ))
    ctx = RuntimeCtx(rank=0, local_rank=0, world_size=1, device=device,
                     run_timestamp="eval")
    val_sets = build_val_sets(cfg, ctx,
                              model_vocab=ckpt.config["model_pt"]["vocab_size"],
                              seq_len=ckpt.config["model_pt"]["sequence_len"])
    out: dict[str, float] = {}
    for eval_fn in build_eval(cfg.eval, ctx, val_sets=val_sets):
        # BPB needs the training-time per-position-loss contract, not the
        # callable-to-logits view `ckpt.model` gives the prompt harnesses.
        out.update(eval_fn(ckpt.model.model))
    return out


def score_blimp(ckpt, device, options: EvalOptions) -> dict[str, float]:
    from nanochat.eval.blimp import run_blimp_eval

    return run_blimp_eval(ckpt.model, ckpt.tokenizer, device,
                          batch_size=options.batch_size,
                          max_per_uid=options.max_per_uid)


def score_core(ckpt, device, options: EvalOptions) -> dict[str, float]:
    from nanochat.eval.core.tasks import run_core_eval

    return run_core_eval(ckpt.model, ckpt.tokenizer, device,
                         max_per_task=options.max_per_task)


SUITES: dict[str, EvalSuite] = {
    s.name: s for s in (
        EvalSuite("bpb", "bits-per-byte on the run's own held-out window", score_bpb),
        EvalSuite("blimp", "BLiMP minimal-pair grammar", score_blimp, headline="blimp"),
        # CORE picks the candidate continuation with the lowest MEAN per-token
        # loss, i.e. length-normalized. Its published numbers were produced
        # inside a bfloat16 autocast region, and float32 moves a handful of
        # borderline examples; bpb and BLiMP were not, and reproduce exactly
        # without it.
        EvalSuite("core", "DCLM CORE, centered accuracy", score_core,
                  headline="core_metric", autocast=True),
    )
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", metavar="DIR",
                        help="a checkpoint directory written by training")
    source.add_argument("--hf-repo", metavar="ORG/NAME",
                        help="a published checkpoint repo, downloaded first")
    ap.add_argument("--step", type=int,
                    help="which step to load (default: the highest present)")
    ap.add_argument("--suite", action="append", choices=sorted(SUITES), metavar="NAME",
                    help="a suite to run, repeatable: "
                         + "; ".join(f"{s.name} — {s.summary}" for s in SUITES.values())
                         + " (default: all of them)")
    ap.add_argument("--eval-tokens", type=int, default=10_485_760, metavar="N",
                    help="primary window size (default: %(default)s, the "
                         "window the reported numbers were scored at)")
    ap.add_argument("--eval-offset-tokens", type=int, default=2_097_152, metavar="N",
                    help="primary window offset (default: %(default)s)")
    ap.add_argument("--aux-eval-tokens", type=int, default=9_437_184, metavar="N",
                    help="window for every non-primary source (default: %(default)s)")
    ap.add_argument("--eval-batch-size", type=int, default=32, metavar="N",
                    help="val batch size; the window is packed into sequences "
                         "by it, so a different value shifts the scored set "
                         "slightly (default: %(default)s)")
    ap.add_argument("--max-per-task", type=int, default=-1, metavar="N",
                    help="cap CORE examples per task, for a quick check. Must "
                         "exceed the largest few-shot count (10), since the "
                         "few-shot prompt is drawn from the same capped set "
                         "(default: all)")
    ap.add_argument("--max-per-uid", type=int, default=-1, metavar="N",
                    help="cap BLiMP pairs per paradigm (default: all)")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="BLiMP scoring batch size (default: %(default)s)")
    ap.add_argument("--out", type=Path, metavar="FILE",
                    help="write the metrics here as JSON (default: beside the checkpoint)")
    ap.add_argument("--wandb-project", metavar="NAME",
                    help="also log the metrics to this W&B project (default: off)")
    ap.add_argument("--wandb-entity", metavar="NAME",
                    help="W&B entity for --wandb-project")
    ap.add_argument("--wandb-run-name", metavar="NAME",
                    help="W&B run name (default: the checkpoint's own name)")
    args = ap.parse_args(argv)

    selected = [SUITES[name] for name in (args.suite or list(SUITES))]
    options = EvalOptions(eval_tokens=args.eval_tokens,
                          eval_offset_tokens=args.eval_offset_tokens,
                          aux_eval_tokens=args.aux_eval_tokens,
                          eval_batch_size=args.eval_batch_size,
                          max_per_task=args.max_per_task,
                          max_per_uid=args.max_per_uid,
                          batch_size=args.batch_size)

    import torch

    from nanochat.common import autodetect_device_type
    from nanochat.eval.core.checkpoint_loading import load_from_dir, load_from_hf

    device = torch.device(autodetect_device_type())
    if args.hf_repo:
        ckpt = load_from_hf(args.hf_repo, device, step=args.step)
    else:
        directory = Path(args.checkpoint)
        if not directory.is_dir():
            sys.exit(f"no checkpoint directory at {directory}")
        ckpt = load_from_dir(str(directory), device, step=args.step)

    print(f"loaded {ckpt.hf_repo} step {ckpt.step}")
    print(f"  arch      : {ckpt.arch_family}")
    print(f"  tokenizer : {ckpt.tokenizer_key} (from data_pt={ckpt.config['data_pt']})")
    print(f"  device    : {device}")
    trained_at = ckpt.config["eval"]["eval_tokens"]
    print(f"  window    : {options.window}"
          + (f"   (this checkpoint trained at eval_tokens={trained_at})"
             if trained_at and trained_at != options.eval_tokens else ""))

    wandb_run = None
    if args.wandb_project:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=args.wandb_run_name or Path(ckpt.hf_repo).name,
            config={"source": ckpt.hf_repo, "step": ckpt.step,
                    "arch_family": ckpt.arch_family, "tokenizer": ckpt.tokenizer_key,
                    "data_pt": ckpt.config["data_pt"], "window": options.window,
                    "eval_batch_size": options.eval_batch_size,
                    "suites": [s.name for s in selected]},
        )

    metrics: dict[str, float] = {}
    for suite in selected:
        print(f"\n{suite.name}: {suite.summary} ..."
              + ("  [bfloat16 autocast]" if suite.autocast and device.type == "cuda" else ""))
        region = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
                  if suite.autocast and device.type == "cuda"
                  else contextlib.nullcontext())
        with region:
            scored = suite.score(ckpt, device, options)
        metrics.update(scored)
        shown = [suite.headline] if suite.headline else sorted(scored)
        for key in shown:
            print(f"  {key:<26} {scored[key]:.6f}")
        # Per suite, so a long run reports as it goes.
        if wandb_run is not None:
            wandb_run.summary.update(scored)

    out = args.out
    if out is None and args.checkpoint:
        out = Path(args.checkpoint) / f"eval_{ckpt.step:06d}.json"
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "source": ckpt.hf_repo,
            "step": ckpt.step,
            "arch_family": ckpt.arch_family,
            "tokenizer": ckpt.tokenizer_key,
            "data_pt": ckpt.config["data_pt"],
            "window": options.window,
            "eval_batch_size": options.eval_batch_size,
            "suites": [s.name for s in selected],
            "metrics": metrics,
        }, indent=1, sort_keys=True) + "\n")
        print(f"\nwrote {out}")
    if wandb_run is not None:
        print(f"logged to {wandb_run.url}")
        wandb_run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
