"""Score a trained checkpoint on the measures the paper reports.

A checkpoint is a directory holding `model_{step:06d}.pt` and
`config_{step:06d}.json`; the model, its tokenizer and its validation sets are
all rebuilt from that snapshot. The highest step present is the end of training.

Three suites, selected by name:

    bpb    Bits-per-byte on the held-out window the paper reports, on FineWeb,
           C4, GitHub code and the run's own PPT corpus.
    blimp  67 minimal-pair paradigms of English grammar, as the mean over
           pairs, with a breakdown by paradigm, phenomenon and field.
    core   22 DCLM multiple-choice and completion tasks, as centered accuracy:
           each task rescaled by its random-guessing baseline, so 0 is chance
           and 1 is perfect.

    uv run python -m reproduce.evaluate --cell scaling__nca_1_1e18 --seed 0
    uv run python -m reproduce.evaluate --checkpoint <dir>
    uv run python -m reproduce.evaluate --hf-repo <org/name> --suite bpb

`--cell` scores the published checkpoint a record in `reproduce/cells.json`
lists for that seed, with the eval settings the record lists.
`scripts/eval.sh` sets up the environment and runs this. Scoring is
single-device.
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
    source over `[0, aux_eval_tokens)`.
    """

    eval_tokens: int = 10_485_760
    eval_offset_tokens: int = 2_097_152
    aux_eval_tokens: int = 9_437_184
    eval_batch_size: int = 32   # val batch
    max_per_task: int = -1      # CORE examples per task
    max_per_uid: int = -1       # BLiMP pairs per paradigm
    batch_size: int = 64        # BLiMP scoring batch
    compile: bool = True        # run bpb through torch.compile, as training did

    @property
    def window(self) -> str:
        return (f"{self.eval_tokens}@{self.eval_offset_tokens}"
                f"+aux{self.aux_eval_tokens}")


@dataclass(frozen=True)
class EvalSuite:
    """One scoring route. `score` returns that suite's flat metric dict;
    `headline` names the key worth printing, and a suite that leaves it unset
    prints every key.

    `autocast` scores the suite inside a bfloat16 autocast region on CUDA.
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

    from experiments.training.config import ExperimentConfig

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
    from nanochat.phase import build_val_sets

    import dataclasses

    cfg = config_from_snapshot(ckpt.config)
    # C4 and GitHub code are scored for every model, whatever it trained on.
    cfg = dataclasses.replace(cfg, extra_eval=[*cfg.extra_eval, *EXTRA_EVAL])
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
        model = ckpt.model.model
        if options.compile and device.type == "cuda":
            import torch
            model = torch.compile(model, dynamic=False)
        out.update(eval_fn(model))
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


EXTRA_EVAL = ("c4-nanochatbpe-10B", "github-code-nanochatbpe-1B")


SUITES: dict[str, EvalSuite] = {
    s.name: s for s in (
        EvalSuite(name="bpb", summary="bits-per-byte on the held-out window the paper reports",
                  score=score_bpb),
        EvalSuite(name="blimp", summary="BLiMP minimal-pair grammar", score=score_blimp,
                  headline="blimp"),
        # CORE picks the candidate continuation with the lowest MEAN per-token
        # loss, i.e. length-normalized.
        EvalSuite(name="core", summary="DCLM CORE, centered accuracy", score=score_core,
                  headline="core_metric", autocast=True),
    )
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--cell", metavar="CELL_ID",
                        help="a reported model: the checkpoint reproduce/cells.json "
                             "lists for this cell at --seed")
    source.add_argument("--checkpoint", metavar="DIR",
                        help="a checkpoint directory written by training")
    source.add_argument("--hf-repo", metavar="ORG/NAME",
                        help="a published checkpoint repo, downloaded first")
    ap.add_argument("--seed", type=int, default=0,
                    help="which of a cell's three seeds (default: %(default)s)")
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
    ap.add_argument("--aux-eval-tokens", type=int, default=None, metavar="N",
                    help="window for every non-primary source (default: the record's "
                         "aux_eval_tokens under --cell, else 9437184)")
    ap.add_argument("--eval-batch-size", type=int, default=None, metavar="N",
                    help="val batch size (default: the record's under --cell, "
                         "else 32)")
    ap.add_argument("--max-per-task", type=int, default=-1, metavar="N",
                    help="cap CORE examples per task, for a quick check. Must "
                         "exceed the largest few-shot count (10), since the "
                         "few-shot prompt is drawn from the same capped set "
                         "(default: all)")
    ap.add_argument("--max-per-uid", type=int, default=-1, metavar="N",
                    help="cap BLiMP pairs per paradigm (default: all)")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="BLiMP scoring batch size (default: %(default)s)")
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None,
                    help="score bpb on the compiled model, CUDA only (default: the "
                         "record's under --cell, else compiled)")
    ap.add_argument("--smoke", action="store_true",
                    help="a few examples per task, to check the path; the numbers "
                         "are not results")
    ap.add_argument("--out", type=Path, metavar="FILE",
                    help="write the metrics here as JSON (default: beside a local "
                         "checkpoint, else under outputs/eval/)")
    ap.add_argument("--wandb-project", metavar="NAME",
                    help="also log the metrics to this W&B project (default: off)")
    ap.add_argument("--wandb-entity", metavar="NAME",
                    help="W&B entity for --wandb-project")
    ap.add_argument("--wandb-run-name", metavar="NAME",
                    help="W&B run name (default: the checkpoint's own name)")
    args = ap.parse_args(argv)

    hf_repo = args.hf_repo
    aux_eval_tokens = args.aux_eval_tokens
    eval_batch_size, compile_model = args.eval_batch_size, args.compile
    if args.cell:
        from reproduce.cells import load_records

        records = load_records()
        if args.cell not in records:
            sys.exit(f"unknown cell_id {args.cell!r} (see reproduce/cells.json)")
        record = records[args.cell]
        entry = record["checkpoints"].get(str(args.seed))
        if entry is None:
            sys.exit(f"{args.cell} lists no checkpoint for seed {args.seed}")
        hf_repo = entry["hf_repo"]
        if aux_eval_tokens is None:
            aux_eval_tokens = record["config"].get("aux_eval_tokens")
        scored_with = record.get("fineweb_bpb_scored_with", {})
        if eval_batch_size is None:
            eval_batch_size = scored_with.get("eval_batch_size")
        if compile_model is None:
            compile_model = scored_with.get("compile")
    if aux_eval_tokens is None:
        aux_eval_tokens = EvalOptions.aux_eval_tokens
    if eval_batch_size is None:
        eval_batch_size = EvalOptions.eval_batch_size
    if compile_model is None:
        compile_model = EvalOptions.compile
    if args.smoke:
        print("=== SMOKE: a few examples per task; the numbers are not results ===")
        # 16 examples per CORE task and per BLiMP paradigm. CORE draws each
        # prompt's few-shot examples from the same 16, so the cap has to stay
        # above the largest few-shot count (10).
        args.max_per_task, args.max_per_uid = 16, 16

    selected = [SUITES[name] for name in (args.suite or list(SUITES))]
    options = EvalOptions(eval_tokens=args.eval_tokens,
                          eval_offset_tokens=args.eval_offset_tokens,
                          aux_eval_tokens=aux_eval_tokens,
                          eval_batch_size=eval_batch_size,
                          max_per_task=args.max_per_task,
                          max_per_uid=args.max_per_uid,
                          batch_size=args.batch_size,
                          compile=compile_model)

    import torch

    from nanochat.common import autodetect_device_type
    from nanochat.eval.core.checkpoint_loading import load_from_dir, load_from_hf

    device = torch.device(autodetect_device_type())
    if hf_repo:
        ckpt = load_from_hf(hf_repo, device, step=args.step)
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
    print(f"  bpb       : eval batch {options.eval_batch_size}, "
          f"{'compiled' if options.compile and device.type == 'cuda' else 'eager'}, "
          f"one device")

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
                    "compile": options.compile and device.type == "cuda",
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
    elif out is None:
        out = Path("outputs") / "eval" / f"{Path(hf_repo).name}_{ckpt.step:06d}.json"
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
            "compile": options.compile and device.type == "cuda",
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
