"""Multi-scale study driver + CLI entry point.

``run_scales`` iterates a list of FLOP scales, delegating each to a shared
``ScaleRunner`` and carrying two things forward between them:

- the prior scale's ``best_cfg`` becomes the next scale's ``warm_start_cfg``,
  setting ``.initial`` on every tuned Param and running as the baseline trial
  at the new budget before the LLM starts proposing;
- the prior scale's ``tuner_<scale>.json`` is handed to ``ContextBuilder``,
  which deserializes its entries onto the next scale's fresh Tuner.

Every knob lives on the ``TaskSpec`` config module named by ``--config``.

    python -m experiments.tuner.llmtuna_driver --config <dotted.task.module>
"""

import argparse
import importlib
import json
from pathlib import Path
from typing import Optional

import llmtuna as lt

from experiments.training.config import PT_VOCAB_SIZE
from experiments.tuner.task import TaskSpec
from .checkpoint import CheckpointStore
from .context_builder import ContextBuilder
from .flops import FlopsEstimator
from .hf_sync import sync_to_hf
from .objective import TorchrunObjective
from .prompts import CONTEXT_SOURCE_FILES, SUMMARIZE_SYSTEM, balanced_system
from .proposal_format import make_log_and_format
from .scale_runner import ScaleRunner


def run_scales(
    *,
    task: TaskSpec,
    runner: ScaleRunner,
    scales: list[str],
    checkpoint_store: CheckpointStore,
    initial_warm_start_cfg: Optional[dict] = None,
    initial_prior_save_path: Optional[Path] = None,
    initial_results: Optional[list[dict]] = None,
    initial_preseed_trials: Optional[list[dict]] = None,
) -> list[dict]:
    """Run every scale in order and return the per-scale summary dicts.

    The ``initial_*`` arguments support resume: a prior run's tuner file and
    results are prepended so the rewritten ``scale_results.json`` stays
    complete, and ``initial_preseed_trials`` (a crashed scale's ledger) is fed
    to the first scale only, observed without re-running. The combined results
    file is rewritten after every scale.
    """
    results: list[dict] = list(initial_results or [])
    prior_save_path = initial_prior_save_path
    warm_start_cfg = initial_warm_start_cfg
    preseed_trials = initial_preseed_trials

    for scale_label in scales:
        result = runner.run(
            task=task,
            scale_label=scale_label,
            warm_start_cfg=warm_start_cfg,
            prior_save_path=prior_save_path,
            preseed_trials=preseed_trials,
        )
        preseed_trials = None
        results.append(result)
        checkpoint_store.save_results(results)

        if task.sync_to_hf:
            # A transient HF issue must not kill the study mid-flight.
            try:
                url = sync_to_hf(
                    folder=checkpoint_store.root,
                    repo_name=task.wandb_project,
                    commit_message=f"scale {scale_label} done ({len(results)}/{len(scales)})",
                )
                print(f"  [hf] synced to {url} (after scale {scale_label})")
            except Exception as e:
                print(f"  [hf] sync failed after scale {scale_label}: {e}")

        if result["save_path"]:
            prior_save_path = Path(result["save_path"])
        if result["best_cfg"]:
            warm_start_cfg = result["best_cfg"]

    return results


def load_task(dotted: str) -> TaskSpec:
    """Import a config module and return its module-level ``TASK``."""
    module = importlib.import_module(dotted)
    task = getattr(module, "TASK", None)
    if not isinstance(task, TaskSpec):
        raise SystemExit(
            f"Config module '{dotted}' must define a module-level TASK "
            f"(a TaskSpec instance); got {type(task).__name__}."
        )
    return task


def main() -> None:
    ap = argparse.ArgumentParser(
        description="LLM-driven multi-scale hparam optimization (llmtuna backend). "
                    "The entire run is defined by one TaskSpec config module."
    )
    ap.add_argument(
        "--config", required=True, metavar="DOTTED.MODULE",
        help="Config module exposing a module-level TASK (TaskSpec), e.g. "
             "experiments.tasks.vanilla",
    )
    ap.add_argument(
        "--llmtuna-resume-from", default=None, metavar="OUT_DIR",
        help="Continue a prior llmtuna run from this output dir: reuse its "
             "scale_results.json + tuner_<scale>.json to warm-start the config's "
             "not-yet-completed scales, appending new outputs to the same dir. "
             "Trials log to the same W&B project as the config.",
    )
    args = ap.parse_args()
    run_task(
        load_task(args.config),
        resume_from=Path(args.llmtuna_resume_from) if args.llmtuna_resume_from else None,
    )


def run_task(task: TaskSpec, *, resume_from: Optional[Path] = None) -> None:
    """Wire the whole llmtuna stack from one fully-specified ``TaskSpec``.

    ``resume_from`` (an existing out_dir) continues a prior run in place:
    completed scales are skipped and the rest warm-start from the last one.
    """
    out_dir = (
        resume_from if resume_from is not None
        else (Path(task.out_dir) if task.out_dir else (Path("outputs") / task.wandb_project))
    )

    provider = lt.OpenRouter(
        model=task.llm_model,
        thinking_budget=task.llm_thinking_budget,
        max_tokens=4000,
    )
    objective = TorchrunObjective(
        task=task,
        flops_estimator=FlopsEstimator(seq_len=task.max_seq_len, vocab_size=PT_VOCAB_SIZE),
        out_dir=out_dir / "metrics",
    )
    checkpoint_store = CheckpointStore(out_dir)

    summarize_paths = CONTEXT_SOURCE_FILES if task.summarize_architecture else None
    context_builder = ContextBuilder(
        provider=provider,
        summarize_paths=summarize_paths,
        summarize_hparam_names=task.hparams,
        summarize_system_prompt=SUMMARIZE_SYSTEM,
        optimizer_description=task.optimizer,
    )

    runner = ScaleRunner(
        provider=provider,
        objective=objective,
        context_builder=context_builder,
        patience=task.patience,
        tol=task.tol,
        # Built from the knobs this study tunes, unless the task overrides it.
        system_prompt=(
            (task.system_prompt or balanced_system(hparams=task.hparams))
            + ("\n\n" + task.system_prompt_extra if task.system_prompt_extra else "")
        ),
        checkpoint_store=checkpoint_store,
        max_iters=task.max_iters_per_scale,
        # THINKING is stripped from the rendered context by default; full
        # reasoning always lands in out_dir/llm_transcript.jsonl.
        format_proposal=make_log_and_format(
            out_dir / "llm_transcript.jsonl",
            strip_thinking=not task.keep_thinking,
            strip_response=task.context_trials_only,
        ),
    )

    if resume_from is not None:
        (scales_to_run, initial_warm_start_cfg, initial_prior_save_path,
         prior_results, preseed_trials) = _resume_state(task, checkpoint_store)
        if not scales_to_run:
            print(f"All config scales {task.scales} already complete in {out_dir}; "
                  f"nothing to run.")
            return
        print(f"Resuming from {out_dir}: "
              f"done={[r['scale_label'] for r in prior_results]}, "
              f"remaining={scales_to_run}, warm-start "
              f"best_cfg={initial_warm_start_cfg}")
        if preseed_trials:
            print(f"Preseeding {scales_to_run[0]} with {len(preseed_trials)} "
                  f"completed trials from its ledger (not re-run)")
    else:
        scales_to_run = task.scales
        initial_warm_start_cfg = _validated_initial_params(task)
        initial_prior_save_path = None
        prior_results = None
        preseed_trials = None
        print(f"Seeding scale 1 baseline with: {initial_warm_start_cfg}")

    print(f"Starting study: task={task.name} scales={scales_to_run}")
    results = run_scales(
        task=task,
        runner=runner,
        scales=scales_to_run,
        checkpoint_store=checkpoint_store,
        initial_warm_start_cfg=initial_warm_start_cfg,
        initial_prior_save_path=initial_prior_save_path,
        initial_results=prior_results,
        initial_preseed_trials=preseed_trials,
    )

    print(f"\nDone. Results at {checkpoint_store.results_path}")
    for r in results:
        print(
            f"  {r['scale_label']}: best_value={r['best_value']} "
            f"best_cfg={r['best_cfg']} n_trials={r['n_trials']}"
        )

    if task.wandb_summary:
        print("\nEmitting W&B summary run...")
        _emit_wandb_summary(task=task, scale_results=results, out_dir=out_dir)

    if task.sync_to_hf:
        # Also picks up files produced after the study loop (summary plots,
        # architecture_summary.txt).
        print(f"\nFinal sync of {out_dir} to HF...")
        url = sync_to_hf(
            folder=out_dir,
            repo_name=task.wandb_project,
            commit_message=(
                f"final: llmtuna {task.name} "
                f"scales={','.join(task.scales)}"
            ),
        )
        print(f"Synced to: {url}")


def _resume_state(task: TaskSpec, checkpoint_store: CheckpointStore):
    """Read a prior run's artifacts to continue its not-yet-completed scales.

    Returns ``(remaining_scales, warm_start_cfg, prior_save_path,
    prior_results, preseed_trials)``. When the first remaining scale crashed
    mid-loop its ledger becomes ``preseed_trials`` and the warm-start is that
    ledger's best entry, with no baseline re-run; a study interrupted on its
    first scale has only the ledger and resumes from that alone. Fails loudly
    if neither artifact exists rather than starting a fresh study.
    """
    results_path = checkpoint_store.results_path
    prior_results = (
        json.loads(results_path.read_text()) if results_path.exists() else []
    )
    completed = {r["scale_label"] for r in prior_results}
    remaining = [s for s in task.scales if s not in completed]
    if prior_results:
        last = prior_results[-1]
        # Rebuild the path from the resume dir, not the possibly-stale
        # absolute path stored in scale_results.json.
        prior_save_path = checkpoint_store.tuner_path(last["scale_label"])
        if not prior_save_path.exists():
            raise SystemExit(
                f"--resume-from: prior tuner file missing: {prior_save_path}"
            )
        warm_start_cfg = last["best_cfg"]
    else:
        prior_save_path = None
        warm_start_cfg = None  # must come from the first scale's ledger below
    preseed_trials = None
    if remaining:
        ledger = checkpoint_store.trials_path(remaining[0])
        if ledger.exists():
            preseed_trials = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
            if not preseed_trials:
                raise SystemExit(f"--resume-from: ledger {ledger} exists but is empty")
            warm_start_cfg = min(preseed_trials, key=lambda t: t["value"])["cfg"]
    if warm_start_cfg is None:
        raise SystemExit(
            f"--resume-from: nothing to resume at {checkpoint_store.root} — "
            f"no completed scales (scale_results.json) and no trial ledger for "
            f"{remaining[0] if remaining else task.scales}"
        )
    return remaining, warm_start_cfg, prior_save_path, prior_results, preseed_trials


def _validated_initial_params(task: TaskSpec) -> dict:
    """Validate the task's search contract and return its ``initial_params``.

    ``initial_params`` must cover every tuned hparam (the first scale's
    baseline runs exactly that config), ``pinned_params`` may not overlap
    ``hparams``, and ``depth`` must be tuned or pinned since the batch plan
    needs it in every trial cfg.
    """
    missing = sorted(set(task.hparams) - set(task.initial_params))
    if missing:
        raise SystemExit(
            f"Task '{task.name}': initial_params is missing tuned hparams "
            f"{missing}. Every name in `hparams` needs an explicit initial "
            f"value in the task config."
        )
    overlap = sorted(set(task.pinned_params) & set(task.hparams))
    if overlap:
        raise SystemExit(
            f"Task '{task.name}': {overlap} appear in both pinned_params and "
            f"hparams — a knob is either tuned or pinned, not both."
        )
    if "depth" not in set(task.hparams) | set(task.pinned_params):
        raise SystemExit(
            f"Task '{task.name}': depth must be tuned (hparams) or pinned "
            f"(pinned_params)."
        )
    return dict(task.initial_params)


def _emit_wandb_summary(
    *,
    task: TaskSpec,
    scale_results: list[dict],
    out_dir: Path,
) -> None:
    """One summary W&B run: per-scale results table, scaling-law fits +
    plots, tuner save files as an artifact."""
    import wandb

    from .scaling_laws import (
        fit_all_scaling_laws,
        format_scaling_law_table,
        plot_scaling_laws,
    )

    laws = fit_all_scaling_laws(scale_results)

    run = wandb.init(
        project=task.wandb_project,
        name=f"llmtuna_summary_{task.name}",
        config={
            "experiment": task.name,
            "scales": task.scales,
            "max_iters_per_scale": task.max_iters_per_scale,
            "patience": task.patience,
            "tol": task.tol,
            "llm_model": task.llm_model,
            "llm_thinking_budget": task.llm_thinking_budget,
            "nproc": task.nproc,
            "total_batch_tokens": task.total_batch_tokens,
            "max_seq_len": task.max_seq_len,
        },
    )

    try:
        hparam_names = sorted({
            k for r in scale_results if r.get("best_cfg") for k in r["best_cfg"].keys()
        })
        results_table = wandb.Table(
            columns=["scale_label", "best_value", "n_trials", *hparam_names],
            data=[
                [
                    r["scale_label"],
                    r["best_value"],
                    r["n_trials"],
                    *[(r["best_cfg"].get(k) if r.get("best_cfg") else None)
                      for k in hparam_names],
                ]
                for r in scale_results
            ],
        )
        run.log({"per_scale_results": results_table})

        if laws:
            laws_table = wandb.Table(
                columns=["name", "slope", "intercept", "r_squared"],
                data=[
                    [name, float(fit["slope"]), float(fit["intercept"]),
                     float(fit["r_squared"])]
                    for name, fit in laws.items()
                ],
            )
            run.log({"scaling_laws": laws_table})
            print("\n" + format_scaling_law_table(laws))

            plots_dir = out_dir / "plots"
            plot_paths = plot_scaling_laws(scale_results, laws, plots_dir)
            for p in plot_paths:
                run.log({p.stem: wandb.Image(str(p))})

        artifact = wandb.Artifact(
            f"llmtuna_study_{task.name}", type="llmtuna"
        )
        for p in out_dir.glob("tuner_*.json"):
            artifact.add_file(str(p))
        for extra in ("scale_results.json", "architecture_summary.txt"):
            p = out_dir / extra
            if p.exists():
                artifact.add_file(str(p))
        run.log_artifact(artifact)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
