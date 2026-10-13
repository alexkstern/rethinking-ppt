"""One-scale orchestrator: builds the Tuner, runs baseline + suggest/observe loop.

``ScaleRunner.run(...)`` builds the search space and a fresh ``lt.Tuner``,
seeds its context, optionally evaluates ``warm_start_cfg`` as an unsuggested
baseline, then loops until ``max_iters`` trials exist over ``suggest -> validate ->
objective.run -> observe -> tracker.update``. A constraint failure adds a
rejection note to the context and re-``suggest()``s without observing
anything.

The runner is stateless across calls, so one instance runs every scale.
"""

from pathlib import Path
from typing import Callable

import llmtuna as lt

from .objective import ObjectiveRunner
from .validators import validate_constraints
from .checkpoint import CheckpointStore
from .context_builder import ContextBuilder
from .convergence import ConvergenceTracker
from .space import build_space


class ScaleRunner:
    """Orchestrates one FLOP scale of the optimization."""

    def __init__(
        self,
        *,
        provider: lt.Provider,
        objective: ObjectiveRunner,
        context_builder: ContextBuilder,
        system_prompt: str,
        patience: int,
        tol: float,
        checkpoint_store: CheckpointStore | None = None,
        max_iters: int = 30,
        max_rejections_per_trial: int = 5,
        format_proposal: Callable[[dict], str] | None = None,
    ):
        """``checkpoint_store=None`` means no on-disk state; ``max_iters``
        caps the trials per scale, baseline and any preseeded trials included; ``run()`` raises after
        ``max_rejections_per_trial`` consecutive constraint rejections.
        """
        if max_iters < 1:
            raise ValueError(f"max_iters must be >= 1, got {max_iters}")
        if max_rejections_per_trial < 0:
            raise ValueError(
                f"max_rejections_per_trial must be >= 0, got {max_rejections_per_trial}"
            )
        self.provider = provider
        self.objective = objective
        self.context_builder = context_builder
        self.system_prompt = system_prompt
        self.patience = patience
        self.tol = tol
        self.checkpoint_store = checkpoint_store
        self.max_iters = max_iters
        self.max_rejections_per_trial = max_rejections_per_trial
        self.format_proposal = format_proposal

    def run(
        self,
        *,
        task,
        scale_label: str,
        warm_start_cfg: dict | None = None,
        prior_save_path: Path | None = None,
        preseed_trials: list[dict] | None = None,
        architecture_summary: str | None = None,
        extra_context: str | None = None,
    ) -> dict:
        """Run one scale; return ``{scale_label, best_cfg, best_value,
        n_trials, all_trials, save_path}``.

        ``warm_start_cfg`` sets ``.initial`` on each Param (clamped to bounds)
        and runs as an unsuggested baseline trial first. ``prior_save_path``
        restores the prior scale's context entries verbatim;
        ``architecture_summary`` / ``extra_context`` apply on the first scale
        only. ``preseed_trials`` from a crashed scale's ledger are observed
        WITHOUT running, and suppress the baseline.
        """
        space = build_space(task.hparams, initials=warm_start_cfg,
                            overrides=task.hparam_overrides)
        tuner_kwargs: dict = dict(
            provider=self.provider,
            space=space,
            objective="minimize",
            system_prompt=self.system_prompt,
        )
        if self.format_proposal is not None:
            tuner_kwargs["format_proposal"] = self.format_proposal
        tuner = lt.Tuner(**tuner_kwargs)
        self.context_builder.seed(
            tuner,
            task=task,
            scale_label=scale_label,
            architecture_summary=architecture_summary,
            extra_context=extra_context,
            prior_save_path=prior_save_path,
        )

        tracker = ConvergenceTracker(patience=self.patience, tol=self.tol)

        # Every trial cfg is the task's pinned constants merged with the tuned
        # values — the one place cfgs are completed, so downstream code indexes
        # them unconditionally. The tracker is updated on the baseline too, so
        # `patience` counts non-improving PROPOSALS from there.
        for t in preseed_trials or []:
            tuner.observe(
                cfg={**task.pinned_params, **t["cfg"]},
                value=t["value"],
                note="completed before interruption (preseeded, not re-run)",
            )
            tracker.update(tuner)

        # With preseeds the warm start only seeds the space initials above;
        # the ledger already holds its result, so no baseline is re-run.
        if preseed_trials is None and warm_start_cfg is not None:
            cfg = {**task.pinned_params, **warm_start_cfg}
            value = self._run_and_record(cfg, scale_label, tuner)
            tuner.observe(
                cfg=cfg,
                value=value,
                note="baseline (warm-start from prior scale)",
            )
            tracker.update(tuner)

        while len(tuner.history) < self.max_iters:
            cfg = {**task.pinned_params, **self._propose_valid(tuner)}
            value = self._run_and_record(cfg, scale_label, tuner)
            tuner.observe(cfg=cfg, value=value)
            if tracker.update(tuner)["should_stop"]:
                break

        save_path = None
        if self.checkpoint_store is not None:
            save_path = self.checkpoint_store.save_tuner(tuner, scale_label)

        best = tuner.best
        return {
            "scale_label": scale_label,
            "best_cfg": best["cfg"] if best else None,
            "best_value": best["value"] if best else None,
            "n_trials": len(tuner.history),
            "all_trials": [
                {"cfg": dict(t.cfg), "value": t.value, "note": t.note}
                for t in tuner.history
            ],
            "save_path": str(save_path) if save_path else None,
        }

    def _run_and_record(self, cfg: dict, scale_label: str, tuner: lt.Tuner) -> float:
        """Score one trial and append it to the scale's crash-resume ledger."""
        value = self.objective.run(
            cfg=cfg, scale_label=scale_label, trial_id=len(tuner.history),
        )
        if self.checkpoint_store is not None:
            self.checkpoint_store.append_trial(scale_label, cfg, value)
        return value

    def _propose_valid(self, tuner: lt.Tuner) -> dict:
        """On constraint failure, add a REJECTED note to the context and
        re-``suggest()``, so the LLM can correct itself on the retry."""
        for _ in range(self.max_rejections_per_trial + 1):
            cfg = tuner.suggest()
            error = validate_constraints(cfg)
            if error is None:
                return cfg
            tuner.context.add(
                text=(
                    f"REJECTED proposal {cfg}: {error} "
                    f"Re-propose with values that satisfy the constraint."
                )
            )
        raise RuntimeError(
            f"ScaleRunner: constraint validator rejected "
            f"{self.max_rejections_per_trial + 1} consecutive proposals"
        )
