"""Seeds a fresh per-scale ``lt.Tuner.context`` with the right framing.

Each scale runs on its own fresh ``lt.Tuner``, so ``tuner.best`` and the
convergence tracker are scale-local; only the context differs across scales.
The first scale gets an architecture summary (from ``lt.Context.add_summary``)
plus extra context, framing and a scale header. Later scales restore the prior
save's entries as structured ``Entry`` objects.

One ``ContextBuilder`` instance serves every scale.
"""

import json
from pathlib import Path
from typing import Optional

import llmtuna as lt


class ContextBuilder:
    """Fills an ``lt.Tuner.context`` for one optimization scale."""

    def __init__(
        self,
        *,
        provider: Optional[lt.Provider] = None,
        summarize_paths: Optional[list[str]] = None,
        summarize_hparam_names: Optional[list[str]] = None,
        summarize_system_prompt: Optional[str] = None,
        summarize_max_tokens: int = 1500,
        optimizer_description: str,
    ):
        """``provider=None`` disables automatic summarization: the caller must
        then pass ``architecture_summary=`` to ``seed()`` or the first scale's
        context has no summary. ``optimizer_description`` is a descriptive
        label for the framing preamble.
        """
        self.provider = provider
        self.summarize_paths = summarize_paths
        self.summarize_hparam_names = summarize_hparam_names
        self.summarize_system_prompt = summarize_system_prompt
        self.summarize_max_tokens = summarize_max_tokens
        self.optimizer_description = optimizer_description

    def seed(
        self,
        tuner: lt.Tuner,
        *,
        task,
        scale_label: str,
        architecture_summary: Optional[str] = None,
        extra_context: Optional[str] = None,
        prior_save_path: Optional[Path] = None,
    ) -> None:
        """Populate ``tuner.context`` for this scale.

        First scale (``prior_save_path=None``): architecture summary +
        ``extra_context`` + framing + scale header. Later scales restore the
        prior save's context entries verbatim, then add the new header.
        """
        if prior_save_path is None:
            if architecture_summary is not None:
                tuner.context.add(
                    text=f"## Architecture\n{architecture_summary}"
                )
            elif self.provider is not None and self.summarize_paths:
                tuner.context.add_summary(
                    provider=self.provider,
                    paths=self.summarize_paths,
                    hparam_names=self.summarize_hparam_names,
                    max_tokens=self.summarize_max_tokens,
                    system_prompt=self.summarize_system_prompt,
                )
            if extra_context:
                tuner.context.add(text=extra_context)
            tuner.context.add(
                text=self._framing(task, optimizer_description=self.optimizer_description)
            )
        else:
            prior_data = json.loads(Path(prior_save_path).read_text())
            tuner.context = lt.Context.from_dict(data=prior_data["context"])

        tuner.context.add(text=self._scale_header(scale_label))

    @staticmethod
    def _framing(task, *, optimizer_description: str) -> str:
        tuned = set(task.hparams)

        preamble = (
            "## Optimization setup\n"
            f"Study: {task.name} — {task.description.rstrip('.')}.\n"
            f"Optimizer: {optimizer_description}.\n"
            "Objective: minimize val_bpb (validation bits-per-byte). "
            "Lower is better.\n"
            "Optimal hparams can shift substantially between FLOP scales "
            "— do not assume the best config at one scale transfers to "
            "the next."
        )

        guidance: list[str] = []

        if "depth" in tuned:
            guidance.append(
                "`depth` is the most important hparam and has a convex "
                "optimum in val_bpb at each scale. Bracket that optimum with "
                "SMALL local moves (1-3 layers) around your first proposed "
                "depth, then pin it — do not keep jumping depth around or "
                "spend the whole budget on it. The learning-rate and schedule "
                "knobs hold the remaining gains once depth is bracketed."
            )


        if "ppt_lr" in tuned:
            guidance.append(
                "This study pre-pretrains on a synthetic corpus, then trains "
                "the real target. The PPT-phase knobs — `ppt_lr` (its learning "
                "rate), the PPT batch multiple, and the PPT schedule ratios "
                "(warmup / warmdown) — are important here, roughly as "
                "important as `depth`: together they set how the synthetic "
                "phase reshapes the weights the target phase inherits. Do NOT "
                "leave any of them at its initial value — actively search each, "
                "across a WIDE range and in BOTH directions (a `ppt_lr` both "
                "well below AND up toward the PT `matrix_lr`; smaller AND larger "
                "PPT batches; shorter AND longer warmdown). A gentler PPT LR is "
                "often a productive region to probe, but confirm it by also "
                "testing hotter values — do not assume the optimum, map it."
            )

        if not guidance:
            return preamble
        return preamble + "\n\n## Priority guidance\n" + "\n\n".join(guidance)

    @staticmethod
    def _scale_header(scale_label: str) -> str:
        bar = "=" * 60
        return f"\n{bar}\n=== CURRENT SCALE: {scale_label} FLOPs ===\n{bar}"
