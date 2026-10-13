"""Search space definitions for llmtuna-based hyperparameter optimization.

Each hparam is an ``lt.Float``, ``lt.Int``, or ``lt.Choice``. A TaskSpec lists
which ones to tune via ``hparams``; the space is built dynamically from the
registry. Registry entries carry no initial values — warm starts come from the
task's ``initial_params`` (first scale) or the prior scale's best config.
"""

import copy
from typing import Any

import llmtuna as lt


HPARAM_REGISTRY: dict[str, lt.Float | lt.Int | lt.Choice] = {
    "depth": lt.Int(
        name="depth",
        description=(
            "Number of transformer layers. Width (d_model) scales with depth "
            "via d_model = round_up(depth * 64, 128). Deeper models have more "
            "parameters but see fewer tokens per FLOP budget."
        ),
        bounds=(1, 26),
    ),
    "matrix_lr": lt.Float(
        name="matrix_lr",
        description=(
            "Muon learning rate for matrix (2D linear) params — maps to "
            "cfg.optim.matrix_lr."
        ),
        bounds=(1e-6, 1.0),
    ),
    "embedding_lr": lt.Float(
        name="embedding_lr",
        description=(
            "AdamW learning rate for token embeddings (wte) — maps to "
            "cfg.optim.embedding_lr."
        ),
        bounds=(1e-6, 5.0),
    ),
    "grad_clip": lt.Float(
        name="grad_clip",
        description=(
            "Maximum gradient norm for clipping. 0 disables clipping."
        ),
        bounds=(0.0, 10.0),
    ),
    "warmdown_ratio": lt.Float(
        name="warmdown_ratio",
        description=(
            "Fraction of training spent in the warmdown (linear decay to 0) "
            "phase. CONSTRAINT: warmup_ratio + warmdown_ratio must be <= 1.0 "
            "(the remaining fraction is the constant-LR plateau). Proposals "
            "violating this will be rejected."
        ),
        bounds=(0.0, 1.0),
    ),
    "warmup_ratio": lt.Float(
        name="warmup_ratio",
        description=(
            "Fraction of PT training spent warming up the learning rate from 0 "
            "to full LR. 0.0 = no warmup (jump to full LR). CONSTRAINT: "
            "warmup_ratio + warmdown_ratio must be <= 1.0. Proposals "
            "violating this will be rejected."
        ),
        bounds=(0.0, 0.5),
    ),
    "alpha_ppt": lt.Float(
        name="alpha_ppt",
        description=(
            "PPT budget as a fraction of target_flops — how much synthetic "
            "pre-pretraining to run before the FineWeb PT phase. This compute "
            "comes out of the same budget, so a larger dose leaves fewer "
            "FLOPs for natural language."
        ),
        bounds=(0.001, 0.5),
    ),
    "ppt_lr": lt.Float(
        name="ppt_lr",
        description=(
            "PPT-phase learning rate (separate-schedule PPT runs only) — maps to cfg.ppt_lr, "
            "which ppt_phase_cfg copies into all optim LRs for the PPT call. "
            "C4 uses a unified schedule and does not tune this."
        ),
        bounds=(1e-6, 1.0),
    ),
    "ppt_warmup_ratio": lt.Float(
        name="ppt_warmup_ratio",
        description=(
            "PPT-phase LR warmup fraction (separate-schedule PPT runs only). "
            "CONSTRAINT: ppt_warmup_ratio + ppt_warmdown_ratio must be <= 1.0."
        ),
        bounds=(0.0, 0.5),
    ),
    "ppt_warmdown_ratio": lt.Float(
        name="ppt_warmdown_ratio",
        description=(
            "PPT-phase LR warmdown fraction (separate-schedule PPT runs only). "
            "CONSTRAINT: ppt_warmup_ratio + ppt_warmdown_ratio must be <= 1.0."
        ),
        bounds=(0.0, 1.0),
    ),
    "ppt_batch_mult": lt.Choice(
        name="ppt_batch_mult",
        description=(
            "PPT-phase effective batch as a multiple of the PT batch "
            "(262,144 tokens/step). The PPT FLOP budget is fixed, so a larger "
            "multiple means the same total PPT tokens in fewer, bigger, "
            "less-noisy optimizer steps, and a smaller multiple means more, "
            "smaller, noisier steps. Search both directions. Memory is "
            "unaffected (absorbed by grad-accum); at small PPT budgets a "
            "large multiple leaves very few optimizer steps. PT batch is "
            "unchanged."
        ),
        options=[0.25, 0.5, 1, 2, 4],
    ),
    "pt_batch_mult": lt.Choice(
        name="pt_batch_mult",
        description=(
            "PT-phase effective batch as a multiple of the base batch (262,144 "
            "tokens/step). The PT token budget is fixed, so a SMALLER multiple "
            "means the same tokens in MORE, smaller optimizer steps (more "
            "updates), and a larger multiple means fewer, bigger steps. Search "
            "both directions. Memory is unaffected "
            "(absorbed by grad-accum); the PPT batch is set separately by "
            "ppt_batch_mult."
        ),
        options=[0.125, 0.25, 0.5, 1, 2],
    ),
}


def build_space(
    hparam_names: list[str],
    *,
    initials: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> list[lt.Float | lt.Int | lt.Choice]:
    """Build a list of lt params from hparam names, with optional warm-start
    initials. Registry entries are deep-copied, so warm-start ``.initial``
    overrides never bleed back into ``HPARAM_REGISTRY``.

    ``initials`` values outside an hparam's ``bounds`` are clamped, and names
    it carries that this scale does not tune are ignored. ``overrides`` is
    consulted before the registry, letting a task replace or add an entry.
    """
    initials = initials or {}
    overrides = overrides or {}
    params: list[lt.Float | lt.Int | lt.Choice] = []
    for name in hparam_names:
        if name not in overrides and name not in HPARAM_REGISTRY:
            available = ", ".join(sorted({*HPARAM_REGISTRY, *overrides}))
            raise ValueError(f"Unknown hparam '{name}'. Available: {available}")
        p = copy.deepcopy(overrides[name] if name in overrides else HPARAM_REGISTRY[name])
        if name in initials:
            val = initials[name]
            if getattr(p, "bounds", None) is not None:
                lo, hi = p.bounds
                val = max(lo, min(hi, val))
            p.initial = val
        params.append(p)
    return params