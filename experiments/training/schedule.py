"""Interleave schedule: pure-PPT prefix, then constant-rate code->text mixing.

Pure arithmetic (no torch) behind the loader in
``nanochat.data.bin.build_interleaved_bin_loader``. It reorders an abrupt
two-phase run while consuming the SAME token budget from each corpus:
``pure_steps`` optimizer steps of pure PPT, then a mixing region spanning the
rest of the run in which each batch draws a constant share of PPT sequences.

``pure_frac`` (share of the PPT budget spent in the pure prefix) is the only
knob; ``pure_frac=1`` leaves nothing to mix in, recovering the abrupt baseline.

Integer per-step counts come from rounding the *cumulative* ideal curve, never
the per-step ideal: per-step rounding drifts, while telescoping differences of
a rounded cumulative hit the total exactly.
"""

from __future__ import annotations

from dataclasses import dataclass


def _round_half_up(x: float) -> int:
    """Python's ``round`` half-rounds to even."""
    return int(x + 0.5)


@dataclass(frozen=True)
class InterleavePlan:
    """Step budgets for one interleaved run.

    All counts are in optimizer steps. Phase-2-local step ``j`` is 0-based and
    runs over ``pt_steps``; the cumulative-code functions clamp at the end, so
    reading one step past the horizon (the train loop prefetches a batch)
    gives pure text rather than an error.
    """

    seqs_per_step: int    # global sequences per optimizer step (batch * world)
    pure_steps: int       # phase 1: pure-PPT steps
    ppt_steps_total: int  # steps the whole PPT budget is worth
    total_steps: int      # whole run (== abrupt baseline's total)

    @property
    def pt_steps(self) -> int:
        """Phase 2's step count, which the mixing region spans."""
        return self.total_steps - self.pure_steps

    def _ideal_code_seqs_before(self, j: int) -> float:
        """Spreads the leftover PPT budget at a constant rate over phase 2."""
        if self.pt_steps == 0:
            return 0.0
        j = min(j, self.pt_steps)
        leftover = (self.ppt_steps_total - self.pure_steps) * self.seqs_per_step
        return leftover * j / self.pt_steps

    def code_seqs_before(self, j: int) -> int:
        """Code sequences consumed before phase-2 step ``j`` (0-based)."""
        return _round_half_up(self._ideal_code_seqs_before(j))

    def code_count_at(self, j: int) -> int:
        """Code sequences in phase-2 step ``j``'s global batch."""
        return self.code_seqs_before(j + 1) - self.code_seqs_before(j)


def plan_interleave(
    *,
    ppt_tokens: int,
    pt_tokens: int,
    pure_frac: float,
    tokens_per_step: int,
    seq_len: int,
) -> InterleavePlan:
    """Build the plan from corpus token budgets and the ``pure_frac`` dial.

    ``ppt_tokens`` / ``pt_tokens`` are the abrupt-baseline run's per-corpus
    budgets. Both must divide ``tokens_per_step`` exactly: the plan reorders
    whole optimizer steps, and a remainder would change what "same tokens
    seen" means. ``pure_frac=0`` is single-phase, mixed from the very first
    batch (no PPT checkpoint to transfer).
    """
    if tokens_per_step % seq_len != 0:
        raise ValueError(f"{tokens_per_step=} not a multiple of {seq_len=}")
    if ppt_tokens % tokens_per_step != 0:
        raise ValueError(f"{ppt_tokens=} not a multiple of {tokens_per_step=}")
    if pt_tokens % tokens_per_step != 0:
        raise ValueError(f"{pt_tokens=} not a multiple of {tokens_per_step=}")
    if not 0.0 <= pure_frac <= 1.0:
        raise ValueError(f"{pure_frac=} outside [0, 1]")

    seqs_per_step = tokens_per_step // seq_len
    ppt_steps_total = ppt_tokens // tokens_per_step
    total_steps = ppt_steps_total + pt_tokens // tokens_per_step
    pure_steps = _round_half_up(pure_frac * ppt_steps_total)
    plan = InterleavePlan(
        seqs_per_step=seqs_per_step,
        pure_steps=pure_steps,
        ppt_steps_total=ppt_steps_total,
        total_steps=total_steps,
    )
    # Conservation: the ramp must deliver exactly the leftover PPT budget.
    leftover = (ppt_steps_total - pure_steps) * seqs_per_step
    assert plan.code_seqs_before(plan.pt_steps) == leftover, (
        plan.code_seqs_before(plan.pt_steps),
        leftover,
    )
    return plan
