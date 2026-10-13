"""Patience/tolerance convergence check for the per-scale suggest loop.

Call ``update(tuner)`` once immediately after every ``tuner.observe(...)``.
Current best and best cfg are read from ``lt.Tuner``, not tracked here.
"""

import llmtuna as lt


class ConvergenceTracker:
    """Patience/tol state for one optimization scale. Minimize-only."""

    def __init__(self, patience: int, tol: float):
        """Stop after ``patience`` consecutive trials that fail to improve the
        best value by more than ``tol``."""
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if tol < 0:
            raise ValueError(f"tol must be >= 0, got {tol}")
        self.patience = patience
        self.tol = tol
        self._prev_best_value: float = float("inf")
        self.no_improve: int = 0

    def update(self, tuner: lt.Tuner) -> dict:
        """Record the effect of the most recent ``tuner.observe(...)``."""
        current = tuner.best
        current_value = current["value"]
        if self._prev_best_value - current_value > self.tol:
            just_improved = True
            self.no_improve = 0
            self._prev_best_value = current_value
        else:
            just_improved = False
            self.no_improve += 1

        return {
            "best_value": current_value,
            "best_cfg": dict(current["cfg"]),
            "just_improved": just_improved,
            "no_improve": self.no_improve,
            "should_stop": self.no_improve >= self.patience,
        }
