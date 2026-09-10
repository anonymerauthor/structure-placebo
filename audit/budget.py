"""Evaluation-budget wrapper.

Kept independent of any audited implementation so that a second audit target
can be run without pulling in the first one's package.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


class Budget:
    """Counts objective evaluations and stops the run at the cap."""

    __slots__ = ("f", "max_evals", "used", "best")

    def __init__(self, f: Callable[[np.ndarray], float], max_evals: int):
        self.f = f
        self.max_evals = int(max_evals)
        self.used = 0
        self.best = np.inf

    def __call__(self, x: np.ndarray) -> float:
        if self.used >= self.max_evals:
            return np.inf
        self.used += 1
        v = float(self.f(x))
        if v < self.best:
            self.best = v
        return v

    @property
    def exhausted(self) -> bool:
        return self.used >= self.max_evals
