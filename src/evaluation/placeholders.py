"""Placeholder evaluation implementations."""

from __future__ import annotations

from typing import Any


class NoOpEvaluator:
    """Evaluator that returns empty diagnostics."""

    def evaluate(self, backtest_result: Any) -> dict[str, float]:
        """Return empty metrics until real evaluation is implemented."""
        return {}
