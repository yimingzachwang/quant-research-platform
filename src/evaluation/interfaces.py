"""Evaluation layer contracts."""

from __future__ import annotations

from typing import Any, Protocol


class Evaluator(Protocol):
    """Computes metrics and diagnostics for research artifacts."""

    def evaluate(self, backtest_result: Any) -> Any:
        """Return structured performance, risk, and validation diagnostics."""
        raise NotImplementedError
