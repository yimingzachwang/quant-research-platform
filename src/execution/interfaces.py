"""Execution layer contracts."""

from __future__ import annotations

from typing import Any, Protocol


class TransactionCostModel(Protocol):
    """Estimates transaction costs under realistic ETF trading assumptions."""

    def estimate(self, orders: Any, market_data: Any) -> Any:
        """Return estimated costs for proposed orders."""
        raise NotImplementedError


class ExecutionSimulator(Protocol):
    """Simulates order fills and portfolio transitions."""

    def execute(self, orders: Any, market_data: Any) -> Any:
        """Return simulated fills and execution diagnostics."""
        raise NotImplementedError
