"""Placeholder execution simulation implementations."""

from __future__ import annotations

from typing import Any


class NoOpTransactionCostModel:
    """Transaction cost model that returns no costs."""

    def estimate(self, orders: Any, market_data: Any) -> dict[str, float]:
        """Return an empty cost summary."""
        return {}


class NoOpExecutionSimulator:
    """Execution simulator that returns orders as placeholder fills."""

    def execute(self, orders: Any, market_data: Any) -> Any:
        """Return placeholder fills."""
        return orders
