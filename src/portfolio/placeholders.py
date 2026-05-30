"""Placeholder portfolio construction implementations."""

from __future__ import annotations

from typing import Any

from src.portfolio.interfaces import PortfolioConstraints


class NoOpPortfolioConstructor:
    """Portfolio constructor that returns signals as placeholder targets."""

    def construct(self, signals: Any, constraints: PortfolioConstraints) -> Any:
        """Return placeholder target weights."""
        return signals
