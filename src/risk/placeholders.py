"""Placeholder risk analysis implementations."""

from __future__ import annotations

from typing import Any

from src.risk.interfaces import RiskReport


class NoOpRiskAnalyzer:
    """Risk analyzer that records no breaches."""

    def analyze(self, portfolio: Any, market_data: Any) -> RiskReport:
        """Return an empty risk report."""
        return RiskReport()
