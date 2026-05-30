"""Risk analysis contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class RiskReport:
    """Structured risk diagnostics."""

    metrics: dict[str, float] = field(default_factory=dict)
    breaches: tuple[str, ...] = ()
    artifacts: dict[str, Any] = field(default_factory=dict)


class RiskAnalyzer(Protocol):
    """Computes ex-ante and ex-post portfolio risk diagnostics."""

    def analyze(self, portfolio: Any, market_data: Any) -> RiskReport:
        """Return structured risk diagnostics."""
        raise NotImplementedError
