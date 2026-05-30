"""Portfolio construction contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class PortfolioConstraints:
    """High-level constraints used when converting signals to targets."""

    max_position_weight: float | None = None
    max_gross_exposure: float | None = None
    max_turnover: float | None = None
    allow_shorting: bool = False


class PortfolioConstructor(Protocol):
    """Converts signals into portfolio targets."""

    def construct(self, signals: Any, constraints: PortfolioConstraints) -> Any:
        """Return target weights or positions."""
        raise NotImplementedError
