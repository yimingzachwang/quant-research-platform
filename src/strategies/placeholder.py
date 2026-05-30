"""Strategy metadata placeholders."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategySpec:
    """Describes a strategy idea without implementing signal logic."""

    strategy_id: str
    hypothesis: str
    owner: str
    status: str = "scaffold"
