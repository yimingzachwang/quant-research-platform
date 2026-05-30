"""Shared domain objects used across platform boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any


@dataclass(frozen=True)
class Universe:
    """A tradable asset universe with an explicit name and membership."""

    name: str
    symbols: tuple[str, ...]
    description: str | None = None


@dataclass(frozen=True)
class Horizon:
    """Prediction or rebalance horizon expressed in trading sessions."""

    name: str
    periods: int


@dataclass(frozen=True)
class DateRange:
    """Inclusive research date range."""

    start: date
    end: date


@dataclass(frozen=True)
class ExperimentContext:
    """Metadata that should travel with every research run."""

    experiment_id: str
    name: str
    created_at: datetime
    universe: Universe
    horizon: Horizon
    date_range: DateRange
    config: Mapping[str, Any] = field(default_factory=dict)
