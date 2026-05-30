"""Schema placeholders for market and reference datasets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PriceSchema:
    """Expected fields for daily ETF price bars."""

    timestamp: str = "timestamp"
    symbol: str = "symbol"
    open: str = "open"
    high: str = "high"
    low: str = "low"
    close: str = "close"
    adjusted_close: str = "adjusted_close"
    volume: str = "volume"


@dataclass(frozen=True)
class DatasetManifest:
    """Metadata required for reproducible dataset use."""

    name: str
    version: str
    source: str
    description: str
