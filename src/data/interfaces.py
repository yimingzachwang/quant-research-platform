"""Data layer contracts."""

from __future__ import annotations

from typing import Any, Protocol

from src.core import DateRange, Universe


class MarketDataSource(Protocol):
    """Source capable of loading market data for a universe and date range."""

    def load_prices(self, universe: Universe, date_range: DateRange) -> Any:
        """Return price data with explicit timestamp and symbol alignment."""
        raise NotImplementedError


class DataCatalog(Protocol):
    """Registry for datasets and their metadata."""

    def get_dataset(self, name: str) -> Any:
        """Return a named dataset or dataset descriptor."""
        raise NotImplementedError

    def validate_dataset(self, name: str) -> None:
        """Validate schema, coverage, freshness, and known leakage risks."""
        raise NotImplementedError
