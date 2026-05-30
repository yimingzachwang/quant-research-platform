"""Feature engineering contracts."""

from __future__ import annotations

from typing import Any, Protocol


class Feature(Protocol):
    """A deterministic feature transformation."""

    name: str
    lookback_periods: int

    def compute(self, data: Any) -> Any:
        """Compute a feature matrix from aligned input data."""
        raise NotImplementedError


class FeaturePipeline(Protocol):
    """Ordered collection of feature transformations."""

    def fit(self, data: Any) -> FeaturePipeline:
        """Fit any stateful feature transforms on training data only."""
        raise NotImplementedError

    def transform(self, data: Any) -> Any:
        """Transform data into a feature matrix with leakage-safe alignment."""
        raise NotImplementedError
