"""Placeholder feature implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NoOpFeature:
    """Feature transform that returns input data unchanged."""

    name: str = "noop_feature"
    lookback_periods: int = 0

    def compute(self, data: Any) -> Any:
        """Return data unchanged."""
        return data


class NoOpFeaturePipeline:
    """Feature pipeline used before real feature engineering exists."""

    def fit(self, data: Any) -> NoOpFeaturePipeline:
        """Return the fitted pipeline placeholder."""
        return self

    def transform(self, data: Any) -> Any:
        """Return data unchanged."""
        return data
