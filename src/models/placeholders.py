"""Placeholder model training implementations."""

from __future__ import annotations

from typing import Any


class NoOpModel:
    """Model that returns features as placeholder predictions."""

    def predict(self, features: Any) -> Any:
        """Return placeholder forecasts."""
        return features


class NoOpModelTrainer:
    """Trainer that returns a no-op model without fitting."""

    def fit(self, features: Any, target: Any) -> NoOpModel:
        """Return a no-op model."""
        return NoOpModel()
