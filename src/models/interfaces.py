"""Modelling layer contracts."""

from __future__ import annotations

from typing import Any, Protocol


class Model(Protocol):
    """Predictive model used by a signal generator."""

    def predict(self, features: Any) -> Any:
        """Return forecasts for aligned feature rows."""
        raise NotImplementedError


class ModelTrainer(Protocol):
    """Fits models using an explicit validation scheme."""

    def fit(self, features: Any, target: Any) -> Model:
        """Train and return a model artifact."""
        raise NotImplementedError


class ValidationScheme(Protocol):
    """Defines train, validation, and test splits."""

    def split(self, data: Any) -> Any:
        """Return leakage-aware temporal splits."""
        raise NotImplementedError
