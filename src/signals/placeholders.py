"""Placeholder signal generation implementations."""

from __future__ import annotations

from typing import Any


class NoOpSignalGenerator:
    """Signal generator that returns features unchanged."""

    def generate(self, features: Any) -> Any:
        """Return placeholder signals."""
        return features
