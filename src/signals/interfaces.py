"""Signal generation contracts."""

from __future__ import annotations

from typing import Any, Protocol


class SignalGenerator(Protocol):
    """Converts model outputs or rules into tradable research signals."""

    def generate(self, features: Any) -> Any:
        """Return signal values indexed by timestamp and symbol."""
        raise NotImplementedError
