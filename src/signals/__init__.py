"""Signal generation interfaces and placeholders."""

from src.signals.interfaces import SignalGenerator
from src.signals.placeholders import NoOpSignalGenerator

__all__ = ["NoOpSignalGenerator", "SignalGenerator"]
