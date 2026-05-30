"""Lightweight research-safe data cleaning utilities.

Philosophy: detect aggressively, modify conservatively.
All functions are explicit, deterministic, and reversible.
"""

from src.cleaning.missing import forward_fill_limited
from src.cleaning.numeric import replace_inf
from src.cleaning.timestamps import remove_duplicate_timestamps, sort_time_index
from src.cleaning.validation import OHLCVValidationResult, validate_ohlcv

__all__ = [
    "sort_time_index",
    "remove_duplicate_timestamps",
    "replace_inf",
    "forward_fill_limited",
    "validate_ohlcv",
    "OHLCVValidationResult",
]
