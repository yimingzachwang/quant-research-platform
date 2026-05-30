"""Research-safe OHLCV validation — detect aggressively, modify conservatively."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

_OHLCV_COLUMNS = {"open", "high", "low", "close", "volume"}


@dataclass
class OHLCVValidationResult:
    """Summary of OHLCV integrity checks.

    All issues are collected before raising so callers see the full picture.
    """

    missing_columns: list[str] = field(default_factory=list)
    negative_prices: dict[str, int] = field(default_factory=dict)
    negative_volume: int = 0
    high_lt_low: int = 0
    high_lt_close: int = 0
    high_lt_open: int = 0
    low_gt_close: int = 0
    low_gt_open: int = 0
    nan_counts: dict[str, int] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Return True only when no issues were detected."""
        return (
            not self.missing_columns
            and not any(self.negative_prices.values())
            and self.negative_volume == 0
            and self.high_lt_low == 0
            and self.high_lt_close == 0
            and self.high_lt_open == 0
            and self.low_gt_close == 0
            and self.low_gt_open == 0
        )

    def summary(self) -> str:
        lines = []
        if self.missing_columns:
            lines.append(f"missing columns: {self.missing_columns}")
        for col, n in self.negative_prices.items():
            if n:
                lines.append(f"{col} negative prices: {n} rows")
        if self.negative_volume:
            lines.append(f"negative volume: {self.negative_volume} rows")
        if self.high_lt_low:
            lines.append(f"high < low: {self.high_lt_low} rows")
        if self.high_lt_close:
            lines.append(f"high < close: {self.high_lt_close} rows")
        if self.high_lt_open:
            lines.append(f"high < open: {self.high_lt_open} rows")
        if self.low_gt_close:
            lines.append(f"low > close: {self.low_gt_close} rows")
        if self.low_gt_open:
            lines.append(f"low > open: {self.low_gt_open} rows")
        for col, n in self.nan_counts.items():
            if n:
                lines.append(f"{col} NaNs: {n} rows")
        return "; ".join(lines) if lines else "OK"


def validate_ohlcv(
    df: pd.DataFrame,
    raise_on_error: bool = True,
) -> OHLCVValidationResult:
    """Run OHLCV integrity checks and return a structured result.

    Column names are matched case-insensitively.

    Args:
        df: DataFrame expected to contain open, high, low, close, volume columns.
        raise_on_error: Raise ValueError when any structural issue is found.
    """
    cols = {c.lower(): c for c in df.columns}
    result = OHLCVValidationResult()

    missing = [c for c in _OHLCV_COLUMNS if c not in cols]
    result.missing_columns = missing
    if missing:
        if raise_on_error:
            raise ValueError(f"OHLCV validation failed: {result.summary()}")
        return result

    o = df[cols["open"]]
    h = df[cols["high"]]
    lo = df[cols["low"]]
    c = df[cols["close"]]
    v = df[cols["volume"]]

    for label, series in [("open", o), ("high", h), ("low", lo), ("close", c)]:
        n = int((series < 0).sum())
        if n:
            result.negative_prices[label] = n

    result.negative_volume = int((v < 0).sum())
    result.high_lt_low = int((h < lo).sum())
    result.high_lt_close = int((h < c).sum())
    result.high_lt_open = int((h < o).sum())
    result.low_gt_close = int((lo > c).sum())
    result.low_gt_open = int((lo > o).sum())

    result.nan_counts = {
        col: int(df[cols[col]].isna().sum())
        for col in ("open", "high", "low", "close", "volume")
    }

    if raise_on_error and not result.is_valid:
        raise ValueError(f"OHLCV validation failed: {result.summary()}")

    return result
