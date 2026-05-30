"""Multi-asset loading and temporal alignment utilities.

Entry points:
    load_universe(symbols, ...) -> dict[str, pd.DataFrame]
    align_prices(universe)      -> pd.DataFrame  (Date × Asset)
    align_returns(prices)       -> pd.DataFrame  (Date × Asset)

All functions are explicit and deterministic.  The caller decides the join
strategy; there are no silent drops or hidden fills.
"""

from __future__ import annotations

import pandas as pd

from src.data.loaders import load_dataset


def load_universe(
    symbols: list[str],
    frequency: str = "1d",
    source: str = "yfinance",
) -> dict[str, pd.DataFrame]:
    """Load one DataFrame per symbol using the existing dataset loader.

    Args:
        symbols: List of ticker symbols to load.
        frequency: Dataset frequency (default '1d').
        source: Data source key (default 'yfinance').

    Returns:
        Dict mapping symbol → raw DataFrame (as returned by load_dataset).
        Symbols that fail to load raise immediately — no silent skipping.
    """
    return {sym: load_dataset(symbol=sym, frequency=frequency, source=source)
            for sym in symbols}


def align_prices(
    universe: dict[str, pd.DataFrame],
    join: str = "inner",
    price_col: str = "close",
) -> pd.DataFrame:
    """Build a Date × Asset price DataFrame from a universe dict.

    Strips UTC timezone from the timestamp index so axes render cleanly.

    Args:
        universe: Output of load_universe().
        join: ``"inner"`` keeps only dates present in all assets (default);
              ``"outer"`` preserves all dates with NaN for missing assets.
        price_col: Column to extract from each raw DataFrame (default 'close').

    Returns:
        DataFrame with DatetimeIndex, one column per symbol.
    """
    if join not in {"inner", "outer"}:
        msg = f"join must be 'inner' or 'outer', got {join!r}"
        raise ValueError(msg)

    series: dict[str, pd.Series] = {}
    for sym, df in universe.items():
        s = _extract_price_series(df, price_col=price_col, name=sym)
        series[sym] = s

    prices = pd.concat(series, axis=1, join=join)
    prices.index.name = "date"
    return prices.sort_index()


def align_returns(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Compute simple period returns from an aligned price DataFrame.

    First row is always NaN (no prior price).  No silent filling.

    Args:
        prices: Date × Asset price DataFrame from align_prices().

    Returns:
        Date × Asset return DataFrame, same shape and index.
    """
    return prices.pct_change()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_price_series(
    df: pd.DataFrame,
    price_col: str,
    name: str,
) -> pd.Series:
    """Convert a raw loaded DataFrame into a tz-naive DatetimeIndex Series."""
    if "timestamp" in df.columns:
        s = df.set_index("timestamp")[price_col]
    elif isinstance(df.index, pd.DatetimeIndex):
        s = df[price_col]
    else:
        msg = f"Cannot determine DatetimeIndex for symbol '{name}'"
        raise ValueError(msg)

    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)

    s.name = name
    return s.sort_index()
