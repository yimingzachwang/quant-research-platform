"""Return computation utilities for price series."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_returns(close: pd.Series) -> pd.Series:
    """Simple period-over-period returns: (p_t / p_{t-1}) - 1."""
    return close.pct_change()


def compute_log_returns(close: pd.Series) -> pd.Series:
    """Log returns: ln(p_t / p_{t-1})."""
    return np.log(close / close.shift(1))


def compute_cumulative_returns(returns: pd.Series) -> pd.Series:
    """Cumulative product of (1 + r_t) - 1, starting from zero at t=0."""
    return (1 + returns).cumprod() - 1
