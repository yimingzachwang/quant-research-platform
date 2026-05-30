"""Abstract base class for portfolio strategies.

A Strategy is a pure function: prices in → weights out.
No plotting, no file I/O, no reporting.

The single required method is generate_weights().  The caller (runner)
is responsible for passing data, executing the backtest, and saving results.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class Strategy(ABC):
    """Minimal contract for portfolio weight generation.

    Subclasses receive a Date × Asset close price DataFrame and return a
    Date × Asset weight DataFrame.  The backtest engine handles the
    look-ahead prevention (weights.shift(1)) — strategies must not apply
    any lag themselves.

    Weights:
        - Index: DatetimeIndex aligned to the price dates.
        - Columns: asset identifiers matching the price DataFrame.
        - Values: portfolio weights.  Long-only strategies produce values
          in [0, 1] with rows summing to ≤ 1.  Rows may sum to 0 when
          the strategy is flat (e.g., during warm-up).

    No weight normalisation is enforced here; the portfolio backtest
    engine accepts any weight scale.
    """

    @abstractmethod
    def generate_weights(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute portfolio weights from price history.

        Args:
            data: Date × Asset close price DataFrame, DatetimeIndex,
                no NaN (caller is responsible for cleaning).

        Returns:
            Date × Asset weight DataFrame, same index as ``data``.
        """

    @property
    def name(self) -> str:
        """Human-readable strategy identifier."""
        return self.__class__.__name__

    def params(self) -> dict[str, Any]:
        """Return strategy parameters as a serialisable dict.

        Override in subclasses to expose hyperparameters.  Used by the
        experiment system to record what was run.
        """
        return {}
