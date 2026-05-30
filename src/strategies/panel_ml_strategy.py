"""Cross-sectional panel ML strategy adapter.

Adapts a BaseMLModel into the Strategy interface for multi-asset cross-sectional
research.  Unlike MLStrategy (single-asset time-series), PanelMLStrategy:

  - Applies the same feature spec to EVERY asset in the universe.
  - Pools all (date, asset) observations into one training matrix.
  - Fits one shared linear model on pooled cross-sectional data.
  - Predicts a score for each (date, asset) pair.
  - Selects the top-N assets by predicted score via equal weighting.

Walk-forward integration
------------------------
run_walk_forward_validation calls strategy.fit(train_prices) before each test
window — PanelMLStrategy implements this hook identically to MLStrategy.

Coefficient semantics in panel mode
-------------------------------------
Coefficients represent how each feature predicts cross-sectional return ranks
across all assets, not time-series returns for a single asset.  A positive
momentum coefficient means high momentum assets tend to outperform in the
cross-section — not that momentum predicts SPY specifically.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from src.ml.contracts import PredictionSeries
from src.strategies.base import Strategy


class PanelMLStrategy(Strategy):
    """Cross-sectional panel ML strategy.

    Pools all (date, asset) feature observations across the universe,
    fits one shared model on cross-sectional ranking labels, and selects
    the top-N assets by predicted score at each date.

    Args:
        model: Any object satisfying the BaseMLModel Protocol.
        tickers: Ordered list of universe tickers.
        feature_fn_builder: Callable(ticker: str) → {name: fn(prices) → pd.Series}.
            Called for each asset to build per-asset feature functions.
        label_fn: Callable(prices: pd.DataFrame) → pd.DataFrame (Date × Asset)
            of cross-sectional ranking labels (e.g., from ranking_target()).
        horizon: Forward horizon in periods; stored for provenance.
        signal_fn: Callable(PredictionSeries) → pd.DataFrame (Date × Asset).
            Converts panel prediction scores to portfolio weights.
            Typically top_n_weights from src.ml.signals.
        label_name: Label name stored for provenance; default "ranking_target".
    """

    def __init__(
        self,
        model: Any,
        tickers: list[str],
        feature_fn_builder: Callable[[str], dict[str, Callable]],
        label_fn: Callable[[pd.DataFrame], pd.DataFrame],
        horizon: int,
        signal_fn: Callable[[PredictionSeries], pd.DataFrame],
        label_name: str = "ranking_target",
    ) -> None:
        self._model = model
        self._tickers = list(tickers)
        self._feature_fn_builder = feature_fn_builder
        self._label_fn = label_fn
        self._horizon = horizon
        self._signal_fn = signal_fn
        self._label_name = label_name
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Predict cross-sectional scores for all assets; return top-N weights.

        Steps:
            1. Build panel feature matrix (MultiIndex(date, asset) × features).
            2. Drop NaN rows.
            3. Predict score per (date, asset) via the fitted model.
            4. Reshape predictions to Date × Asset DataFrame.
            5. Apply signal_fn (typically top_n_weights) → Date × Asset weights.
            6. Reindex to full price index; warm-up rows are flat (0).

        Args:
            prices: Date × Asset close price DataFrame.

        Returns:
            Date × Asset weight DataFrame compatible with run_portfolio_backtest.

        Raises:
            RuntimeError: If called before fit().
        """
        if not self._is_fitted:
            raise RuntimeError(
                f"PanelMLStrategy({self._model.name}) must be fit() before "
                "generate_weights()."
            )

        from src.ml.panel import build_panel_feature_matrix

        X_panel = build_panel_feature_matrix(prices, self._feature_fn_builder, self._tickers)
        X_clean = X_panel.dropna()

        if X_clean.empty:
            return pd.DataFrame(0.0, index=prices.index, columns=self._tickers)

        predictions = self._model.predict(X_clean)
        # predictions.values is pd.Series with MultiIndex(date, asset)
        pred_scores = predictions.values
        if not isinstance(pred_scores, pd.Series):
            return pd.DataFrame(0.0, index=prices.index, columns=self._tickers)

        # Reshape MultiIndex Series to Date × Asset DataFrame
        try:
            pred_df = pred_scores.unstack(level="asset")
        except Exception:
            return pd.DataFrame(0.0, index=prices.index, columns=self._tickers)

        panel_preds = PredictionSeries(
            values=pred_df,
            label_name=self._label_name,
            model_name=self._model.name,
        )
        weights = self._signal_fn(panel_preds)

        # Reindex to the full price index; warm-up rows become flat (0)
        return weights.reindex(prices.index, fill_value=0.0).reindex(
            columns=self._tickers, fill_value=0.0
        )

    # ------------------------------------------------------------------
    # Optional fit hook — matched by run_walk_forward_validation
    # ------------------------------------------------------------------

    def fit(self, train_prices: pd.DataFrame) -> None:
        """Pool panel features + cross-sectional labels and fit the model.

        For each ticker in the universe, builds the feature matrix using
        feature_fn_builder(ticker).  Stacks all (date, asset) observations
        into a pooled matrix, aligns with cross-sectional ranking labels,
        and fits the model once on the pooled dataset.

        Args:
            train_prices: Date × Asset price DataFrame for the training window.
        """
        from src.ml.datasets import build_supervised_dataset
        from src.ml.panel import build_panel_feature_matrix

        X_panel = build_panel_feature_matrix(
            train_prices, self._feature_fn_builder, self._tickers
        )

        y_frame = self._label_fn(train_prices)  # Date × Asset
        y_stacked = y_frame.stack(future_stack=True)  # pd.Series with MultiIndex
        if y_stacked.index.names != ["date", "asset"]:
            y_stacked.index.names = ["date", "asset"]

        dataset = build_supervised_dataset(
            X_panel, y_stacked, horizon=self._horizon, label_name=self._label_name
        )
        self._model.fit(dataset)
        self._is_fitted = True

    # ------------------------------------------------------------------
    # Strategy metadata
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        getattr(self._signal_fn, "__name__", "panel")
        return f"PanelMLStrategy({self._model.name})"

    def params(self) -> dict[str, Any]:
        return {
            "model": self._model.name,
            "n_tickers": len(self._tickers),
            "tickers": self._tickers,
            "horizon": self._horizon,
            "label_name": self._label_name,
        }
