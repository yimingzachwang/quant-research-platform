"""ML-driven strategy adapter.

Adapts a BaseMLModel into the existing Strategy interface so that ML models
can be evaluated with the existing backtesting and walk-forward validation
infrastructure without any changes to those systems.

The only novel responsibility here is wiring feature construction + dataset
construction + model fitting + prediction + signal conversion into the
Strategy.generate_weights() contract.

All heavy lifting is delegated to existing infrastructure:
    build_feature_matrix     → src.ml.feature_matrix
    build_supervised_dataset → src.ml.datasets
    model.fit / model.predict → BaseMLModel contract (src.ml.models)
    signal_fn                → caller supplies (e.g. from src.ml.signals)

No plotting, no persistence, no experiment tracking, no orchestration.

Walk-forward integration
------------------------
run_walk_forward_validation already calls:
    if hasattr(strategy, "fit"):
        strategy.fit(train_prices)
before each test window.  MLStrategy implements this hook — no changes to
the validation runner are needed.

Single-asset vs. panel workflows
---------------------------------
E1 models (linear, logistic) produce PredictionSeries with pd.Series values —
compatible with sign_signal and threshold_signal.

For cross-sectional ranking via top_n_weights / long_short_weights (which
require pd.DataFrame predictions), a model that produces panel PredictionSeries
is required.  MLStrategy is general: signal_fn is injected by the caller, so
both single-asset and panel workflows use the same adapter class.
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from src.ml.contracts import PredictionSeries
from src.ml.datasets import build_supervised_dataset
from src.ml.feature_matrix import build_feature_matrix
from src.ml.models.base import BaseMLModel
from src.strategies.base import Strategy


class MLStrategy(Strategy):
    """Adapts a BaseMLModel into the Strategy interface.

    Implements Strategy.generate_weights(prices) → weights so the model can be
    passed to run_strategy, compare_strategies, and run_walk_forward_validation
    without any changes to those callers.

    Implements fit(train_prices) matching the optional hook signature expected
    by run_walk_forward_validation — no changes to the validation runner needed.

    Args:
        model:       Any object satisfying the BaseMLModel Protocol.
        feature_fns: Dict passed directly to build_feature_matrix.  Keys become
                     column names (or prefixes for multi-column outputs).
        label_fn:    Callable(prices: pd.DataFrame) → pd.Series | pd.DataFrame.
                     Applied to train_prices during fit().  Must not use future
                     prices — the caller is responsible for correct horizon.
        horizon:     Forward horizon in periods; stored in SupervisedDataset for
                     provenance.  Must match what label_fn actually computes.
        signal_fn:   Callable(PredictionSeries) → pd.DataFrame (Date × Asset).
                     Converts model predictions to portfolio weights.  Compose
                     from src.ml.signals (sign_signal, top_n_weights, etc.).
        label_name:  Name stored in SupervisedDataset; default "target".
    """

    def __init__(
        self,
        model: BaseMLModel,
        feature_fns: dict[str, Callable[[pd.DataFrame], pd.Series | pd.DataFrame]],
        label_fn: Callable[[pd.DataFrame], pd.Series | pd.DataFrame],
        horizon: int,
        signal_fn: Callable[[PredictionSeries], pd.DataFrame],
        label_name: str = "target",
    ) -> None:
        self._model = model
        self._feature_fns = feature_fns
        self._label_fn = label_fn
        self._horizon = horizon
        self._signal_fn = signal_fn
        self._label_name = label_name
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Build features from prices, predict, apply signal_fn → weights.

        Called by run_strategy and run_walk_forward_validation.  The
        walk-forward runner passes prices[:test_end] — no data beyond the
        current evaluation horizon ever reaches this method.

        Feature rows with NaN (warm-up period) are dropped before prediction
        and produce flat (zero) weights in the returned DataFrame.

        Args:
            prices: Date × Asset close price DataFrame passed by the caller.

        Returns:
            Date × Asset weight DataFrame compatible with run_portfolio_backtest.
            Rows during the feature warm-up period are all zeros (flat).

        Raises:
            RuntimeError: If called before fit().
        """
        if not self._is_fitted:
            raise RuntimeError(
                f"MLStrategy({self._model.name}) must be fit() before "
                "generate_weights().  Call fit(train_prices) first, or use "
                "run_walk_forward_validation which invokes fit() automatically."
            )

        X = build_feature_matrix(prices, self._feature_fns)
        X_clean = X.dropna()

        if X_clean.empty:
            # All rows are in the warm-up window — return zero weights
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        predictions = self._model.predict(X_clean)
        weights = self._signal_fn(predictions)

        # Reindex to the full price index; warm-up rows become flat (0)
        return weights.reindex(prices.index, fill_value=0.0)

    # ------------------------------------------------------------------
    # Optional fit hook — matched by run_walk_forward_validation
    # ------------------------------------------------------------------

    def fit(self, train_prices: pd.DataFrame) -> None:
        """Build a SupervisedDataset from train_prices and fit the model.

        Delegates to build_feature_matrix → label_fn → build_supervised_dataset
        → model.fit().  Model state is overwritten on each call, matching the
        E1 walk-forward pipeline convention (re-fit on every split).

        Args:
            train_prices: Date × Asset price DataFrame for the training window.
                The walk-forward runner passes prices[:train_end].
        """
        X = build_feature_matrix(train_prices, self._feature_fns)
        y = self._label_fn(train_prices)
        dataset = build_supervised_dataset(
            X, y, horizon=self._horizon, label_name=self._label_name
        )
        self._model.fit(dataset)
        self._is_fitted = True

    # ------------------------------------------------------------------
    # Strategy metadata
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return f"MLStrategy({self._model.name})"

    def params(self) -> dict[str, Any]:
        return {
            "model": self._model.name,
            "horizon": self._horizon,
            "label_name": self._label_name,
            "n_feature_fns": len(self._feature_fns),
        }
