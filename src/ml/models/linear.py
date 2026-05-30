"""Thin sklearn wrappers for linear regression models.

Each class is a minimal adapter: constructor takes only model-relevant
hyperparameters; fit() accepts SupervisedDataset; predict() returns
PredictionSeries.

Models assume dataset.X and dataset.y are already aligned, NaN-free, and
leakage-safe — they do not modify index semantics or drop rows.
"""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

from src.ml.contracts import PredictionSeries
from src.ml.datasets import SupervisedDataset


class LinearRegressionModel:
    """Ordinary least-squares linear regression.

    Wrapper around sklearn.linear_model.LinearRegression with no
    regularisation and no hyperparameters to tune.  Use when the feature
    matrix is low-dimensional relative to the number of observations.
    """

    def __init__(self) -> None:
        self._model = LinearRegression()
        self._label_name: str = ""
        self._is_fitted: bool = False

    @property
    def name(self) -> str:
        return "LinearRegression"

    def fit(self, dataset: SupervisedDataset) -> None:
        """Train on dataset.X (features) and dataset.y (targets).

        Raises:
            TypeError: If dataset.y is not a pd.Series.
        """
        _require_series_y(dataset, self.name)
        self._model.fit(dataset.X.to_numpy(), dataset.y.to_numpy())
        self._label_name = dataset.label_name
        self._is_fitted = True

    def predict(self, X: pd.DataFrame) -> PredictionSeries:
        """Return regression predictions as a PredictionSeries.

        Raises:
            RuntimeError: If called before fit().
        """
        _require_fitted(self._is_fitted, self.name)
        raw = self._model.predict(X.to_numpy()).astype("float64")
        return PredictionSeries(
            values=pd.Series(raw, index=X.index, dtype="float64"),
            label_name=self._label_name,
            model_name=self.name,
        )


class RidgeRegressionModel:
    """L2-regularised linear regression (Ridge).

    Wrapper around sklearn.linear_model.Ridge.  Suitable when features
    are correlated or the feature matrix is high-dimensional.

    Args:
        alpha: Regularisation strength.  Larger values shrink coefficients
            more aggressively.  Default: 1.0.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self._alpha = alpha
        self._model = Ridge(alpha=alpha)
        self._label_name: str = ""
        self._is_fitted: bool = False

    @property
    def name(self) -> str:
        return f"Ridge(alpha={self._alpha})"

    def fit(self, dataset: SupervisedDataset) -> None:
        _require_series_y(dataset, self.name)
        self._model.fit(dataset.X.to_numpy(), dataset.y.to_numpy())
        self._label_name = dataset.label_name
        self._is_fitted = True

    def predict(self, X: pd.DataFrame) -> PredictionSeries:
        _require_fitted(self._is_fitted, self.name)
        raw = self._model.predict(X.to_numpy()).astype("float64")
        return PredictionSeries(
            values=pd.Series(raw, index=X.index, dtype="float64"),
            label_name=self._label_name,
            model_name=self.name,
        )


class LassoRegressionModel:
    """L1-regularised linear regression (Lasso).

    Wrapper around sklearn.linear_model.Lasso.  Performs implicit feature
    selection by driving low-signal coefficients to exactly zero.

    Args:
        alpha: Regularisation strength.  Default: 1.0.
        max_iter: Maximum iterations for coordinate descent.  Increase if
            convergence warnings appear.  Default: 1000.
    """

    def __init__(self, alpha: float = 1.0, max_iter: int = 1000) -> None:
        self._alpha = alpha
        self._max_iter = max_iter
        self._model = Lasso(alpha=alpha, max_iter=max_iter)
        self._label_name: str = ""
        self._is_fitted: bool = False

    @property
    def name(self) -> str:
        return f"Lasso(alpha={self._alpha})"

    def fit(self, dataset: SupervisedDataset) -> None:
        _require_series_y(dataset, self.name)
        self._model.fit(dataset.X.to_numpy(), dataset.y.to_numpy())
        self._label_name = dataset.label_name
        self._is_fitted = True

    def predict(self, X: pd.DataFrame) -> PredictionSeries:
        _require_fitted(self._is_fitted, self.name)
        raw = self._model.predict(X.to_numpy()).astype("float64")
        return PredictionSeries(
            values=pd.Series(raw, index=X.index, dtype="float64"),
            label_name=self._label_name,
            model_name=self.name,
        )


class ElasticNetRegressionModel:
    """Elastic net regularisation (L1 + L2 combined).

    Wrapper around sklearn.linear_model.ElasticNet.  Combines Ridge and Lasso
    penalties; useful when both feature correlation and sparsity are concerns.

    Args:
        alpha: Overall regularisation strength.  Default: 1.0.
        l1_ratio: Mix between L1 (1.0) and L2 (0.0).  Default: 0.5.
        max_iter: Maximum iterations for coordinate descent.  Default: 1000.
    """

    def __init__(
        self, alpha: float = 1.0, l1_ratio: float = 0.5, max_iter: int = 1000
    ) -> None:
        self._alpha = alpha
        self._l1_ratio = l1_ratio
        self._max_iter = max_iter
        self._model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter)
        self._label_name: str = ""
        self._is_fitted: bool = False

    @property
    def name(self) -> str:
        return f"ElasticNet(alpha={self._alpha}, l1_ratio={self._l1_ratio})"

    def fit(self, dataset: SupervisedDataset) -> None:
        _require_series_y(dataset, self.name)
        self._model.fit(dataset.X.to_numpy(), dataset.y.to_numpy())
        self._label_name = dataset.label_name
        self._is_fitted = True

    def predict(self, X: pd.DataFrame) -> PredictionSeries:
        _require_fitted(self._is_fitted, self.name)
        raw = self._model.predict(X.to_numpy()).astype("float64")
        return PredictionSeries(
            values=pd.Series(raw, index=X.index, dtype="float64"),
            label_name=self._label_name,
            model_name=self.name,
        )


# ---------------------------------------------------------------------------
# Internal guards
# ---------------------------------------------------------------------------


def _require_fitted(is_fitted: bool, model_name: str) -> None:
    if not is_fitted:
        raise RuntimeError(f"{model_name} must be fit before calling predict()")


def _require_series_y(dataset: SupervisedDataset, model_name: str) -> None:
    if not isinstance(dataset.y, pd.Series):
        raise TypeError(
            f"{model_name} requires dataset.y to be pd.Series; "
            f"got {type(dataset.y).__name__}. "
            "For cross-sectional targets, use a model that accepts DataFrame y."
        )
