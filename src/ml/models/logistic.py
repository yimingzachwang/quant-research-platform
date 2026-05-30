"""Thin sklearn wrapper for logistic classification models.

Designed for use with binary_direction_label() from src.ml.labels.
Predictions are class-1 probability scores (not hard 0/1 labels).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.ml.contracts import PredictionSeries
from src.ml.datasets import SupervisedDataset
from src.ml.models.linear import _require_fitted, _require_series_y


class LogisticRegressionModel:
    """L2-regularised logistic regression for binary classification.

    Wrapper around sklearn.linear_model.LogisticRegression.  Intended for
    use with binary direction labels (0.0 / 1.0) produced by
    binary_direction_label() from src.ml.labels.

    Prediction output:
        predict() returns the class-1 probability score (P(y=1 | X)), NOT
        a hard 0/1 label.  This is more useful for ranking and downstream
        portfolio construction than a hard decision boundary.

    Args:
        C: Inverse regularisation strength (smaller = stronger regularisation).
            Default: 1.0.
        max_iter: Maximum solver iterations.  Increase if convergence warnings
            appear on small or ill-conditioned datasets.  Default: 1000.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000) -> None:
        self._C = C
        self._max_iter = max_iter
        self._model = LogisticRegression(C=C, max_iter=max_iter)
        self._label_name: str = ""
        self._is_fitted: bool = False

    @property
    def name(self) -> str:
        return f"LogisticRegression(C={self._C})"

    def fit(self, dataset: SupervisedDataset) -> None:
        """Train on binary labels in dataset.y.

        Expects dataset.y values to be 0.0 or 1.0 (as produced by
        binary_direction_label).  Values are cast to int before fitting
        to avoid sklearn dtype warnings.

        Raises:
            TypeError: If dataset.y is not a pd.Series.
            ValueError: If dataset.y contains fewer than 2 distinct classes
                (e.g., a training window where all returns were positive).
        """
        _require_series_y(dataset, self.name)

        y_vals = dataset.y.to_numpy()
        unique_classes = np.unique(y_vals[~np.isnan(y_vals)])
        if len(unique_classes) < 2:
            raise ValueError(
                f"{self.name}.fit() requires at least 2 classes in dataset.y; "
                f"found only: {unique_classes.tolist()}. "
                "The training window may be too short or too homogeneous."
            )

        # Cast to int: sklearn warns on float 0.0/1.0 labels
        self._model.fit(dataset.X.to_numpy(), y_vals.astype(int))
        self._label_name = dataset.label_name
        self._is_fitted = True

    def predict(self, X: pd.DataFrame) -> PredictionSeries:
        """Return class-1 probability scores (P(y=1 | X)).

        The returned values are in [0, 1].  Higher values indicate a higher
        predicted probability of a positive forward return.  These scores can
        be used directly for ranking or thresholding in downstream portfolio
        logic.

        Raises:
            RuntimeError: If called before fit().
        """
        _require_fitted(self._is_fitted, self.name)
        # predict_proba returns shape (n_obs, 2); column 1 is P(class=1)
        proba = self._model.predict_proba(X.to_numpy())[:, 1].astype("float64")
        return PredictionSeries(
            values=pd.Series(proba, index=X.index, dtype="float64"),
            label_name=self._label_name,
            model_name=self.name,
        )
