"""Lightweight protocol for E1 ML models.

All models in src/ml/models/ must satisfy this interface.
No inheritance required — duck typing via Protocol is sufficient.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from src.ml.contracts import PredictionSeries
from src.ml.datasets import SupervisedDataset


@runtime_checkable
class BaseMLModel(Protocol):
    """Protocol satisfied by all E1 ML models.

    Guarantees:
    - fit() accepts a SupervisedDataset whose X and y are already aligned,
      NaN-free, and leakage-safe (E0 invariant).
    - predict() accepts a raw feature DataFrame and returns a PredictionSeries.
    - Models do NOT create splits, shuffle, or alter index semantics.
    - All temporal correctness is owned by E0 + src.validation, not by models.

    runtime_checkable=True allows isinstance(model, BaseMLModel) assertions
    in tests without requiring inheritance.
    """

    def fit(self, dataset: SupervisedDataset) -> None:
        """Train on dataset.X and dataset.y.

        Assumes dataset has already been aligned and cleaned by E0.
        Do NOT drop NaN, shift labels, or reindex inside this method.
        """
        ...

    def predict(self, X: pd.DataFrame) -> PredictionSeries:
        """Return predictions for the rows in X.

        Returned PredictionSeries.values.index must equal X.index.
        dtype must be float64.
        model_name must identify this model.
        """
        ...
