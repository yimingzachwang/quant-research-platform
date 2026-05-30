"""E1 baseline ML models and evaluation metrics.

Models are thin sklearn wrappers with a common fit/predict interface.
All models assume dataset.X and dataset.y are already aligned and NaN-free
(E0 invariant).  No split logic lives here.
"""

from src.ml.models.base import BaseMLModel
from src.ml.models.linear import (
    ElasticNetRegressionModel,
    LassoRegressionModel,
    LinearRegressionModel,
    RidgeRegressionModel,
)
from src.ml.models.logistic import LogisticRegressionModel
from src.ml.models.metrics import (
    correlation_coefficient,
    directional_accuracy,
    mae,
    mse,
    r2_score,
    rmse,
)

__all__ = [
    # Protocol
    "BaseMLModel",
    # Linear regression
    "LinearRegressionModel",
    "RidgeRegressionModel",
    "LassoRegressionModel",
    "ElasticNetRegressionModel",
    # Classification
    "LogisticRegressionModel",
    # Metrics
    "mse",
    "rmse",
    "mae",
    "r2_score",
    "correlation_coefficient",
    "directional_accuracy",
]
