"""ML research infrastructure (E0), signal translation (F1), and diagnostics (F2).

E0: supervised dataset construction, label generation, feature matrix
composition, and prediction contracts.
F1: pure functions that translate PredictionSeries outputs into portfolio-
compatible signals and weights.
F2: pure diagnostic functions for prediction quality, stability, and turnover.

Key invariants:
- Labels use shift(-horizon) only — lookahead is always explicit.
- Splits come from src.validation; E0 does not own split logic.
- SupervisedDataset is in-memory only; no persistence layer.
- Validation is advisory: validators return list[str], never raise.
- Signal functions are pure: no state, no fitting, no I/O.
- Diagnostic functions are pure: no state, no fitting, no I/O.
"""

from src.ml.contracts import PredictionSeries, validate_prediction_index_alignment
from src.ml.datasets import SupervisedDataset, build_supervised_dataset, dataset_hash
from src.ml.feature_matrix import align_features_and_labels, build_feature_matrix
from src.ml.labels import (
    binary_direction_label,
    forward_returns,
    ranking_target,
    volatility_target,
)
from src.ml.signals import (
    long_short_weights,
    normalize_to_weights,
    sign_signal,
    threshold_signal,
    top_n_weights,
)
from src.ml.diagnostics import (
    average_turnover,
    coefficient_stability,
    information_coefficient,
    prediction_correlation,
    prediction_drift,
    prediction_quantiles,
    rolling_directional_accuracy,
    signal_turnover,
    split_metric_table,
    turnover_by_split,
)

__all__ = [
    # Labels
    "forward_returns",
    "binary_direction_label",
    "volatility_target",
    "ranking_target",
    # Feature matrix
    "build_feature_matrix",
    "align_features_and_labels",
    # Datasets
    "SupervisedDataset",
    "build_supervised_dataset",
    "dataset_hash",
    # Contracts
    "PredictionSeries",
    "validate_prediction_index_alignment",
    # Signal translation (F1)
    "sign_signal",
    "threshold_signal",
    "top_n_weights",
    "long_short_weights",
    "normalize_to_weights",
    # Diagnostics (F2)
    "prediction_correlation",
    "information_coefficient",
    "rolling_directional_accuracy",
    "prediction_quantiles",
    "split_metric_table",
    "coefficient_stability",
    "prediction_drift",
    "signal_turnover",
    "average_turnover",
    "turnover_by_split",
]
