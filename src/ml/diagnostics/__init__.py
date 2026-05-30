"""ML research diagnostics.

Pure functions for evaluating prediction quality, signal stability,
and portfolio turnover from ML-driven strategies.

No orchestration, no frameworks, no plotting.  All functions return
plain pandas objects so results can be inspected or passed to
src.visualization.ml_plots for display.
"""

from src.ml.diagnostics.prediction import (
    information_coefficient,
    prediction_correlation,
    prediction_quantiles,
    rolling_directional_accuracy,
)
from src.ml.diagnostics.stability import (
    coefficient_stability,
    prediction_drift,
    split_metric_table,
)
from src.ml.diagnostics.turnover import (
    average_turnover,
    signal_turnover,
    turnover_by_split,
)

__all__ = [
    # prediction
    "prediction_correlation",
    "information_coefficient",
    "rolling_directional_accuracy",
    "prediction_quantiles",
    # stability
    "split_metric_table",
    "coefficient_stability",
    "prediction_drift",
    # turnover
    "signal_turnover",
    "average_turnover",
    "turnover_by_split",
]
