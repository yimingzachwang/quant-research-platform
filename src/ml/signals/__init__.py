"""ML signal translation layer.

Pure functions that convert PredictionSeries outputs into portfolio-compatible
signals and weight DataFrames.  No state, no fitting, no I/O.

Dependency direction:
    src.ml.signals → src.ml.contracts (PredictionSeries)
    src.ml.signals → src.portfolio.ranking, src.portfolio.allocation
    Nothing in src.portfolio depends on src.ml.
"""

from src.ml.signals.prediction import (
    long_short_weights,
    normalize_to_weights,
    sign_signal,
    threshold_signal,
    top_n_weights,
)

__all__ = [
    "sign_signal",
    "threshold_signal",
    "top_n_weights",
    "long_short_weights",
    "normalize_to_weights",
]
