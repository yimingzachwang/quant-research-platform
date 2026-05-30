"""Feature engineering layer — pure, deterministic transformations."""

from src.features.interfaces import Feature, FeaturePipeline
from src.features.momentum import momentum, momentum_20d, momentum_60d, momentum_252d
from src.features.normalization import minmax_normalize, robust_normalize, zscore_normalize
from src.features.placeholders import NoOpFeature, NoOpFeaturePipeline
from src.features.returns import compute_cumulative_returns, compute_log_returns, compute_returns
from src.features.rolling import (
    bollinger_distance,
    rolling_autocorrelation,
    rolling_minmax,
    rolling_rank,
    rolling_skewness,
    rolling_zscore,
)
from src.features.trend import ema, sma, sma_crossover, trend_strength
from src.features.volatility import (
    downside_volatility,
    ewm_volatility,
    rolling_volatility,
    vol_of_vol,
    vol_percentile,
)

__all__ = [
    # contracts
    "Feature",
    "FeaturePipeline",
    "NoOpFeature",
    "NoOpFeaturePipeline",
    # returns
    "compute_returns",
    "compute_log_returns",
    "compute_cumulative_returns",
    # momentum
    "momentum",
    "momentum_20d",
    "momentum_60d",
    "momentum_252d",
    # volatility
    "rolling_volatility",
    "ewm_volatility",
    "downside_volatility",
    "vol_of_vol",
    "vol_percentile",
    # rolling utilities
    "rolling_zscore",
    "rolling_rank",
    "rolling_minmax",
    "bollinger_distance",
    "rolling_skewness",
    "rolling_autocorrelation",
    # trend
    "sma",
    "ema",
    "sma_crossover",
    "trend_strength",
    # normalization
    "zscore_normalize",
    "minmax_normalize",
    "robust_normalize",
]
