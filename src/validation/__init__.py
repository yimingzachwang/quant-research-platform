"""Walk-forward validation infrastructure.

Provides time-aware train/test splits, walk-forward runners, and stability
analytics for research-grade validation of quantitative strategies.

Designed to be compatible with future ML model integration: strategies or
models implementing an optional ``fit(train_data)`` method will have it called
automatically during walk-forward runs.
"""

from src.validation.splits import (
    TimeSplit,
    expanding_time_splits,
    rolling_time_splits,
)
from src.validation.stability import (
    parameter_robustness_summary,
    rolling_sharpe_by_split,
    split_metrics_table,
    summarize_stability,
)
from src.validation.walk_forward import (
    SplitResult,
    WalkForwardResult,
    run_walk_forward_validation,
)

__all__ = [
    # splits
    "TimeSplit",
    "rolling_time_splits",
    "expanding_time_splits",
    # walk-forward
    "SplitResult",
    "WalkForwardResult",
    "run_walk_forward_validation",
    # stability
    "split_metrics_table",
    "summarize_stability",
    "rolling_sharpe_by_split",
    "parameter_robustness_summary",
]
