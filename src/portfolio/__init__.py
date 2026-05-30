"""Multi-asset portfolio research layer."""

# Legacy stubs — preserved for compatibility
# Alignment and loading
from src.portfolio.alignment import align_prices, align_returns, load_universe

# Allocation
from src.portfolio.allocation import equal_weight, resample_weights_to_daily, volatility_scaled
from src.portfolio.interfaces import PortfolioConstraints, PortfolioConstructor

# Panel feature computation
from src.portfolio.panel import (
    universe_momentum,
    universe_returns,
    universe_rolling_volatility,
    universe_rolling_zscore,
)
from src.portfolio.placeholders import NoOpPortfolioConstructor

# Backtesting
from src.portfolio.portfolio_backtest import PortfolioBacktestResult, run_portfolio_backtest

# Cross-sectional ranking
from src.portfolio.ranking import rank_assets, select_bottom_n, select_top_n

# Weighting policies
from src.portfolio.weighting_policy import VALID_WEIGHTING_SCHEMES, apply_weighting_policy

__all__ = [
    # legacy
    "PortfolioConstraints",
    "PortfolioConstructor",
    "NoOpPortfolioConstructor",
    # alignment
    "load_universe",
    "align_prices",
    "align_returns",
    # panel
    "universe_returns",
    "universe_momentum",
    "universe_rolling_volatility",
    "universe_rolling_zscore",
    # ranking
    "rank_assets",
    "select_top_n",
    "select_bottom_n",
    # allocation
    "equal_weight",
    "volatility_scaled",
    "resample_weights_to_daily",
    # backtest
    "run_portfolio_backtest",
    "PortfolioBacktestResult",
    # weighting policies
    "apply_weighting_policy",
    "VALID_WEIGHTING_SCHEMES",
]
