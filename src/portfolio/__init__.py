"""Multi-asset portfolio research layer."""

# Legacy stubs — preserved for compatibility
from src.portfolio.interfaces import PortfolioConstraints, PortfolioConstructor
from src.portfolio.placeholders import NoOpPortfolioConstructor

# Alignment and loading
from src.portfolio.alignment import align_prices, align_returns, load_universe

# Panel feature computation
from src.portfolio.panel import (
    universe_momentum,
    universe_returns,
    universe_rolling_volatility,
    universe_rolling_zscore,
)

# Cross-sectional ranking
from src.portfolio.ranking import rank_assets, select_bottom_n, select_top_n

# Allocation
from src.portfolio.allocation import equal_weight, resample_weights_to_daily, volatility_scaled

# Weighting policies
from src.portfolio.weighting_policy import apply_weighting_policy, VALID_WEIGHTING_SCHEMES

# Backtesting
from src.portfolio.portfolio_backtest import PortfolioBacktestResult, run_portfolio_backtest

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
