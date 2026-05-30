# Backtesting

Status: STABLE for vectorized functions; PLACEHOLDER/PARTIAL for
`BacktestEngine` orchestration.

## Single-Asset Backtest

`src/backtesting/engine.py::run_backtest` accepts aligned returns and signals.
It returns a DataFrame with:

- `position`
- `gross_return`
- `turnover`
- `transaction_cost`
- `net_return`
- `equity_curve`
- `drawdown`

Anti-lookahead invariant: `position = signal.shift(1).fillna(0.0)`.

## Portfolio Backtest

`src/portfolio/portfolio_backtest.py::run_portfolio_backtest` accepts Date x
Asset returns and weights. It returns `PortfolioBacktestResult`:

- `backtest`: daily returns/cost/equity/drawdown DataFrame.
- `weights`: lagged weights actually applied.
- `metrics`: scalar metrics computed on `net_return`.

Anti-lookahead invariant: `w_lagged = weights.shift(1).fillna(0.0)`.

## Metrics

`src/backtesting/metrics.py` implements annualized return, annualized
volatility, Sharpe ratio, max drawdown, Calmar ratio, hit rate, and
`compute_metrics`.

## Placeholder Engine

`BacktestEngine.run(context)` currently returns a `BacktestResult` containing
the input `ExperimentContext` and empty artifacts/metrics. It is likely
intended as a future orchestrator, but direct functions are the real
implemented backtesting path.

## Limitations

- Execution simulation is limited to turnover-based transaction costs.
- No fill model, slippage model, borrow model, tax model, or liquidity model.
- No broker/live execution.

## Should Not Do

- Generate strategy signals.
- Load data directly.
- Persist experiment artifacts.
- Create validation splits.

