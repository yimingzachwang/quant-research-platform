# Signals And Strategies

Status: STABLE for implemented helpers and strategies; PLACEHOLDER for the
generic `src/signals` package.

## Signal Helpers

`src/backtesting/signals.py` contains deterministic helpers:

- `long_only_signal`
- `signal_from_threshold`
- `crossover_signal`
- `volatility_target_signal`

These helpers do not lag signals. The backtest layer owns lagging.

## Strategy Contract

`src/strategies/base.py::Strategy` defines:

- input: Date x Asset close price DataFrame.
- output: Date x Asset weight DataFrame.
- no file I/O, plotting, reporting, or persistence.
- strategies must not apply their own anti-lookahead lag.

## Implemented Strategies

| Strategy | Status | Evidence |
|---|---:|---|
| `BuyAndHoldStrategy` | STABLE | static first-asset or user-specified weights |
| `EqualWeightStrategy` | STABLE | periodic equal-weight rebalancing |
| `MomentumRotationStrategy` | STABLE | trailing momentum, top-N selection, equal weighting |

## Runner And Comparison

`run_strategy(prices, strategy, transaction_cost_bps)`:

1. Calls `strategy.generate_weights(prices)`.
2. Computes universe returns.
3. Runs `run_portfolio_backtest()`.
4. Returns `StrategyResult`.

`compare_strategies()`, `metrics_table()`, and `rank_strategies()` compare
multiple strategy results without I/O.

## Placeholder Signal Package

`src/signals` currently contains a `SignalGenerator` protocol and no-op
implementation. It does not yet implement a production signal layer.

## Should Not Do

- Fetch data.
- Persist results.
- Generate plots.
- Own transaction cost or fill simulation.
- Apply the one-period lag that belongs to the backtest engine.

