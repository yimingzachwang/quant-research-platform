# Visualization

Status: STABLE.

## Purpose

The visualization subsystem is a read-only matplotlib layer for diagnostics.
It consumes computed series, DataFrames, strategy results, or validation
results and returns figures.

## Implemented Plot Families

| Family | Examples |
|---|---|
| Backtest | equity curve, drawdown, equity/drawdown combo, rolling Sharpe, rolling volatility |
| Distribution | return distribution, monthly return heatmap |
| Signal | SMA strategy, position state |
| Portfolio | weights, rolling weights, correlations, turnover, concentration, contribution |
| Comparison | strategy equity curves, drawdowns, metric bars, metrics table |
| Validation | walk-forward equity, stitched OOS equity, split Sharpe, stability, train/test |
| Diagnostics | turnover, concentration metrics, rolling average correlation |
| Utilities | style context, save figure, datetime index conversion |

## Boundaries

Visualization should:

- Accept already-computed artifacts.
- Return matplotlib figure objects where practical.
- Save only through explicit utility calls or caller-owned paths.

Visualization should not:

- Load market data.
- Compute strategy weights.
- Run backtests or validation.
- Mutate experiment artifacts.
- Recompute report metrics.

## Reporting Integration

D1 experiment orchestration saves plots under experiment artifacts.
D2 reporting copies saved PNGs and references them in markdown/HTML reports.

