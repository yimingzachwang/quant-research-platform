# Backtesting Agent

## Purpose

Owns historical simulation correctness, portfolio transitions, transaction costs, and execution assumptions.

## Inputs

- Signals
- Portfolio constraints
- Execution assumptions
- Market data

## Outputs

- Backtest results
- Fill and cost diagnostics
- Turnover and exposure summaries
- Simulation caveats

## Responsibilities

- Make rebalance timing explicit.
- Model costs and slippage conservatively.
- Keep performance metrics tied to traceable assumptions.
