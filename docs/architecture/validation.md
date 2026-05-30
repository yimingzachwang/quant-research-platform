# Validation

Status: STABLE.

## Split Generation

`src/validation/splits.py` defines:

- `TimeSplit`: frozen train/test window dataclass.
- `rolling_time_splits`: fixed-width sliding train window.
- `expanding_time_splits`: anchored expanding train window.

Both split generators snap calendar windows to available dates and require
strict chronological ordering: `train_end < test_start`.

## Strategy Walk-Forward

`src/validation/walk_forward.py::run_walk_forward_validation`:

1. Receives prices, a `Strategy`, splits, and cost assumptions.
2. Optionally calls `strategy.fit(train_prices)` when a strategy implements it.
3. Runs the strategy only on data through `split.test_end`.
4. Extracts test-window backtest rows.
5. Computes metrics on test-window net returns.
6. Returns `WalkForwardResult`.

## Result Structures

- `SplitResult`: split metadata, strategy name, metrics, equity curve, weights.
- `WalkForwardResult`: strategy name and list of split results.

## Stability Analytics

`src/validation/stability.py` provides:

- `split_metrics_table`
- `summarize_stability`
- `rolling_sharpe_by_split`
- `parameter_robustness_summary`

## Anti-Leakage Protections

- Splits are chronological and never shuffled.
- Optional strategy fitting uses train windows only.
- Strategy run receives no prices after `test_end`.
- Portfolio backtest still applies one-period weight lag.

## Should Not Do

- Create features or labels.
- Own model APIs.
- Persist experiment artifacts.
- Generate plots directly.

