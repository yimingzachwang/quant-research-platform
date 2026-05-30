# Research Workflows

## 1. Single Asset Workflow

Status: STABLE for primitives.

1. Load a dataset through `load_dataset(...)`.
2. Extract or compute close prices/returns.
3. Compute feature or signal series.
4. Run `run_backtest(returns, signal, transaction_cost_bps)`.
5. Compute metrics from `net_return`.
6. Plot with `src.visualization.backtest_plots`.

Safeguard: `run_backtest` shifts the signal by one period.

Failure points: missing registry entry, missing columns, non-overlapping
returns/signal indices.

## 2. Multi Asset Strategy Workflow

Status: STABLE.

1. `load_universe(symbols)` loads registered datasets.
2. `align_prices(...)` produces Date x Asset close prices.
3. Strategy generates Date x Asset weights.
4. `run_strategy(...)` computes returns and calls portfolio backtest.
5. `StrategyResult` carries weights, backtest DataFrame, and metrics.

Safeguard: `run_portfolio_backtest` shifts weights by one period.

## 3. Walk Forward Validation Workflow

Status: STABLE.

1. Generate `TimeSplit` objects with rolling or expanding split functions.
2. Run `run_walk_forward_validation(prices, strategy, splits, costs)`.
3. Evaluate each test window separately.
4. Summarize with `split_metrics_table` or `summarize_stability`.
5. Plot with validation visualization helpers.

Safeguards: chronological splits, optional fit on train only, no prices after
`test_end`, backtest lagging.

## 4. ML Dataset Workflow

Status: EXPERIMENTAL/STABLE.

1. Compute features with `src/features` or callables passed to
   `build_feature_matrix`.
2. Generate labels with explicit `horizon` using `src/ml/labels.py`.
3. Call `build_supervised_dataset(X, y, horizon)`.
4. Fit a model implementing `BaseMLModel`.
5. Produce `PredictionSeries`.
6. Validate alignment with `validate_prediction_index_alignment`.
7. Optionally run walk-forward predictions over `TimeSplit` objects.

Safeguards: labels make lookahead explicit, alignment drops NaN warm-up/future
rows, prediction concatenation rejects overlapping test windows.

## 5. Portfolio Construction Workflow

Status: STABLE/PARTIAL.

1. Align prices.
2. Compute panel features such as momentum or volatility.
3. Rank assets cross-sectionally.
4. Select top/bottom names.
5. Allocate equal or inverse-vol weights.
6. Resample periodic weights to daily.
7. Backtest with lagged weights.

Limitations: no optimizer or constraint engine.

## 6. Visualization Workflow

Status: STABLE.

1. Run strategy/backtest/validation first.
2. Pass results into visualization functions.
3. Save figures explicitly with `save_figure` or through experiment tracking.
4. Reports consume saved figures.

Safeguard: plotting functions do not own core computation.

