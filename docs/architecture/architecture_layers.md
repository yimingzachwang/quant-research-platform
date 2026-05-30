# Architecture Layers

Status labels used here: COMPLETE, STABLE, PARTIAL, EXPERIMENTAL,
PLACEHOLDER, PLANNED/FUTURE.

## Layer Summary

| Layer | Status | Primary Modules |
|---|---:|---|
| Core domain objects | STABLE | `src/core.py` |
| Data ingestion and storage | STABLE/PARTIAL | `src/data/*` |
| Dataset registry and loading | STABLE | `src/data/manifest.py`, `src/data/registry/*`, `src/data/loaders/dataset_loader.py` |
| Cleaning | STABLE | `src/cleaning/*` |
| Feature engineering | STABLE | `src/features/*` |
| Signals and strategies | STABLE/PARTIAL | `src/backtesting/signals.py`, `src/strategies/*`, `src/signals/*` |
| Portfolio construction | STABLE/PARTIAL | `src/portfolio/*` |
| Backtesting | STABLE/PARTIAL | `src/backtesting/*` |
| Validation | STABLE | `src/validation/*` |
| ML research | EXPERIMENTAL/STABLE | `src/ml/*` |
| Experiments and tracking | STABLE/PARTIAL | `src/experiments/*` |
| Visualization | STABLE | `src/visualization/*` |
| Reporting | STABLE | `src/reporting/*` |
| Risk, execution, evaluation | PLACEHOLDER | `src/risk`, `src/execution`, `src/evaluation` |
| LLM interface | EXPERIMENTAL | `src/llm/*` |

## Core

Purpose: shared domain objects: `Universe`, `Horizon`, `DateRange`,
`ExperimentContext`.

Responsibilities: carry explicit identifiers, symbols, date ranges, horizons,
and config metadata across modules.

Should not do: load data, generate features, run backtests, or mutate run state.

## Data Layer

Purpose: turn declarative data profiles into materialized raw/processed parquet
datasets with manifests.

Inputs: YAML configs, universes, date ranges, `DataRequest` objects.

Outputs: raw parquet extracts, processed parquet datasets, validation reports,
`DatasetManifest` entries.

Dependencies: pydantic, pandas, pyarrow, yfinance/FRED download adapters,
`DatasetValidator`, `DataStorage`, `DatasetRegistry`.

Anti-leakage protections: none specific to strategy timing; this layer is about
dataset integrity and reproducibility.

Should not do: strategy logic, feature engineering, ML, backtesting, plotting,
or alpha research.

## Registry And Loading

Purpose: resolve persisted datasets through metadata, not filenames alone.

Inputs: `DatasetQuery` or exact keyword query fields.

Outputs: one parquet-backed pandas DataFrame or explicit typed exception.

Invariants: no silent selection on zero/multiple matches; schema version can be
checked; required columns can be checked after loading.

Should not do: infer identity from storage paths, download data, mutate the
registry during reads, or compute features.

## Cleaning

Purpose: small research-safe utilities for timestamp ordering, duplicate index
removal, finite numeric values, bounded forward-fill, and OHLCV validation.

Should not do: silently repair severe OHLCV errors, infer missing data policy
for a research workflow, or mutate inputs in place.

## Feature Engineering

Purpose: deterministic transformations from price/return series into features.

Inputs: pandas Series/DataFrames.

Outputs: pandas Series/DataFrames with preserved index semantics.

Anti-leakage: functions use trailing windows or contemporaneous transforms.
Forward-looking labels are in `src/ml/labels.py`, not `src/features`.

Should not do: load data, define strategy weights, split train/test windows,
fit models, or persist datasets.

## Signals And Strategies

Purpose: convert conditions or price matrices into tradable signal/weight
series for evaluation.

Implemented: signal helpers in `src/backtesting/signals.py`, `Strategy` base,
buy-and-hold, equal-weight, momentum rotation, runner, comparison utilities.

Partial: `src/signals` package contains protocol/no-op placeholder only.

Anti-leakage: strategy outputs are not lagged internally; backtest functions
apply the one-period lag.

Should not do: file I/O, plotting, persistence, or experiment registration.

## Portfolio

Purpose: multi-asset alignment, panel returns/features, ranking, selection,
allocation, resampling, and vectorized portfolio backtesting.

Partial: no optimizer, constraint solver, covariance model, or production
portfolio construction layer found.

Should not do: data ingestion, experiment orchestration, reporting, or hidden
execution simulation.

## Backtesting

Purpose: compute historical returns, costs, equity curves, drawdowns, and
metrics from returns and signals/weights.

Anti-leakage: `run_backtest()` uses `signal.shift(1)` and
`run_portfolio_backtest()` uses `weights.shift(1)`.

Partial: `BacktestEngine` is a legacy/future wrapper and does not orchestrate a
full pipeline.

Should not do: invent signals, fetch data, optimize portfolios, or persist
experiments directly.

## Validation

Purpose: create chronological rolling/expanding train/test splits and evaluate
strategies out-of-sample.

Anti-leakage: split generators enforce `train_end < test_start`; validation
passes no prices after `test_end` into a strategy.

Should not do: create labels, create features, fit arbitrary models outside the
documented optional `fit(train_data)` hook, or save reports.

## ML Research Layer

Purpose: early supervised ML contracts, label builders, feature matrices,
dataset hashing, sklearn wrappers, prediction metrics, and walk-forward
prediction loops.

Status: experimental but tested. It is more implemented than a placeholder,
but not yet integrated into D1 experiment orchestration or artifact registry.

Should not do: split time series internally, shuffle data, silently align
predictions outside documented helpers, or bypass validation splits.

## Experiments, Visualization, Reporting

Experiments own config validation, orchestration, artifact persistence, and
experiment registry metadata.

Visualization returns/saves matplotlib figures from already-computed data.

Reporting reads saved artifacts and produces markdown/HTML/provenance without
recomputation.

