# Design Invariants

Only invariants supported by repository evidence are listed.

## Timing And Leakage

- Single-asset backtests apply `signal.shift(1)`.
- Portfolio backtests apply `weights.shift(1)`.
- Strategies generate weights at time `t`; they do not apply their own lag.
- Validation splits are chronological and require train windows before test
  windows.
- Walk-forward validation passes no prices after `split.test_end` into the
  strategy run.
- ML labels use explicit `shift(-horizon)` and keep future-unavailable rows as
  NaN until alignment drops them.

## Reproducibility

- Data requests are typed and hashable through stable JSON + SHA256.
- Processed data paths are deterministic by dataset type, symbol, frequency.
- Dataset registry manifests record request hash, schema version, path, dates,
  symbol, source, frequency, and row count.
- Experiment specs have deterministic hashes.
- Experiment artifacts use JSON/parquet filesystem outputs.

## Boundaries

- Data ingestion does not contain strategy, ML, or backtesting logic.
- Feature functions do not load/persist datasets.
- Strategies generate weights only.
- Backtests evaluate returns and weights; they do not invent signals.
- Validation creates/evaluates chronological splits; it does not own features.
- Visualization and reporting are read-only consumers of computed artifacts.

## Simplicity

- Prefer pure functions, dataclasses, pydantic models, protocols, and explicit
  pandas objects.
- No database, ORM, async orchestration, service layer, or plugin system is
  implemented for core research flows.
- Filesystem artifacts remain inspectable.

## Evidence-Backed Non-Invariants

- Adjusted-close is not yet an enforced runtime OHLCV invariant.
- `BacktestEngine` is not yet a complete orchestration invariant.
- `src/models`, `src/signals`, `src/risk`, `src/execution`, and
  `src/evaluation` are not production implementations.

