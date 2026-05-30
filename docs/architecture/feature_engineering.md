# Feature Engineering

Status: STABLE for pure feature functions; PARTIAL for materialized feature
dataset infrastructure.

## Implemented Feature Modules

| Module | Examples | Contract |
|---|---|---|
| `returns.py` | simple, log, cumulative returns | price Series in, return Series out |
| `momentum.py` | `momentum`, `momentum_20d`, `momentum_60d`, `momentum_252d` | trailing price ratios |
| `rolling.py` | rolling z-score, rank, min-max | trailing rolling windows |
| `volatility.py` | rolling and EWMA volatility | return Series in, volatility out |
| `trend.py` | SMA, EMA, crossover, trend strength | trailing/recursive trend features |
| `normalization.py` | z-score, robust, min-max | deterministic normalization helpers |

## ML Feature Matrix Adapter

`src/ml/feature_matrix.py` composes feature callables into a wide matrix and
aligns features with labels. It deliberately reuses feature functions rather
than reimplementing feature logic.

## Anti-Leakage Evidence

- Feature modules use trailing windows, shifts, or contemporaneous transforms.
- Forward-looking target construction lives in `src/ml/labels.py`, where
  `shift(-horizon)` is explicit.
- `align_features_and_labels()` drops warm-up and future-unavailable rows only
  after both features and labels are aligned.

## Limitations

- Feature datasets are not yet materialized in `data/features`.
- Feature lineage is not persisted in a registry manifest.
- No feature pipeline object is productionized beyond placeholder interfaces.

## Should Not Do

- Load datasets directly.
- Generate portfolio weights.
- Run backtests or validation splits.
- Persist artifacts unless a future feature dataset registry path is added.

