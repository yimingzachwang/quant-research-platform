# ML Layer

Status: EXPERIMENTAL/STABLE. This layer is implemented and tested, but it is
not yet integrated into the main D1 experiment orchestration or registry
artifact workflow.

## Implemented Areas

| Area | Modules | Status |
|---|---|---:|
| Prediction contract | `src/ml/contracts.py` | STABLE |
| Feature matrix composition | `src/ml/feature_matrix.py` | STABLE |
| Label generation | `src/ml/labels.py` | STABLE |
| Supervised dataset | `src/ml/datasets.py` | STABLE |
| Model protocol | `src/ml/models/base.py` | STABLE |
| Linear/logistic wrappers | `src/ml/models/linear.py`, `logistic.py` | EXPERIMENTAL/STABLE |
| ML metrics | `src/ml/models/metrics.py` | STABLE |
| Walk-forward predictions | `src/ml/pipelines/walk_forward.py` | EXPERIMENTAL/STABLE |
| Generic `src/models` package | `src/models/*` | PLACEHOLDER |

## Core Contracts

`PredictionSeries` is a frozen dataclass:

- `values`: pandas Series or DataFrame, expected float dtype.
- `label_name`: target being predicted.
- `model_name`: producing model identity.

`validate_prediction_index_alignment()` is advisory: it returns violation
strings and never raises.

`SupervisedDataset` stores aligned `X`, `y`, `feature_names`, `label_name`,
and prediction `horizon`. It has a `slice(start, end)` method for validation
loops.

## Label Semantics

Forward-looking labels are explicit:

- `forward_returns`: `pct_change(horizon).shift(-horizon)`.
- `binary_direction_label`: probability/class target from forward returns.
- `volatility_target`: future realized volatility.
- `ranking_target`: cross-sectional percentile rank of forward returns.

The last `horizon` rows are NaN and later removed by alignment.

## Model Semantics

`BaseMLModel` requires:

- `fit(dataset: SupervisedDataset) -> None`.
- `predict(X: pd.DataFrame) -> PredictionSeries`.

Model wrappers assume `X` and `y` are already aligned, NaN-free, and
leakage-safe. They do not split, shuffle, or reindex.

## Limitations

- No ML experiment registry integration.
- No model artifact persistence found.
- No hyperparameter search framework.
- No model-based portfolio construction integration.
- `src/models` remains a separate placeholder package.

## Should Not Do

- Generate time splits internally.
- Shuffle time-series data.
- Hide label lookahead.
- Bypass `SupervisedDataset` alignment.

