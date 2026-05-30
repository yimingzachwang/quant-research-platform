"""Prediction contracts for future model outputs.

Defines the exchange type between trained models and downstream consumers
(backtesting, evaluation, reporting).  E0 does not include model training —
PredictionSeries is a placeholder contract that future models will satisfy.

Validation is advisory: validate_prediction_index_alignment() returns a list
of violation strings and never raises.  Callers decide whether to warn, error,
or proceed.  This matches the style of check_artefact_dir() in contracts.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from src.ml.datasets import SupervisedDataset


@dataclass(frozen=True)
class PredictionSeries:
    """Output contract for future predictive model outputs.

    values supports both:

    - pd.Series: single-target predictions (regression, binary classification,
      single-asset scores).
    - pd.DataFrame: panel predictions (cross-sectional scores, multi-asset
      signals, ranking outputs).

    Both forms share the same validation interface so that models producing
    either can plug into the same downstream evaluation infrastructure.

    model_name identifies which model produced the predictions — essential for
    multi-model comparison workflows and provenance.
    """

    values: pd.Series | pd.DataFrame  # model outputs, float dtype
    label_name: str                    # what was predicted
    model_name: str                    # which model produced this


def validate_prediction_index_alignment(
    predictions: PredictionSeries,
    dataset: "SupervisedDataset",
) -> list[str]:
    """Advisory alignment check between prediction outputs and a dataset.

    Returns a list of human-readable violation strings.  An empty list means
    all checks passed.  Never raises — callers decide severity.

    Checks performed:
    1. Prediction index is a subset of the dataset label index.
    2. All value columns have float dtype.
    3. No NaN values in predictions.

    Args:
        predictions: The model output to validate.
        dataset: The SupervisedDataset the model was evaluated against.

    Returns:
        List of violation strings; empty = no issues detected.
    """
    violations: list[str] = []

    pred_idx = predictions.values.index
    ds_idx = dataset.y.index

    extra = pred_idx.difference(ds_idx)
    if len(extra) > 0:
        violations.append(
            f"{len(extra)} prediction date(s) not present in dataset index"
        )

    if isinstance(predictions.values, pd.Series):
        if not pd.api.types.is_float_dtype(predictions.values):
            violations.append(
                f"values dtype is {predictions.values.dtype!r}; expected float"
            )
        nan_count = int(predictions.values.isna().sum())
        if nan_count > 0:
            violations.append(f"values contains {nan_count} NaN row(s)")
    else:
        for col in predictions.values.columns:
            col_series = predictions.values[col]
            if not pd.api.types.is_float_dtype(col_series):
                violations.append(
                    f"column {col!r} dtype is {col_series.dtype!r}; expected float"
                )
        nan_count = int(predictions.values.isna().sum().sum())
        if nan_count > 0:
            violations.append(f"values contains {nan_count} NaN cell(s)")

    return violations
