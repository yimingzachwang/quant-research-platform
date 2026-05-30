"""Minimal walk-forward prediction helper.

A thin loop that sequences fit/predict calls across time splits.
No orchestration framework, no callbacks, no event system.

Splits come from src.validation.splits — this module does not generate them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from src.ml.contracts import PredictionSeries
from src.ml.datasets import SupervisedDataset
from src.ml.models.base import BaseMLModel
from src.validation.splits import TimeSplit


@dataclass
class WalkForwardPredictions:
    """Collected out-of-sample predictions from a walk-forward run.

    predictions[i] corresponds to splits[i].test window.
    Lengths are equal; empty windows are skipped during the run.
    """

    predictions: list[PredictionSeries]
    splits: list[TimeSplit]

    @property
    def n_splits(self) -> int:
        return len(self.splits)


def run_walk_forward_predictions(
    model: BaseMLModel,
    dataset: SupervisedDataset,
    splits: Sequence[TimeSplit],
) -> WalkForwardPredictions:
    """Run walk-forward fit/predict loop.

    For each TimeSplit:
      1. Slice dataset to the train window.
      2. Fit the model on the train slice.
      3. Slice dataset to the test window.
      4. Predict on the test features.

    Train and test slices that are empty (no observations after alignment)
    are skipped silently — this can happen at the edges of short datasets.
    The model object is re-fitted on each split; its internal state is
    overwritten in place (standard sklearn stateful API).

    Temporal integrity guarantee: split.train_end < split.test_start for
    every TimeSplit produced by src.validation.splits rolling/expanding
    generators.  This module does not verify it — callers are responsible for
    using chronologically valid splits.

    Args:
        model: Any object satisfying BaseMLModel (fit + predict).
        dataset: Full SupervisedDataset; sliced internally per split.
        splits: Chronological TimeSplit sequence.

    Returns:
        WalkForwardPredictions with per-split PredictionSeries and splits.
    """
    preds: list[PredictionSeries] = []
    valid_splits: list[TimeSplit] = []

    for split in splits:
        train_ds = dataset.slice(split.train_start, split.train_end)
        test_ds = dataset.slice(split.test_start, split.test_end)
        if len(train_ds.X) == 0 or len(test_ds.X) == 0:
            continue
        model.fit(train_ds)
        preds.append(model.predict(test_ds.X))
        valid_splits.append(split)

    return WalkForwardPredictions(predictions=preds, splits=valid_splits)


def concatenate_predictions(wf: WalkForwardPredictions) -> PredictionSeries:
    """Stitch out-of-sample predictions from all splits into one PredictionSeries.

    The predictions are sorted chronologically.  Raises if any index dates
    appear in more than one split (which would indicate overlapping test
    windows — a configuration error in the split generator, not a bug here).

    Args:
        wf: Result from run_walk_forward_predictions.

    Returns:
        PredictionSeries with all test-window predictions concatenated.

    Raises:
        ValueError: If wf has no predictions or indices overlap across splits.
    """
    if not wf.predictions:
        raise ValueError("WalkForwardPredictions contains no predictions to concatenate.")

    all_values = pd.concat([p.values for p in wf.predictions]).sort_index()

    if all_values.index.duplicated().any():
        dup_count = int(all_values.index.duplicated().sum())
        raise ValueError(
            f"Duplicate index entries ({dup_count}) detected across walk-forward splits. "
            "Test windows must not overlap."
        )

    return PredictionSeries(
        values=all_values,
        label_name=wf.predictions[0].label_name,
        model_name=wf.predictions[0].model_name,
    )
