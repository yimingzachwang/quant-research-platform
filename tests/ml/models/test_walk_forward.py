"""Tests for walk-forward prediction pipeline.

Focus: correct fit/predict sequencing, chronological integrity, concatenation,
no-overlap guarantee.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.ml.contracts import PredictionSeries
from src.ml.datasets import SupervisedDataset, build_supervised_dataset
from src.ml.labels import forward_returns
from src.ml.models.linear import LinearRegressionModel, RidgeRegressionModel
from src.ml.pipelines.walk_forward import (
    WalkForwardPredictions,
    concatenate_predictions,
    run_walk_forward_predictions,
)
from src.validation.splits import rolling_time_splits

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _price_df(n: int = 400, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"SPY": 400.0 * np.exp(rng.normal(0.0003, 0.010, n).cumsum())},
        index=idx,
    )


def _dataset(n: int = 400, horizon: int = 10, seed: int = 1) -> SupervisedDataset:
    prices = _price_df(n=n, seed=seed)
    X_raw = prices.pct_change().dropna().rename(columns={"SPY": "ret_SPY"})
    y_raw = forward_returns(prices["SPY"], horizon=horizon)
    return build_supervised_dataset(X_raw, y_raw, horizon=horizon, label_name="fwd_SPY")


# ---------------------------------------------------------------------------
# run_walk_forward_predictions
# ---------------------------------------------------------------------------


def test_wf_returns_correct_type():
    ds = _dataset()
    splits = rolling_time_splits(ds.X.index, train_months=12, test_months=3)
    model = LinearRegressionModel()
    result = run_walk_forward_predictions(model, ds, splits)
    assert isinstance(result, WalkForwardPredictions)


def test_wf_predictions_count_matches_splits():
    ds = _dataset()
    splits = rolling_time_splits(ds.X.index, train_months=12, test_months=3)
    model = LinearRegressionModel()
    result = run_walk_forward_predictions(model, ds, splits)
    assert result.n_splits == len(result.predictions)


def test_wf_each_prediction_is_prediction_series():
    ds = _dataset()
    splits = rolling_time_splits(ds.X.index, train_months=12, test_months=3)
    model = LinearRegressionModel()
    result = run_walk_forward_predictions(model, ds, splits)
    for pred in result.predictions:
        assert isinstance(pred, PredictionSeries)


def test_wf_prediction_indices_within_test_windows():
    ds = _dataset()
    splits = rolling_time_splits(ds.X.index, train_months=12, test_months=3)
    model = LinearRegressionModel()
    result = run_walk_forward_predictions(model, ds, splits)
    for pred, split in zip(result.predictions, result.splits, strict=False):
        assert pred.values.index.min() >= split.test_start
        assert pred.values.index.max() <= split.test_end


def test_wf_predictions_float_dtype():
    ds = _dataset()
    splits = rolling_time_splits(ds.X.index, train_months=12, test_months=3)
    model = LinearRegressionModel()
    result = run_walk_forward_predictions(model, ds, splits)
    for pred in result.predictions:
        assert pred.values.dtype == np.dtype("float64")


def test_wf_no_nan_in_predictions():
    ds = _dataset()
    splits = rolling_time_splits(ds.X.index, train_months=12, test_months=3)
    model = LinearRegressionModel()
    result = run_walk_forward_predictions(model, ds, splits)
    for pred in result.predictions:
        assert pred.values.notna().all()


def test_wf_chronological_integrity():
    """Train window end must precede test window start for every split."""
    ds = _dataset()
    splits = rolling_time_splits(ds.X.index, train_months=12, test_months=3)
    for split in splits:
        assert split.train_end < split.test_start


def test_wf_test_windows_non_overlapping():
    """No date should appear in two test windows."""
    ds = _dataset()
    splits = rolling_time_splits(ds.X.index, train_months=12, test_months=3)
    model = LinearRegressionModel()
    result = run_walk_forward_predictions(model, ds, splits)
    all_indices = pd.Index([])
    for pred in result.predictions:
        overlap = all_indices.intersection(pred.values.index)
        assert len(overlap) == 0, f"Overlapping test index dates: {overlap}"
        all_indices = all_indices.append(pred.values.index)


def test_wf_skips_empty_windows():
    """Walk-forward should handle tiny datasets gracefully."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    X = pd.DataFrame({"f0": rng.standard_normal(10)}, index=idx)
    y = pd.Series(rng.standard_normal(10), index=idx, name="y")
    ds = SupervisedDataset(X=X, y=y, feature_names=("f0",), label_name="y", horizon=1)
    # Splits that are larger than the dataset — all windows should be skipped
    splits = rolling_time_splits(ds.X.index, train_months=24, test_months=12)
    model = LinearRegressionModel()
    result = run_walk_forward_predictions(model, ds, splits)
    # Either no splits, or all predictions were skipped
    assert result.n_splits == 0 or all(len(p.values) == 0 for p in result.predictions)


def test_wf_model_name_consistent_across_splits():
    ds = _dataset()
    splits = rolling_time_splits(ds.X.index, train_months=12, test_months=3)
    model = RidgeRegressionModel(alpha=0.5)
    result = run_walk_forward_predictions(model, ds, splits)
    names = {p.model_name for p in result.predictions}
    assert len(names) == 1  # same model, same name


# ---------------------------------------------------------------------------
# concatenate_predictions
# ---------------------------------------------------------------------------


def test_concatenate_produces_prediction_series():
    ds = _dataset()
    splits = rolling_time_splits(ds.X.index, train_months=12, test_months=3)
    model = LinearRegressionModel()
    wf = run_walk_forward_predictions(model, ds, splits)
    stitched = concatenate_predictions(wf)
    assert isinstance(stitched, PredictionSeries)


def test_concatenate_total_length():
    ds = _dataset()
    splits = rolling_time_splits(ds.X.index, train_months=12, test_months=3)
    model = LinearRegressionModel()
    wf = run_walk_forward_predictions(model, ds, splits)
    stitched = concatenate_predictions(wf)
    expected = sum(len(p.values) for p in wf.predictions)
    assert len(stitched.values) == expected


def test_concatenate_sorted_chronologically():
    ds = _dataset()
    splits = rolling_time_splits(ds.X.index, train_months=12, test_months=3)
    model = LinearRegressionModel()
    wf = run_walk_forward_predictions(model, ds, splits)
    stitched = concatenate_predictions(wf)
    assert stitched.values.index.is_monotonic_increasing


def test_concatenate_propagates_metadata():
    ds = _dataset()
    splits = rolling_time_splits(ds.X.index, train_months=12, test_months=3)
    model = LinearRegressionModel()
    wf = run_walk_forward_predictions(model, ds, splits)
    stitched = concatenate_predictions(wf)
    assert stitched.label_name == wf.predictions[0].label_name
    assert stitched.model_name == wf.predictions[0].model_name


def test_concatenate_raises_on_empty():
    empty_wf = WalkForwardPredictions(predictions=[], splits=[])
    with pytest.raises(ValueError, match="no predictions"):
        concatenate_predictions(empty_wf)


def test_concatenate_raises_on_duplicate_index():
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    p1 = PredictionSeries(pd.Series(1.0, index=idx, dtype=float), "y", "m")
    p2 = PredictionSeries(pd.Series(2.0, index=idx, dtype=float), "y", "m")
    from src.validation.splits import TimeSplit
    split = TimeSplit(
        split_index=0,
        train_start=idx[0], train_end=idx[2],
        test_start=idx[3], test_end=idx[4],
    )
    wf = WalkForwardPredictions(predictions=[p1, p2], splits=[split, split])
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        concatenate_predictions(wf)


# ---------------------------------------------------------------------------
# Package imports
# ---------------------------------------------------------------------------


def test_importable_from_pipelines_package():
    from src.ml.pipelines import (
        concatenate_predictions,
        run_walk_forward_predictions,
    )
    assert callable(run_walk_forward_predictions)
    assert callable(concatenate_predictions)
