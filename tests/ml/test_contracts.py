"""Tests for src.ml.contracts.

Focus areas: PredictionSeries with Series and DataFrame values, advisory
validation correctness, and never-raise guarantee.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.ml.contracts import PredictionSeries, validate_prediction_index_alignment
from src.ml.datasets import SupervisedDataset, build_supervised_dataset
from src.ml.labels import forward_returns

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _prices(n: int = 100, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"A": 100.0 * np.exp(rng.normal(0.0002, 0.01, n).cumsum()),
         "B": 100.0 * np.exp(rng.normal(0.0001, 0.01, n).cumsum())},
        index=idx,
    )


def _make_dataset(n: int = 100, horizon: int = 10) -> SupervisedDataset:
    prices = _prices(n=n)
    X = prices.pct_change().dropna()
    y = forward_returns(prices["A"], horizon=horizon)
    return build_supervised_dataset(X, y, horizon=horizon, label_name="fwd_A")


def _clean_predictions(dataset: SupervisedDataset) -> PredictionSeries:
    values = pd.Series(0.5, index=dataset.y.index, dtype="float64", name="pred")
    return PredictionSeries(values=values, label_name="fwd_A", model_name="TestModel")


# ---------------------------------------------------------------------------
# PredictionSeries construction
# ---------------------------------------------------------------------------


def test_prediction_series_series_values():
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    values = pd.Series(np.random.default_rng(0).standard_normal(10), index=idx)
    ps = PredictionSeries(values=values, label_name="target", model_name="m1")
    assert isinstance(ps.values, pd.Series)


def test_prediction_series_dataframe_values():
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    values = pd.DataFrame(
        {"A": np.ones(10, dtype=float), "B": np.zeros(10, dtype=float)}, index=idx
    )
    ps = PredictionSeries(values=values, label_name="rank", model_name="panel_model")
    assert isinstance(ps.values, pd.DataFrame)


def test_prediction_series_is_frozen():
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    values = pd.Series(1.0, index=idx, dtype=float)
    ps = PredictionSeries(values=values, label_name="t", model_name="m")
    with pytest.raises((Exception,)):
        ps.model_name = "other"  # type: ignore[misc]


def test_prediction_series_stores_metadata():
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    values = pd.Series(1.0, index=idx, dtype=float)
    ps = PredictionSeries(values=values, label_name="my_label", model_name="my_model")
    assert ps.label_name == "my_label"
    assert ps.model_name == "my_model"


# ---------------------------------------------------------------------------
# validate_prediction_index_alignment — clean cases
# ---------------------------------------------------------------------------


def test_validate_clean_predictions_no_violations():
    ds = _make_dataset()
    ps = _clean_predictions(ds)
    violations = validate_prediction_index_alignment(ps, ds)
    assert violations == []


def test_validate_returns_list():
    ds = _make_dataset()
    ps = _clean_predictions(ds)
    result = validate_prediction_index_alignment(ps, ds)
    assert isinstance(result, list)


def test_validate_never_raises_on_bad_input():
    ds = _make_dataset()
    # Completely disjoint index — should return violations, not raise
    bad_idx = pd.date_range("2000-01-01", periods=10, freq="B")
    ps = PredictionSeries(
        values=pd.Series(1.0, index=bad_idx, dtype=float),
        label_name="x",
        model_name="m",
    )
    result = validate_prediction_index_alignment(ps, ds)
    assert isinstance(result, list)
    assert len(result) >= 1


# ---------------------------------------------------------------------------
# validate_prediction_index_alignment — violation detection
# ---------------------------------------------------------------------------


def test_validate_extra_dates_detected():
    ds = _make_dataset()
    # Predictions include dates not in the dataset
    future_idx = pd.date_range("2030-01-01", periods=5, freq="B")
    extended_idx = ds.y.index.append(future_idx)
    ps = PredictionSeries(
        values=pd.Series(0.5, index=extended_idx, dtype=float),
        label_name="fwd_A",
        model_name="m",
    )
    violations = validate_prediction_index_alignment(ps, ds)
    assert any("5" in v for v in violations)


def test_validate_detects_non_float_series():
    ds = _make_dataset()
    int_values = pd.Series(1, index=ds.y.index, dtype="int64")
    ps = PredictionSeries(values=int_values, label_name="fwd_A", model_name="m")
    violations = validate_prediction_index_alignment(ps, ds)
    assert any("float" in v.lower() or "dtype" in v.lower() for v in violations)


def test_validate_detects_nan_in_series():
    ds = _make_dataset()
    values = pd.Series(0.5, index=ds.y.index, dtype=float)
    values.iloc[0] = float("nan")
    ps = PredictionSeries(values=values, label_name="fwd_A", model_name="m")
    violations = validate_prediction_index_alignment(ps, ds)
    assert any("nan" in v.lower() or "NaN" in v for v in violations)


def test_validate_dataframe_clean():
    from src.ml.labels import ranking_target

    prices = _prices(n=100)
    X = prices.pct_change().dropna()
    y = ranking_target(prices, horizon=10)
    ds = build_supervised_dataset(X, y, horizon=10, label_name="rank")
    values = pd.DataFrame(
        {"A": 0.6, "B": 0.4}, index=ds.y.index, dtype=float
    )
    ps = PredictionSeries(values=values, label_name="rank", model_name="panel")
    violations = validate_prediction_index_alignment(ps, ds)
    assert violations == []


def test_validate_detects_non_float_dataframe_column():
    from src.ml.labels import ranking_target

    prices = _prices(n=100)
    X = prices.pct_change().dropna()
    y = ranking_target(prices, horizon=10)
    ds = build_supervised_dataset(X, y, horizon=10, label_name="rank")
    values = pd.DataFrame(
        {"A": pd.Series(1, index=ds.y.index, dtype="int64"),
         "B": pd.Series(0.5, index=ds.y.index, dtype=float)},
    )
    ps = PredictionSeries(values=values, label_name="rank", model_name="panel")
    violations = validate_prediction_index_alignment(ps, ds)
    assert any("A" in v or "dtype" in v.lower() or "float" in v.lower() for v in violations)


def test_validate_detects_nan_in_dataframe():
    from src.ml.labels import ranking_target

    prices = _prices(n=100)
    X = prices.pct_change().dropna()
    y = ranking_target(prices, horizon=10)
    ds = build_supervised_dataset(X, y, horizon=10, label_name="rank")
    values = pd.DataFrame({"A": 0.5, "B": 0.5}, index=ds.y.index, dtype=float)
    values.iloc[0, 0] = float("nan")
    ps = PredictionSeries(values=values, label_name="rank", model_name="panel")
    violations = validate_prediction_index_alignment(ps, ds)
    assert any("nan" in v.lower() or "NaN" in v for v in violations)


# ---------------------------------------------------------------------------
# Package-level imports
# ---------------------------------------------------------------------------


def test_importable_from_package():
    from src.ml import PredictionSeries as PS
    from src.ml import validate_prediction_index_alignment as vpa
    assert callable(vpa)
    assert PS is PredictionSeries
