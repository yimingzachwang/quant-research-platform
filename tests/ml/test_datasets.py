"""Tests for src.ml.datasets.

Focus areas: SupervisedDataset construction, slice(), frozen semantics,
dataset_hash() determinism and metadata-only hashing.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.ml.datasets import SupervisedDataset, build_supervised_dataset, dataset_hash
from src.ml.labels import forward_returns


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _prices(n: int = 120, tickers: list[str] | None = None, seed: int = 3) -> pd.DataFrame:
    tickers = tickers or ["A", "B", "C"]
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    data = {t: 100.0 * np.exp(rng.normal(0.0001, 0.01, n).cumsum()) for t in tickers}
    return pd.DataFrame(data, index=idx)


def _simple_dataset(horizon: int = 10, n: int = 120) -> SupervisedDataset:
    prices = _prices(n=n)
    X = prices.pct_change().dropna().to_frame("A_ret") if len(prices.columns) == 1 else prices.pct_change().dropna()
    y = forward_returns(prices["A"], horizon=horizon)
    return build_supervised_dataset(X, y, horizon=horizon, label_name="fwd_ret")


# ---------------------------------------------------------------------------
# build_supervised_dataset
# ---------------------------------------------------------------------------


def test_build_supervised_dataset_returns_correct_type():
    ds = _simple_dataset()
    assert isinstance(ds, SupervisedDataset)


def test_build_supervised_dataset_no_nan_in_X():
    ds = _simple_dataset()
    assert ds.X.notna().all().all()


def test_build_supervised_dataset_no_nan_in_y():
    ds = _simple_dataset()
    if isinstance(ds.y, pd.Series):
        assert ds.y.notna().all()
    else:
        assert ds.y.notna().all().all()


def test_build_supervised_dataset_X_y_same_index():
    ds = _simple_dataset()
    assert ds.X.index.equals(ds.y.index)


def test_build_supervised_dataset_horizon_stored():
    ds = build_supervised_dataset(
        _prices().pct_change().dropna(),
        forward_returns(_prices()["A"], horizon=20),
        horizon=20,
    )
    assert ds.horizon == 20


def test_build_supervised_dataset_feature_names_match_columns():
    prices = _prices()
    X = prices.pct_change().dropna()
    y = forward_returns(prices["A"], horizon=5)
    ds = build_supervised_dataset(X, y, horizon=5)
    assert ds.feature_names == tuple(str(c) for c in X.columns)


def test_build_supervised_dataset_label_name_explicit():
    prices = _prices()
    X = prices.pct_change().dropna()
    y = forward_returns(prices["A"], horizon=5)
    ds = build_supervised_dataset(X, y, horizon=5, label_name="custom_label")
    assert ds.label_name == "custom_label"


def test_build_supervised_dataset_label_name_from_series_name():
    prices = _prices()
    X = prices.pct_change().dropna()
    y = forward_returns(prices["A"], horizon=5)
    assert y.name is not None  # forward_returns inherits prices["A"].name
    ds = build_supervised_dataset(X, y, horizon=5)
    assert ds.label_name == str(y.name)


def test_build_supervised_dataset_label_name_default_target():
    prices = _prices()
    X = prices.pct_change().dropna()
    y = forward_returns(prices["A"], horizon=5)
    y.name = None
    ds = build_supervised_dataset(X, y, horizon=5)
    assert ds.label_name == "target"


def test_build_supervised_dataset_shrinks_index_by_horizon():
    n = 100
    horizon = 15
    prices = _prices(n=n)
    X = prices.pct_change().dropna()
    y = forward_returns(prices["A"], horizon=horizon)
    ds = build_supervised_dataset(X, y, horizon=horizon)
    # Dataset must be strictly smaller than input (trailing NaN rows removed)
    assert len(ds.X) < n


def test_build_supervised_dataset_with_dataframe_y():
    from src.ml.labels import ranking_target

    prices = _prices()
    X = prices.pct_change().dropna()
    y = ranking_target(prices, horizon=10)
    ds = build_supervised_dataset(X, y, horizon=10, label_name="cross_rank")
    assert isinstance(ds.y, pd.DataFrame)
    assert ds.X.index.equals(ds.y.index)


# ---------------------------------------------------------------------------
# SupervisedDataset.slice()
# ---------------------------------------------------------------------------


def test_slice_returns_new_dataset():
    ds = _simple_dataset()
    start = ds.X.index[10]
    end = ds.X.index[40]
    sliced = ds.slice(start, end)
    assert sliced is not ds


def test_slice_restricts_date_range():
    ds = _simple_dataset()
    start = ds.X.index[10]
    end = ds.X.index[40]
    sliced = ds.slice(start, end)
    assert sliced.X.index[0] >= start
    assert sliced.X.index[-1] <= end


def test_slice_preserves_metadata():
    ds = _simple_dataset(horizon=7)
    sliced = ds.slice(ds.X.index[5], ds.X.index[30])
    assert sliced.feature_names == ds.feature_names
    assert sliced.label_name == ds.label_name
    assert sliced.horizon == ds.horizon


def test_slice_X_y_remain_aligned():
    ds = _simple_dataset()
    sliced = ds.slice(ds.X.index[5], ds.X.index[50])
    assert sliced.X.index.equals(sliced.y.index)


def test_slice_accepts_string_dates():
    ds = _simple_dataset()
    sliced = ds.slice("2020-03-01", "2020-06-30")
    assert isinstance(sliced, SupervisedDataset)


# ---------------------------------------------------------------------------
# SupervisedDataset frozen semantics
# ---------------------------------------------------------------------------


def test_supervised_dataset_is_frozen():
    ds = _simple_dataset()
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        ds.horizon = 999  # type: ignore[misc]


def test_supervised_dataset_feature_names_is_tuple():
    ds = _simple_dataset()
    assert isinstance(ds.feature_names, tuple)


# ---------------------------------------------------------------------------
# dataset_hash
# ---------------------------------------------------------------------------


def test_dataset_hash_returns_string():
    ds = _simple_dataset()
    assert isinstance(dataset_hash(ds), str)


def test_dataset_hash_12_chars():
    ds = _simple_dataset()
    assert len(dataset_hash(ds)) == 12


def test_dataset_hash_lowercase_hex():
    ds = _simple_dataset()
    h = dataset_hash(ds)
    assert all(c in "0123456789abcdef" for c in h)


def test_dataset_hash_deterministic():
    ds = _simple_dataset()
    assert dataset_hash(ds) == dataset_hash(ds)


def test_dataset_hash_changes_with_horizon():
    prices = _prices()
    X = prices.pct_change().dropna()
    ds5 = build_supervised_dataset(X, forward_returns(prices["A"], 5), horizon=5)
    ds10 = build_supervised_dataset(X, forward_returns(prices["A"], 10), horizon=10)
    assert dataset_hash(ds5) != dataset_hash(ds10)


def test_dataset_hash_changes_with_feature_names():
    prices = _prices()
    X = prices.pct_change().dropna()
    y = forward_returns(prices["A"], 5)
    ds_original = build_supervised_dataset(X, y, horizon=5)
    X_renamed = X.rename(columns=lambda c: f"new_{c}")
    ds_renamed = build_supervised_dataset(X_renamed, y, horizon=5)
    assert dataset_hash(ds_original) != dataset_hash(ds_renamed)


def test_dataset_hash_changes_with_date_range():
    prices = _prices(n=200)
    X = prices.pct_change().dropna()
    y = forward_returns(prices["A"], 10)
    ds_full = build_supervised_dataset(X, y, horizon=10)
    ds_sliced = ds_full.slice(ds_full.X.index[20], ds_full.X.index[80])
    assert dataset_hash(ds_full) != dataset_hash(ds_sliced)


def test_dataset_hash_metadata_only_not_data_values():
    # Two datasets with different data but same structural metadata should have
    # different hashes only if shape or index differs.
    prices1 = _prices(seed=1)
    prices2 = _prices(seed=99)  # different values, same shape
    X1 = prices1.pct_change().dropna()
    X2 = prices2.pct_change().dropna()
    y1 = forward_returns(prices1["A"], 10)
    y2 = forward_returns(prices2["A"], 10)
    ds1 = build_supervised_dataset(X1, y1, horizon=10)
    ds2 = build_supervised_dataset(X2, y2, horizon=10)
    # Same metadata (shape, columns, horizon, index range) → same hash
    # (index values are the same since both use the same date_range)
    assert dataset_hash(ds1) == dataset_hash(ds2)


def test_dataset_hash_empty_dataset():
    prices = _prices(n=20)
    X = prices.pct_change().dropna()
    y = forward_returns(prices["A"], 10)
    ds = build_supervised_dataset(X, y, horizon=10)
    empty = ds.slice("1900-01-01", "1900-01-02")
    h = dataset_hash(empty)
    assert isinstance(h, str)
    assert len(h) == 12
