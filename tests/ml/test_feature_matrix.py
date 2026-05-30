"""Tests for src.ml.feature_matrix.

Focus areas: callable composition, column naming, NaN alignment,
leakage timing, and index identity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.feature_matrix import align_features_and_labels, build_feature_matrix


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _prices(n: int = 100, tickers: list[str] | None = None, seed: int = 7) -> pd.DataFrame:
    tickers = tickers or ["SPY", "QQQ", "IWM"]
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    data = {t: 100.0 * np.exp(rng.normal(0.0002, 0.01, n).cumsum()) for t in tickers}
    return pd.DataFrame(data, index=idx)


def _feature_df(n: int = 80, n_features: int = 3, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    data = {f"f{i}": rng.standard_normal(n) for i in range(n_features)}
    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------------------
# build_feature_matrix
# ---------------------------------------------------------------------------


def test_build_feature_matrix_single_series_fn():
    prices = _prices()
    fns = {"mean_price": lambda p: p.mean(axis=1)}
    X = build_feature_matrix(prices, fns)
    assert "mean_price" in X.columns
    assert X.shape[1] == 1


def test_build_feature_matrix_series_renamed_to_key():
    prices = _prices()
    fn = lambda p: pd.Series(p.mean(axis=1), name="something_else")
    X = build_feature_matrix(prices, {"my_feature": fn})
    assert "my_feature" in X.columns
    assert "something_else" not in X.columns


def test_build_feature_matrix_multiple_series_fns():
    prices = _prices()
    fns = {
        "mean": lambda p: p.mean(axis=1),
        "std": lambda p: p.std(axis=1),
    }
    X = build_feature_matrix(prices, fns)
    assert set(X.columns) == {"mean", "std"}
    assert X.shape == (len(prices), 2)


def test_build_feature_matrix_single_col_dataframe_renamed():
    prices = _prices(tickers=["SPY"])
    fn = lambda p: p.pct_change().rename(columns={"SPY": "original_name"})
    X = build_feature_matrix(prices, {"my_col": fn})
    assert list(X.columns) == ["my_col"]


def test_build_feature_matrix_multi_col_dataframe_prefixed():
    prices = _prices(tickers=["A", "B", "C"])
    fn = lambda p: p.pct_change()
    X = build_feature_matrix(prices, {"ret": fn})
    expected_cols = {"ret_A", "ret_B", "ret_C"}
    assert set(X.columns) == expected_cols


def test_build_feature_matrix_mixed_series_and_dataframe():
    prices = _prices(tickers=["A", "B"])
    fns = {
        "avg": lambda p: p.mean(axis=1),
        "ret": lambda p: p.pct_change(),
    }
    X = build_feature_matrix(prices, fns)
    assert "avg" in X.columns
    assert "ret_A" in X.columns
    assert "ret_B" in X.columns


def test_build_feature_matrix_empty_fns_returns_empty_df():
    prices = _prices()
    X = build_feature_matrix(prices, {})
    assert isinstance(X, pd.DataFrame)
    assert len(X) == len(prices)
    assert X.shape[1] == 0


def test_build_feature_matrix_index_matches_prices():
    prices = _prices()
    fns = {"f": lambda p: p.mean(axis=1)}
    X = build_feature_matrix(prices, fns)
    assert X.index.equals(prices.index)


def test_build_feature_matrix_does_not_drop_nans():
    # build_feature_matrix should NOT drop NaN rows — that is align's job
    prices = _prices(n=50)
    # rolling mean introduces leading NaN
    fns = {"roll": lambda p: p.rolling(20).mean().iloc[:, 0]}
    X = build_feature_matrix(prices, fns)
    assert X["roll"].isna().any()
    assert len(X) == len(prices)


def test_build_feature_matrix_reuses_existing_feature_fn():
    from src.features.momentum import momentum

    prices = _prices(tickers=["SPY"])
    fns = {"mom_20": lambda p: momentum(p["SPY"], window=20)}
    X = build_feature_matrix(prices, fns)
    assert "mom_20" in X.columns


# ---------------------------------------------------------------------------
# align_features_and_labels
# ---------------------------------------------------------------------------


def test_align_no_nan_unchanged():
    n = 50
    X = _feature_df(n=n)
    y = pd.Series(np.random.default_rng(0).standard_normal(n), index=X.index)
    X_a, y_a = align_features_and_labels(X, y)
    assert len(X_a) == n
    assert X_a.index.equals(y_a.index)


def test_align_drops_rows_where_X_has_nan():
    X = _feature_df(n=50)
    X.iloc[10, 0] = float("nan")
    y = pd.Series(1.0, index=X.index)
    X_a, y_a = align_features_and_labels(X, y)
    assert 50 - 1 == len(X_a)
    assert X_a.notna().all().all()


def test_align_drops_rows_where_y_is_nan():
    n = 60
    X = _feature_df(n=n)
    y = pd.Series(np.random.default_rng(0).standard_normal(n), index=X.index)
    # Simulate trailing NaN from shift(-horizon)
    y.iloc[-10:] = float("nan")
    X_a, y_a = align_features_and_labels(X, y)
    assert len(X_a) == n - 10
    assert y_a.notna().all()


def test_align_inner_join_on_index():
    idx_a = pd.date_range("2020-01-01", periods=50, freq="B")
    idx_b = pd.date_range("2020-02-01", periods=50, freq="B")
    X = pd.DataFrame({"f": 1.0}, index=idx_a)
    y = pd.Series(1.0, index=idx_b)
    X_a, y_a = align_features_and_labels(X, y)
    common = idx_a.intersection(idx_b)
    assert len(X_a) == len(common)


def test_align_identical_index_on_outputs():
    X = _feature_df(n=80)
    y = pd.Series(np.random.default_rng(5).standard_normal(80), index=X.index)
    y.iloc[-5:] = float("nan")
    X_a, y_a = align_features_and_labels(X, y)
    assert X_a.index.equals(y_a.index)


def test_align_preserves_series_type():
    X = _feature_df(n=50)
    y = pd.Series(1.0, index=X.index, name="my_label")
    X_a, y_a = align_features_and_labels(X, y)
    assert isinstance(y_a, pd.Series)


def test_align_preserves_dataframe_label():
    X = _feature_df(n=50)
    y_df = pd.DataFrame({"rank_A": 0.8, "rank_B": 0.3}, index=X.index)
    X_a, y_a = align_features_and_labels(X, y_df)
    assert isinstance(y_a, pd.DataFrame)
    assert list(y_a.columns) == ["rank_A", "rank_B"]


def test_align_dataframe_label_drops_nan_rows():
    n = 60
    X = _feature_df(n=n)
    y_df = pd.DataFrame(
        {"a": np.random.default_rng(0).standard_normal(n), "b": 1.0},
        index=X.index,
    )
    y_df.iloc[-8:, 0] = float("nan")
    X_a, y_a = align_features_and_labels(X, y_df)
    assert len(X_a) == n - 8
    assert y_a.notna().all().all()


def test_align_no_mutation_of_inputs():
    X = _feature_df(n=50)
    y = pd.Series(np.random.default_rng(0).standard_normal(50), index=X.index)
    y.iloc[-5:] = float("nan")
    X_copy = X.copy()
    y_copy = y.copy()
    align_features_and_labels(X, y)
    pd.testing.assert_frame_equal(X, X_copy)
    pd.testing.assert_series_equal(y, y_copy)
