"""Tests for src.ml.diagnostics.stability."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.ml.diagnostics.stability import (
    coefficient_stability,
    prediction_drift,
    split_metric_table,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dates(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="B")


def _make_wf_result():
    """Minimal WalkForwardResult for split_metric_table tests."""
    import numpy as np
    import pandas as pd
    from src.backtesting.metrics import compute_metrics
    from src.validation.splits import TimeSplit
    from src.validation.walk_forward import SplitResult, WalkForwardResult

    dates = pd.date_range("2020-01-01", periods=200, freq="B")
    returns = pd.Series(np.random.randn(200) * 0.01, index=dates, name="SPY")

    splits = []
    for i in range(3):
        start = i * 40
        mid = start + 60
        end = mid + 40
        train_ret = returns.iloc[start:mid]
        test_ret = returns.iloc[mid:end]
        metrics = compute_metrics(test_ret)
        ts = TimeSplit(
            split_index=i,
            train_start=train_ret.index[0],
            train_end=train_ret.index[-1],
            test_start=test_ret.index[0],
            test_end=test_ret.index[-1],
        )
        splits.append(SplitResult(
            split=ts,
            strategy_name="test",
            metrics=metrics,
            equity_curve=pd.Series(dtype=float),
            weights=pd.DataFrame(),
        ))
    return WalkForwardResult(strategy_name="test", splits=splits)


# ---------------------------------------------------------------------------
# split_metric_table
# ---------------------------------------------------------------------------

class TestSplitMetricTable:
    def test_returns_dataframe(self):
        wf = _make_wf_result()
        df = split_metric_table(wf)
        assert isinstance(df, pd.DataFrame)

    def test_has_sharpe_column(self):
        wf = _make_wf_result()
        df = split_metric_table(wf)
        assert "sharpe_ratio" in df.columns

    def test_row_count_matches_splits(self):
        wf = _make_wf_result()
        df = split_metric_table(wf)
        assert len(df) == 3

    def test_empty_result_on_no_splits(self):
        from src.validation.walk_forward import WalkForwardResult
        wf = WalkForwardResult(strategy_name="test", splits=[])
        df = split_metric_table(wf)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# coefficient_stability
# ---------------------------------------------------------------------------

class TestCoefficientStability:
    def _make_coeff_df(self, n_splits: int = 5, n_features: int = 3,
                       seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        features = [f"feat_{i}" for i in range(n_features)]
        data = rng.standard_normal((n_splits, n_features))
        return pd.DataFrame(data, columns=features)

    def test_returns_dataframe(self):
        df = self._make_coeff_df()
        result = coefficient_stability(df)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns_present(self):
        df = self._make_coeff_df()
        result = coefficient_stability(df)
        for col in ["mean", "std", "sign_consistency", "min", "max"]:
            assert col in result.columns

    def test_index_is_feature_names(self):
        df = self._make_coeff_df(n_features=3)
        result = coefficient_stability(df)
        assert set(result.index) == {"feat_0", "feat_1", "feat_2"}

    def test_mean_is_correct(self):
        data = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [-1.0, -2.0, -3.0]})
        result = coefficient_stability(data)
        assert abs(result.loc["A", "mean"] - 2.0) < 1e-9
        assert abs(result.loc["B", "mean"] - (-2.0)) < 1e-9

    def test_std_is_correct(self):
        data = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
        result = coefficient_stability(data)
        expected_std = pd.Series([1.0, 2.0, 3.0]).std(ddof=1)
        assert abs(result.loc["A", "std"] - expected_std) < 1e-9

    def test_sign_consistency_all_positive(self):
        data = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
        result = coefficient_stability(data)
        assert abs(result.loc["A", "sign_consistency"] - 1.0) < 1e-9

    def test_sign_consistency_mixed(self):
        # 2 positive, 1 negative → mean positive → consistency = 2/3
        data = pd.DataFrame({"A": [1.0, 2.0, -0.5]})
        result = coefficient_stability(data)
        expected = 2.0 / 3.0
        assert abs(result.loc["A", "sign_consistency"] - expected) < 1e-9

    def test_sign_consistency_zero_mean(self):
        data = pd.DataFrame({"A": [1.0, -1.0]})
        result = coefficient_stability(data)
        assert pd.isna(result.loc["A", "sign_consistency"])

    def test_min_max_correct(self):
        data = pd.DataFrame({"A": [1.0, 5.0, 3.0]})
        result = coefficient_stability(data)
        assert abs(result.loc["A", "min"] - 1.0) < 1e-9
        assert abs(result.loc["A", "max"] - 5.0) < 1e-9

    def test_empty_input_returns_empty(self):
        df = pd.DataFrame(columns=["A", "B"])
        result = coefficient_stability(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_single_split_std_is_nan(self):
        data = pd.DataFrame({"A": [1.5]})
        result = coefficient_stability(data)
        assert pd.isna(result.loc["A", "std"])

    def test_nan_values_dropped_per_feature(self):
        data = pd.DataFrame({"A": [1.0, np.nan, 3.0]})
        result = coefficient_stability(data)
        # mean should be (1.0 + 3.0) / 2 = 2.0
        assert abs(result.loc["A", "mean"] - 2.0) < 1e-9


# ---------------------------------------------------------------------------
# prediction_drift
# ---------------------------------------------------------------------------

class TestPredictionDrift:
    def test_returns_series(self):
        idx = _make_dates(30)
        preds = pd.Series(np.random.randn(30), index=idx)
        drift = prediction_drift(preds, window=5)
        assert isinstance(drift, pd.Series)

    def test_series_named_correctly(self):
        idx = _make_dates(30)
        preds = pd.Series(np.random.randn(30), index=idx)
        drift = prediction_drift(preds, window=7)
        assert drift.name == "prediction_drift_7d"

    def test_first_window_minus_one_is_nan(self):
        idx = _make_dates(20)
        preds = pd.Series(np.random.randn(20), index=idx)
        drift = prediction_drift(preds, window=5)
        assert drift.iloc[:4].isna().all()
        assert not pd.isna(drift.iloc[4])

    def test_constant_series_drift_equals_constant(self):
        idx = _make_dates(20)
        preds = pd.Series(np.full(20, 3.5), index=idx)
        drift = prediction_drift(preds, window=5)
        clean = drift.dropna()
        assert (clean - 3.5).abs().max() < 1e-9

    def test_index_preserved(self):
        idx = _make_dates(25)
        preds = pd.Series(np.random.randn(25), index=idx)
        drift = prediction_drift(preds, window=5)
        assert drift.index.equals(idx)

    def test_raises_on_window_less_than_1(self):
        idx = _make_dates(10)
        preds = pd.Series(np.ones(10), index=idx)
        with pytest.raises(ValueError):
            prediction_drift(preds, window=0)

    def test_window_1_returns_original(self):
        idx = _make_dates(10)
        vals = np.random.randn(10)
        preds = pd.Series(vals, index=idx)
        drift = prediction_drift(preds, window=1)
        np.testing.assert_array_almost_equal(drift.to_numpy(), vals)

    def test_rolling_mean_is_correct(self):
        idx = _make_dates(10)
        preds = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0,
                           6.0, 7.0, 8.0, 9.0, 10.0], index=idx)
        drift = prediction_drift(preds, window=3)
        # Index 4 (0-based): mean of [3, 4, 5] = 4.0
        assert abs(drift.iloc[4] - 4.0) < 1e-9


# ---------------------------------------------------------------------------
# Package import
# ---------------------------------------------------------------------------

def test_import_from_src_ml():
    from src.ml import (
        coefficient_stability,
        prediction_drift,
        split_metric_table,
    )
    assert callable(coefficient_stability)
    assert callable(prediction_drift)
    assert callable(split_metric_table)
