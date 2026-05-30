"""Tests for linear regression model wrappers.

Focus: fit/predict correctness, PredictionSeries contract, index preservation,
leakage safety, dtype enforcement, fit-before-predict guard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.contracts import PredictionSeries
from src.ml.datasets import SupervisedDataset
from src.ml.models.linear import (
    ElasticNetRegressionModel,
    LassoRegressionModel,
    LinearRegressionModel,
    RidgeRegressionModel,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _regression_dataset(
    n_obs: int = 150, n_features: int = 4, seed: int = 42
) -> SupervisedDataset:
    """Clean regression dataset: y = 2*f0 - f1 + noise."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_obs, freq="B")
    X = pd.DataFrame(
        rng.standard_normal((n_obs, n_features)),
        index=idx,
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = pd.Series(
        2.0 * X["f0"] - X["f1"] + rng.normal(0, 0.05, n_obs),
        index=idx,
        name="target",
    )
    return SupervisedDataset(
        X=X, y=y, feature_names=tuple(X.columns), label_name="target", horizon=5
    )


def _df_y_dataset(n_obs: int = 100) -> SupervisedDataset:
    """Dataset with DataFrame y — should be rejected by linear models."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=n_obs, freq="B")
    X = pd.DataFrame({"f0": rng.standard_normal(n_obs)}, index=idx)
    y = pd.DataFrame({"a": rng.standard_normal(n_obs)}, index=idx)
    return SupervisedDataset(
        X=X, y=y, feature_names=("f0",), label_name="multi", horizon=5
    )


def _train_test_split(
    ds: SupervisedDataset, train_frac: float = 0.7
) -> tuple[SupervisedDataset, SupervisedDataset]:
    n = len(ds.X)
    cutoff = ds.X.index[int(n * train_frac)]
    return ds.slice(ds.X.index[0], cutoff), ds.slice(ds.X.index[int(n * train_frac) + 1], ds.X.index[-1])


# ---------------------------------------------------------------------------
# LinearRegressionModel
# ---------------------------------------------------------------------------


def test_linear_fit_predict_returns_prediction_series():
    ds = _regression_dataset()
    train, test = _train_test_split(ds)
    model = LinearRegressionModel()
    model.fit(train)
    result = model.predict(test.X)
    assert isinstance(result, PredictionSeries)


def test_linear_prediction_index_matches_input():
    ds = _regression_dataset()
    train, test = _train_test_split(ds)
    model = LinearRegressionModel()
    model.fit(train)
    result = model.predict(test.X)
    assert result.values.index.equals(test.X.index)


def test_linear_prediction_dtype_float64():
    ds = _regression_dataset()
    train, test = _train_test_split(ds)
    model = LinearRegressionModel()
    model.fit(train)
    result = model.predict(test.X)
    assert result.values.dtype == np.dtype("float64")


def test_linear_prediction_no_nan():
    ds = _regression_dataset()
    train, test = _train_test_split(ds)
    model = LinearRegressionModel()
    model.fit(train)
    result = model.predict(test.X)
    assert result.values.notna().all()


def test_linear_model_name_populated():
    ds = _regression_dataset()
    train, test = _train_test_split(ds)
    model = LinearRegressionModel()
    model.fit(train)
    result = model.predict(test.X)
    assert result.model_name == "LinearRegression"


def test_linear_label_name_matches_dataset():
    ds = _regression_dataset()
    train, test = _train_test_split(ds)
    model = LinearRegressionModel()
    model.fit(train)
    result = model.predict(test.X)
    assert result.label_name == ds.label_name


def test_linear_predict_before_fit_raises():
    model = LinearRegressionModel()
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    X = pd.DataFrame({"f0": [1.0] * 5}, index=idx)
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(X)


def test_linear_rejects_dataframe_y():
    model = LinearRegressionModel()
    with pytest.raises(TypeError, match="pd.Series"):
        model.fit(_df_y_dataset())


def test_linear_predictions_correlated_with_truth():
    # With a clean linear target, predictions should correlate > 0.9
    ds = _regression_dataset(n_obs=500, seed=0)
    train, test = _train_test_split(ds, train_frac=0.8)
    model = LinearRegressionModel()
    model.fit(train)
    result = model.predict(test.X)
    corr = np.corrcoef(result.values.to_numpy(), test.y.to_numpy())[0, 1]
    assert corr > 0.9


def test_linear_no_future_timestamps():
    ds = _regression_dataset()
    train, test = _train_test_split(ds)
    model = LinearRegressionModel()
    model.fit(train)
    result = model.predict(test.X)
    # Prediction index must not extend beyond the input's index
    assert result.values.index.max() <= test.X.index.max()
    assert result.values.index.min() >= test.X.index.min()


def test_linear_multi_feature_support():
    ds = _regression_dataset(n_features=10)
    train, test = _train_test_split(ds)
    model = LinearRegressionModel()
    model.fit(train)
    result = model.predict(test.X)
    assert len(result.values) == len(test.X)


# ---------------------------------------------------------------------------
# RidgeRegressionModel
# ---------------------------------------------------------------------------


def test_ridge_default_alpha():
    model = RidgeRegressionModel()
    assert model._alpha == 1.0


def test_ridge_name_includes_alpha():
    model = RidgeRegressionModel(alpha=0.5)
    assert "0.5" in model.name


def test_ridge_fit_predict():
    ds = _regression_dataset()
    train, test = _train_test_split(ds)
    model = RidgeRegressionModel(alpha=1.0)
    model.fit(train)
    result = model.predict(test.X)
    assert isinstance(result, PredictionSeries)
    assert result.values.dtype == np.dtype("float64")
    assert result.values.index.equals(test.X.index)


def test_ridge_predict_before_fit_raises():
    model = RidgeRegressionModel()
    idx = pd.date_range("2020-01-01", periods=3, freq="B")
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(pd.DataFrame({"f0": [1.0] * 3}, index=idx))


# ---------------------------------------------------------------------------
# LassoRegressionModel
# ---------------------------------------------------------------------------


def test_lasso_default_params():
    model = LassoRegressionModel()
    assert model._alpha == 1.0
    assert model._max_iter == 1000


def test_lasso_name_includes_alpha():
    model = LassoRegressionModel(alpha=0.1)
    assert "0.1" in model.name


def test_lasso_fit_predict():
    ds = _regression_dataset()
    train, test = _train_test_split(ds)
    model = LassoRegressionModel(alpha=0.01)
    model.fit(train)
    result = model.predict(test.X)
    assert isinstance(result, PredictionSeries)
    assert result.values.index.equals(test.X.index)
    assert result.values.dtype == np.dtype("float64")


def test_lasso_predict_before_fit_raises():
    model = LassoRegressionModel()
    idx = pd.date_range("2020-01-01", periods=3, freq="B")
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(pd.DataFrame({"f0": [1.0] * 3}, index=idx))


# ---------------------------------------------------------------------------
# ElasticNetRegressionModel
# ---------------------------------------------------------------------------


def test_elasticnet_default_params():
    model = ElasticNetRegressionModel()
    assert model._alpha == 1.0
    assert model._l1_ratio == 0.5


def test_elasticnet_name_includes_both_params():
    model = ElasticNetRegressionModel(alpha=0.2, l1_ratio=0.7)
    assert "0.2" in model.name
    assert "0.7" in model.name


def test_elasticnet_fit_predict():
    ds = _regression_dataset()
    train, test = _train_test_split(ds)
    model = ElasticNetRegressionModel(alpha=0.01, l1_ratio=0.5)
    model.fit(train)
    result = model.predict(test.X)
    assert isinstance(result, PredictionSeries)
    assert result.values.index.equals(test.X.index)
    assert result.values.dtype == np.dtype("float64")


def test_elasticnet_predict_before_fit_raises():
    model = ElasticNetRegressionModel()
    idx = pd.date_range("2020-01-01", periods=3, freq="B")
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(pd.DataFrame({"f0": [1.0] * 3}, index=idx))


# ---------------------------------------------------------------------------
# Re-fit overwrites prior state
# ---------------------------------------------------------------------------


def test_refit_overwrites_previous_model():
    ds1 = _regression_dataset(seed=1)
    ds2 = _regression_dataset(seed=99)
    train1, test = _train_test_split(ds1)
    train2, _ = _train_test_split(ds2)
    model = LinearRegressionModel()
    model.fit(train1)
    pred1 = model.predict(test.X)
    model.fit(train2)
    pred2 = model.predict(test.X)
    # Predictions should differ after refitting on different data
    assert not pred1.values.equals(pred2.values)
