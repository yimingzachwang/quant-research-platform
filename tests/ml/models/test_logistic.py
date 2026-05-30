"""Tests for LogisticRegressionModel.

Focus: probability output, binary label compatibility, fit-before-predict
guard, single-class guard, index and dtype correctness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.contracts import PredictionSeries
from src.ml.datasets import SupervisedDataset
from src.ml.labels import binary_direction_label, forward_returns
from src.ml.models.logistic import LogisticRegressionModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _binary_dataset(n_obs: int = 200, n_features: int = 3, seed: int = 7) -> SupervisedDataset:
    """Dataset with binary direction labels produced by E0 label function."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_obs, freq="B")
    X = pd.DataFrame(
        rng.standard_normal((n_obs, n_features)),
        index=idx,
        columns=[f"f{i}" for i in range(n_features)],
    )
    # Binary y: 0.0 / 1.0 using E0 label function on synthetic prices
    prices = pd.Series(
        100.0 * np.exp(rng.normal(0.0002, 0.01, n_obs).cumsum()), index=idx
    )
    y = binary_direction_label(prices, horizon=5)
    # Manually align and drop NaN (last 5 rows)
    valid = y.notna()
    return SupervisedDataset(
        X=X.loc[valid],
        y=y.loc[valid],
        feature_names=tuple(X.columns),
        label_name="direction",
        horizon=5,
    )


def _split(ds: SupervisedDataset, frac: float = 0.7):
    n = len(ds.X)
    cut = int(n * frac)
    return ds.slice(ds.X.index[0], ds.X.index[cut]), ds.slice(ds.X.index[cut + 1], ds.X.index[-1])


# ---------------------------------------------------------------------------
# Basic interface
# ---------------------------------------------------------------------------


def test_logistic_returns_prediction_series():
    ds = _binary_dataset()
    train, test = _split(ds)
    model = LogisticRegressionModel()
    model.fit(train)
    result = model.predict(test.X)
    assert isinstance(result, PredictionSeries)


def test_logistic_index_matches_input():
    ds = _binary_dataset()
    train, test = _split(ds)
    model = LogisticRegressionModel()
    model.fit(train)
    result = model.predict(test.X)
    assert result.values.index.equals(test.X.index)


def test_logistic_dtype_float64():
    ds = _binary_dataset()
    train, test = _split(ds)
    model = LogisticRegressionModel()
    model.fit(train)
    result = model.predict(test.X)
    assert result.values.dtype == np.dtype("float64")


def test_logistic_no_nan_in_predictions():
    ds = _binary_dataset()
    train, test = _split(ds)
    model = LogisticRegressionModel()
    model.fit(train)
    result = model.predict(test.X)
    assert result.values.notna().all()


def test_logistic_model_name_populated():
    model = LogisticRegressionModel(C=0.5)
    ds = _binary_dataset()
    train, test = _split(ds)
    model.fit(train)
    result = model.predict(test.X)
    assert "LogisticRegression" in result.model_name
    assert "0.5" in result.model_name


def test_logistic_label_name_propagated():
    ds = _binary_dataset()
    train, test = _split(ds)
    model = LogisticRegressionModel()
    model.fit(train)
    result = model.predict(test.X)
    assert result.label_name == "direction"


# ---------------------------------------------------------------------------
# Probability output
# ---------------------------------------------------------------------------


def test_logistic_predictions_are_probabilities():
    """Predictions must be in [0, 1] — class-1 probability scores."""
    ds = _binary_dataset()
    train, test = _split(ds)
    model = LogisticRegressionModel()
    model.fit(train)
    result = model.predict(test.X)
    assert (result.values >= 0.0).all()
    assert (result.values <= 1.0).all()


def test_logistic_does_not_return_hard_labels():
    """Verify predictions are not restricted to {0.0, 1.0}."""
    ds = _binary_dataset(n_obs=300)
    train, test = _split(ds)
    model = LogisticRegressionModel()
    model.fit(train)
    result = model.predict(test.X)
    unique_vals = result.values.unique()
    # Should have more than 2 unique values if returning probabilities
    assert len(unique_vals) > 2


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_logistic_predict_before_fit_raises():
    model = LogisticRegressionModel()
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    X = pd.DataFrame({"f0": [1.0] * 5}, index=idx)
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(X)


def test_logistic_rejects_dataframe_y():
    idx = pd.date_range("2020-01-01", periods=20, freq="B")
    X = pd.DataFrame({"f0": np.ones(20)}, index=idx)
    y_df = pd.DataFrame({"a": np.ones(20)}, index=idx)
    ds = SupervisedDataset(X=X, y=y_df, feature_names=("f0",), label_name="t", horizon=1)
    model = LogisticRegressionModel()
    with pytest.raises(TypeError, match="pd.Series"):
        model.fit(ds)


def test_logistic_raises_on_single_class():
    """Training data with only one class should raise a clear error."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=50, freq="B")
    X = pd.DataFrame(rng.standard_normal((50, 2)), index=idx, columns=["f0", "f1"])
    y = pd.Series(np.ones(50), index=idx, name="dir")  # all 1.0
    ds = SupervisedDataset(X=X, y=y, feature_names=("f0", "f1"), label_name="dir", horizon=5)
    model = LogisticRegressionModel()
    with pytest.raises(ValueError, match="2 classes"):
        model.fit(ds)


# ---------------------------------------------------------------------------
# Leakage / temporal safety
# ---------------------------------------------------------------------------


def test_logistic_no_future_timestamps():
    ds = _binary_dataset()
    train, test = _split(ds)
    model = LogisticRegressionModel()
    model.fit(train)
    result = model.predict(test.X)
    assert result.values.index.max() <= test.X.index.max()
    assert result.values.index.min() >= test.X.index.min()


def test_logistic_default_C():
    model = LogisticRegressionModel()
    assert model._C == 1.0


def test_logistic_name_includes_C():
    model = LogisticRegressionModel(C=2.5)
    assert "2.5" in model.name
