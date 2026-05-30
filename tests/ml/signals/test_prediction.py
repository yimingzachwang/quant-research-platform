"""Tests for src/ml/signals/prediction.py.

Focus: index preservation, dtype correctness, row normalization,
long/short exposure, TypeError on wrong input type, flat-row handling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.ml.contracts import PredictionSeries
from src.ml.signals.prediction import (
    long_short_weights,
    normalize_to_weights,
    sign_signal,
    threshold_signal,
    top_n_weights,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _series_preds(values: list[float], name: str = "m") -> PredictionSeries:
    idx = pd.date_range("2020-01-01", periods=len(values), freq="B")
    return PredictionSeries(
        values=pd.Series(values, index=idx, dtype="float64"),
        label_name="y",
        model_name=name,
    )


def _frame_preds(data: dict[str, list[float]], name: str = "m") -> PredictionSeries:
    idx = pd.date_range("2020-01-01", periods=len(next(iter(data.values()))), freq="B")
    df = pd.DataFrame(data, index=idx, dtype="float64")
    return PredictionSeries(values=df, label_name="y", model_name=name)


# ===========================================================================
# sign_signal
# ===========================================================================


def test_sign_signal_positive_values():
    p = _series_preds([0.5, 1.0, 2.0])
    result = sign_signal(p)
    assert (result == 1.0).all()


def test_sign_signal_negative_values():
    p = _series_preds([-0.5, -1.0])
    result = sign_signal(p)
    assert (result == -1.0).all()


def test_sign_signal_zero():
    p = _series_preds([0.0, 0.5, -0.5])
    result = sign_signal(p)
    assert result.iloc[0] == 0.0
    assert result.iloc[1] == 1.0
    assert result.iloc[2] == -1.0


def test_sign_signal_preserves_index():
    p = _series_preds([1.0, -1.0, 0.0])
    result = sign_signal(p)
    assert result.index.equals(p.values.index)


def test_sign_signal_float64_dtype():
    p = _series_preds([1.0, -1.0])
    assert sign_signal(p).dtype == np.dtype("float64")


def test_sign_signal_raises_on_dataframe():
    p = _frame_preds({"A": [1.0, 2.0], "B": [-1.0, -2.0]})
    with pytest.raises(TypeError, match="pd.Series"):
        sign_signal(p)


def test_sign_signal_nan_propagates():
    """NaN predictions produce NaN signal — the backtest engine treats NaN as flat."""
    p = _series_preds([float("nan"), 1.0])
    result = sign_signal(p)
    assert pd.isna(result.iloc[0])
    assert result.iloc[1] == 1.0


# ===========================================================================
# threshold_signal
# ===========================================================================


def test_threshold_signal_above_default():
    p = _series_preds([0.1, -0.1, 0.0])
    result = threshold_signal(p)  # default threshold=0.0
    assert result.iloc[0] == 1.0
    assert result.iloc[1] == 0.0
    assert result.iloc[2] == 0.0  # strictly greater than 0.0


def test_threshold_signal_custom_threshold():
    p = _series_preds([0.4, 0.6, 0.5])
    result = threshold_signal(p, threshold=0.5)
    assert result.iloc[0] == 0.0  # 0.4 not > 0.5
    assert result.iloc[1] == 1.0  # 0.6 > 0.5
    assert result.iloc[2] == 0.0  # 0.5 not strictly > 0.5


def test_threshold_signal_preserves_index():
    p = _series_preds([0.8, 0.2])
    result = threshold_signal(p, threshold=0.5)
    assert result.index.equals(p.values.index)


def test_threshold_signal_float64_dtype():
    p = _series_preds([0.9, 0.1])
    assert threshold_signal(p).dtype == np.dtype("float64")


def test_threshold_signal_raises_on_dataframe():
    p = _frame_preds({"A": [0.5, 0.6]})
    with pytest.raises(TypeError, match="pd.Series"):
        threshold_signal(p)


def test_threshold_signal_all_above():
    p = _series_preds([1.0, 2.0, 3.0])
    result = threshold_signal(p, threshold=0.0)
    assert (result == 1.0).all()


def test_threshold_signal_all_below():
    p = _series_preds([-1.0, -2.0])
    result = threshold_signal(p, threshold=0.0)
    assert (result == 0.0).all()


# ===========================================================================
# top_n_weights
# ===========================================================================


def test_top_n_weights_rows_sum_to_one():
    p = _frame_preds({"A": [3.0, 1.0], "B": [2.0, 4.0], "C": [1.0, 2.0]})
    result = top_n_weights(p, n=2)
    np.testing.assert_allclose(result.sum(axis=1).values, [1.0, 1.0])


def test_top_n_weights_selects_top_n():
    p = _frame_preds({"A": [3.0], "B": [2.0], "C": [1.0]})
    result = top_n_weights(p, n=2)
    # A and B are top-2; C should be 0
    assert result.loc[:, "C"].iloc[0] == 0.0
    assert result.loc[:, "A"].iloc[0] > 0.0
    assert result.loc[:, "B"].iloc[0] > 0.0


def test_top_n_weights_equal_weight_per_row():
    p = _frame_preds({"A": [3.0], "B": [2.0], "C": [1.0]})
    result = top_n_weights(p, n=2)
    # Equal weight for 2 selected assets = 0.5 each
    np.testing.assert_allclose(result.loc[:, "A"].iloc[0], 0.5)
    np.testing.assert_allclose(result.loc[:, "B"].iloc[0], 0.5)


def test_top_n_weights_preserves_index():
    p = _frame_preds({"A": [1.0, 2.0], "B": [3.0, 0.5]})
    result = top_n_weights(p, n=1)
    assert result.index.equals(p.values.index)


def test_top_n_weights_preserves_columns():
    p = _frame_preds({"SPY": [1.0], "TLT": [2.0]})
    result = top_n_weights(p, n=1)
    assert set(result.columns) == {"SPY", "TLT"}


def test_top_n_weights_raises_on_series():
    p = _series_preds([1.0, 2.0])
    with pytest.raises(TypeError, match="pd.DataFrame"):
        top_n_weights(p, n=1)


def test_top_n_weights_raises_on_n_zero():
    p = _frame_preds({"A": [1.0]})
    with pytest.raises(ValueError, match="n must be >= 1"):
        top_n_weights(p, n=0)


def test_top_n_weights_float_non_negative():
    p = _frame_preds({"A": [3.0, 1.0], "B": [1.0, 2.0], "C": [2.0, 3.0]})
    result = top_n_weights(p, n=2)
    assert (result >= 0).all().all()


def test_top_n_weights_n_equals_all_assets():
    p = _frame_preds({"A": [1.0], "B": [2.0], "C": [3.0]})
    result = top_n_weights(p, n=3)
    np.testing.assert_allclose(result.sum(axis=1).values, [1.0])
    np.testing.assert_allclose(result.values, [[1 / 3, 1 / 3, 1 / 3]], atol=1e-10)


# ===========================================================================
# long_short_weights
# ===========================================================================


def test_long_short_weights_net_exposure_zero():
    p = _frame_preds({"A": [4.0, 1.0], "B": [3.0, 2.0], "C": [2.0, 3.0], "D": [1.0, 4.0]})
    result = long_short_weights(p, n_long=2, n_short=2)
    np.testing.assert_allclose(result.sum(axis=1).values, [0.0, 0.0], atol=1e-10)


def test_long_short_weights_gross_exposure_two():
    p = _frame_preds({"A": [4.0], "B": [3.0], "C": [2.0], "D": [1.0]})
    result = long_short_weights(p, n_long=2, n_short=2)
    gross = result.abs().sum(axis=1)
    np.testing.assert_allclose(gross.values, [2.0], atol=1e-10)


def test_long_short_weights_long_positive_short_negative():
    p = _frame_preds({"A": [4.0], "B": [3.0], "C": [2.0], "D": [1.0]})
    result = long_short_weights(p, n_long=2, n_short=2)
    # A, B are long (+); C, D are short (-)
    assert result.loc[:, "A"].iloc[0] > 0.0
    assert result.loc[:, "B"].iloc[0] > 0.0
    assert result.loc[:, "C"].iloc[0] < 0.0
    assert result.loc[:, "D"].iloc[0] < 0.0


def test_long_short_weights_equal_magnitude_each_side():
    p = _frame_preds({"A": [4.0], "B": [3.0], "C": [2.0], "D": [1.0]})
    result = long_short_weights(p, n_long=2, n_short=2)
    np.testing.assert_allclose(result.loc[:, "A"].iloc[0], 0.5)
    np.testing.assert_allclose(result.loc[:, "B"].iloc[0], 0.5)
    np.testing.assert_allclose(result.loc[:, "C"].iloc[0], -0.5)
    np.testing.assert_allclose(result.loc[:, "D"].iloc[0], -0.5)


def test_long_short_weights_preserves_index():
    p = _frame_preds({"A": [1.0, 2.0], "B": [3.0, 0.0], "C": [2.0, 1.0]})
    result = long_short_weights(p, n_long=1, n_short=1)
    assert result.index.equals(p.values.index)


def test_long_short_weights_raises_on_series():
    p = _series_preds([1.0, 2.0])
    with pytest.raises(TypeError, match="pd.DataFrame"):
        long_short_weights(p, n_long=1, n_short=1)


def test_long_short_weights_raises_on_n_long_zero():
    p = _frame_preds({"A": [1.0], "B": [2.0]})
    with pytest.raises(ValueError, match="n_long"):
        long_short_weights(p, n_long=0, n_short=1)


def test_long_short_weights_raises_on_n_short_zero():
    p = _frame_preds({"A": [1.0], "B": [2.0]})
    with pytest.raises(ValueError, match="n_short"):
        long_short_weights(p, n_long=1, n_short=0)


# ===========================================================================
# normalize_to_weights
# ===========================================================================


def test_normalize_to_weights_rows_sum_to_one():
    p = _frame_preds({"A": [3.0, 1.0], "B": [1.0, 2.0], "C": [2.0, 0.5]})
    result = normalize_to_weights(p)
    np.testing.assert_allclose(result.sum(axis=1).values, [1.0, 1.0])


def test_normalize_to_weights_negatives_clipped():
    p = _frame_preds({"A": [2.0], "B": [-1.0]})
    result = normalize_to_weights(p)
    # B is negative → clipped to 0, A gets full weight
    np.testing.assert_allclose(result.loc[:, "A"].iloc[0], 1.0)
    np.testing.assert_allclose(result.loc[:, "B"].iloc[0], 0.0)


def test_normalize_to_weights_flat_row_when_all_negative():
    p = _frame_preds({"A": [-1.0], "B": [-2.0]})
    result = normalize_to_weights(p)
    assert (result.iloc[0] == 0.0).all()


def test_normalize_to_weights_preserves_index():
    p = _frame_preds({"A": [1.0, 2.0], "B": [3.0, 4.0]})
    result = normalize_to_weights(p)
    assert result.index.equals(p.values.index)


def test_normalize_to_weights_proportional():
    p = _frame_preds({"A": [1.0], "B": [3.0]})
    result = normalize_to_weights(p)
    np.testing.assert_allclose(result.loc[:, "A"].iloc[0], 0.25)
    np.testing.assert_allclose(result.loc[:, "B"].iloc[0], 0.75)


def test_normalize_to_weights_raises_on_series():
    p = _series_preds([1.0, 2.0])
    with pytest.raises(TypeError, match="pd.DataFrame"):
        normalize_to_weights(p)


def test_normalize_to_weights_all_positive_values_non_negative():
    p = _frame_preds({"A": [0.5, 1.0], "B": [0.5, 0.0], "C": [1.0, 2.0]})
    result = normalize_to_weights(p)
    assert (result >= 0).all().all()


# ===========================================================================
# Package imports
# ===========================================================================


def test_importable_from_ml_signals():
    from src.ml.signals import (
        long_short_weights,
        normalize_to_weights,
        sign_signal,
        threshold_signal,
        top_n_weights,
    )
    for fn in (sign_signal, threshold_signal, top_n_weights, long_short_weights, normalize_to_weights):
        assert callable(fn)


def test_importable_from_ml_top_level():
    from src.ml import (
        long_short_weights,
        normalize_to_weights,
        sign_signal,
        threshold_signal,
        top_n_weights,
    )
    for fn in (sign_signal, threshold_signal, top_n_weights, long_short_weights, normalize_to_weights):
        assert callable(fn)
