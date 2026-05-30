"""Tests for src/features/normalization.py."""

import numpy as np
import pandas as pd
import pytest

from src.features.normalization import minmax_normalize, robust_normalize, zscore_normalize


@pytest.fixture()
def series() -> pd.Series:
    rng = np.random.default_rng(99)
    return pd.Series(rng.normal(50, 10, 100), name="feature")


def test_zscore_mean_zero(series: pd.Series) -> None:
    z = zscore_normalize(series)
    assert z.mean() == pytest.approx(0.0, abs=1e-10)


def test_zscore_std_one(series: pd.Series) -> None:
    z = zscore_normalize(series)
    assert z.std() == pytest.approx(1.0, abs=1e-10)


def test_zscore_constant_series() -> None:
    s = pd.Series([5.0] * 20)
    z = zscore_normalize(s)
    assert z.isna().all()


def test_minmax_range(series: pd.Series) -> None:
    mm = minmax_normalize(series)
    assert mm.min() == pytest.approx(0.0)
    assert mm.max() == pytest.approx(1.0)


def test_minmax_constant_series() -> None:
    s = pd.Series([3.0] * 20)
    mm = minmax_normalize(s)
    assert mm.isna().all()


def test_robust_median_zero(series: pd.Series) -> None:
    r = robust_normalize(series)
    assert r.median() == pytest.approx(0.0, abs=1e-10)


def test_robust_iqr_one(series: pd.Series) -> None:
    r = robust_normalize(series)
    iqr = r.quantile(0.75) - r.quantile(0.25)
    assert iqr == pytest.approx(1.0, abs=1e-10)


def test_robust_constant_series() -> None:
    s = pd.Series([7.0] * 20)
    r = robust_normalize(s)
    assert r.isna().all()
