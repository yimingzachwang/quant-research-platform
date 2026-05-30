"""Tests for src/backtesting/portfolio.py."""

import pandas as pd
import pytest
from src.backtesting.portfolio import compute_exposure, compute_turnover, position_sizing


@pytest.fixture()
def positions() -> pd.Series:
    return pd.Series([0.0, 1.0, 1.0, -1.0, 0.0, 0.5])


def test_compute_turnover_first_is_nan(positions: pd.Series) -> None:
    t = compute_turnover(positions)
    assert pd.isna(t.iloc[0])


def test_compute_turnover_values(positions: pd.Series) -> None:
    t = compute_turnover(positions)
    # 0→1: 1, 1→1: 0, 1→-1: 2, -1→0: 1, 0→0.5: 0.5
    assert t.iloc[1] == pytest.approx(1.0)
    assert t.iloc[2] == pytest.approx(0.0)
    assert t.iloc[3] == pytest.approx(2.0)
    assert t.iloc[4] == pytest.approx(1.0)
    assert t.iloc[5] == pytest.approx(0.5)


def test_compute_turnover_non_negative(positions: pd.Series) -> None:
    t = compute_turnover(positions).dropna()
    assert (t >= 0).all()


def test_compute_exposure_absolute(positions: pd.Series) -> None:
    e = compute_exposure(positions)
    assert (e >= 0).all()
    assert e.iloc[3] == pytest.approx(1.0)  # |-1| = 1


def test_compute_exposure_matches_abs() -> None:
    s = pd.Series([-2.0, 0.0, 1.5, -0.5])
    e = compute_exposure(s)
    pd.testing.assert_series_equal(e, s.abs(), check_names=False)


def test_position_sizing_scales_to_max_leverage() -> None:
    s = pd.Series([1.0, 2.0, -3.0, 0.5])
    sized = position_sizing(s, max_leverage=1.0)
    assert sized.abs().max() == pytest.approx(1.0)


def test_position_sizing_preserves_sign() -> None:
    s = pd.Series([1.0, -2.0, 3.0])
    sized = position_sizing(s, max_leverage=1.0)
    assert (sized * s >= 0).all()


def test_position_sizing_custom_leverage() -> None:
    s = pd.Series([1.0, -1.0, 2.0])
    sized = position_sizing(s, max_leverage=2.0)
    assert sized.abs().max() == pytest.approx(2.0)


def test_position_sizing_all_zero_unchanged() -> None:
    s = pd.Series([0.0, 0.0, 0.0])
    sized = position_sizing(s)
    pd.testing.assert_series_equal(sized, s, check_names=False)


def test_position_sizing_preserves_ratios() -> None:
    s = pd.Series([1.0, 2.0, 4.0])
    sized = position_sizing(s, max_leverage=1.0)
    # Ratios should be 1:2:4 → 0.25 : 0.5 : 1.0
    assert sized.iloc[0] == pytest.approx(0.25)
    assert sized.iloc[1] == pytest.approx(0.50)
    assert sized.iloc[2] == pytest.approx(1.00)
