"""Tests for src/strategies/baselines.py."""

import numpy as np
import pandas as pd
import pytest

from src.strategies.base import Strategy
from src.strategies.baselines import BuyAndHoldStrategy, EqualWeightStrategy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def prices_3asset() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=200, freq="B")
    rng = np.random.default_rng(0)
    data = {
        "A": 100 * np.cumprod(1 + rng.normal(0.001, 0.01, 200)),
        "B": 100 * np.cumprod(1 + rng.normal(0.000, 0.01, 200)),
        "C": 100 * np.cumprod(1 + rng.normal(-0.001, 0.01, 200)),
    }
    return pd.DataFrame(data, index=idx)


@pytest.fixture()
def prices_2asset() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    return pd.DataFrame({"X": 100.0, "Y": 200.0}, index=idx)


# ---------------------------------------------------------------------------
# BuyAndHoldStrategy — ABC contract
# ---------------------------------------------------------------------------


def test_buyandhold_is_strategy_subclass() -> None:
    assert issubclass(BuyAndHoldStrategy, Strategy)


def test_buyandhold_name_default() -> None:
    s = BuyAndHoldStrategy()
    assert "first_asset" in s.name.lower() or "BuyAndHold" in s.name


def test_buyandhold_name_custom() -> None:
    s = BuyAndHoldStrategy(weights={"A": 0.6, "B": 0.4})
    assert "BuyAndHold" in s.name
    assert "A" in s.name


def test_buyandhold_params_none() -> None:
    s = BuyAndHoldStrategy()
    assert s.params() == {"weights": None}


def test_buyandhold_params_custom() -> None:
    s = BuyAndHoldStrategy(weights={"A": 0.5, "B": 0.5})
    assert s.params()["weights"] == {"A": 0.5, "B": 0.5}


# ---------------------------------------------------------------------------
# BuyAndHoldStrategy — weight correctness
# ---------------------------------------------------------------------------


def test_buyandhold_default_allocates_first_asset_100(
    prices_2asset: pd.DataFrame,
) -> None:
    w = BuyAndHoldStrategy().generate_weights(prices_2asset)
    assert (w["X"] == 1.0).all()
    assert (w["Y"] == 0.0).all()


def test_buyandhold_weights_constant_across_rows(prices_3asset: pd.DataFrame) -> None:
    w = BuyAndHoldStrategy().generate_weights(prices_3asset)
    # All rows identical — static allocation
    assert (w == w.iloc[0]).all().all()


def test_buyandhold_custom_weights_applied(prices_3asset: pd.DataFrame) -> None:
    custom = {"A": 0.6, "B": 0.4, "C": 0.0}
    w = BuyAndHoldStrategy(weights=custom).generate_weights(prices_3asset)
    assert ((w["A"] - 0.6).abs() < 1e-12).all()
    assert ((w["B"] - 0.4).abs() < 1e-12).all()
    assert (w["C"] == 0.0).all()


def test_buyandhold_custom_weights_unknown_asset_is_zero(
    prices_2asset: pd.DataFrame,
) -> None:
    # "Z" is not in prices — should be ignored / mapped to 0
    w = BuyAndHoldStrategy(weights={"X": 1.0, "Z": 0.5}).generate_weights(prices_2asset)
    assert (w["Y"] == 0.0).all()


def test_buyandhold_same_index_as_prices(prices_3asset: pd.DataFrame) -> None:
    w = BuyAndHoldStrategy().generate_weights(prices_3asset)
    assert (w.index == prices_3asset.index).all()


def test_buyandhold_columns_match_prices(prices_3asset: pd.DataFrame) -> None:
    w = BuyAndHoldStrategy().generate_weights(prices_3asset)
    assert list(w.columns) == list(prices_3asset.columns)


# ---------------------------------------------------------------------------
# EqualWeightStrategy — ABC contract
# ---------------------------------------------------------------------------


def test_equalweight_is_strategy_subclass() -> None:
    assert issubclass(EqualWeightStrategy, Strategy)


def test_equalweight_name_contains_freq() -> None:
    s = EqualWeightStrategy(rebalance_freq="QE")
    assert "QE" in s.name


def test_equalweight_params() -> None:
    s = EqualWeightStrategy(rebalance_freq="W-FRI")
    assert s.params() == {"rebalance_freq": "W-FRI"}


# ---------------------------------------------------------------------------
# EqualWeightStrategy — weight correctness
# ---------------------------------------------------------------------------


def test_equalweight_sums_to_one(prices_3asset: pd.DataFrame) -> None:
    w = EqualWeightStrategy().generate_weights(prices_3asset)
    invested = w[w.sum(axis=1) > 0]
    assert ((invested.sum(axis=1) - 1.0).abs() < 1e-9).all()


def test_equalweight_all_assets_equal_weight(prices_3asset: pd.DataFrame) -> None:
    w = EqualWeightStrategy().generate_weights(prices_3asset)
    invested = w[w.sum(axis=1) > 0]
    # Each asset should receive 1/3
    for col in invested.columns:
        assert ((invested[col] - 1 / 3).abs() < 1e-9).all()


def test_equalweight_two_assets_half_each(prices_2asset: pd.DataFrame) -> None:
    w = EqualWeightStrategy().generate_weights(prices_2asset)
    invested = w[w.sum(axis=1) > 0]
    assert ((invested["X"] - 0.5).abs() < 1e-9).all()
    assert ((invested["Y"] - 0.5).abs() < 1e-9).all()


def test_equalweight_invested_from_first_day(prices_3asset: pd.DataFrame) -> None:
    w = EqualWeightStrategy().generate_weights(prices_3asset)
    # First row should already be invested (synthetic rebalance on day 1)
    assert w.iloc[0].sum() == pytest.approx(1.0)


def test_equalweight_same_index_as_prices(prices_3asset: pd.DataFrame) -> None:
    w = EqualWeightStrategy().generate_weights(prices_3asset)
    assert len(w) == len(prices_3asset)
    assert (w.index == prices_3asset.index).all()


def test_equalweight_non_negative(prices_3asset: pd.DataFrame) -> None:
    w = EqualWeightStrategy().generate_weights(prices_3asset)
    assert (w >= 0).all().all()


def test_equalweight_deterministic(prices_3asset: pd.DataFrame) -> None:
    s = EqualWeightStrategy()
    w1 = s.generate_weights(prices_3asset)
    w2 = s.generate_weights(prices_3asset)
    pd.testing.assert_frame_equal(w1, w2)
