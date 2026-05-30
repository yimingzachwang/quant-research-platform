"""Tests for src/strategies/momentum_rotation.py and src/strategies/base.py."""

import numpy as np
import pandas as pd
import pytest
from src.strategies.base import Strategy
from src.strategies.momentum_rotation import MomentumRotationStrategy

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def prices_3asset() -> pd.DataFrame:
    """300-day price DataFrame, 3 assets with distinct trends."""
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    rng = np.random.default_rng(0)
    # A trends up, B flat, C trends down
    a = 100 * np.cumprod(1 + rng.normal(0.001, 0.01, 300))
    b = 100 * np.cumprod(1 + rng.normal(0.000, 0.01, 300))
    c = 100 * np.cumprod(1 + rng.normal(-0.001, 0.01, 300))
    return pd.DataFrame({"A": a, "B": b, "C": c}, index=idx)


@pytest.fixture()
def prices_flat() -> pd.DataFrame:
    """All assets with identical prices — ties in ranking."""
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    return pd.DataFrame({"A": 100.0, "B": 100.0}, index=idx)


@pytest.fixture()
def strategy_default() -> MomentumRotationStrategy:
    return MomentumRotationStrategy(lookback=60, top_n=2, rebalance_freq="ME")


# ---------------------------------------------------------------------------
# Strategy ABC contract
# ---------------------------------------------------------------------------


def test_is_strategy_subclass() -> None:
    assert issubclass(MomentumRotationStrategy, Strategy)


def test_name_contains_parameters() -> None:
    s = MomentumRotationStrategy(lookback=120, top_n=2, rebalance_freq="ME")
    assert "120" in s.name
    assert "2" in s.name
    assert "ME" in s.name


def test_params_returns_dict() -> None:
    s = MomentumRotationStrategy(lookback=60, top_n=1, rebalance_freq="QE")
    p = s.params()
    assert p["lookback"] == 60
    assert p["top_n"] == 1
    assert p["rebalance_freq"] == "QE"


# ---------------------------------------------------------------------------
# Weight structure
# ---------------------------------------------------------------------------


def test_weights_is_dataframe(
    prices_3asset: pd.DataFrame, strategy_default: MomentumRotationStrategy
) -> None:
    w = strategy_default.generate_weights(prices_3asset)
    assert isinstance(w, pd.DataFrame)


def test_weights_same_index_as_prices(
    prices_3asset: pd.DataFrame, strategy_default: MomentumRotationStrategy
) -> None:
    w = strategy_default.generate_weights(prices_3asset)
    assert len(w) == len(prices_3asset)
    assert (w.index == prices_3asset.index).all()


def test_weights_columns_match_prices(
    prices_3asset: pd.DataFrame, strategy_default: MomentumRotationStrategy
) -> None:
    w = strategy_default.generate_weights(prices_3asset)
    assert list(w.columns) == list(prices_3asset.columns)


def test_weights_non_negative(
    prices_3asset: pd.DataFrame, strategy_default: MomentumRotationStrategy
) -> None:
    w = strategy_default.generate_weights(prices_3asset)
    assert (w >= 0).all().all()


# ---------------------------------------------------------------------------
# Top-N selection
# ---------------------------------------------------------------------------


def test_at_most_top_n_assets_selected(
    prices_3asset: pd.DataFrame, strategy_default: MomentumRotationStrategy
) -> None:
    w = strategy_default.generate_weights(prices_3asset)
    # After warm-up, at most top_n assets should have non-zero weight
    n_active = (w > 0).sum(axis=1)
    assert (n_active <= strategy_default.top_n).all()


def test_exactly_top_n_after_warmup(prices_3asset: pd.DataFrame) -> None:
    s = MomentumRotationStrategy(lookback=60, top_n=2, rebalance_freq="ME")
    w = s.generate_weights(prices_3asset)
    # After the first rebalance following warm-up, should hold exactly top_n
    warmed = w.iloc[65:]  # comfortably past the 60-day lookback
    n_active = (warmed > 0).sum(axis=1)
    assert (n_active[n_active > 0] == s.top_n).all()


# ---------------------------------------------------------------------------
# Weight normalisation
# ---------------------------------------------------------------------------


def test_weights_sum_to_one_after_warmup(
    prices_3asset: pd.DataFrame, strategy_default: MomentumRotationStrategy
) -> None:
    w = strategy_default.generate_weights(prices_3asset)
    row_sums = w.sum(axis=1)
    # After warm-up, invested rows should sum to 1
    invested = row_sums[row_sums > 0]
    assert ((invested - 1.0).abs() < 1e-9).all()


def test_warmup_rows_are_zero(prices_3asset: pd.DataFrame) -> None:
    s = MomentumRotationStrategy(lookback=60, top_n=1, rebalance_freq="ME")
    w = s.generate_weights(prices_3asset)
    # First 60 rows must be zero (no momentum signal yet)
    assert (w.iloc[:60] == 0.0).all().all()


# ---------------------------------------------------------------------------
# Look-ahead prevention
# ---------------------------------------------------------------------------


def test_weight_at_t_does_not_use_future_prices() -> None:
    """Strategy should not be sensitive to prices AFTER date t.

    Construct prices where a trend reversal occurs at a known split point.
    The weights before the split should favour the then-dominant asset.
    Reversing prices after the split must not change weights before the split.
    """
    idx = pd.date_range("2020-01-01", periods=200, freq="B")
    # A up, B flat for the whole window
    prices = pd.DataFrame(
        {
            "A": np.linspace(100, 150, 200),
            "B": np.full(200, 100.0),
        },
        index=idx,
    )

    # Tamper with the last 10 rows — must not affect earlier weights
    prices_tampered = prices.copy()
    prices_tampered.iloc[-10:] = prices_tampered.iloc[-10:] * 2.0

    s = MomentumRotationStrategy(lookback=60, top_n=1, rebalance_freq="ME")
    w_original = s.generate_weights(prices)
    w_tampered = s.generate_weights(prices_tampered)

    # Weights for the first 190 rows should be identical
    pd.testing.assert_frame_equal(w_original.iloc[:190], w_tampered.iloc[:190])


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_generate_weights_is_deterministic(
    prices_3asset: pd.DataFrame, strategy_default: MomentumRotationStrategy
) -> None:
    w1 = strategy_default.generate_weights(prices_3asset)
    w2 = strategy_default.generate_weights(prices_3asset)
    pd.testing.assert_frame_equal(w1, w2)


# ---------------------------------------------------------------------------
# Rebalance frequency
# ---------------------------------------------------------------------------


def test_monthly_vs_quarterly_different_weights() -> None:
    idx = pd.date_range("2020-01-01", periods=400, freq="B")
    rng = np.random.default_rng(42)
    prices = pd.DataFrame(
        {"A": 100 * np.cumprod(1 + rng.normal(0.001, 0.01, 400)),
         "B": 100 * np.cumprod(1 + rng.normal(0.000, 0.01, 400))},
        index=idx,
    )
    monthly = MomentumRotationStrategy(lookback=60, top_n=1, rebalance_freq="ME")
    quarterly = MomentumRotationStrategy(lookback=60, top_n=1, rebalance_freq="QE")
    wm = monthly.generate_weights(prices)
    wq = quarterly.generate_weights(prices)
    # Turnover (sum of weight changes) should differ
    assert not wm.equals(wq)
