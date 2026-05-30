"""Tests for src/strategies/comparison.py."""

import numpy as np
import pandas as pd
import pytest

from src.strategies.baselines import BuyAndHoldStrategy, EqualWeightStrategy
from src.strategies.comparison import compare_strategies, metrics_table, rank_strategies
from src.strategies.momentum_rotation import MomentumRotationStrategy
from src.strategies.runner import StrategyResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def prices() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    rng = np.random.default_rng(99)
    return pd.DataFrame(
        {
            "A": 100 * np.cumprod(1 + rng.normal(0.001, 0.01, 300)),
            "B": 100 * np.cumprod(1 + rng.normal(0.000, 0.01, 300)),
            "C": 100 * np.cumprod(1 + rng.normal(-0.001, 0.01, 300)),
        },
        index=idx,
    )


@pytest.fixture()
def strategies() -> list:
    return [
        BuyAndHoldStrategy(),
        EqualWeightStrategy(),
        MomentumRotationStrategy(lookback=60, top_n=2, rebalance_freq="ME"),
    ]


@pytest.fixture()
def results(prices: pd.DataFrame, strategies: list) -> dict:
    return compare_strategies(prices, strategies, transaction_cost_bps=0.0)


# ---------------------------------------------------------------------------
# compare_strategies
# ---------------------------------------------------------------------------


def test_compare_returns_dict(prices: pd.DataFrame, strategies: list) -> None:
    out = compare_strategies(prices, strategies)
    assert isinstance(out, dict)


def test_compare_all_strategies_present(
    results: dict, strategies: list
) -> None:
    for s in strategies:
        assert s.name in results


def test_compare_result_types(results: dict) -> None:
    for v in results.values():
        assert isinstance(v, StrategyResult)


def test_compare_preserves_insertion_order(
    prices: pd.DataFrame, strategies: list
) -> None:
    out = compare_strategies(prices, strategies)
    assert list(out.keys()) == [s.name for s in strategies]


def test_compare_duplicate_name_raises(prices: pd.DataFrame) -> None:
    s1 = BuyAndHoldStrategy()
    s2 = BuyAndHoldStrategy()
    with pytest.raises(ValueError, match="Duplicate strategy name"):
        compare_strategies(prices, [s1, s2])


def test_compare_single_strategy(prices: pd.DataFrame) -> None:
    out = compare_strategies(prices, [BuyAndHoldStrategy()])
    assert len(out) == 1


def test_compare_cost_bps_reduces_net_return(
    prices: pd.DataFrame, strategies: list
) -> None:
    no_cost = compare_strategies(prices, strategies, transaction_cost_bps=0.0)
    with_cost = compare_strategies(prices, strategies, transaction_cost_bps=20.0)
    for name in no_cost:
        nc_total = no_cost[name].backtest["net_return"].sum()
        wc_total = with_cost[name].backtest["net_return"].sum()
        assert wc_total <= nc_total


# ---------------------------------------------------------------------------
# metrics_table
# ---------------------------------------------------------------------------


def test_metrics_table_is_dataframe(results: dict) -> None:
    t = metrics_table(results)
    assert isinstance(t, pd.DataFrame)


def test_metrics_table_rows_are_strategies(results: dict) -> None:
    t = metrics_table(results)
    assert set(t.index) == set(results.keys())


def test_metrics_table_columns(results: dict) -> None:
    t = metrics_table(results)
    expected = {"annualized_return", "annualized_volatility", "sharpe_ratio",
                "max_drawdown", "calmar_ratio", "hit_rate"}
    assert set(t.columns) == expected


def test_metrics_table_numeric(results: dict) -> None:
    t = metrics_table(results)
    assert t.dtypes.apply(lambda d: pd.api.types.is_float_dtype(d)).all()


# ---------------------------------------------------------------------------
# rank_strategies
# ---------------------------------------------------------------------------


def test_rank_strategies_returns_dataframe(results: dict) -> None:
    r = rank_strategies(results, by="sharpe_ratio")
    assert isinstance(r, pd.DataFrame)


def test_rank_strategies_has_rank_column(results: dict) -> None:
    r = rank_strategies(results, by="sharpe_ratio")
    assert "rank" in r.columns


def test_rank_strategies_rank_starts_at_1(results: dict) -> None:
    r = rank_strategies(results, by="sharpe_ratio")
    assert r["rank"].iloc[0] == 1


def test_rank_strategies_invalid_metric(results: dict) -> None:
    with pytest.raises(ValueError, match="not found"):
        rank_strategies(results, by="nonexistent_metric")


def test_rank_strategies_descending_sharpe(results: dict) -> None:
    r = rank_strategies(results, by="sharpe_ratio", ascending=False)
    sharpes = r["sharpe_ratio"].dropna()
    if len(sharpes) > 1:
        assert (sharpes.values[:-1] >= sharpes.values[1:]).all()
