"""Tests for src/portfolio/ranking.py."""

import numpy as np
import pandas as pd
import pytest
from src.portfolio.ranking import rank_assets, select_bottom_n, select_top_n


@pytest.fixture()
def scores() -> pd.DataFrame:
    """3-asset score DataFrame where A > B > C on every row."""
    idx = pd.date_range("2021-01-01", periods=20, freq="B")
    return pd.DataFrame(
        {"A": 3.0, "B": 2.0, "C": 1.0},
        index=idx,
    )


@pytest.fixture()
def random_scores() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2021-01-01", periods=50, freq="B")
    return pd.DataFrame(rng.normal(0, 1, (50, 5)), index=idx,
                        columns=["A", "B", "C", "D", "E"])


# ---------------------------------------------------------------------------
# rank_assets
# ---------------------------------------------------------------------------


def test_rank_assets_range(scores: pd.DataFrame) -> None:
    ranks = rank_assets(scores)
    assert (ranks >= 0).all().all()
    assert (ranks <= 1).all().all()


def test_rank_assets_ordering(scores: pd.DataFrame) -> None:
    ranks = rank_assets(scores, ascending=False)
    # A > B > C so rank(A) > rank(B) > rank(C) on every row
    assert (ranks["A"] > ranks["B"]).all()
    assert (ranks["B"] > ranks["C"]).all()


def test_rank_assets_ascending_reverses(scores: pd.DataFrame) -> None:
    ranks = rank_assets(scores, ascending=True)
    # Now lower score = higher rank
    assert (ranks["C"] > ranks["B"]).all()
    assert (ranks["B"] > ranks["A"]).all()


def test_rank_assets_nan_propagates() -> None:
    idx = pd.date_range("2021-01-01", periods=5, freq="B")
    df = pd.DataFrame({"A": [1.0, float("nan"), 3.0, 4.0, 5.0],
                       "B": [2.0, 3.0, 2.0, 3.0, 4.0]}, index=idx)
    ranks = rank_assets(df)
    assert pd.isna(ranks.loc[idx[1], "A"])


# ---------------------------------------------------------------------------
# select_top_n
# ---------------------------------------------------------------------------


def test_select_top_n_count(random_scores: pd.DataFrame) -> None:
    ranks = rank_assets(random_scores)
    mask = select_top_n(ranks, n=3)
    assert (mask.sum(axis=1) == 3).all()


def test_select_top_n_selects_highest(scores: pd.DataFrame) -> None:
    ranks = rank_assets(scores, ascending=False)
    mask = select_top_n(ranks, n=1)
    # Only A (highest score) should be selected
    assert mask["A"].all()
    assert not mask["B"].any()
    assert not mask["C"].any()


def test_select_top_n_top_2(scores: pd.DataFrame) -> None:
    ranks = rank_assets(scores, ascending=False)
    mask = select_top_n(ranks, n=2)
    assert mask["A"].all()
    assert mask["B"].all()
    assert not mask["C"].any()


def test_select_top_n_invalid_n(scores: pd.DataFrame) -> None:
    ranks = rank_assets(scores)
    with pytest.raises(ValueError, match="n must be >= 1"):
        select_top_n(ranks, n=0)


def test_select_top_n_returns_bool(random_scores: pd.DataFrame) -> None:
    ranks = rank_assets(random_scores)
    mask = select_top_n(ranks, n=2)
    assert mask.dtypes.apply(lambda d: d == bool).all()  # noqa: E721


# ---------------------------------------------------------------------------
# select_bottom_n
# ---------------------------------------------------------------------------


def test_select_bottom_n_selects_lowest(scores: pd.DataFrame) -> None:
    ranks = rank_assets(scores, ascending=False)
    mask = select_bottom_n(ranks, n=1)
    # C has the lowest score → should be selected
    assert mask["C"].all()
    assert not mask["A"].any()


def test_select_bottom_n_count(random_scores: pd.DataFrame) -> None:
    ranks = rank_assets(random_scores)
    mask = select_bottom_n(ranks, n=2)
    assert (mask.sum(axis=1) == 2).all()
