"""Tests for src.portfolio.weighting_policy.

Covers apply_weighting_policy(), _zscore_softmax(), _confidence_weighted(),
_apply_prediction_normalization(), and backward-compatibility with equal_weight.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.portfolio.allocation import equal_weight
from src.portfolio.weighting_policy import (
    VALID_PREDICTION_NORMALIZATIONS,
    VALID_WEIGHTING_SCHEMES,
    _apply_prediction_normalization,
    _confidence_weighted,
    _zscore_softmax,
    apply_weighting_policy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scores(n_rows: int = 4, n_cols: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_rows, n_cols))
    dates = pd.date_range("2020-01-01", periods=n_rows, freq="ME")
    tickers = [f"A{i}" for i in range(n_cols)]
    return pd.DataFrame(data, index=dates, columns=tickers)


def _make_top2_mask(scores: pd.DataFrame) -> pd.DataFrame:
    """Boolean mask selecting the top-2 assets by score per row."""
    from src.portfolio.ranking import rank_assets, select_top_n
    ranks = rank_assets(scores, ascending=False)
    return select_top_n(ranks, n=2)


# ---------------------------------------------------------------------------
# VALID_WEIGHTING_SCHEMES set
# ---------------------------------------------------------------------------

def test_valid_weighting_schemes_contains_required():
    assert "equal_weight" in VALID_WEIGHTING_SCHEMES
    assert "zscore_softmax" in VALID_WEIGHTING_SCHEMES
    assert "confidence_weighted" in VALID_WEIGHTING_SCHEMES


def test_valid_prediction_normalizations_contains_required():
    assert "none" in VALID_PREDICTION_NORMALIZATIONS
    assert "zscore" in VALID_PREDICTION_NORMALIZATIONS


# ---------------------------------------------------------------------------
# apply_weighting_policy — equal_weight (backward compat)
# ---------------------------------------------------------------------------

def test_equal_weight_matches_allocation_equal_weight():
    scores = _make_scores()
    mask = _make_top2_mask(scores)
    result = apply_weighting_policy(scores, mask, scheme="equal_weight")
    expected = equal_weight(mask)
    pd.testing.assert_frame_equal(result, expected)


def test_equal_weight_rows_sum_to_one():
    scores = _make_scores()
    mask = _make_top2_mask(scores)
    result = apply_weighting_policy(scores, mask)  # default scheme
    row_sums = result.sum(axis=1)
    for s in row_sums:
        assert s == pytest.approx(1.0, abs=1e-9)


def test_equal_weight_ignores_prediction_normalization():
    scores = _make_scores()
    mask = _make_top2_mask(scores)
    r_none = apply_weighting_policy(scores, mask, scheme="equal_weight", prediction_normalization="none")
    r_zscore = apply_weighting_policy(scores, mask, scheme="equal_weight", prediction_normalization="zscore")
    pd.testing.assert_frame_equal(r_none, r_zscore)


# ---------------------------------------------------------------------------
# apply_weighting_policy — unknown scheme
# ---------------------------------------------------------------------------

def test_unknown_scheme_raises():
    scores = _make_scores()
    mask = _make_top2_mask(scores)
    with pytest.raises(ValueError, match="Unknown weighting scheme"):
        apply_weighting_policy(scores, mask, scheme="magic_scheme")


# ---------------------------------------------------------------------------
# _apply_prediction_normalization
# ---------------------------------------------------------------------------

def test_none_normalization_is_passthrough():
    scores = _make_scores()
    result = _apply_prediction_normalization(scores, "none")
    pd.testing.assert_frame_equal(result, scores)


def test_zscore_normalization_row_mean_is_zero():
    scores = _make_scores()
    result = _apply_prediction_normalization(scores, "zscore")
    row_means = result.mean(axis=1)
    for m in row_means:
        assert m == pytest.approx(0.0, abs=1e-9)


def test_zscore_normalization_row_std_is_one():
    scores = _make_scores(n_cols=4)
    result = _apply_prediction_normalization(scores, "zscore")
    row_stds = result.std(axis=1)
    for s in row_stds:
        assert s == pytest.approx(1.0, abs=1e-6)


def test_unknown_normalization_raises():
    scores = _make_scores()
    with pytest.raises(ValueError, match="Unknown prediction_normalization"):
        _apply_prediction_normalization(scores, "absolute_value")


# ---------------------------------------------------------------------------
# _zscore_softmax
# ---------------------------------------------------------------------------

def test_zscore_softmax_rows_sum_to_one():
    scores = _make_scores()
    mask = _make_top2_mask(scores)
    result = _zscore_softmax(scores, mask, temperature=1.0)
    row_sums = result.sum(axis=1)
    for s in row_sums:
        assert s == pytest.approx(1.0, abs=1e-9)


def test_zscore_softmax_non_selected_are_zero():
    scores = _make_scores()
    mask = _make_top2_mask(scores)
    result = _zscore_softmax(scores, mask, temperature=1.0)
    # Non-selected assets must have weight 0 (use numpy indexing to avoid NaN propagation)
    non_selected_weights = result.values[~mask.values.astype(bool)]
    assert (non_selected_weights == 0.0).all()


def test_zscore_softmax_high_temperature_approaches_equal_weight():
    scores = _make_scores(n_cols=3)
    mask = _make_top2_mask(scores)
    result = _zscore_softmax(scores, mask, temperature=1000.0)
    expected = equal_weight(mask)
    pd.testing.assert_frame_equal(result, expected, atol=0.01, check_exact=False)


def test_zscore_softmax_concentration_increases_with_low_temperature():
    scores = _make_scores(n_cols=4)
    mask = _make_top2_mask(scores)
    high_t = _zscore_softmax(scores, mask, temperature=10.0)
    low_t = _zscore_softmax(scores, mask, temperature=0.1)
    # Low temperature concentrates weight more → max weight per row larger
    assert low_t.max(axis=1).mean() >= high_t.max(axis=1).mean()


def test_zscore_softmax_invalid_temperature_raises():
    scores = _make_scores()
    mask = _make_top2_mask(scores)
    with pytest.raises(ValueError, match="temperature must be positive"):
        _zscore_softmax(scores, mask, temperature=0.0)


def test_zscore_softmax_via_apply_weighting_policy():
    scores = _make_scores()
    mask = _make_top2_mask(scores)
    result = apply_weighting_policy(scores, mask, scheme="zscore_softmax", temperature=1.0)
    direct = _zscore_softmax(scores, mask, temperature=1.0)
    pd.testing.assert_frame_equal(result, direct)


# ---------------------------------------------------------------------------
# _confidence_weighted
# ---------------------------------------------------------------------------

def test_confidence_weighted_rows_sum_to_one():
    # Use scores that are positive to ensure no fallback
    scores = _make_scores().abs() + 0.1
    mask = _make_top2_mask(scores)
    result = _confidence_weighted(scores, mask)
    row_sums = result.sum(axis=1)
    for s in row_sums:
        assert s == pytest.approx(1.0, abs=1e-9)


def test_confidence_weighted_non_selected_are_zero():
    scores = _make_scores().abs() + 0.1
    mask = _make_top2_mask(scores)
    result = _confidence_weighted(scores, mask)
    non_selected_weights = result.values[~mask.values.astype(bool)]
    assert (non_selected_weights == 0.0).all()


def test_confidence_weighted_proportional_to_positive_scores():
    # Construct a simple 2-asset case with known positive scores
    scores = pd.DataFrame(
        {"A": [3.0, 1.0], "B": [1.0, 3.0]},
        index=pd.date_range("2020-01-01", periods=2, freq="ME"),
    )
    mask = pd.DataFrame(
        {"A": [True, True], "B": [True, True]},
        index=scores.index,
    )
    result = _confidence_weighted(scores, mask)
    # Row 0: A=3, B=1 → A weight = 3/4 = 0.75
    assert result.loc[result.index[0], "A"] == pytest.approx(0.75, rel=1e-6)
    assert result.loc[result.index[0], "B"] == pytest.approx(0.25, rel=1e-6)


def test_confidence_weighted_all_negative_fallback_equal_weight():
    # When all selected scores are negative, fallback to equal_weight
    scores = pd.DataFrame(
        {"A": [-1.0, -2.0], "B": [-3.0, -4.0]},
        index=pd.date_range("2020-01-01", periods=2, freq="ME"),
    )
    mask = pd.DataFrame(
        {"A": [True, True], "B": [True, True]},
        index=scores.index,
    )
    result = _confidence_weighted(scores, mask)
    expected = equal_weight(mask)
    pd.testing.assert_frame_equal(result, expected)


def test_confidence_weighted_via_apply_weighting_policy():
    scores = _make_scores().abs() + 0.1
    mask = _make_top2_mask(scores)
    result = apply_weighting_policy(scores, mask, scheme="confidence_weighted")
    direct = _confidence_weighted(scores, mask)
    pd.testing.assert_frame_equal(result, direct)


# ---------------------------------------------------------------------------
# Row-wise invariant — no cross-date contamination
# ---------------------------------------------------------------------------

def test_zscore_softmax_is_row_wise():
    """Weights on row t must not change when data on row t+1 changes."""
    scores = _make_scores()
    mask = _make_top2_mask(scores)
    result_original = _zscore_softmax(scores, mask, temperature=1.0)

    scores_modified = scores.copy()
    scores_modified.iloc[-1] *= 100.0  # change only last row
    _make_top2_mask(scores_modified)
    # Only recompute for equal mask rows
    result_modified = _zscore_softmax(scores_modified, mask, temperature=1.0)

    # First n-1 rows should be unchanged (mask is the same, scores unchanged)
    pd.testing.assert_frame_equal(
        result_original.iloc[:-1],
        result_modified.iloc[:-1],
    )


def test_confidence_weighted_is_row_wise():
    scores = _make_scores().abs() + 0.1
    mask = _make_top2_mask(scores)
    result_original = _confidence_weighted(scores, mask)

    scores_modified = scores.copy()
    scores_modified.iloc[-1] *= 100.0
    result_modified = _confidence_weighted(scores_modified, mask)

    pd.testing.assert_frame_equal(
        result_original.iloc[:-1],
        result_modified.iloc[:-1],
    )


# ---------------------------------------------------------------------------
# Public export from src.portfolio
# ---------------------------------------------------------------------------

def test_apply_weighting_policy_exported_from_portfolio():
    from src.portfolio import VALID_WEIGHTING_SCHEMES, apply_weighting_policy
    assert callable(apply_weighting_policy)
    assert "equal_weight" in VALID_WEIGHTING_SCHEMES
