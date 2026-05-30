"""Tests for src/cleaning/missing.py."""

import math

import pandas as pd
import pytest
from src.cleaning.missing import forward_fill_limited


def test_forward_fill_fills_within_limit() -> None:
    df = pd.DataFrame({"a": [1.0, float("nan"), float("nan"), 4.0]})
    result = forward_fill_limited(df, limit=2)
    assert result["a"].iloc[1] == 1.0
    assert result["a"].iloc[2] == 1.0
    assert result["a"].iloc[3] == 4.0


def test_forward_fill_leaves_long_gaps() -> None:
    df = pd.DataFrame({"a": [1.0, float("nan"), float("nan"), float("nan"), 5.0]})
    result = forward_fill_limited(df, limit=2)
    assert result["a"].iloc[1] == 1.0
    assert result["a"].iloc[2] == 1.0
    # gap of 3 → third NaN is beyond limit=2
    assert math.isnan(result["a"].iloc[3])


def test_forward_fill_limit_zero_raises() -> None:
    df = pd.DataFrame({"a": [1.0, float("nan")]})
    with pytest.raises(ValueError, match="limit must be >= 1"):
        forward_fill_limited(df, limit=0)


def test_forward_fill_no_nans_unchanged() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    result = forward_fill_limited(df, limit=5)
    assert list(result["a"]) == [1.0, 2.0, 3.0]
