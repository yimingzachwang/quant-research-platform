"""Tests for src/cleaning/numeric.py."""

import math

import pandas as pd
import pytest

from src.cleaning.numeric import replace_inf


def test_replace_inf_with_nan() -> None:
    df = pd.DataFrame({"a": [1.0, float("inf"), -float("inf"), 4.0]})
    result = replace_inf(df)
    assert math.isnan(result["a"].iloc[1])
    assert math.isnan(result["a"].iloc[2])
    assert result["a"].iloc[0] == 1.0


def test_replace_inf_custom_value() -> None:
    df = pd.DataFrame({"a": [float("inf"), -float("inf"), 0.0]})
    result = replace_inf(df, value=0.0)
    assert result["a"].iloc[0] == 0.0
    assert result["a"].iloc[1] == 0.0


def test_replace_inf_no_inf_unchanged() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    result = replace_inf(df)
    assert list(result["a"]) == [1.0, 2.0, 3.0]


def test_replace_inf_preserves_nan() -> None:
    df = pd.DataFrame({"a": [float("nan"), float("inf")]})
    result = replace_inf(df)
    assert math.isnan(result["a"].iloc[0])
    assert math.isnan(result["a"].iloc[1])
