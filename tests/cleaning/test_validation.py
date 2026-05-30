"""Tests for src/cleaning/validation.py."""

import pandas as pd
import pytest
from src.cleaning.validation import validate_ohlcv


@pytest.fixture()
def valid_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [105.0, 106.0],
            "low": [98.0, 99.0],
            "close": [103.0, 104.0],
            "volume": [1_000_000.0, 1_100_000.0],
        }
    )


def test_valid_ohlcv_passes(valid_ohlcv: pd.DataFrame) -> None:
    result = validate_ohlcv(valid_ohlcv, raise_on_error=False)
    assert result.is_valid
    assert result.summary() == "OK"


def test_missing_column_detected() -> None:
    df = pd.DataFrame({"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5]})
    result = validate_ohlcv(df, raise_on_error=False)
    assert "volume" in result.missing_columns
    assert not result.is_valid


def test_missing_column_raises() -> None:
    df = pd.DataFrame({"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5]})
    with pytest.raises(ValueError, match="missing columns"):
        validate_ohlcv(df, raise_on_error=True)


def test_negative_price_detected(valid_ohlcv: pd.DataFrame) -> None:
    valid_ohlcv.loc[0, "close"] = -1.0
    result = validate_ohlcv(valid_ohlcv, raise_on_error=False)
    assert result.negative_prices.get("close", 0) == 1
    assert not result.is_valid


def test_negative_volume_detected(valid_ohlcv: pd.DataFrame) -> None:
    valid_ohlcv.loc[0, "volume"] = -500.0
    result = validate_ohlcv(valid_ohlcv, raise_on_error=False)
    assert result.negative_volume == 1
    assert not result.is_valid


def test_high_lt_low_detected(valid_ohlcv: pd.DataFrame) -> None:
    valid_ohlcv.loc[0, "high"] = 90.0  # below low=98
    result = validate_ohlcv(valid_ohlcv, raise_on_error=False)
    assert result.high_lt_low == 1
    assert not result.is_valid


def test_high_lt_close_detected(valid_ohlcv: pd.DataFrame) -> None:
    valid_ohlcv.loc[0, "high"] = 102.0  # below close=103
    result = validate_ohlcv(valid_ohlcv, raise_on_error=False)
    assert result.high_lt_close == 1


def test_nan_counts_recorded(valid_ohlcv: pd.DataFrame) -> None:
    valid_ohlcv.loc[0, "close"] = float("nan")
    result = validate_ohlcv(valid_ohlcv, raise_on_error=False)
    assert result.nan_counts["close"] == 1


def test_raise_on_error(valid_ohlcv: pd.DataFrame) -> None:
    valid_ohlcv.loc[0, "high"] = 90.0
    with pytest.raises(ValueError, match="OHLCV validation failed"):
        validate_ohlcv(valid_ohlcv, raise_on_error=True)


def test_case_insensitive_columns() -> None:
    df = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [105.0],
            "Low": [98.0],
            "Close": [103.0],
            "Volume": [1_000_000.0],
        }
    )
    result = validate_ohlcv(df, raise_on_error=False)
    assert result.is_valid
