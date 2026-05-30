from datetime import date

import pytest
from pydantic import ValidationError
from src.data import DataFrequency, DataRequest, DataSource, DataType


def test_data_request_builds_stable_dataset_id() -> None:
    request = DataRequest(
        symbol="spy",
        data_type=DataType.OHLCV,
        source=DataSource.YFINANCE,
        start_date=date(2020, 1, 1),
        end_date=date(2020, 1, 31),
        frequency=DataFrequency.DAILY,
    )

    assert request.normalized_symbol == "SPY"
    assert request.dataset_id == "ohlcv_yfinance_SPY_1d"


def test_data_request_rejects_invalid_date_range() -> None:
    with pytest.raises(ValidationError):
        DataRequest(
            symbol="SPY",
            data_type=DataType.OHLCV,
            source=DataSource.YFINANCE,
            start_date=date(2020, 2, 1),
            end_date=date(2020, 1, 1),
            frequency=DataFrequency.DAILY,
        )


def test_data_request_rejects_unsupported_source_pair() -> None:
    with pytest.raises(ValidationError):
        DataRequest(
            symbol="SPY",
            data_type=DataType.OHLCV,
            source=DataSource.FRED,
            start_date=date(2020, 1, 1),
            end_date=date(2020, 1, 31),
            frequency=DataFrequency.DAILY,
        )
