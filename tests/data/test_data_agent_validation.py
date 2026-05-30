from datetime import date

import pandas as pd
from src.data import DataFrequency, DataRequest, DataSource, DataType
from src.data.models import ValidationStatus
from src.data.validators import DatasetValidator


def _request() -> DataRequest:
    return DataRequest(
        symbol="SPY",
        data_type=DataType.OHLCV,
        source=DataSource.YFINANCE,
        start_date=date(2020, 1, 1),
        end_date=date(2020, 1, 3),
        frequency=DataFrequency.DAILY,
    )


def test_validator_fails_duplicate_timestamps() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-01-01", "2020-01-01"], utc=True),
            "open": [1.0, 1.0],
            "high": [2.0, 2.0],
            "low": [0.5, 0.5],
            "close": [1.5, 1.5],
            "volume": [100, 100],
            "symbol": ["SPY", "SPY"],
            "source": ["yfinance", "yfinance"],
            "frequency": ["1d", "1d"],
        }
    )

    report = DatasetValidator().validate(frame, _request())

    assert report.status == ValidationStatus.FAILED
    assert any(check.name == "duplicate_timestamps" for check in report.checks)


def test_validator_warns_on_missing_business_day() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-01-01", "2020-01-03"], utc=True),
            "open": [1.0, 1.0],
            "high": [2.0, 2.0],
            "low": [0.5, 0.5],
            "close": [1.5, 1.5],
            "volume": [100, 100],
            "symbol": ["SPY", "SPY"],
            "source": ["yfinance", "yfinance"],
            "frequency": ["1d", "1d"],
        }
    )

    report = DatasetValidator().validate(frame, _request())

    assert report.status == ValidationStatus.WARNING
    missing_check = next(check for check in report.checks if check.name == "missing_timestamps")
    assert missing_check.observed_value == 1
