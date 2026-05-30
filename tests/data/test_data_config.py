from datetime import date

import pytest
from src.core import DateRange, Universe
from src.data import (
    DataAgentV1Config,
    DataFrequency,
    DataSource,
    DataType,
    build_data_requests,
    expand_profile_data_requests,
    load_data_agent_v1_config,
    load_dataset_profile_config,
)


def test_load_data_agent_v1_config_parses_canonical_yaml() -> None:
    config = load_data_agent_v1_config("configs/data/data_agent_v1.yaml")

    assert config.name == "data_agent_v1"
    assert config.schema_version == "v1"
    assert config.frequency == DataFrequency.DAILY
    assert DataType.OHLCV in config.supported_data_types
    assert config.supported_sources[DataType.OHLCV] == (DataSource.YFINANCE,)
    assert "adjusted_close" in config.canonical_schemas[DataType.OHLCV]


def test_load_dataset_profile_config_parses_daily_prices_profile() -> None:
    profile = load_dataset_profile_config("configs/data/daily_prices.yaml")

    assert profile.name == "daily_prices"
    assert profile.schema_version == "v1"
    assert profile.ingestion_config.as_posix() == "configs/data/data_agent_v1.yaml"
    assert profile.data_type == DataType.OHLCV
    assert profile.source == DataSource.YFINANCE
    assert profile.frequency == DataFrequency.DAILY


def test_expand_profile_data_requests_uses_universe_and_date_range() -> None:
    config = load_data_agent_v1_config("configs/data/data_agent_v1.yaml")
    profile = load_dataset_profile_config("configs/data/daily_prices.yaml")
    universe = Universe(name="test_etfs", symbols=("spy", "QQQ"))
    date_range = DateRange(start=date(2020, 1, 1), end=date(2020, 1, 31))

    requests = expand_profile_data_requests(
        profile=profile,
        config=config,
        universe=universe,
        date_range=date_range,
    )

    assert [request.symbol for request in requests] == ["spy", "QQQ"]
    assert [request.normalized_symbol for request in requests] == ["SPY", "QQQ"]
    assert [request.dataset_id for request in requests] == [
        "ohlcv_yfinance_SPY_1d",
        "ohlcv_yfinance_QQQ_1d",
    ]
    assert all(request.start_date == date(2020, 1, 1) for request in requests)
    assert all(request.end_date == date(2020, 1, 31) for request in requests)


def test_build_data_requests_rejects_source_not_supported_by_config() -> None:
    config = load_data_agent_v1_config("configs/data/data_agent_v1.yaml")

    with pytest.raises(ValueError, match="unsupported configured source"):
        build_data_requests(
            universe=("SPY",),
            start_date="2020-01-01",
            end_date="2020-01-31",
            data_type=DataType.OHLCV,
            source=DataSource.FRED,
            frequency=DataFrequency.DAILY,
            config=config,
        )


def test_data_agent_v1_config_requires_supported_sources_for_each_data_type() -> None:
    payload = {
        "name": "bad_config",
        "schema_version": "v1",
        "supported_data_types": ["ohlcv"],
        "supported_sources": {},
        "frequency": "1d",
        "storage": {
            "raw": "data/raw",
            "processed": "data/processed",
            "metadata": "data/external/metadata",
            "registry": "data/external/registry/datasets.json",
        },
        "validation": {"max_nan_ratio": 0.05},
    }

    with pytest.raises(ValueError, match="supported_sources missing entries"):
        DataAgentV1Config.model_validate(payload)
