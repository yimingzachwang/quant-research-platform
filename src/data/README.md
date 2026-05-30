# Data Agent V1

The data package provides deterministic infrastructure for ETF OHLCV and
macroeconomic daily datasets. Research code should call `DataAgent.load(...)`
instead of calling vendor APIs directly.

V1 deliberately does not contain alpha logic, signal generation, forecasting,
portfolio construction, execution, or live trading hooks.

## Supported Scope

- Data types: daily ETF OHLCV and daily macroeconomic series
- Sources: `yfinance` for OHLCV, FRED CSV endpoint for macro series
- Storage: pyarrow-backed parquet under `data/processed/`
- Registry: JSON metadata under `data/external/registry/datasets.json`
- Raw extracts: immutable parquet snapshots under `data/raw/`

## Canonical Schemas

OHLCV:

```text
timestamp, open, high, low, close, volume, symbol, source, frequency
```

Macro:

```text
timestamp, value, series_id, source, frequency
```

## Example Usage

```python
from datetime import date

from src.data import DataAgent, DataFrequency, DataRequest, DataSource, DataType

agent = DataAgent(project_root=".")

spy = agent.load(
    DataRequest(
        symbol="SPY",
        data_type=DataType.OHLCV,
        source=DataSource.YFINANCE,
        start_date=date(2020, 1, 1),
        end_date=date(2020, 12, 31),
        frequency=DataFrequency.DAILY,
    )
)

cpi = agent.load(
    DataRequest(
        symbol="CPIAUCSL",
        data_type=DataType.MACRO,
        source=DataSource.FRED,
        start_date=date(2020, 1, 1),
        end_date=date(2020, 12, 31),
        frequency=DataFrequency.DAILY,
    )
)
```

## Architecture

The facade is intentionally thin:

- `contracts/`: Pydantic request models and supported enum values
- `downloaders/`: source adapters with retries and normalized timestamps
- `transformers/`: vendor-to-canonical schema standardization
- `validators/`: structured validation reports for empty data, duplicates,
  ordering, missing timestamps, required columns, and NaN ratios
- `loaders/`: parquet storage and `DataAgent`
- `registry/`: JSON metadata registry
- `update_engine/`: incremental update orchestration
- `models/`: metadata and validation report models

Raw datasets are never overwritten. Processed datasets are merged, deduplicated
by timestamp, revalidated, and only then written back to parquet.
