# Data Layer

Status: STABLE/PARTIAL.

## Implemented Components

| Component | Status | Evidence |
|---|---:|---|
| `DataRequest` | STABLE | Pydantic model in `src/data/contracts/requests.py` |
| V1 config models | STABLE | `src/data/config.py` |
| Dataset profile execution | STABLE | `src/data/engine.py::DatasetUpdateEngine` |
| Yahoo/FRED downloaders | STABLE/PARTIAL | source-specific adapters, network-dependent in real use |
| Standardizers | STABLE/PARTIAL | OHLCV/macro canonical transforms |
| `DatasetValidator` | STABLE | structured validation report |
| `DataStorage` | STABLE | deterministic raw/processed/metadata paths |
| `DatasetManifest` | STABLE | manifest pydantic model |
| `DatasetRegistry` | STABLE | JSON-backed registry |
| `load_dataset` | STABLE | registry-backed parquet loader |
| `JsonDatasetRegistry`/`DatasetMetadata` | PARTIAL/LEGACY | overlapping metadata path |

## Data Flow

1. `DatasetUpdateEngine.run_profile(profile_path)` loads a dataset profile.
2. The profile references `configs/data/data_agent_v1.yaml`.
3. The engine loads a universe and date range.
4. `expand_profile_data_requests()` creates one `DataRequest` per symbol.
5. `hash_request()` computes a deterministic SHA256 request hash.
6. Existing request hashes are skipped.
7. Downloader fetches source-shaped data.
8. Standardizer converts vendor columns into canonical shape.
9. `DatasetValidator` validates required columns, empty data, duplicate and
   monotonic timestamps, expected timestamps, and NaN ratios.
10. `DataStorage` writes raw and processed parquet files.
11. The registry stores a `DatasetManifest`.

## Contracts

`DataRequest` fields: `symbol`, `data_type`, `source`, `start_date`,
`end_date`, `frequency`. V1 supports `ohlcv/yfinance`, `macro/fred`, and
daily frequency.

`DatasetManifest` fields include identity, symbol, source, frequency, date
range, schema version, storage path, row count, created time, and request hash.

`DatasetQuery` supports exact matching by dataset family, symbol, frequency,
source, dataset id/name, schema version, and required columns.

## Assumptions And Limitations

- The canonical config lists `adjusted_close` for OHLCV, but
  `OHLCVStandardizer.canonical_columns` and `DatasetValidator._required_columns`
  do not currently include it. This is an evidence-backed inconsistency.
- Raw parquet filenames include a UTC timestamp, so raw extracts are not
  deterministic paths even though processed paths are deterministic.
- Registry writes are JSON-file based, append/replace by request hash, and not
  a concurrent multi-process database.
- Downloaders require network access and vendor dependencies in real runs.

## Should Not Do

- Strategy logic.
- Feature engineering.
- ML training.
- Backtesting.
- Visualization/reporting.

