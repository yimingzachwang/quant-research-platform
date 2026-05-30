# Cleaning Layer

Status: STABLE.

## Purpose

`src/cleaning` contains small, explicit utilities for local DataFrame hygiene
and OHLCV validation. It is separate from the ingestion validator in
`src/data/validators`.

## Implemented Functions

| Module | Functions / Types | Responsibility |
|---|---|---|
| `timestamps.py` | `sort_time_index`, `remove_duplicate_timestamps` | Sort and deduplicate DatetimeIndex rows |
| `numeric.py` | `replace_inf` | Replace positive/negative infinity in numeric data |
| `missing.py` | `forward_fill_limited` | Forward-fill bounded gaps only |
| `validation.py` | `OHLCVValidationResult`, `validate_ohlcv` | Check OHLCV columns, negative prices/volume, high/low consistency, NaNs |

## Contracts

- Functions return new pandas objects or validation result dataclasses.
- `validate_ohlcv(..., raise_on_error=True)` raises on structural OHLCV issues.
- Bounded forward-fill keeps longer gaps visible as NaN.

## Invariants

- Cleaning is conservative and explicit.
- No function performs hidden imputation beyond the requested bounded fill.
- OHLCV validation detects problems; it does not silently repair them.

## Should Not Do

- Load data from disk or vendors.
- Decide research-specific imputation policy.
- Generate alpha features or labels.
- Persist cleaned datasets without going through data/storage contracts.

