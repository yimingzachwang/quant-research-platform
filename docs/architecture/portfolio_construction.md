# Portfolio Construction

Status: STABLE/PARTIAL.

## Implemented Utilities

| Area | Modules | Status |
|---|---|---:|
| Data alignment | `src/portfolio/alignment.py` | STABLE |
| Panel features | `src/portfolio/panel.py` | STABLE |
| Ranking/selection | `src/portfolio/ranking.py` | STABLE |
| Allocation | `src/portfolio/allocation.py` | STABLE |
| Portfolio backtest | `src/portfolio/portfolio_backtest.py` | STABLE |
| Interfaces/no-op | `src/portfolio/interfaces.py`, `placeholders.py` | PLACEHOLDER |

## Flow

1. `load_universe(symbols)` loads registered datasets through `load_dataset`.
2. `align_prices(universe)` extracts close prices and aligns by date.
3. Panel functions compute returns, momentum, volatility, or z-scores.
4. Ranking functions select top/bottom assets.
5. Allocation functions produce weights.
6. `run_portfolio_backtest` evaluates lagged weights.

## Anti-Leakage

- Allocation and strategy code emits weights observable at date `t`.
- Backtesting applies weights at `t+1` via `weights.shift(1)`.
- `resample_weights_to_daily()` forward-fills periodic weights; the backtest
  lag handles next-period application.

## Limitations

- No optimizer or constraint solver.
- No covariance/risk model based allocation.
- No production portfolio construction service.
- Current implemented loading/alignment lives in `alignment.py`; there are no
  `src.portfolio.universe` or `src.portfolio.returns` modules.

## Should Not Do

- Download data.
- Own experiment tracking.
- Perform reporting.
- Hide strategy timing assumptions.

