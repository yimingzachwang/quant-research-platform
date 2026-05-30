from datetime import UTC, date, datetime

from src.backtesting import BacktestEngine
from src.core import DateRange, ExperimentContext, Horizon, Universe


def test_backtest_engine_returns_empty_result_for_skeleton_context() -> None:
    context = ExperimentContext(
        experiment_id="exp_test",
        name="skeleton test",
        created_at=datetime.now(UTC),
        universe=Universe(name="core_etfs", symbols=("SPY", "QQQ")),
        horizon=Horizon(name="daily", periods=1),
        date_range=DateRange(start=date(2020, 1, 1), end=date(2020, 12, 31)),
    )

    result = BacktestEngine().run(context)

    assert result.context == context
    assert result.artifacts == {}
    assert result.metrics == {}
