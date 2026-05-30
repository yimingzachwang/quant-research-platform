"""Smoke tests for src.visualization.ml_plots.

All tests verify that functions return a matplotlib Figure without error.
No pixel-level assertions — these are structural smoke tests only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
import pytest

matplotlib.use("Agg")

from src.visualization.ml_plots import (
    plot_coefficient_stability,
    plot_information_coefficient,
    plot_prediction_distribution,
    plot_prediction_vs_actual,
    plot_signal_turnover,
    plot_split_metric_stability,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dates(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="B")


def _make_weights(n_dates: int = 20, n_assets: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = _make_dates(n_dates)
    assets = [f"A{i}" for i in range(n_assets)]
    raw = np.abs(rng.standard_normal((n_dates, n_assets))) + 0.1
    w = raw / raw.sum(axis=1, keepdims=True)
    return pd.DataFrame(w, index=dates, columns=assets)


def _make_metric_df(n_splits: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = {
        "sharpe_ratio": rng.standard_normal(n_splits),
        "annualized_return": rng.standard_normal(n_splits) * 0.1,
        "max_drawdown": -np.abs(rng.standard_normal(n_splits)) * 0.2,
    }
    return pd.DataFrame(data, index=range(n_splits))


def _make_coeff_df(n_features: int = 6, n_splits: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    features = [f"feat_{i}" for i in range(n_features)]
    data = rng.standard_normal((n_splits, n_features))
    coeff_splits = pd.DataFrame(data, columns=features)

    from src.ml.diagnostics.stability import coefficient_stability
    return coefficient_stability(coeff_splits)


# ---------------------------------------------------------------------------
# plot_prediction_vs_actual
# ---------------------------------------------------------------------------

class TestPlotPredictionVsActual:
    def test_returns_figure(self):
        import matplotlib.pyplot as plt
        idx = _make_dates(60)
        actual = pd.Series(np.random.randn(60), index=idx)
        predicted = actual + np.random.randn(60) * 0.3
        fig = plot_prediction_vs_actual(actual, predicted)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_two_axes(self):
        import matplotlib.pyplot as plt
        idx = _make_dates(40)
        actual = pd.Series(np.random.randn(40), index=idx)
        predicted = pd.Series(np.random.randn(40), index=idx)
        fig = plot_prediction_vs_actual(actual, predicted)
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_custom_title(self):
        import matplotlib.pyplot as plt
        idx = _make_dates(30)
        a = pd.Series(np.random.randn(30), index=idx)
        p = pd.Series(np.random.randn(30), index=idx)
        fig = plot_prediction_vs_actual(a, p, title="My Title")
        # Title on first axes
        assert "My Title" in fig.axes[0].get_title()
        plt.close(fig)

    def test_nan_rows_dropped(self):
        import matplotlib.pyplot as plt
        idx = _make_dates(20)
        vals = np.random.randn(20)
        vals[3] = np.nan
        actual = pd.Series(vals, index=idx)
        predicted = pd.Series(np.random.randn(20), index=idx)
        fig = plot_prediction_vs_actual(actual, predicted)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_prediction_distribution
# ---------------------------------------------------------------------------

class TestPlotPredictionDistribution:
    def test_returns_figure(self):
        import matplotlib.pyplot as plt
        idx = _make_dates(100)
        preds = pd.Series(np.random.randn(100), index=idx)
        fig = plot_prediction_distribution(preds)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_bins(self):
        import matplotlib.pyplot as plt
        idx = _make_dates(100)
        preds = pd.Series(np.random.randn(100), index=idx)
        fig = plot_prediction_distribution(preds, bins=20)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_axis(self):
        import matplotlib.pyplot as plt
        idx = _make_dates(50)
        preds = pd.Series(np.random.randn(50), index=idx)
        fig = plot_prediction_distribution(preds)
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_nan_handling(self):
        import matplotlib.pyplot as plt
        idx = _make_dates(50)
        vals = np.random.randn(50)
        vals[::5] = np.nan
        preds = pd.Series(vals, index=idx)
        fig = plot_prediction_distribution(preds)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_information_coefficient
# ---------------------------------------------------------------------------

class TestPlotInformationCoefficient:
    def _make_ic(self, n: int = 60) -> pd.Series:
        idx = _make_dates(n)
        return pd.Series(np.random.randn(n) * 0.1, index=idx, name="IC")

    def test_returns_figure(self):
        import matplotlib.pyplot as plt
        ic = self._make_ic()
        fig = plot_information_coefficient(ic)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_axis(self):
        import matplotlib.pyplot as plt
        ic = self._make_ic()
        fig = plot_information_coefficient(ic)
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_custom_rolling_window(self):
        import matplotlib.pyplot as plt
        ic = self._make_ic(n=80)
        fig = plot_information_coefficient(ic, rolling_window=10)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_short_series_no_rolling_line(self):
        import matplotlib.pyplot as plt
        # Fewer dates than rolling_window → should not crash
        ic = self._make_ic(n=10)
        fig = plot_information_coefficient(ic, rolling_window=21)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_split_metric_stability
# ---------------------------------------------------------------------------

class TestPlotSplitMetricStability:
    def test_returns_figure(self):
        import matplotlib.pyplot as plt
        df = _make_metric_df()
        fig = plot_split_metric_stability(df, metric="sharpe_ratio")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_axis(self):
        import matplotlib.pyplot as plt
        df = _make_metric_df()
        fig = plot_split_metric_stability(df)
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_raises_on_missing_metric(self):
        df = _make_metric_df()
        with pytest.raises(ValueError, match="not found"):
            plot_split_metric_stability(df, metric="nonexistent_metric")

    def test_custom_metric(self):
        import matplotlib.pyplot as plt
        df = _make_metric_df()
        fig = plot_split_metric_stability(df, metric="annualized_return")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_default_metric_is_sharpe(self):
        import matplotlib.pyplot as plt
        df = _make_metric_df()
        # should not raise
        fig = plot_split_metric_stability(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_coefficient_stability
# ---------------------------------------------------------------------------

class TestPlotCoefficientStability:
    def test_returns_figure(self):
        import matplotlib.pyplot as plt
        df = _make_coeff_df()
        fig = plot_coefficient_stability(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_axis(self):
        import matplotlib.pyplot as plt
        df = _make_coeff_df()
        fig = plot_coefficient_stability(df)
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_top_n_filter(self):
        import matplotlib.pyplot as plt
        df = _make_coeff_df(n_features=10)
        fig = plot_coefficient_stability(df, top_n=3)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_raises_on_missing_mean_column(self):
        df = pd.DataFrame({"std": [0.1, 0.2]}, index=["A", "B"])
        with pytest.raises(ValueError, match="missing required columns"):
            plot_coefficient_stability(df)

    def test_raises_on_missing_std_column(self):
        df = pd.DataFrame({"mean": [0.1, 0.2]}, index=["A", "B"])
        with pytest.raises(ValueError, match="missing required columns"):
            plot_coefficient_stability(df)


# ---------------------------------------------------------------------------
# plot_signal_turnover
# ---------------------------------------------------------------------------

class TestPlotSignalTurnover:
    def _make_turnover(self, n: int = 30) -> pd.Series:
        from src.ml.diagnostics.turnover import signal_turnover
        w = _make_weights(n_dates=n)
        return signal_turnover(w)

    def test_returns_figure(self):
        import matplotlib.pyplot as plt
        to = self._make_turnover()
        fig = plot_signal_turnover(to)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_axis(self):
        import matplotlib.pyplot as plt
        to = self._make_turnover()
        fig = plot_signal_turnover(to)
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_custom_title(self):
        import matplotlib.pyplot as plt
        to = self._make_turnover()
        fig = plot_signal_turnover(to, title="Custom Title")
        assert "Custom Title" in fig.axes[0].get_title()
        plt.close(fig)

    def test_nan_prefix_handled(self):
        import matplotlib.pyplot as plt
        # signal_turnover has NaN in first row — should be dropped without error
        to = self._make_turnover(n=20)
        assert pd.isna(to.iloc[0])
        fig = plot_signal_turnover(to)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_all_nan_series(self):
        import matplotlib.pyplot as plt
        idx = _make_dates(5)
        to = pd.Series([np.nan] * 5, index=idx, name="turnover")
        # Should not crash — mean annotation is omitted
        fig = plot_signal_turnover(to)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Package import
# ---------------------------------------------------------------------------

def test_import_from_src_visualization():
    from src.visualization import (
        plot_coefficient_stability,
        plot_information_coefficient,
        plot_prediction_distribution,
        plot_prediction_vs_actual,
        plot_signal_turnover,
        plot_split_metric_stability,
    )
    assert callable(plot_prediction_vs_actual)
    assert callable(plot_prediction_distribution)
    assert callable(plot_information_coefficient)
    assert callable(plot_split_metric_stability)
    assert callable(plot_coefficient_stability)
    assert callable(plot_signal_turnover)
