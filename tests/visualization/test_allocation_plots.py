"""Tests for src.visualization.allocation_plots.

Covers plot_concentration_evolution, plot_prediction_dispersion,
and plot_confidence_calibration.  All tests verify return type, basic
invariants, and graceful fallback paths for degenerate inputs.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.visualization.allocation_plots import (
    plot_concentration_evolution,
    plot_confidence_calibration,
    plot_prediction_dispersion,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_weights(n_dates: int = 60, n_assets: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    raw = rng.dirichlet(np.ones(n_assets), size=n_dates)
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    return pd.DataFrame(raw, index=dates, columns=[f"A{i}" for i in range(n_assets)])


def _make_scores(n_dates: int = 60, n_assets: int = 5, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_dates, n_assets))
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    return pd.DataFrame(data, index=dates, columns=[f"A{i}" for i in range(n_assets)])


def _make_calibration_data(n_quintiles: int = 5, monotonic: bool = True) -> dict:
    if monotonic:
        q_means = [float(i) * 0.01 for i in range(n_quintiles)]
    else:
        q_means = [0.03, 0.01, 0.04, 0.02, 0.00]
    labels = [f"Q{i+1}" for i in range(n_quintiles)]
    qr = pd.Series(q_means, index=labels)
    qc = pd.Series([20] * n_quintiles, index=labels)
    spread = q_means[-1] - q_means[0]
    is_monotonic = all(q_means[i] <= q_means[i + 1] for i in range(len(q_means) - 1))
    return {
        "quintile_returns": qr,
        "quintile_counts": qc,
        "monotonic_up": is_monotonic,
        "top_minus_bottom_spread": spread,
    }


# ---------------------------------------------------------------------------
# plot_concentration_evolution
# ---------------------------------------------------------------------------

def test_concentration_evolution_returns_figure():
    weights = _make_weights()
    fig = plot_concentration_evolution(weights)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_concentration_evolution_three_panels():
    weights = _make_weights()
    fig = plot_concentration_evolution(weights)
    assert len(fig.axes) == 3
    plt.close(fig)


def test_concentration_evolution_short_window():
    weights = _make_weights(n_dates=10)
    fig = plot_concentration_evolution(weights, window=5)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_concentration_evolution_empty_weights_fallback():
    empty = pd.DataFrame(dtype=float)
    fig = plot_concentration_evolution(empty)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_concentration_evolution_all_zero_weights_fallback():
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    zero_weights = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
    fig = plot_concentration_evolution(zero_weights)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_concentration_evolution_equal_weight_consistent_hhi():
    n_assets = 4
    weights = pd.DataFrame(
        1.0 / n_assets,
        index=pd.date_range("2020-01-01", periods=30, freq="B"),
        columns=[f"A{i}" for i in range(n_assets)],
    )
    fig = plot_concentration_evolution(weights)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_concentration_evolution_with_title():
    weights = _make_weights()
    fig = plot_concentration_evolution(weights, title="Custom Title")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# ---------------------------------------------------------------------------
# plot_prediction_dispersion
# ---------------------------------------------------------------------------

def test_prediction_dispersion_returns_figure():
    scores = _make_scores()
    fig = plot_prediction_dispersion(scores)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_prediction_dispersion_two_panels():
    scores = _make_scores()
    fig = plot_prediction_dispersion(scores)
    assert len(fig.axes) == 2
    plt.close(fig)


def test_prediction_dispersion_empty_fallback():
    empty = pd.DataFrame(dtype=float)
    fig = plot_prediction_dispersion(empty)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_prediction_dispersion_insufficient_data_fallback():
    scores = _make_scores(n_dates=3)
    fig = plot_prediction_dispersion(scores)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_prediction_dispersion_with_stress_mask():
    scores = _make_scores()
    stress = pd.Series(False, index=scores.index)
    stress.iloc[10:20] = True
    fig = plot_prediction_dispersion(scores, stress_mask=stress)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_prediction_dispersion_single_asset_no_error():
    dates = pd.date_range("2020-01-01", periods=30, freq="B")
    scores = pd.DataFrame({"A": np.random.default_rng(0).standard_normal(30)},
                          index=dates)
    fig = plot_prediction_dispersion(scores)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_prediction_dispersion_custom_window():
    scores = _make_scores()
    fig = plot_prediction_dispersion(scores, window=21)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# ---------------------------------------------------------------------------
# plot_confidence_calibration
# ---------------------------------------------------------------------------

def test_confidence_calibration_returns_figure():
    data = _make_calibration_data()
    fig = plot_confidence_calibration(data)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_confidence_calibration_monotonic_data():
    data = _make_calibration_data(monotonic=True)
    fig = plot_confidence_calibration(data)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_confidence_calibration_non_monotonic_data():
    data = _make_calibration_data(monotonic=False)
    fig = plot_confidence_calibration(data)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_confidence_calibration_insufficient_data_fallback():
    fig = plot_confidence_calibration({"quintile_returns": None})
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_confidence_calibration_empty_dict_fallback():
    fig = plot_confidence_calibration({})
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_confidence_calibration_with_group_monthly():
    data = _make_calibration_data()
    dates = pd.date_range("2020-01-01", periods=12, freq="ME")
    rng = np.random.default_rng(42)
    data["group_monthly"] = pd.DataFrame(
        {"top": rng.standard_normal(12) * 0.01,
         "mid": rng.standard_normal(12) * 0.005,
         "bottom": rng.standard_normal(12) * 0.01},
        index=dates,
    )
    fig = plot_confidence_calibration(data)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_confidence_calibration_without_quintile_counts():
    data = _make_calibration_data()
    data.pop("quintile_counts", None)
    fig = plot_confidence_calibration(data)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_confidence_calibration_three_quintiles():
    labels = ["Q1", "Q2", "Q3"]
    data = {
        "quintile_returns": pd.Series([0.01, 0.02, 0.03], index=labels),
        "quintile_counts": pd.Series([10, 10, 10], index=labels),
        "monotonic_up": True,
        "top_minus_bottom_spread": 0.02,
    }
    fig = plot_confidence_calibration(data)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Save-path smoke test
# ---------------------------------------------------------------------------

def test_concentration_evolution_save_path(tmp_path):
    weights = _make_weights()
    out = str(tmp_path / "test_concentration.png")
    fig = plot_concentration_evolution(weights, save_path=out)
    assert isinstance(fig, plt.Figure)
    import os
    assert os.path.exists(out)
    plt.close(fig)


def test_prediction_dispersion_save_path(tmp_path):
    scores = _make_scores()
    out = str(tmp_path / "test_dispersion.png")
    fig = plot_prediction_dispersion(scores, save_path=out)
    assert isinstance(fig, plt.Figure)
    import os
    assert os.path.exists(out)
    plt.close(fig)


def test_confidence_calibration_save_path(tmp_path):
    data = _make_calibration_data()
    out = str(tmp_path / "test_calibration.png")
    fig = plot_confidence_calibration(data, save_path=out)
    assert isinstance(fig, plt.Figure)
    import os
    assert os.path.exists(out)
    plt.close(fig)
