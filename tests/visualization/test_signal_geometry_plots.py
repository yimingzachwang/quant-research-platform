"""Tests for src/visualization/signal_geometry_plots.py (Phase 3A)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from src.visualization.signal_geometry_plots import (
    plot_calibration_sweep,
    plot_concentration_emergence,
    plot_dispersion_evolution,
    plot_dispersion_sweep,
    plot_intrabasket_geometry,
    plot_robustness_tradeoff,
    plot_turnover_by_alpha,
    plot_wf_stability_heatmap,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def dispersion_data() -> dict[str, dict]:
    return {
        "α=0.50": {"mean_cs_std": 0.036, "mean_cs_spread": 0.102, "min_cs_std": 0.006, "max_cs_std": 0.080},
        "α=0.10": {"mean_cs_std": 0.085, "mean_cs_spread": 0.240, "min_cs_std": 0.015, "max_cs_std": 0.190},
        "α=0.05": {"mean_cs_std": 0.120, "mean_cs_spread": 0.340, "min_cs_std": 0.020, "max_cs_std": 0.270},
        "α=0.01": {"mean_cs_std": 0.200, "mean_cs_spread": 0.560, "min_cs_std": 0.030, "max_cs_std": 0.450},
    }


@pytest.fixture()
def cs_std_series() -> dict[str, pd.Series]:
    idx = pd.date_range("2018-01-01", periods=300, freq="B")
    rng = np.random.default_rng(42)
    return {
        "α=0.50": pd.Series(rng.normal(0.036, 0.005, 300), index=idx).clip(0.001),
        "α=0.10": pd.Series(rng.normal(0.085, 0.015, 300), index=idx).clip(0.001),
        "α=0.05": pd.Series(rng.normal(0.120, 0.025, 300), index=idx).clip(0.001),
        "α=0.01": pd.Series(rng.normal(0.200, 0.050, 300), index=idx).clip(0.001),
    }


@pytest.fixture()
def calibration_data() -> dict[str, dict]:
    return {
        "α=0.50": {
            "quintile_returns": pd.Series({"Q1": -0.003, "Q2": 0.005, "Q3": 0.011, "Q4": 0.012, "Q5": 0.013}),
            "monotonic_up": True,
            "top_minus_bottom_spread": 0.016,
        },
        "α=0.10": {
            "quintile_returns": pd.Series({"Q1": -0.006, "Q2": 0.004, "Q3": 0.013, "Q4": 0.018, "Q5": 0.025}),
            "monotonic_up": True,
            "top_minus_bottom_spread": 0.031,
        },
        "α=0.05": {
            "quintile_returns": pd.Series({"Q1": -0.004, "Q2": 0.002, "Q3": 0.007, "Q4": 0.010, "Q5": 0.014}),
            "monotonic_up": True,
            "top_minus_bottom_spread": 0.018,
        },
        "α=0.01": {
            "quintile_returns": pd.Series({"Q1": 0.002, "Q2": 0.003, "Q3": 0.001, "Q4": 0.008, "Q5": 0.010}),
            "monotonic_up": False,
            "top_minus_bottom_spread": 0.008,
        },
    }


@pytest.fixture()
def split_sharpe_data() -> tuple[dict[str, list[float]], list[str]]:
    data = {
        "α=0.50": [2.58, 0.45, 0.89, -0.32, 1.12, 0.67, 0.95],
        "α=0.10": [2.70, 0.51, 1.02, -0.18, 1.35, 0.82, 1.10],
        "α=0.05": [2.45, 0.38, 0.75, -0.60, 1.05, 0.50, 0.88],
        "α=0.01": [2.10, 0.20, 0.45, -1.20, 0.70, 0.30, 0.55],
    }
    labels = ["2017", "2018", "2019", "2020", "2021", "2022", "2023"]
    return data, labels


@pytest.fixture()
def summary_data() -> dict[str, dict]:
    return {
        "α=0.50": {"mean_cs_std": 0.036, "oos_mean_sharpe": 0.645, "oos_sharpe_std": 0.85, "mean_turnover": 0.127},
        "α=0.10": {"mean_cs_std": 0.085, "oos_mean_sharpe": 0.720, "oos_sharpe_std": 0.90, "mean_turnover": 0.145},
        "α=0.05": {"mean_cs_std": 0.120, "oos_mean_sharpe": 0.680, "oos_sharpe_std": 1.10, "mean_turnover": 0.162},
        "α=0.01": {"mean_cs_std": 0.200, "oos_mean_sharpe": 0.580, "oos_sharpe_std": 1.45, "mean_turnover": 0.195},
    }


@pytest.fixture()
def turnover_series() -> dict[str, pd.Series]:
    idx = pd.date_range("2018-01-01", periods=500, freq="B")
    rng = np.random.default_rng(10)
    return {
        "α=0.50": pd.Series(rng.exponential(0.127, 500), index=idx),
        "α=0.10": pd.Series(rng.exponential(0.145, 500), index=idx),
        "α=0.05": pd.Series(rng.exponential(0.162, 500), index=idx),
        "α=0.01": pd.Series(rng.exponential(0.195, 500), index=idx),
    }


@pytest.fixture()
def hhi_series() -> dict[str, pd.Series]:
    idx = pd.date_range("2018-01-01", periods=500, freq="B")
    rng = np.random.default_rng(7)
    return {
        "α=0.50": pd.Series(0.200 + rng.normal(0, 0.002, 500), index=idx).clip(0.1),
        "α=0.10": pd.Series(0.215 + rng.normal(0, 0.005, 500), index=idx).clip(0.1),
        "α=0.05": pd.Series(0.235 + rng.normal(0, 0.010, 500), index=idx).clip(0.1),
        "α=0.01": pd.Series(0.260 + rng.normal(0, 0.020, 500), index=idx).clip(0.1),
    }


# ---------------------------------------------------------------------------
# plot_dispersion_sweep
# ---------------------------------------------------------------------------


def test_dispersion_sweep_returns_figure(dispersion_data):
    fig = plot_dispersion_sweep(dispersion_data)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_dispersion_sweep_has_two_panels(dispersion_data):
    fig = plot_dispersion_sweep(dispersion_data)
    assert len(fig.axes) == 2
    plt.close(fig)


def test_dispersion_sweep_empty_input():
    fig = plot_dispersion_sweep({})
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_dispersion_sweep_single_alpha():
    fig = plot_dispersion_sweep({"α=0.50": {"mean_cs_std": 0.036, "mean_cs_spread": 0.102}})
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_dispersion_sweep_nan_values():
    data = {"α=0.50": {"mean_cs_std": float("nan"), "mean_cs_spread": 0.102}}
    fig = plot_dispersion_sweep(data)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_dispersion_sweep_save_path(tmp_path, dispersion_data):
    p = tmp_path / "disp_sweep.png"
    fig = plot_dispersion_sweep(dispersion_data, save_path=str(p))
    assert p.exists()
    plt.close(fig)


# ---------------------------------------------------------------------------
# plot_dispersion_evolution
# ---------------------------------------------------------------------------


def test_dispersion_evolution_returns_figure(cs_std_series):
    fig = plot_dispersion_evolution(cs_std_series)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_dispersion_evolution_single_panel(cs_std_series):
    fig = plot_dispersion_evolution(cs_std_series)
    assert len(fig.axes) == 1
    plt.close(fig)


def test_dispersion_evolution_empty_input():
    fig = plot_dispersion_evolution({})
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_dispersion_evolution_custom_window(cs_std_series):
    fig = plot_dispersion_evolution(cs_std_series, window=21)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_dispersion_evolution_none_series():
    fig = plot_dispersion_evolution({"α=0.50": None, "α=0.10": pd.Series(dtype=float)})
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_dispersion_evolution_save_path(tmp_path, cs_std_series):
    p = tmp_path / "disp_evo.png"
    fig = plot_dispersion_evolution(cs_std_series, save_path=str(p))
    assert p.exists()
    plt.close(fig)


# ---------------------------------------------------------------------------
# plot_calibration_sweep
# ---------------------------------------------------------------------------


def test_calibration_sweep_returns_figure(calibration_data):
    fig = plot_calibration_sweep(calibration_data)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_calibration_sweep_single_panel(calibration_data):
    fig = plot_calibration_sweep(calibration_data)
    assert len(fig.axes) == 1
    plt.close(fig)


def test_calibration_sweep_empty_input():
    fig = plot_calibration_sweep({})
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_calibration_sweep_non_monotonic(calibration_data):
    # Ensure non-monotonic α=0.01 doesn't raise
    fig = plot_calibration_sweep(calibration_data)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_calibration_sweep_dict_quintile_returns():
    # Dict-style quintile returns (from JSON)
    data = {"α=0.50": {
        "quintile_returns": {"Q1": -0.003, "Q2": 0.005, "Q3": 0.011, "Q4": 0.012, "Q5": 0.013},
        "monotonic_up": True,
    }}
    fig = plot_calibration_sweep(data)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_calibration_sweep_save_path(tmp_path, calibration_data):
    p = tmp_path / "calib_sweep.png"
    fig = plot_calibration_sweep(calibration_data, save_path=str(p))
    assert p.exists()
    plt.close(fig)


# ---------------------------------------------------------------------------
# plot_wf_stability_heatmap
# ---------------------------------------------------------------------------


def test_wf_stability_heatmap_returns_figure(split_sharpe_data):
    data, labels = split_sharpe_data
    fig = plot_wf_stability_heatmap(data, labels)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_wf_stability_heatmap_empty_input():
    fig = plot_wf_stability_heatmap({})
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_wf_stability_heatmap_no_split_labels(split_sharpe_data):
    data, _ = split_sharpe_data
    fig = plot_wf_stability_heatmap(data)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_wf_stability_heatmap_single_split():
    data = {"α=0.50": [0.8], "α=0.10": [1.2]}
    fig = plot_wf_stability_heatmap(data, ["2023"])
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_wf_stability_heatmap_save_path(tmp_path, split_sharpe_data):
    data, labels = split_sharpe_data
    p = tmp_path / "wf_heat.png"
    fig = plot_wf_stability_heatmap(data, labels, save_path=str(p))
    assert p.exists()
    plt.close(fig)


# ---------------------------------------------------------------------------
# plot_robustness_tradeoff
# ---------------------------------------------------------------------------


def test_robustness_tradeoff_returns_figure(summary_data):
    fig = plot_robustness_tradeoff(summary_data)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_robustness_tradeoff_empty_input():
    fig = plot_robustness_tradeoff({})
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_robustness_tradeoff_nan_values():
    data = {"α=0.50": {"mean_cs_std": float("nan"), "oos_mean_sharpe": 0.6}}
    fig = plot_robustness_tradeoff(data)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_robustness_tradeoff_save_path(tmp_path, summary_data):
    p = tmp_path / "robust.png"
    fig = plot_robustness_tradeoff(summary_data, save_path=str(p))
    assert p.exists()
    plt.close(fig)


# ---------------------------------------------------------------------------
# plot_turnover_by_alpha
# ---------------------------------------------------------------------------


def test_turnover_by_alpha_returns_figure(turnover_series):
    fig = plot_turnover_by_alpha(turnover_series)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_turnover_by_alpha_empty_input():
    fig = plot_turnover_by_alpha({})
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_turnover_by_alpha_custom_window(turnover_series):
    fig = plot_turnover_by_alpha(turnover_series, window=5)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_turnover_by_alpha_save_path(tmp_path, turnover_series):
    p = tmp_path / "to.png"
    fig = plot_turnover_by_alpha(turnover_series, save_path=str(p))
    assert p.exists()
    plt.close(fig)


# ---------------------------------------------------------------------------
# plot_intrabasket_geometry
# ---------------------------------------------------------------------------


def test_intrabasket_geometry_returns_figure(dispersion_data):
    fig = plot_intrabasket_geometry(dispersion_data)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_intrabasket_geometry_two_panels(dispersion_data):
    fig = plot_intrabasket_geometry(dispersion_data)
    assert len(fig.axes) == 2
    plt.close(fig)


def test_intrabasket_geometry_empty_input():
    fig = plot_intrabasket_geometry({})
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_intrabasket_geometry_save_path(tmp_path, dispersion_data):
    p = tmp_path / "ib.png"
    fig = plot_intrabasket_geometry(dispersion_data, save_path=str(p))
    assert p.exists()
    plt.close(fig)


# ---------------------------------------------------------------------------
# plot_concentration_emergence
# ---------------------------------------------------------------------------


def test_concentration_emergence_returns_figure(hhi_series):
    fig = plot_concentration_emergence(hhi_series)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_concentration_emergence_empty_input():
    fig = plot_concentration_emergence({})
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_concentration_emergence_custom_window(hhi_series):
    fig = plot_concentration_emergence(hhi_series, window=21)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_concentration_emergence_save_path(tmp_path, hhi_series):
    p = tmp_path / "conc.png"
    fig = plot_concentration_emergence(hhi_series, save_path=str(p))
    assert p.exists()
    plt.close(fig)
