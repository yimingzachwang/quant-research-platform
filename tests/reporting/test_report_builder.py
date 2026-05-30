"""Tests for src/reporting/report_builder.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from src.reporting.report_builder import (
    ExperimentArtefacts,
    ReportPaths,
    generate_experiment_report,
    load_experiment_artefacts,
)

# ---------------------------------------------------------------------------
# Helpers — minimal artefact directory builders
# ---------------------------------------------------------------------------

_METADATA = {
    "experiment_name": "test_exp",
    "strategy_name": "EqualWeight(freq=ME)",
    "parameters": {"rebalance_freq": "ME"},
    "created_at": "2026-05-23T12:00:00+00:00",
}

_METRICS = {
    "annualized_return": 0.08,
    "annualized_volatility": 0.12,
    "sharpe_ratio": 0.65,
    "max_drawdown": -0.15,
    "calmar_ratio": 0.53,
    "hit_rate": 0.52,
}

_CONFIG = {
    "name": "test_exp",
    "version": "1",
    "universe": {"tickers": ["SPY", "QQQ"]},
    "date_range": {"start": "2020-01-01", "end": "2023-12-31"},
    "strategy": {"type": "EqualWeight", "parameters": {"rebalance_freq": "ME"}},
    "validation": {"type": "none", "parameters": {}},
    "execution": {"transaction_cost_bps": 5.0},
    "output": {},
    "tags": [],
    "description": "",
}


def _make_artefact_dir(
    tmp_path: Path,
    *,
    include_config: bool = True,
    config_name: str = "normalized_config.json",
    include_plots: bool = False,
    n_plots: int = 1,
) -> Path:
    """Create a minimal experiment artefact directory."""
    d = tmp_path / "exp_dir"
    d.mkdir()
    (d / "metadata.json").write_text(json.dumps(_METADATA), encoding="utf-8")
    (d / "metrics.json").write_text(json.dumps(_METRICS), encoding="utf-8")
    if include_config:
        (d / config_name).write_text(json.dumps(_CONFIG), encoding="utf-8")
    if include_plots:
        plots = d / "plots"
        plots.mkdir()
        for i in range(n_plots):
            (plots / f"figure_{i}.png").write_bytes(b"\x89PNG\r\n")  # minimal PNG header
    return d


# ---------------------------------------------------------------------------
# load_experiment_artefacts
# ---------------------------------------------------------------------------


def test_load_returns_dataclass(tmp_path: Path) -> None:
    d = _make_artefact_dir(tmp_path)
    artefacts = load_experiment_artefacts(d, output_dir=tmp_path / "reports")
    assert isinstance(artefacts, ExperimentArtefacts)


def test_load_metadata_populated(tmp_path: Path) -> None:
    d = _make_artefact_dir(tmp_path)
    artefacts = load_experiment_artefacts(d, output_dir=tmp_path / "reports")
    assert artefacts.metadata["experiment_name"] == "test_exp"
    assert artefacts.metadata["strategy_name"] == "EqualWeight(freq=ME)"


def test_load_metrics_populated(tmp_path: Path) -> None:
    d = _make_artefact_dir(tmp_path)
    artefacts = load_experiment_artefacts(d, output_dir=tmp_path / "reports")
    assert pytest.approx(artefacts.metrics["sharpe_ratio"]) == 0.65


def test_load_config_prefers_normalized(tmp_path: Path) -> None:
    d = _make_artefact_dir(tmp_path, config_name="normalized_config.json")
    # Also write a config.json with different content to confirm preference
    (d / "config.json").write_text(json.dumps({"name": "wrong"}), encoding="utf-8")
    artefacts = load_experiment_artefacts(d, output_dir=tmp_path / "reports")
    assert artefacts.config["name"] == "test_exp"


def test_load_config_fallback_to_config_json(tmp_path: Path) -> None:
    d = _make_artefact_dir(tmp_path, config_name="config.json")
    artefacts = load_experiment_artefacts(d, output_dir=tmp_path / "reports")
    assert artefacts.config is not None
    assert artefacts.config["name"] == "test_exp"


def test_load_config_absent_returns_none(tmp_path: Path) -> None:
    d = _make_artefact_dir(tmp_path, include_config=False)
    artefacts = load_experiment_artefacts(d, output_dir=tmp_path / "reports")
    assert artefacts.config is None


def test_load_discovers_figures(tmp_path: Path) -> None:
    d = _make_artefact_dir(tmp_path, include_plots=True, n_plots=2)
    artefacts = load_experiment_artefacts(d, output_dir=tmp_path / "reports")
    assert len(artefacts.source_figures) == 2
    assert all(p.suffix == ".png" for p in artefacts.source_figures)


def test_load_no_plots_dir_gives_empty_list(tmp_path: Path) -> None:
    d = _make_artefact_dir(tmp_path, include_plots=False)
    artefacts = load_experiment_artefacts(d, output_dir=tmp_path / "reports")
    assert artefacts.source_figures == []


def test_load_artefact_dir_is_resolved(tmp_path: Path) -> None:
    d = _make_artefact_dir(tmp_path)
    artefacts = load_experiment_artefacts(d, output_dir=tmp_path / "reports")
    assert artefacts.artefact_dir == d.resolve()


def test_load_markdown_path_precomputed(tmp_path: Path) -> None:
    d = _make_artefact_dir(tmp_path)
    out = tmp_path / "reports"
    artefacts = load_experiment_artefacts(d, output_dir=out)
    assert artefacts.markdown_path == (out.resolve() / "markdown" / "test_exp.md")


def test_load_html_path_precomputed(tmp_path: Path) -> None:
    d = _make_artefact_dir(tmp_path)
    out = tmp_path / "reports"
    artefacts = load_experiment_artefacts(d, output_dir=out)
    assert artefacts.html_path == (out.resolve() / "html" / "test_exp.html")


def test_load_figure_dir_precomputed(tmp_path: Path) -> None:
    d = _make_artefact_dir(tmp_path)
    out = tmp_path / "reports"
    artefacts = load_experiment_artefacts(d, output_dir=out)
    assert artefacts.figure_dir == (out.resolve() / "figures" / "test_exp")


def test_load_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Experiment directory"):
        load_experiment_artefacts(tmp_path / "ghost", output_dir=tmp_path)


def test_load_missing_metadata_raises(tmp_path: Path) -> None:
    d = tmp_path / "exp"
    d.mkdir()
    (d / "metrics.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="metadata.json"):
        load_experiment_artefacts(d, output_dir=tmp_path)


def test_load_missing_metrics_raises(tmp_path: Path) -> None:
    d = tmp_path / "exp"
    d.mkdir()
    (d / "metadata.json").write_text(json.dumps(_METADATA), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="metrics.json"):
        load_experiment_artefacts(d, output_dir=tmp_path)


def test_load_figures_sorted(tmp_path: Path) -> None:
    d = _make_artefact_dir(tmp_path, include_plots=True, n_plots=3)
    artefacts = load_experiment_artefacts(d, output_dir=tmp_path / "reports")
    names = [p.name for p in artefacts.source_figures]
    assert names == sorted(names)


# ---------------------------------------------------------------------------
# generate_experiment_report
# ---------------------------------------------------------------------------


def test_generate_returns_report_paths(tmp_path: Path) -> None:
    d = _make_artefact_dir(tmp_path)
    paths = generate_experiment_report(d, output_dir=tmp_path / "reports")
    assert isinstance(paths, ReportPaths)


def test_generate_creates_markdown_file(tmp_path: Path) -> None:
    d = _make_artefact_dir(tmp_path)
    paths = generate_experiment_report(d, output_dir=tmp_path / "reports")
    assert paths.markdown.exists()
    assert paths.markdown.suffix == ".md"


def test_generate_creates_html_file(tmp_path: Path) -> None:
    d = _make_artefact_dir(tmp_path)
    paths = generate_experiment_report(d, output_dir=tmp_path / "reports")
    assert paths.html is not None
    assert paths.html.exists()
    assert paths.html.suffix == ".html"


def test_generate_creates_provenance_file(tmp_path: Path) -> None:
    d = _make_artefact_dir(tmp_path)
    paths = generate_experiment_report(d, output_dir=tmp_path / "reports")
    assert paths.provenance.exists()
    data = json.loads(paths.provenance.read_text())
    assert "report_version" in data
    assert "generated_at" in data
    assert "source_experiment" in data
    assert data["source_experiment"] == "test_exp"


def test_generate_provenance_has_correct_fields(tmp_path: Path) -> None:
    d = _make_artefact_dir(tmp_path)
    paths = generate_experiment_report(d, output_dir=tmp_path / "reports")
    data = json.loads(paths.provenance.read_text())
    assert set(data.keys()) == {"report_version", "artefact_version", "generated_at", "source_experiment", "config_hash"}


def test_generate_no_html_returns_none(tmp_path: Path) -> None:
    d = _make_artefact_dir(tmp_path)
    paths = generate_experiment_report(d, output_dir=tmp_path / "reports", include_html=False)
    assert paths.html is None


def test_generate_no_html_file_written(tmp_path: Path) -> None:
    d = _make_artefact_dir(tmp_path)
    out = tmp_path / "reports"
    generate_experiment_report(d, output_dir=out, include_html=False)
    assert not (out / "html" / "test_exp.html").exists()


def test_generate_copies_figures(tmp_path: Path) -> None:
    d = _make_artefact_dir(tmp_path, include_plots=True, n_plots=2)
    out = tmp_path / "reports"
    generate_experiment_report(d, output_dir=out)
    figure_dir = out / "figures" / "test_exp"
    assert figure_dir.is_dir()
    assert len(list(figure_dir.glob("*.png"))) == 2


def test_generate_no_figures_skips_figure_dir(tmp_path: Path) -> None:
    d = _make_artefact_dir(tmp_path, include_plots=False)
    out = tmp_path / "reports"
    generate_experiment_report(d, output_dir=out)
    assert not (out / "figures" / "test_exp").exists()


def test_generate_deterministic_markdown(tmp_path: Path) -> None:
    d = _make_artefact_dir(tmp_path)
    out1 = tmp_path / "r1"
    out2 = tmp_path / "r2"
    # Can't guarantee identical timestamps, so check structure only
    p1 = generate_experiment_report(d, output_dir=out1)
    p2 = generate_experiment_report(d, output_dir=out2)
    md1 = p1.markdown.read_text()
    md2 = p2.markdown.read_text()
    # Structure is identical (headings, metrics) even if timestamp differs
    assert "## Performance Metrics" in md1
    assert "## Performance Metrics" in md2
    assert md1.split("Generated:")[0] == md2.split("Generated:")[0]


def test_generate_markdown_contains_experiment_name(tmp_path: Path) -> None:
    d = _make_artefact_dir(tmp_path)
    paths = generate_experiment_report(d, output_dir=tmp_path / "reports")
    content = paths.markdown.read_text()
    assert "test_exp" in content


def test_generate_html_valid_structure(tmp_path: Path) -> None:
    d = _make_artefact_dir(tmp_path)
    paths = generate_experiment_report(d, output_dir=tmp_path / "reports")
    html = paths.html.read_text()
    assert "<!DOCTYPE html>" in html
    assert "<body>" in html
    assert "</body>" in html


def test_generate_missing_artefact_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        generate_experiment_report(tmp_path / "ghost", output_dir=tmp_path)
