"""Filesystem path resolution for experiment artefacts.

All path logic for the experiments output tree lives here.  No computation,
no data loading — pure path construction so every other module stays DRY.

Layout assumed::

    results/experiments/<experiment_name>/
        metadata.json
        metrics.json
        config.json
        raw_config.yaml
        normalized_config.json
        ml_provenance.json
        equity_curve.parquet
        returns.parquet
        weights.parquet
        diagnostics/
            backtest_diagnostics.json
            ml_diagnostics.json
            ml_model_diagnostics.json
            split_metrics.json
            universe_coverage.json
            wf_equity_curves.json
            alignment_diagnostics.json
            data_summary.json
            feature_correlations.json
            feature_families.json
            feature_registry.json
            feature_summary.json
            signal_transitions.json
        research/
            (same files as diagnostics — separate subdirectory)
        plots/
            *.png
            plot_index.json

    reports/
        markdown/<experiment_name>.md
        markdown/<experiment_name>_manifest.json
        markdown/<experiment_name>_provenance.json
        html/<experiment_name>.html
        figures/<experiment_name>/*.png
"""

from __future__ import annotations

from pathlib import Path

_RESULTS_ROOT = Path("results/experiments")
_REPORTS_ROOT = Path("reports")

# ---------------------------------------------------------------------------
# Experiment root
# ---------------------------------------------------------------------------


def experiment_root(name: str, base: Path | str | None = None) -> Path:
    root = Path(base) if base else _RESULTS_ROOT
    return root / name


def experiments_root(base: Path | str | None = None) -> Path:
    return Path(base) if base else _RESULTS_ROOT


# ---------------------------------------------------------------------------
# Core artefact paths
# ---------------------------------------------------------------------------


def metadata_path(name: str, base: Path | str | None = None) -> Path:
    return experiment_root(name, base) / "metadata.json"


def metrics_path(name: str, base: Path | str | None = None) -> Path:
    return experiment_root(name, base) / "metrics.json"


def config_path(name: str, base: Path | str | None = None) -> Path:
    return experiment_root(name, base) / "config.json"


def equity_curve_path(name: str, base: Path | str | None = None) -> Path:
    return experiment_root(name, base) / "equity_curve.parquet"


def returns_path(name: str, base: Path | str | None = None) -> Path:
    return experiment_root(name, base) / "returns.parquet"


def weights_path(name: str, base: Path | str | None = None) -> Path:
    return experiment_root(name, base) / "weights.parquet"


def ml_provenance_path(name: str, base: Path | str | None = None) -> Path:
    return experiment_root(name, base) / "ml_provenance.json"


# ---------------------------------------------------------------------------
# Diagnostics artefacts
# ---------------------------------------------------------------------------

_DIAGNOSTICS_FILES = {
    "backtest_diagnostics": "backtest_diagnostics.json",
    "ml_diagnostics": "ml_diagnostics.json",
    "ml_model_diagnostics": "ml_model_diagnostics.json",
    "split_metrics": "split_metrics.json",
    "universe_coverage": "universe_coverage.json",
    "wf_equity_curves": "wf_equity_curves.json",
}

_RESEARCH_FILES = {
    "alignment_diagnostics": "alignment_diagnostics.json",
    "data_summary": "data_summary.json",
    "feature_correlations": "feature_correlations.json",
    "feature_families": "feature_families.json",
    "feature_registry": "feature_registry.json",
    "feature_summary": "feature_summary.json",
    "signal_transitions": "signal_transitions.json",
}


def diagnostics_dir(name: str, base: Path | str | None = None) -> Path:
    return experiment_root(name, base) / "diagnostics"


def research_dir(name: str, base: Path | str | None = None) -> Path:
    return experiment_root(name, base) / "research"


def diagnostics_path(name: str, key: str, base: Path | str | None = None) -> Path | None:
    filename = _DIAGNOSTICS_FILES.get(key) or _RESEARCH_FILES.get(key)
    if filename is None:
        return None
    if key in _DIAGNOSTICS_FILES:
        return diagnostics_dir(name, base) / filename
    return research_dir(name, base) / filename


def all_diagnostics_paths(name: str, base: Path | str | None = None) -> dict[str, Path]:
    diag_dir = diagnostics_dir(name, base)
    res_dir = research_dir(name, base)
    result: dict[str, Path] = {}
    for key, fname in _DIAGNOSTICS_FILES.items():
        result[key] = diag_dir / fname
    for key, fname in _RESEARCH_FILES.items():
        result[key] = res_dir / fname
    return result


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plots_dir(name: str, base: Path | str | None = None) -> Path:
    return experiment_root(name, base) / "plots"


def plot_index_path(name: str, base: Path | str | None = None) -> Path:
    return plots_dir(name, base) / "plot_index.json"


def plot_path(name: str, plot_stem: str, base: Path | str | None = None) -> Path:
    return plots_dir(name, base) / f"{plot_stem}.png"


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

_MARKDOWN_DIR = _REPORTS_ROOT / "markdown"
_HTML_DIR = _REPORTS_ROOT / "html"
_FIGURES_DIR = _REPORTS_ROOT / "figures"


def reports_markdown_dir(reports_base: Path | str | None = None) -> Path:
    base = Path(reports_base) / "markdown" if reports_base else _MARKDOWN_DIR
    return base


def report_markdown_path(name: str, reports_base: Path | str | None = None) -> Path:
    return reports_markdown_dir(reports_base) / f"{name}.md"


def report_manifest_path(name: str, reports_base: Path | str | None = None) -> Path:
    return reports_markdown_dir(reports_base) / f"{name}_manifest.json"


def report_html_path(name: str, reports_base: Path | str | None = None) -> Path:
    base = Path(reports_base) / "html" if reports_base else _HTML_DIR
    return base / f"{name}.html"


def report_figures_dir(name: str, reports_base: Path | str | None = None) -> Path:
    base = Path(reports_base) / "figures" if reports_base else _FIGURES_DIR
    return base / name


def report_figure_path(
    experiment_name: str,
    figure_stem: str,
    reports_base: Path | str | None = None,
) -> Path:
    return report_figures_dir(experiment_name, reports_base) / f"{figure_stem}.png"


# ---------------------------------------------------------------------------
# Experiment config paths (configs/experiments/)
# ---------------------------------------------------------------------------

_CONFIGS_ROOT = Path("configs") / "experiments"


def experiment_config_path(name: str, configs_base: Path | str | None = None) -> Path:
    """Return the YAML config path for a named experiment.

    configs_base overrides the default configs/experiments/ root.
    Used by all three Phase 3 config_generation modules so the root is
    defined in exactly one place.
    """
    root = Path(configs_base) if configs_base else _CONFIGS_ROOT
    return root / f"{name}.yaml"


# ---------------------------------------------------------------------------
# LLM review output paths
# ---------------------------------------------------------------------------

_LLM_REVIEWS_DIR = Path("results/llm_reviews")


def llm_reviews_root(base: Path | str | None = None) -> Path:
    """Root directory holding per-experiment LLM review subdirectories."""
    return Path(base) if base else _LLM_REVIEWS_DIR


def llm_review_dir(name: str, base: Path | str | None = None) -> Path:
    return llm_reviews_root(base) / name


def llm_context_path(name: str, base: Path | str | None = None) -> Path:
    return llm_review_dir(name, base) / "llm_context.json"


def llm_review_path(name: str, base: Path | str | None = None) -> Path:
    return llm_review_dir(name, base) / "llm_review.json"


def iteration_proposal_json_path(name: str, base: Path | str | None = None) -> Path:
    return llm_review_dir(name, base) / "iteration_proposal.json"


def draft_json_path(name: str, draft_id: str, base: Path | str | None = None) -> Path:
    return llm_review_dir(name, base) / f"draft_{draft_id}.json"


def iteration_proposal_md_path(name: str, base: Path | str | None = None) -> Path:
    return llm_review_dir(name, base) / "iteration_proposal.md"


# ---------------------------------------------------------------------------
# Comparative review output paths
# ---------------------------------------------------------------------------

_COMPARISONS_DIR = Path("results/comparisons")


def comparisons_root(base: Path | str | None = None) -> Path:
    return Path(base) if base else _COMPARISONS_DIR


def comparison_dir(
    baseline: str,
    candidate: str,
    base: Path | str | None = None,
) -> Path:
    return comparisons_root(base) / f"{baseline}__vs__{candidate}"


def comparative_review_json_path(
    baseline: str,
    candidate: str,
    base: Path | str | None = None,
) -> Path:
    return comparison_dir(baseline, candidate, base) / "comparative_review.json"


def comparative_review_md_path(
    baseline: str,
    candidate: str,
    base: Path | str | None = None,
) -> Path:
    return comparison_dir(baseline, candidate, base) / "comparative_review.md"


def comparison_evidence_json_path(
    baseline: str,
    candidate: str,
    base: Path | str | None = None,
) -> Path:
    return comparison_dir(baseline, candidate, base) / "comparison_evidence.json"


# ---------------------------------------------------------------------------
# Lineage metadata paths (within experiment directory)
# ---------------------------------------------------------------------------


def lineage_path(name: str, base: Path | str | None = None) -> Path:
    return experiment_root(name, base) / "lineage.json"


# ---------------------------------------------------------------------------
# Evolution chain output paths
# ---------------------------------------------------------------------------

_EVOLUTION_DIR = Path("results/evolution")


def evolution_root(base: Path | str | None = None) -> Path:
    return Path(base) if base else _EVOLUTION_DIR


def evolution_dir(root_experiment: str, base: Path | str | None = None) -> Path:
    return evolution_root(base) / root_experiment


def evolution_chain_json_path(root_experiment: str, base: Path | str | None = None) -> Path:
    return evolution_dir(root_experiment, base) / "evolution_chain.json"


def evolution_chain_md_path(root_experiment: str, base: Path | str | None = None) -> Path:
    return evolution_dir(root_experiment, base) / "evolution_chain.md"


# ---------------------------------------------------------------------------
# Research session paths
# ---------------------------------------------------------------------------

_SESSIONS_DIR = Path("results/research_sessions")


def sessions_root(base: Path | str | None = None) -> Path:
    return Path(base) if base else _SESSIONS_DIR


def session_dir(session_id: str, base: Path | str | None = None) -> Path:
    return sessions_root(base) / session_id


def session_json_path(session_id: str, base: Path | str | None = None) -> Path:
    return session_dir(session_id, base) / "session.json"


def list_session_ids(base: Path | str | None = None) -> list[str]:
    """Return session IDs for all sessions present on disk."""
    root = sessions_root(base)
    if not root.exists():
        return []
    return sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and (p / "session.json").exists()
    )


# ---------------------------------------------------------------------------
# Research memory paths (Phase 1 metadata/keyword RAG)
# ---------------------------------------------------------------------------

_RESEARCH_MEMORY_DIR = Path("results/research_memory")


def research_memory_root(base: Path | str | None = None) -> Path:
    return Path(base) if base else _RESEARCH_MEMORY_DIR


def memory_index_path(base: Path | str | None = None) -> Path:
    """Path to the local JSONL research-memory index."""
    return research_memory_root(base) / "memory_index.jsonl"


def semantic_memory_index_path(base: Path | str | None = None) -> Path:
    """Path to the local JSONL semantic (embedding) research-memory index."""
    return research_memory_root(base) / "semantic_memory_index.jsonl"


def semantic_memory_manifest_path(base: Path | str | None = None) -> Path:
    """Path to the semantic-memory manifest (model, dim, counts, timestamp)."""
    return research_memory_root(base) / "semantic_memory_manifest.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def list_experiments(base: Path | str | None = None) -> list[str]:
    """Return names of all experiment directories present on disk."""
    root = experiments_root(base)
    if not root.exists():
        return []
    return sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and (p / "metadata.json").exists()
    )
