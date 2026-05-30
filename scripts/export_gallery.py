"""Export experiment reports into the gallery directory.

Reads the experiment registry, generates a report for each registered
experiment, and copies the outputs into the gallery.

Usage:
    python scripts/export_gallery.py
    python scripts/export_gallery.py --registry results/experiments/registry.json
    python scripts/export_gallery.py --output exports/experiment_gallery/examples
    python scripts/export_gallery.py --no-html
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.registry import load_registry
from src.reporting.report_builder import generate_experiment_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Export experiment reports to gallery.")
    parser.add_argument(
        "--registry",
        default="results/experiments/registry.json",
        help="Path to registry.json (default: results/experiments/registry.json).",
    )
    parser.add_argument(
        "--output",
        default="exports/experiment_gallery/examples",
        help="Gallery output directory (default: exports/experiment_gallery/examples).",
    )
    parser.add_argument(
        "--no-html", action="store_true", help="Skip HTML generation."
    )
    args = parser.parse_args()

    registry_path = Path(args.registry)
    if not registry_path.exists():
        print(f"Registry not found: {registry_path}")
        sys.exit(1)

    experiments = load_registry(registry_path)
    if not experiments:
        print("No experiments found in registry.")
        return

    gallery_dir = Path(args.output)
    gallery_dir.mkdir(parents=True, exist_ok=True)

    # Use a temporary report dir then copy into gallery
    tmp_reports = PROJECT_ROOT / "_gallery_tmp"
    include_html = not args.no_html

    exported = 0
    skipped = 0
    index_entries: list[dict] = []

    for entry in experiments:
        exp_dir = entry.get("path") or entry.get("artefact_dir", "")
        exp_name = entry.get("experiment_name", "")
        if not exp_dir or not Path(exp_dir).is_dir():
            print(f"  SKIP  {exp_name or exp_dir!r} — artefact directory not found")
            skipped += 1
            continue

        try:
            paths = generate_experiment_report(
                artefact_path=exp_dir,
                output_dir=tmp_reports,
                include_html=include_html,
            )
        except Exception as exc:
            print(f"  ERROR {exp_name!r}: {exc}")
            skipped += 1
            continue

        # Copy report outputs into gallery/<exp_name>/
        dest_dir = gallery_dir / exp_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(paths.markdown, dest_dir / paths.markdown.name)
        shutil.copy2(paths.provenance, dest_dir / paths.provenance.name)
        if paths.html:
            shutil.copy2(paths.html, dest_dir / paths.html.name)
        if paths.manifest:
            shutil.copy2(paths.manifest, dest_dir / paths.manifest.name)

        # Copy figures directory if it was created
        tmp_figs = tmp_reports / "figures" / exp_name
        if tmp_figs.is_dir():
            dest_figs = dest_dir / "figures"
            if dest_figs.exists():
                shutil.rmtree(dest_figs)
            shutil.copytree(tmp_figs, dest_figs)

        # Collect gallery index entry from artefact metadata
        index_entry = _build_index_entry(
            exp_dir=Path(exp_dir),
            exp_name=exp_name,
            dest_dir=dest_dir,
            gallery_dir=gallery_dir,
            has_html=include_html,
        )
        index_entries.append(index_entry)

        print(f"  OK    {exp_name} → {dest_dir}")
        exported += 1

    # Write gallery_index.json for frontend discovery
    if index_entries:
        index_path = gallery_dir / "gallery_index.json"
        with index_path.open("w", encoding="utf-8") as f:
            json.dump(index_entries, f, indent=2)
        print(f"\nGallery index → {index_path}")

    # Clean up temp directory
    if tmp_reports.exists():
        shutil.rmtree(tmp_reports)

    print(f"Exported {exported} experiment(s), skipped {skipped}.")


def _build_index_entry(
    exp_dir: Path,
    exp_name: str,
    dest_dir: Path,
    gallery_dir: Path,
    has_html: bool,
) -> dict:
    """Build a single gallery index entry from an experiment artefact directory."""
    # Read lightweight artefact metadata
    metadata: dict = {}
    metrics: dict = {}
    md_path = exp_dir / "metadata.json"
    me_path = exp_dir / "metrics.json"
    if md_path.exists():
        with md_path.open(encoding="utf-8") as f:
            metadata = json.load(f)
    if me_path.exists():
        with me_path.open(encoding="utf-8") as f:
            metrics = json.load(f)

    config: dict = {}
    for cfg_name in ("normalized_config.json", "config.json"):
        cfg_path = exp_dir / cfg_name
        if cfg_path.exists():
            with cfg_path.open(encoding="utf-8") as f:
                config = json.load(f)
            break

    # Relative directory path from gallery root
    rel_dir = dest_dir.relative_to(gallery_dir).as_posix()

    files: dict[str, str] = {
        "markdown": f"{rel_dir}/{exp_name}.md",
        "provenance": f"{rel_dir}/{exp_name}_provenance.json",
        "manifest": f"{rel_dir}/{exp_name}_manifest.json",
    }
    if has_html:
        files["html"] = f"{rel_dir}/{exp_name}.html"

    figs_dir = dest_dir / "figures"
    figures = (
        [f"{rel_dir}/figures/{p.name}" for p in sorted(figs_dir.glob("*.png"))]
        if figs_dir.is_dir() else []
    )

    _SUMMARY_KEYS = ("sharpe_ratio", "annualized_return", "max_drawdown",
                     "calmar_ratio", "hit_rate", "annualized_volatility")
    metrics_summary = {k: round(float(v), 6) for k, v in metrics.items()
                       if k in _SUMMARY_KEYS and v is not None}

    return {
        "experiment_name": exp_name,
        "created_at": metadata.get("created_at", ""),
        "strategy_name": metadata.get("strategy_name", ""),
        "tags": (config.get("tags") or []),
        "has_ml": (exp_dir / "ml_provenance.json").exists(),
        "has_validation": (exp_dir / "diagnostics" / "split_metrics.json").exists(),
        "has_diagnostics": (exp_dir / "diagnostics").is_dir(),
        "metrics_summary": metrics_summary,
        "directory": rel_dir,
        "files": files,
        "figures": figures,
    }


if __name__ == "__main__":
    main()
