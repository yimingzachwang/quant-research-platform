"""Generate a static markdown and HTML report from a saved experiment directory.

Usage:
    python scripts/generate_report.py results/experiments/my_experiment
    python scripts/generate_report.py results/experiments/my_experiment --no-html
    python scripts/generate_report.py results/experiments/my_experiment --output reports
    python scripts/generate_report.py results/experiments/my_experiment --preset canonical
"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reporting.report_builder import generate_experiment_report
from src.reporting.report_spec import (
    AUDIT_REPORT,
    CANONICAL_SHOWCASE,
    COMPACT_REPORT,
    DIAGNOSTICS_REPORT,
    STANDARD_REPORT,
)

PRESET_MAP = {
    "standard": STANDARD_REPORT,
    "canonical": CANONICAL_SHOWCASE,
    "compact": COMPACT_REPORT,
    "diagnostics": DIAGNOSTICS_REPORT,
    "audit": AUDIT_REPORT,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate experiment report.")
    parser.add_argument("experiment_dir", help="Path to saved experiment directory.")
    parser.add_argument(
        "--output", default="reports", help="Root output directory (default: reports)."
    )
    parser.add_argument(
        "--no-html", action="store_true", help="Skip HTML generation."
    )
    parser.add_argument(
        "--preset",
        choices=list(PRESET_MAP),
        default="standard",
        help="Report preset (default: standard).",
    )
    args = parser.parse_args()

    paths = generate_experiment_report(
        artefact_path=args.experiment_dir,
        output_dir=args.output,
        include_html=not args.no_html,
        report_spec=PRESET_MAP[args.preset],
    )

    print(f"Markdown  : {paths.markdown}")
    if paths.html:
        print(f"HTML      : {paths.html}")
    print(f"Provenance: {paths.provenance}")


if __name__ == "__main__":
    main()
