"""Run an experiment from a YAML or JSON config file.

Usage:
    python scripts/run_from_config.py configs/experiments/momentum_rotation.yaml
    python scripts/run_from_config.py configs/experiments/equal_weight.yaml
    python scripts/run_from_config.py configs/experiments/momentum_rotation.yaml --report
    python scripts/run_from_config.py configs/experiments/momentum_rotation.yaml --report --preset canonical
    python scripts/run_from_config.py configs/experiments/momentum_rotation.yaml --report --output reports/custom
    python scripts/run_from_config.py configs/experiments/canonical_ml.yaml --report --render-profile frontend
"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.orchestrator import (
    format_run_summary,
    run_and_report,
    run_experiment_from_config,
)
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
    parser = argparse.ArgumentParser(
        description="Run an experiment from a YAML or JSON config file."
    )
    parser.add_argument("config", type=Path, help="Path to the config file")
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate a markdown/HTML report after the experiment completes",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports"),
        metavar="DIR",
        help="Report output directory (default: reports/)",
    )
    parser.add_argument(
        "--no-html",
        dest="html",
        action="store_false",
        help="Skip HTML report generation",
    )
    parser.add_argument(
        "--preset",
        choices=list(PRESET_MAP),
        default="standard",
        help="Report preset (default: standard). Only used with --report.",
    )
    parser.add_argument(
        "--render-profile",
        dest="render_profile",
        choices=["report", "frontend"],
        default="report",
        help="Rendering/export profile for canonical figures (default: report).",
    )
    args = parser.parse_args()

    if args.report:
        run, paths = run_and_report(
            args.config,
            report_output_dir=args.output,
            include_html=args.html,
            report_spec=PRESET_MAP[args.preset],
            profile=args.render_profile,
        )
        print(format_run_summary(run))
        print(f"\nReport     : {paths.markdown}")
        if paths.html:
            print(f"HTML       : {paths.html}")
        print(f"Provenance : {paths.provenance}")
    else:
        run = run_experiment_from_config(args.config, profile=args.render_profile)
        print(format_run_summary(run))


if __name__ == "__main__":
    main()
