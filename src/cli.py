"""Console entry points for platform workflows."""

from __future__ import annotations

import argparse
import json

from src.experiments import ExperimentRunner, load_experiment_config
from src.reporting import MarkdownReportGenerator

DEFAULT_EXPERIMENT_CONFIG = "configs/experiments/example_daily_momentum.yaml"
DEFAULT_DATA_CONFIG = "configs/data/daily_prices.yaml"


def _add_config_argument(parser: argparse.ArgumentParser, default: str) -> None:
    parser.add_argument("config", nargs="?", default=None, help="Path to a YAML config.")
    parser.add_argument("--config", dest="config_option", help="Path to a YAML config.")


def _resolve_config_arg(args: argparse.Namespace, default: str) -> str:
    return args.config_option or args.config or default


def run_experiment() -> None:
    """Run an experiment from configuration."""
    parser = argparse.ArgumentParser(description="Run a configured research experiment.")
    _add_config_argument(parser, DEFAULT_EXPERIMENT_CONFIG)
    args = parser.parse_args()
    config_path = _resolve_config_arg(args, DEFAULT_EXPERIMENT_CONFIG)

    result = ExperimentRunner().run_config(config_path)
    payload = {
        "experiment_id": result.context.experiment_id,
        "metrics": result.metrics,
        "artifacts": result.artifacts,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def generate_report() -> None:
    """Generate a markdown report from a scaffold experiment run."""
    parser = argparse.ArgumentParser(description="Generate a placeholder experiment report.")
    _add_config_argument(parser, DEFAULT_EXPERIMENT_CONFIG)
    args = parser.parse_args()
    config_path = _resolve_config_arg(args, DEFAULT_EXPERIMENT_CONFIG)

    experiment_config = load_experiment_config(config_path)
    result = ExperimentRunner().run(experiment_config.context)
    report = MarkdownReportGenerator().generate(experiment_config.context, result)
    print(report.content)


def validate_data() -> None:
    """Print a placeholder validation result for a dataset config."""
    parser = argparse.ArgumentParser(description="Validate a configured dataset.")
    _add_config_argument(parser, DEFAULT_DATA_CONFIG)
    args = parser.parse_args()
    config_path = _resolve_config_arg(args, DEFAULT_DATA_CONFIG)
    print(json.dumps({"config": config_path, "status": "placeholder"}, indent=2))
