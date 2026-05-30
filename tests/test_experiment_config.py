from src.experiments import ExperimentRunner, load_experiment_config


def test_load_experiment_config_builds_context_from_yaml() -> None:
    config = load_experiment_config("configs/experiments/example_daily_momentum.yaml")

    assert config.context.experiment_id == "exp_scaffold_20260520_example"
    assert config.context.name == "example_daily_momentum"
    assert config.context.universe.name == "core_etfs"
    assert "SPY" in config.context.universe.symbols
    assert config.context.horizon.periods == 1


def test_experiment_runner_runs_configured_placeholder() -> None:
    result = ExperimentRunner().run_config("configs/experiments/example_daily_momentum.yaml")

    assert result.context.experiment_id == "exp_scaffold_20260520_example"
    assert result.metrics == {}
    assert result.artifacts["tracking_run_id"] == "exp_scaffold_20260520_example"
