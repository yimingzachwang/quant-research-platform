"""Unit tests for the governed execution bridge in research_api.

These tests mock the quant engine entry points — no experiment is ever run.
They verify:
  - the bridge reuses the existing engine runner (run_and_report /
    run_experiment_from_config), not new engine logic
  - dry-run plans only and never executes
  - a missing config never reaches the engine
  - engine errors are captured as a result (never raised)
  - exactly one config is run per call (no loop, no retry)
  - post-run review is skipped on failed / dry-run execution
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.orchestration.api import research_api as api
from src.orchestration.api.research_api import ExecutionResult, ExecutionReviewResult
from src.orchestration.api.schemas import LLMReviewOutput

_RUNNER = "src.experiments.orchestrator.run_and_report"
_PLAIN_RUNNER = "src.experiments.orchestrator.run_experiment_from_config"


# ---------------------------------------------------------------------------
# execute_approved_config
# ---------------------------------------------------------------------------


def test_dry_run_does_not_call_engine():
    with patch(_RUNNER) as m:
        result = api.execute_approved_config("any/path.yaml", dry_run=True)
    m.assert_not_called()
    assert isinstance(result, ExecutionResult)
    assert result.success is True
    assert result.experiment_name is None
    assert result.artefact_root is None
    assert "any/path.yaml" in result.command_hint


def test_missing_config_never_reaches_engine():
    with patch(_RUNNER) as m:
        result = api.execute_approved_config("does/not/exist.yaml", dry_run=False)
    m.assert_not_called()
    assert result.success is False
    assert "not found" in (result.error or "").lower()


def test_calls_existing_runner_and_returns_result(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("version: '2'\n")
    fake_run = MagicMock()
    fake_run.output_path = tmp_path / "results" / "experiments" / "foo"
    fake_paths = MagicMock()
    fake_paths.markdown = tmp_path / "reports" / "foo.md"
    with patch(_RUNNER, return_value=(fake_run, fake_paths)) as m:
        result = api.execute_approved_config(cfg, report=True, preset="canonical")
    # Exactly one engine invocation — no loop, no retry.
    m.assert_called_once()
    assert result.success is True
    assert result.experiment_name == "foo"
    assert result.artefact_root.endswith("foo")
    assert result.report_path.endswith("foo.md")


def test_no_report_uses_plain_runner(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("version: '2'\n")
    fake_run = MagicMock()
    fake_run.output_path = tmp_path / "results" / "experiments" / "bar"
    with (
        patch(_PLAIN_RUNNER, return_value=fake_run) as m_plain,
        patch(_RUNNER) as m_report,
    ):
        result = api.execute_approved_config(cfg, report=False)
    m_plain.assert_called_once()
    m_report.assert_not_called()
    assert result.experiment_name == "bar"
    assert result.report_path is None


def test_engine_error_is_captured_not_raised(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("version: '2'\n")
    with patch(_RUNNER, side_effect=RuntimeError("boom")):
        result = api.execute_approved_config(cfg)
    assert result.success is False
    assert "boom" in (result.error or "")


# ---------------------------------------------------------------------------
# execute_and_review_approved_config
# ---------------------------------------------------------------------------


def test_review_skipped_on_failed_execution():
    failing = ExecutionResult(
        config_path="c.yaml",
        experiment_name=None,
        success=False,
        artefact_root=None,
        report_path=None,
        error="nope",
    )
    with (
        patch.object(api, "execute_approved_config", return_value=failing),
        patch.object(api, "build_llm_context") as m_ctx,
        patch.object(api, "run_llm_review") as m_review,
    ):
        out = api.execute_and_review_approved_config("c.yaml")
    assert isinstance(out, ExecutionReviewResult)
    assert out.review is None
    assert out.context_hash is None
    m_ctx.assert_not_called()
    m_review.assert_not_called()


def test_review_runs_on_successful_execution():
    ok = ExecutionResult(
        config_path="c.yaml",
        experiment_name="foo",
        success=True,
        artefact_root="results/experiments/foo",
        report_path=None,
        error=None,
    )
    fake_review = LLMReviewOutput(
        experiment_name="foo",
        provider="stub",
        model="stub",
        prompt_template="experiment_review",
        review_text="ok",
    )
    with (
        patch.object(api, "execute_approved_config", return_value=ok),
        patch.object(api, "build_llm_context", return_value=MagicMock()),
        patch.object(api, "compute_context_hash", return_value="hash123"),
        patch.object(api, "run_llm_review", return_value=fake_review) as m_review,
    ):
        out = api.execute_and_review_approved_config("c.yaml", provider="stub")
    assert out.review is fake_review
    assert out.context_hash == "hash123"
    m_review.assert_called_once()


def test_dry_run_skips_review_entirely():
    with (
        patch.object(api, "run_llm_review") as m_review,
        patch(_RUNNER) as m_engine,
    ):
        out = api.execute_and_review_approved_config("c.yaml", dry_run=True)
    assert out.review is None
    m_review.assert_not_called()
    m_engine.assert_not_called()
