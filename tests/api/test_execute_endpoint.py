"""Governed execution endpoint tests.

POST /api/sessions/{id}/execute-approved-config

All Research API calls are mocked — no disk I/O, no engine, no LLM.
Verifies the governance contract:
  - dry run plans only, records nothing, executes nothing
  - real execution records REQUESTED -> COMPLETED -> POST_RUN_REVIEW_GENERATED
  - failed execution records REQUESTED only and skips post-run review
"""

from __future__ import annotations

from unittest.mock import patch

from src.orchestration.api.research_api import ExecutionResult, ExecutionReviewResult
from src.orchestration.session.session_schema import SessionEventType
from tests.api.conftest import make_stub_review, make_stub_session

_MODULE = "src.orchestration.api.research_api"
_PATH = "/api/sessions/test-session-id/execute-approved-config"


def _exec_result(success: bool = True, experiment_name: str | None = "exp_a_v2") -> ExecutionResult:
    return ExecutionResult(
        config_path="configs/experiments/exp_a_v2.yaml",
        experiment_name=experiment_name if success else None,
        success=success,
        artefact_root="results/experiments/exp_a_v2" if success else None,
        report_path="reports/markdown/exp_a_v2.md" if success else None,
        error=None if success else "Config not found: x",
        command_hint="python scripts/run_from_config.py configs/experiments/exp_a_v2.yaml --report --preset canonical",
    )


def test_missing_session_returns_404(client):
    with patch(f"{_MODULE}.load_research_session", return_value=None):
        resp = client.post(_PATH, json={"config_path": "configs/experiments/exp_a_v2.yaml"})
    assert resp.status_code == 404


def test_dry_run_plans_only_records_nothing(client):
    planned = ExecutionReviewResult(execution=_exec_result(), review=None, context_hash=None)
    with (
        patch(f"{_MODULE}.load_research_session", return_value=make_stub_session()),
        patch(f"{_MODULE}.summarize_research_session", return_value={"event_count": 0}),
        patch(f"{_MODULE}.execute_and_review_approved_config", return_value=planned) as m_exec,
        patch(f"{_MODULE}.record_session_event") as m_event,
    ):
        resp = client.post(
            _PATH, json={"config_path": "configs/experiments/exp_a_v2.yaml", "dry_run": True}
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["planned"] is True
    assert body["review"] is None
    # Dry run must record NO session events.
    m_event.assert_not_called()
    # The bridge is invoked with dry_run=True.
    assert m_exec.call_args.kwargs["dry_run"] is True


def test_real_execution_records_full_lifecycle(client):
    result = ExecutionReviewResult(
        execution=_exec_result(success=True),
        review=make_stub_review(experiment_name="exp_a_v2"),
        context_hash="abc" * 21 + "x",
    )
    with (
        patch(f"{_MODULE}.load_research_session", return_value=make_stub_session()),
        patch(f"{_MODULE}.summarize_research_session", return_value={"event_count": 3}),
        patch(f"{_MODULE}.execute_and_review_approved_config", return_value=result),
        patch(f"{_MODULE}.record_session_event", side_effect=lambda session, **kw: session) as m_event,
    ):
        resp = client.post(
            _PATH, json={"config_path": "configs/experiments/exp_a_v2.yaml", "provider": "stub"}
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["execution"]["success"] is True
    assert body["review"] is not None
    recorded = [c.kwargs["event_type"] for c in m_event.call_args_list]
    assert recorded == [
        SessionEventType.EXECUTION_REQUESTED,
        SessionEventType.EXECUTION_COMPLETED,
        SessionEventType.POST_RUN_REVIEW_GENERATED,
    ]


def test_failed_execution_skips_post_run_review(client):
    result = ExecutionReviewResult(
        execution=_exec_result(success=False),
        review=None,
        context_hash=None,
    )
    with (
        patch(f"{_MODULE}.load_research_session", return_value=make_stub_session()),
        patch(f"{_MODULE}.summarize_research_session", return_value={"event_count": 1}),
        patch(f"{_MODULE}.execute_and_review_approved_config", return_value=result),
        patch(f"{_MODULE}.record_session_event", side_effect=lambda session, **kw: session) as m_event,
    ):
        resp = client.post(
            _PATH, json={"config_path": "configs/experiments/missing.yaml"}
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["execution"]["success"] is False
    assert body["review"] is None
    assert "error" in body
    # Only the authorisation event is recorded — no COMPLETED, no POST_RUN review.
    recorded = [c.kwargs["event_type"] for c in m_event.call_args_list]
    assert recorded == [SessionEventType.EXECUTION_REQUESTED]
