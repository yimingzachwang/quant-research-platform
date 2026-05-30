"""Routing endpoint tests.

parse() and route() are mocked so no LLM calls or quant engine access occur.
"""

from __future__ import annotations

from unittest.mock import patch

from src.orchestration.intents.intent_schema import ReviewExperimentIntent, UnrecognisedIntent
from src.orchestration.router.routing_schema import WorkflowResult

_PARSE = "src.api.routers.routing.parse"
_ROUTE = "src.api.routers.routing.route"
_LIST  = "src.orchestration.api.research_api.list_all_experiments"


def _make_review_intent() -> ReviewExperimentIntent:
    return ReviewExperimentIntent(experiment_name="exp_a", provider="stub")


def _make_unrecognised_intent() -> UnrecognisedIntent:
    return UnrecognisedIntent(raw_text="gibberish", reason="no match")


def _make_workflow_result(intent, result=None, error="") -> WorkflowResult:
    return WorkflowResult(
        intent=intent,
        api_function="run_llm_review",
        result=result,
        error=error,
    )


class TestRouteEndpoint:
    def _post_route(self, client, text="review exp_a", intent=None, result=None, error=""):
        intent = intent or _make_review_intent()
        workflow = _make_workflow_result(intent, result=result, error=error)
        with (
            patch(_LIST, return_value=["exp_a"]),
            patch(_PARSE, return_value=intent),
            patch(_ROUTE, return_value=workflow),
        ):
            return client.post("/api/route", json={"text": text, "provider": "stub"})

    def test_returns_200(self, client):
        assert self._post_route(client).status_code == 200

    def test_response_contains_intent_type(self, client):
        data = self._post_route(client).json()
        assert data["intent_type"] == "ReviewExperimentIntent"

    def test_response_success_true_when_no_error(self, client):
        data = self._post_route(client).json()
        assert data["success"] is True

    def test_response_success_false_when_error(self, client):
        intent = _make_review_intent()
        data = self._post_route(client, intent=intent, error="LLM call failed").json()
        assert data["success"] is False

    def test_response_error_null_when_no_error(self, client):
        data = self._post_route(client).json()
        assert data["error"] is None

    def test_response_error_set_when_error(self, client):
        intent = _make_review_intent()
        data = self._post_route(client, intent=intent, error="LLM call failed").json()
        assert data["error"] == "LLM call failed"

    def test_result_is_json_serialisable(self, client):
        import json
        data = self._post_route(client)
        json.dumps(data.json())  # must not raise

    def test_result_null_when_none_result(self, client):
        data = self._post_route(client, result=None).json()
        assert data["result"] is None

    def test_unrecognised_intent_returns_200(self, client):
        intent = _make_unrecognised_intent()
        workflow = WorkflowResult(
            intent=intent,
            api_function="",
            result=None,
            error="Unrecognised intent",
        )
        with (
            patch(_LIST, return_value=[]),
            patch(_PARSE, return_value=intent),
            patch(_ROUTE, return_value=workflow),
        ):
            data = client.post("/api/route", json={"text": "gibberish"}).json()
        assert data["intent_type"] == "UnrecognisedIntent"
        assert data["success"] is False

    def test_does_not_mutate_session(self, client):
        intent = _make_review_intent()
        workflow = _make_workflow_result(intent)
        with (
            patch(_LIST, return_value=["exp_a"]),
            patch(_PARSE, return_value=intent),
            patch(_ROUTE, return_value=workflow),
            patch("src.orchestration.api.research_api.record_session_event") as mock_record,
        ):
            client.post("/api/route", json={"text": "review exp_a"})
        mock_record.assert_not_called()

    def test_calls_list_experiments_for_context(self, client):
        intent = _make_review_intent()
        workflow = _make_workflow_result(intent)
        with (
            patch(_LIST, return_value=["exp_a"]) as mock_list,
            patch(_PARSE, return_value=intent),
            patch(_ROUTE, return_value=workflow),
        ):
            client.post("/api/route", json={"text": "review exp_a"})
        mock_list.assert_called_once()

    def test_passes_text_to_parse(self, client):
        intent = _make_review_intent()
        workflow = _make_workflow_result(intent)
        with (
            patch(_LIST, return_value=["exp_a"]),
            patch(_PARSE, return_value=intent) as mock_parse,
            patch(_ROUTE, return_value=workflow),
        ):
            client.post("/api/route", json={"text": "review exp_a", "provider": "stub"})
        args, kwargs = mock_parse.call_args
        assert "review exp_a" in args or kwargs.get("text") == "review exp_a"
