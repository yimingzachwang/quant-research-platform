"""Draft lifecycle endpoint tests.

All backend functions are mocked.
"""

from __future__ import annotations

from unittest.mock import patch

from tests.api.conftest import (
    make_stub_draft,
    make_stub_session,
    make_stub_validation,
)

_MODULE = "src.orchestration.api.research_api"


class TestGenerateDraftEndpoint:
    def _post_draft(self, client, stub_session, stub_draft):
        with (
            patch(f"{_MODULE}.load_research_session", return_value=stub_session),
            patch(f"{_MODULE}.generate_experiment_draft", return_value=stub_draft),
            patch(f"{_MODULE}.record_session_event", return_value=stub_session),
        ):
            return client.post(
                "/api/sessions/test-session-id/draft",
                json={"experiment_name": "exp_a", "provider": "stub"},
            )

    def test_returns_200(self, client):
        assert self._post_draft(
            client, make_stub_session(), make_stub_draft()
        ).status_code == 200

    def test_response_contains_draft(self, client):
        stub_draft = make_stub_draft()
        data = self._post_draft(client, make_stub_session(), stub_draft).json()
        assert "draft" in data
        assert data["draft"]["draft_id"] == "draft-001"
        assert data["draft"]["proposed_name"] == "exp_a_v2"

    def test_response_contains_session_and_summary(self, client):
        data = self._post_draft(client, make_stub_session(), make_stub_draft()).json()
        assert "session" in data
        assert "summary" in data

    def test_records_draft_generated_event(self, client):
        stub = make_stub_session()
        draft = make_stub_draft()
        with (
            patch(f"{_MODULE}.load_research_session", return_value=stub),
            patch(f"{_MODULE}.generate_experiment_draft", return_value=draft),
            patch(f"{_MODULE}.record_session_event", return_value=stub) as mock_record,
        ):
            client.post(
                "/api/sessions/test-session-id/draft",
                json={"experiment_name": "exp_a", "provider": "stub"},
            )
        mock_record.assert_called_once()
        assert mock_record.call_args.kwargs["event_type"] == "DRAFT_GENERATED"

    def test_missing_session_returns_404(self, client):
        with patch(f"{_MODULE}.load_research_session", return_value=None):
            response = client.post(
                "/api/sessions/ghost/draft",
                json={"experiment_name": "exp_a"},
            )
        assert response.status_code == 404


class TestValidateDraftEndpoint:
    def _post_validate(self, client, stub_session, stub_draft, validation_result):
        with (
            patch(f"{_MODULE}.load_research_session", return_value=stub_session),
            patch(f"{_MODULE}.load_experiment_draft", return_value=stub_draft),
            patch(f"{_MODULE}.validate_experiment_draft", return_value=validation_result),
            patch(f"{_MODULE}.record_session_event", return_value=stub_session),
        ):
            return client.post(
                "/api/sessions/test-session-id/draft/validate",
                json={"experiment_name": "exp_a", "draft_id": "draft-001"},
            )

    def test_returns_200(self, client):
        assert self._post_validate(
            client, make_stub_session(), make_stub_draft(), make_stub_validation()
        ).status_code == 200

    def test_response_contains_validation(self, client):
        data = self._post_validate(
            client, make_stub_session(), make_stub_draft(), make_stub_validation()
        ).json()
        assert "validation" in data
        assert data["validation"]["is_valid"] is True
        assert data["validation"]["errors"] == []

    def test_invalid_draft_shows_errors(self, client):
        data = self._post_validate(
            client, make_stub_session(), make_stub_draft(), make_stub_validation(False)
        ).json()
        assert data["validation"]["is_valid"] is False
        assert len(data["validation"]["errors"]) > 0

    def test_records_draft_validated_event(self, client):
        stub = make_stub_session()
        draft = make_stub_draft()
        result = make_stub_validation()
        with (
            patch(f"{_MODULE}.load_research_session", return_value=stub),
            patch(f"{_MODULE}.load_experiment_draft", return_value=draft),
            patch(f"{_MODULE}.validate_experiment_draft", return_value=result),
            patch(f"{_MODULE}.record_session_event", return_value=stub) as mock_record,
        ):
            client.post(
                "/api/sessions/test-session-id/draft/validate",
                json={"experiment_name": "exp_a", "draft_id": "draft-001"},
            )
        mock_record.assert_called_once()
        assert mock_record.call_args.kwargs["event_type"] == "DRAFT_VALIDATED"

    def test_missing_draft_returns_404(self, client):
        with (
            patch(f"{_MODULE}.load_research_session", return_value=make_stub_session()),
            patch(f"{_MODULE}.load_experiment_draft", return_value=None),
        ):
            response = client.post(
                "/api/sessions/test-session-id/draft/validate",
                json={"experiment_name": "exp_a", "draft_id": "nonexistent"},
            )
        assert response.status_code == 404


class TestApproveDraftEndpoint:
    def _post_approve(self, client, stub_session, stub_draft, approved_draft):
        with (
            patch(f"{_MODULE}.load_research_session", return_value=stub_session),
            patch(f"{_MODULE}.load_experiment_draft", return_value=stub_draft),
            patch(f"{_MODULE}.approve_experiment_draft", return_value=approved_draft),
            patch(f"{_MODULE}.record_session_event", return_value=stub_session),
        ):
            return client.post(
                "/api/sessions/test-session-id/draft/approve",
                json={"experiment_name": "exp_a", "draft_id": "draft-001"},
            )

    def test_returns_200(self, client):
        assert self._post_approve(
            client, make_stub_session(), make_stub_draft(), make_stub_draft(approved=True)
        ).status_code == 200

    def test_response_draft_is_approved(self, client):
        data = self._post_approve(
            client, make_stub_session(), make_stub_draft(), make_stub_draft(approved=True)
        ).json()
        assert data["draft"]["approved"] is True

    def test_records_draft_approved_event(self, client):
        stub = make_stub_session()
        draft = make_stub_draft()
        approved = make_stub_draft(approved=True)
        with (
            patch(f"{_MODULE}.load_research_session", return_value=stub),
            patch(f"{_MODULE}.load_experiment_draft", return_value=draft),
            patch(f"{_MODULE}.approve_experiment_draft", return_value=approved),
            patch(f"{_MODULE}.record_session_event", return_value=stub) as mock_record,
        ):
            client.post(
                "/api/sessions/test-session-id/draft/approve",
                json={"experiment_name": "exp_a", "draft_id": "draft-001"},
            )
        mock_record.assert_called_once()
        assert mock_record.call_args.kwargs["event_type"] == "DRAFT_APPROVED"

    def test_missing_draft_returns_404(self, client):
        with (
            patch(f"{_MODULE}.load_research_session", return_value=make_stub_session()),
            patch(f"{_MODULE}.load_experiment_draft", return_value=None),
        ):
            response = client.post(
                "/api/sessions/test-session-id/draft/approve",
                json={"experiment_name": "exp_a", "draft_id": "nonexistent"},
            )
        assert response.status_code == 404


class TestRenderDraftEndpoint:
    def test_unapproved_draft_returns_400(self, client):
        unapproved = make_stub_draft(approved=False)
        with (
            patch(f"{_MODULE}.load_research_session", return_value=make_stub_session()),
            patch(f"{_MODULE}.load_experiment_draft", return_value=unapproved),
        ):
            response = client.post(
                "/api/sessions/test-session-id/draft/render",
                json={"experiment_name": "exp_a", "draft_id": "draft-001"},
            )
        assert response.status_code == 400
        assert "approved" in response.json()["detail"].lower()

    def test_approved_draft_returns_200(self, client):
        approved = make_stub_draft(approved=True)
        stub = make_stub_session()
        with (
            patch(f"{_MODULE}.load_research_session", return_value=stub),
            patch(f"{_MODULE}.load_experiment_draft", return_value=approved),
            patch(f"{_MODULE}.render_draft_to_yaml", return_value="version: 2\nname: exp_a_v2\n"),
            patch(f"{_MODULE}.record_session_event", return_value=stub),
        ):
            response = client.post(
                "/api/sessions/test-session-id/draft/render",
                json={"experiment_name": "exp_a", "draft_id": "draft-001"},
            )
        assert response.status_code == 200

    def test_render_response_contains_yaml_and_config_path(self, client):
        approved = make_stub_draft(approved=True)
        stub = make_stub_session()
        with (
            patch(f"{_MODULE}.load_research_session", return_value=stub),
            patch(f"{_MODULE}.load_experiment_draft", return_value=approved),
            patch(
                f"{_MODULE}.render_draft_to_yaml",
                return_value="version: 2\nname: exp_a_v2\n",
            ),
            patch(f"{_MODULE}.record_session_event", return_value=stub),
        ):
            data = client.post(
                "/api/sessions/test-session-id/draft/render",
                json={"experiment_name": "exp_a", "draft_id": "draft-001"},
            ).json()
        assert "yaml" in data
        assert "exp_a_v2" in data["yaml"]
        assert "config_path" in data
        assert "exp_a_v2" in data["config_path"]

    def test_render_records_yaml_rendered_event(self, client):
        approved = make_stub_draft(approved=True)
        stub = make_stub_session()
        with (
            patch(f"{_MODULE}.load_research_session", return_value=stub),
            patch(f"{_MODULE}.load_experiment_draft", return_value=approved),
            patch(f"{_MODULE}.render_draft_to_yaml", return_value="version: 2\n"),
            patch(f"{_MODULE}.record_session_event", return_value=stub) as mock_record,
        ):
            client.post(
                "/api/sessions/test-session-id/draft/render",
                json={"experiment_name": "exp_a", "draft_id": "draft-001"},
            )
        mock_record.assert_called_once()
        assert mock_record.call_args.kwargs["event_type"] == "YAML_RENDERED"

    def test_missing_draft_returns_404(self, client):
        with (
            patch(f"{_MODULE}.load_research_session", return_value=make_stub_session()),
            patch(f"{_MODULE}.load_experiment_draft", return_value=None),
        ):
            response = client.post(
                "/api/sessions/test-session-id/draft/render",
                json={"experiment_name": "exp_a", "draft_id": "nonexistent"},
            )
        assert response.status_code == 404

    def test_render_does_not_execute_experiment(self, client):
        approved = make_stub_draft(approved=True)
        stub = make_stub_session()
        with (
            patch(f"{_MODULE}.load_research_session", return_value=stub),
            patch(f"{_MODULE}.load_experiment_draft", return_value=approved),
            patch(f"{_MODULE}.render_draft_to_yaml", return_value="version: 2\n"),
            patch(f"{_MODULE}.record_session_event", return_value=stub),
        ):
            data = client.post(
                "/api/sessions/test-session-id/draft/render",
                json={"experiment_name": "exp_a", "draft_id": "draft-001"},
            ).json()
        # Response should not contain any "execute" or "run" confirmation
        assert "execute" not in str(data).lower()
        assert "ran experiment" not in str(data).lower()
