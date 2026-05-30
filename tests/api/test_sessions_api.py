"""Session endpoint tests.

All Research API calls are mocked — no disk I/O occurs.
"""

from __future__ import annotations

from unittest.mock import patch

from tests.api.conftest import make_stub_session

_MODULE = "src.orchestration.api.research_api"


class TestListSessions:
    def test_returns_200(self, client):
        with patch(f"{_MODULE}.list_research_sessions", return_value=["s1", "s2"]):
            response = client.get("/api/sessions")
        assert response.status_code == 200

    def test_returns_session_ids(self, client):
        with patch(f"{_MODULE}.list_research_sessions", return_value=["s1", "s2"]):
            data = client.get("/api/sessions").json()
        assert data["sessions"] == ["s1", "s2"]

    def test_empty_list(self, client):
        with patch(f"{_MODULE}.list_research_sessions", return_value=[]):
            data = client.get("/api/sessions").json()
        assert data["sessions"] == []


class TestCreateSession:
    def _create(self, client, stub_session):
        with patch(f"{_MODULE}.create_research_session", return_value=stub_session):
            return client.post(
                "/api/sessions",
                json={"root_experiment": "exp_a", "research_goal": "Test goal"},
            )

    def test_returns_201(self, client):
        stub = make_stub_session()
        assert self._create(client, stub).status_code == 201

    def test_response_contains_session(self, client):
        stub = make_stub_session()
        data = self._create(client, stub).json()
        assert "session" in data
        assert data["session"]["session_id"] == "test-session-id"

    def test_response_contains_summary(self, client):
        stub = make_stub_session()
        data = self._create(client, stub).json()
        assert "summary" in data
        assert "session_id" in data["summary"]


class TestGetSession:
    def test_returns_200_when_found(self, client):
        stub = make_stub_session()
        with patch(f"{_MODULE}.load_research_session", return_value=stub):
            response = client.get("/api/sessions/test-session-id")
        assert response.status_code == 200

    def test_response_contains_session_and_summary(self, client):
        stub = make_stub_session()
        with patch(f"{_MODULE}.load_research_session", return_value=stub):
            data = client.get("/api/sessions/test-session-id").json()
        assert "session" in data
        assert "summary" in data

    def test_missing_session_returns_404(self, client):
        with patch(f"{_MODULE}.load_research_session", return_value=None):
            response = client.get("/api/sessions/nonexistent")
        assert response.status_code == 404

    def test_404_detail_is_readable(self, client):
        with patch(f"{_MODULE}.load_research_session", return_value=None):
            data = client.get("/api/sessions/nonexistent").json()
        assert "detail" in data
        assert "nonexistent" in data["detail"]


class TestUpdateStatus:
    def test_returns_200(self, client):
        stub = make_stub_session()
        updated = make_stub_session(status="paused")
        with (
            patch(f"{_MODULE}.load_research_session", return_value=stub),
            patch(f"{_MODULE}.update_research_session_status", return_value=updated),
        ):
            response = client.put(
                "/api/sessions/test-session-id/status",
                json={"status": "paused"},
            )
        assert response.status_code == 200

    def test_response_shows_updated_status(self, client):
        stub = make_stub_session()
        updated = make_stub_session(status="paused")
        with (
            patch(f"{_MODULE}.load_research_session", return_value=stub),
            patch(f"{_MODULE}.update_research_session_status", return_value=updated),
        ):
            data = client.put(
                "/api/sessions/test-session-id/status",
                json={"status": "paused"},
            ).json()
        assert data["session"]["status"] == "paused"

    def test_invalid_status_returns_400(self, client):
        response = client.put(
            "/api/sessions/test-session-id/status",
            json={"status": "not_a_real_status"},
        )
        assert response.status_code == 400

    def test_invalid_status_detail_mentions_value(self, client):
        data = client.put(
            "/api/sessions/test-session-id/status",
            json={"status": "invalid"},
        ).json()
        assert "invalid" in data["detail"]

    def test_valid_status_still_returns_200(self, client):
        stub = make_stub_session()
        paused = make_stub_session(status="paused")
        with (
            patch(f"{_MODULE}.load_research_session", return_value=stub),
            patch(f"{_MODULE}.update_research_session_status", return_value=paused),
        ):
            response = client.put(
                "/api/sessions/test-session-id/status",
                json={"status": "paused"},
            )
        assert response.status_code == 200

    def test_missing_session_returns_404(self, client):
        with patch(f"{_MODULE}.load_research_session", return_value=None):
            response = client.put(
                "/api/sessions/ghost/status",
                json={"status": "paused"},
            )
        assert response.status_code == 404


class TestRecordEvent:
    def _record(self, client, stub, updated):
        with (
            patch(f"{_MODULE}.load_research_session", return_value=stub),
            patch(f"{_MODULE}.record_session_event", return_value=updated),
        ):
            return client.post(
                "/api/sessions/test-session-id/events",
                json={
                    "event_type": "REVIEW_GENERATED",
                    "experiment_name": "exp_a",
                    "data": {"provider": "stub"},
                },
            )

    def test_returns_200(self, client):
        stub = make_stub_session()
        assert self._record(client, stub, stub).status_code == 200

    def test_response_contains_session_and_summary(self, client):
        stub = make_stub_session()
        data = self._record(client, stub, stub).json()
        assert "session" in data
        assert "summary" in data

    def test_missing_session_returns_404(self, client):
        with patch(f"{_MODULE}.load_research_session", return_value=None):
            response = client.post(
                "/api/sessions/ghost/events",
                json={"event_type": "REVIEW_GENERATED", "experiment_name": "exp_a"},
            )
        assert response.status_code == 404
