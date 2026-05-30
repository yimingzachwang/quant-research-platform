"""Health endpoint tests."""

from fastapi.testclient import TestClient

from src.api.app import app


def test_health_returns_ok():
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_is_json():
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.headers["content-type"].startswith("application/json")
