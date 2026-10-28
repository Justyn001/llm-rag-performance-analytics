"""
Smoke tests for the FastAPI application wiring.

These tests do not call any external services (Ollama, Qdrant, cloud APIs).
They verify that the app can be imported and that local-only endpoints work.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture()
def client():
    """Create a test client with the application lifespan enabled."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "rag-analytics"}


def test_rag_configs_endpoint(client):
    response = client.get("/api/rag-configs")

    assert response.status_code == 200
    configs = response.json()["configs"]
    assert {config["name"] for config in configs} == {
        "basic-256",
        "basic-512",
        "basic-1024",
        "reranked-512",
    }


def test_test_sets_endpoint(client):
    response = client.get("/api/test-sets")

    assert response.status_code == 200
    test_sets = response.json()["test_sets"]
    assert {test_set["name"] for test_set in test_sets} == {
        "wikipedia-nq",
        "squad",
        "trivia",
    }
    assert all(test_set["total_available"] == 50 for test_set in test_sets)


def test_get_test_set(client):
    response = client.get("/api/test-sets/wikipedia-nq", params={"count": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["total_available"] == 50
    assert len(body["questions"]) == 1
    assert {
        "question",
        "expected_answer",
        "source",
        "answerable",
    } <= body["questions"][0].keys()


def test_get_unknown_test_set_returns_404(client):
    response = client.get("/api/test-sets/does-not-exist")

    assert response.status_code == 404
