"""
Smoke tests for the FastAPI application wiring.

These tests do not call any external services (Ollama, Qdrant, cloud APIs).
They verify that the app can be imported and that local-only endpoints work.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.routes import query as query_routes
from src.api.routes.datasets import _collection_chunk_sizes
from src.rag.retriever import CollectionNotFoundError


@pytest.fixture()
def client():
    """Create a test client with the application lifespan enabled."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "rag-analytics"}


def test_health_alias_endpoint(client):
    """Monitoring tools commonly probe /health instead of /api/health."""
    response = client.get("/health")

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


def test_datasets_endpoint_lists_ingested_collections(client, monkeypatch):
    class FakeCollection:
        def __init__(self, name):
            self.name = name

    class FakeCollections:
        collections = [
            FakeCollection("rag_wiki_256"),
            FakeCollection("rag_wiki_512"),
            FakeCollection("rag_squad_512"),
            FakeCollection("rag_trivia_1024"),
            FakeCollection("rag_documents"),
        ]

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def get_collections(self):
            return FakeCollections()

    monkeypatch.setattr("src.api.routes.datasets.QdrantClient", FakeClient)

    response = client.get("/api/datasets")

    assert response.status_code == 200
    datasets = {item["name"]: item for item in response.json()["datasets"]}
    assert datasets["wiki"]["chunk_sizes"] == [256, 512]
    assert datasets["squad"]["chunk_sizes"] == [512]
    assert datasets["trivia"]["chunk_sizes"] == [1024]


def test_datasets_endpoint_returns_empty_when_qdrant_is_down(client, monkeypatch):
    class BrokenClient:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Qdrant unavailable")

    monkeypatch.setattr("src.api.routes.datasets.QdrantClient", BrokenClient)

    response = client.get("/api/datasets")

    assert response.status_code == 200
    assert response.json()["datasets"] == []


def test_collection_chunk_sizes_ignores_unrelated_collections():
    names = {
        "rag_wiki_256",
        "rag_wiki_512",
        "rag_squad_512",
        "rag_documents",
        "rag_wiki_backup",
    }

    assert _collection_chunk_sizes(names, "wiki") == [256, 512]
    assert _collection_chunk_sizes(names, "trivia") == []


def test_test_sets_endpoint(client):
    response = client.get("/api/test-sets")

    assert response.status_code == 200
    test_sets = {item["name"]: item for item in response.json()["test_sets"]}
    assert set(test_sets) == {"wikipedia-simple", "squad", "trivia"}
    assert test_sets["wikipedia-simple"]["dataset"] == "wiki"
    assert test_sets["wikipedia-simple"]["total_available"] == 40
    assert test_sets["squad"]["dataset"] == "squad"
    assert test_sets["squad"]["total_available"] == 50
    assert test_sets["trivia"]["dataset"] == "trivia"
    assert test_sets["trivia"]["total_available"] == 50


def test_get_test_set(client):
    response = client.get("/api/test-sets/wikipedia-simple", params={"count": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["total_available"] == 40
    assert len(body["questions"]) == 1
    question = body["questions"][0]
    assert {
        "question_id",
        "question",
        "expected_answer",
        "gold_context",
        "gold_span",
        "dataset",
        "split",
        "set_type",
        "source",
        "answerable",
    } <= question.keys()
    assert question["answerable"] is True
    assert question["gold_span"] in question["gold_context"]


def test_test_set_contains_unanswerable_controls(client):
    response = client.get("/api/test-sets/wikipedia-simple", params={"count": 40})

    assert response.status_code == 200
    controls = [q for q in response.json()["questions"] if not q["answerable"]]
    assert len(controls) == 10
    assert all(q["gold_context"] is None for q in controls)
    assert all(q["gold_span"] is None for q in controls)


def test_get_unknown_test_set_returns_404(client):
    response = client.get("/api/test-sets/does-not-exist")

    assert response.status_code == 404


@pytest.mark.parametrize(
    ("name", "dataset", "total"),
    [("squad", "squad", 50), ("trivia", "trivia", 50)],
)
def test_official_test_sets_are_exposed(client, name, dataset, total):
    response = client.get(f"/api/test-sets/{name}", params={"count": total})

    assert response.status_code == 200
    body = response.json()
    assert body["dataset"] == dataset
    assert body["total_available"] == total
    assert len(body["questions"]) == total
    assert all(question["dataset"] == dataset for question in body["questions"])
    assert all(question["split"] == "validation" for question in body["questions"])
    assert all(question["set_type"] == "official" for question in body["questions"])

    answerable = [question for question in body["questions"] if question["answerable"]]
    assert answerable
    assert all(question["gold_span"] in question["gold_context"] for question in answerable)


def test_query_returns_clear_error_when_collection_is_missing(client, monkeypatch):
    message = (
        "Qdrant collection 'rag_squad_512' does not exist. Available collections: rag_wiki_512."
    )

    def fake_get_pipeline(*args, **kwargs):
        raise CollectionNotFoundError(message)

    monkeypatch.setattr(query_routes, "_get_pipeline", fake_get_pipeline)

    response = client.post(
        "/api/query",
        json={
            "question": "What is the capital of France?",
            "model_name": "Gemma 3 4B",
            "dataset": "squad",
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"] == message
