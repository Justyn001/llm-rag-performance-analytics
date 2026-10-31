"""
Tests for the Qdrant collection selection logic.

These tests use a fake Qdrant client, so they do not require a running
Qdrant instance or any external services.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.rag.retriever import CollectionNotFoundError, _resolve_collection_name


def _fake_client(*collection_names: str) -> Mock:
    """Create a fake Qdrant client exposing the given collection names."""
    client = Mock()
    client.get_collections.return_value = SimpleNamespace(
        collections=[SimpleNamespace(name=name) for name in collection_names]
    )
    return client


def test_existing_collection_is_returned():
    client = _fake_client("rag_wiki_512")

    assert _resolve_collection_name(client, "wiki", 512) == "rag_wiki_512"


def test_missing_collection_raises_clear_error():
    client = _fake_client("rag_documents", "rag_wiki_512")

    with pytest.raises(CollectionNotFoundError) as exc_info:
        _resolve_collection_name(client, "squad", 512)

    message = str(exc_info.value)
    assert "rag_squad_512" in message
    assert "rag_documents" in message
    assert "uv run python -m src.data.ingest --dataset squad --all-configs" in message


def test_legacy_collection_is_not_used_as_fallback():
    client = _fake_client("rag_documents")

    with pytest.raises(CollectionNotFoundError):
        _resolve_collection_name(client, "wiki", 512)
