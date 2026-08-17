"""
Embedding models wrapper.

Embeddings = converting text into numbers (vectors).
WHY? Because computers can't search by "meaning" using raw text.
Vectors let us find documents that are SEMANTICALLY similar to a question,
even if they don't share exact words.

Example:
  "What is the capital of France?" → [0.12, -0.45, 0.78, ...]
  "Paris is the capital city of France" → [0.11, -0.44, 0.77, ...]
  These vectors are very close! So retriever finds the right document.
"""

import logging
from enum import StrEnum

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class EmbeddingProvider(StrEnum):
    """Available embedding model providers."""

    LOCAL = "local"  # sentence-transformers (runs on your GPU)
    OPENAI = "openai"  # OpenAI embeddings API
    GOOGLE = "google"  # Google Gemini embeddings API


def get_embedding_model(
    provider: EmbeddingProvider = EmbeddingProvider.LOCAL,
    model_name: str | None = None,
) -> Embeddings:
    """
    Get an embedding model instance.

    Args:
        provider: Which provider to use (local, openai, google)
        model_name: Override the default model name

    Returns:
        LangChain Embeddings instance (works with Qdrant, RAGAS, etc.)
    """
    if provider == EmbeddingProvider.LOCAL:
        return _get_local_embeddings(model_name)
    if provider == EmbeddingProvider.OPENAI:
        return _get_openai_embeddings(model_name)
    if provider == EmbeddingProvider.GOOGLE:
        return _get_google_embeddings(model_name)

    msg = f"Unknown embedding provider: {provider}"
    raise ValueError(msg)


def _get_local_embeddings(model_name: str | None = None) -> Embeddings:
    """
    Local embeddings using sentence-transformers (runs on your GPU).

    Default model: all-MiniLM-L6-v2
    - Fast, lightweight, good quality
    - 384-dimensional vectors
    - Works in English (and decent in other languages)
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings

    if model_name is None:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"

    logger.info(f"Loading local embedding model: {model_name}")

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cuda"},  # Use your RTX 5070 Ti
        encode_kwargs={"normalize_embeddings": True},
    )


def _get_openai_embeddings(model_name: str | None = None) -> Embeddings:
    """OpenAI embeddings (cloud API, costs money but tiny amounts)."""
    from langchain_openai import OpenAIEmbeddings

    from src.config.settings import get_settings

    settings = get_settings()
    if not settings.has_openai():
        msg = "OpenAI API key not configured. Set OPENAI_API_KEY in .env"
        raise ValueError(msg)

    if model_name is None:
        model_name = "text-embedding-3-small"

    logger.info(f"Using OpenAI embedding model: {model_name}")

    return OpenAIEmbeddings(
        model=model_name,
        openai_api_key=settings.openai_api_key,
    )


def _get_google_embeddings(model_name: str | None = None) -> Embeddings:
    """Google Gemini embeddings (cloud API, free tier available)."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    from src.config.settings import get_settings

    settings = get_settings()
    if not settings.has_google():
        msg = "Google API key not configured. Set GOOGLE_API_KEY in .env"
        raise ValueError(msg)

    if model_name is None:
        model_name = "models/text-embedding-004"

    logger.info(f"Using Google embedding model: {model_name}")

    return GoogleGenerativeAIEmbeddings(
        model=model_name,
        google_api_key=settings.google_api_key,
    )
