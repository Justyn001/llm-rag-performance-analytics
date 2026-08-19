"""
Query endpoint — ask questions to the RAG system.

POST /api/query
"""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.config.models import ALL_MODELS
from src.data.embeddings import EmbeddingProvider, get_embedding_model
from src.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

router = APIRouter()

# Cache: reuse pipelines to avoid re-loading models on every request
_pipelines: dict[str, RAGPipeline] = {}
_embeddings = None


def _get_embeddings():
    """Get or create cached embedding model."""
    global _embeddings
    if _embeddings is None:
        _embeddings = get_embedding_model(provider=EmbeddingProvider.LOCAL)
    return _embeddings


def _get_pipeline(model_name: str, language: str = "en") -> RAGPipeline:
    """Get or create a cached RAG pipeline for the given model."""
    cache_key = f"{model_name}_{language}"
    if cache_key not in _pipelines:
        # Find model config by name
        model_config = None
        for m in ALL_MODELS:
            if m.name == model_name:
                model_config = m
                break

        if model_config is None:
            msg = f"Model '{model_name}' not found"
            raise ValueError(msg)

        _pipelines[cache_key] = RAGPipeline(
            model_config=model_config,
            embeddings=_get_embeddings(),
            language=language,
        )

    return _pipelines[cache_key]


# --- Request/Response schemas ---


class QueryRequest(BaseModel):
    """What the frontend sends."""

    question: str = Field(..., min_length=1, max_length=1000, description="The question to ask")
    model_name: str = Field(..., description="Name of the model to use")
    language: str = Field(default="en", description="Language: 'en' or 'pl'")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context chunks")


class ContextChunk(BaseModel):
    """A single retrieved context chunk."""

    content: str
    title: str
    score: float | None = None


class QueryResponse(BaseModel):
    """What the frontend receives."""

    answer: str
    question: str
    model_name: str
    language: str
    contexts: list[ContextChunk]
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    top_k: int


# --- Endpoint ---


@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Ask a question to the RAG system.

    The system will:
    1. Find relevant documents in the vector database
    2. Use the selected LLM to generate an answer
    3. Return the answer with timing metrics
    """
    try:
        pipeline = _get_pipeline(request.model_name, request.language)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Failed to initialize model '{request.model_name}'. Is the service running?",
        ) from e

    try:
        response = pipeline.ask(
            question=request.question,
            top_k=request.top_k,
            language=request.language,
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}") from e

    # Build context chunks for response
    contexts = [
        ContextChunk(
            content=ctx,
            title=meta.get("document_title", "Unknown"),
        )
        for ctx, meta in zip(response.contexts, response.context_metadata, strict=True)
    ]

    return QueryResponse(
        answer=response.answer,
        question=response.question,
        model_name=response.model_name,
        language=response.language,
        contexts=contexts,
        retrieval_time_ms=response.retrieval_time_ms,
        generation_time_ms=response.generation_time_ms,
        total_time_ms=response.total_time_ms,
        top_k=response.top_k,
    )
