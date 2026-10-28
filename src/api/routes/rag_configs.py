"""
RAG configurations endpoint.

GET /api/rag-configs → List available RAG configurations
"""

from fastapi import APIRouter
from pydantic import BaseModel

from src.config.rag_config import RAG_CONFIGS

router = APIRouter()


class RAGConfigResponse(BaseModel):
    """A single RAG config for the frontend."""

    name: str
    display_name: str
    chunk_size: int
    use_reranking: bool
    description: str


class RAGConfigsListResponse(BaseModel):
    """List of available RAG configs."""

    configs: list[RAGConfigResponse]


@router.get("/rag-configs", response_model=RAGConfigsListResponse)
async def list_rag_configs():
    """List all available RAG configurations."""
    return RAGConfigsListResponse(
        configs=[
            RAGConfigResponse(
                name=c.name,
                display_name=c.display_name,
                chunk_size=c.chunk_size,
                use_reranking=c.use_reranking,
                description=c.description,
            )
            for c in RAG_CONFIGS
        ]
    )
