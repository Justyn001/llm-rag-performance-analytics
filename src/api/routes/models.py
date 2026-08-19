"""
Models endpoint — list available LLM models.

GET /api/models
"""

from fastapi import APIRouter
from pydantic import BaseModel

from src.rag.llm_provider import list_available_models

router = APIRouter()


class ModelInfo(BaseModel):
    """Model information for the frontend."""

    name: str
    provider: str
    model_id: str
    description: str
    is_local: bool
    parameters: int | None = None


class ModelsResponse(BaseModel):
    """List of available models."""

    models: list[ModelInfo]
    total: int


@router.get("/models", response_model=ModelsResponse)
async def get_models():
    """List all currently available LLM models."""
    available = list_available_models()

    models = [
        ModelInfo(
            name=m.name,
            provider=m.provider.value,
            model_id=m.model_id,
            description=m.description,
            is_local=m.is_local,
            parameters=m.parameters,
        )
        for m in available
    ]

    return ModelsResponse(models=models, total=len(models))
