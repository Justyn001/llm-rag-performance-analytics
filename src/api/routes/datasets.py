"""
Datasets endpoint — knowledge bases that are ready to query.

GET /api/datasets → List datasets with at least one ingested Qdrant collection

The frontend uses this endpoint to populate the dataset dropdown.  A dataset
is listed only when its collection exists, so the UI never offers a knowledge
base that would fail with a 503 on the first query.
"""

import logging

from fastapi import APIRouter
from pydantic import BaseModel
from qdrant_client import QdrantClient

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Display names are shared with the test-set registry.  The collection prefix
# is always ``rag_<dataset>_<chunk_size>`` (see src.config.rag_config).
DATASET_REGISTRY = {
    "wiki": "Simple Wikipedia",
    "squad": "SQuAD 2.0",
    "trivia": "TriviaQA",
}


class DatasetInfo(BaseModel):
    """One knowledge base that can be selected in the UI."""

    name: str
    display_name: str
    chunk_sizes: list[int]


class DatasetsResponse(BaseModel):
    """List of ingested knowledge bases."""

    datasets: list[DatasetInfo]


def _collection_chunk_sizes(collection_names: set[str], dataset: str) -> list[int]:
    """Return chunk sizes for ``rag_<dataset>_<size>`` collections."""
    prefix = f"rag_{dataset}_"
    sizes: list[int] = []
    for name in collection_names:
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix) :]
        if suffix.isdigit():
            sizes.append(int(suffix))
    return sorted(set(sizes))


@router.get("/datasets", response_model=DatasetsResponse)
async def list_datasets():
    """
    List knowledge bases whose Qdrant collections exist.

    If Qdrant is unreachable, the endpoint returns an empty list instead of
    failing the whole frontend.  Queries will still fail fast with a clear
    error when the user tries to retrieve anything.
    """
    try:
        client = QdrantClient(url=get_settings().qdrant_url, timeout=3)
        collection_names = {item.name for item in client.get_collections().collections}
    except Exception as exc:  # noqa: BLE001 - Qdrant may be down at startup
        logger.warning("Could not list Qdrant collections: %s", exc)
        collection_names = set()

    datasets = [
        DatasetInfo(
            name=name,
            display_name=display_name,
            chunk_sizes=sizes,
        )
        for name, display_name in DATASET_REGISTRY.items()
        if (sizes := _collection_chunk_sizes(collection_names, name))
    ]
    return DatasetsResponse(datasets=datasets)
