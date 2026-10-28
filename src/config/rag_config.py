"""
RAG configuration — defines retrieval strategies for experiments.

Each RAG config specifies:
- chunk_size: how large each text chunk is
- chunk_overlap: overlap between chunks
- use_reranking: whether to rerank results with a cross-encoder
- collection_suffix: Qdrant collection name suffix (e.g., "_256")

Usage:
    from src.config.rag_config import RAG_CONFIGS, get_rag_config
    config = get_rag_config("basic-512")
    print(config.chunk_size)  # 512
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RAGConfig:
    """Configuration for a single RAG retrieval strategy."""

    name: str
    display_name: str
    chunk_size: int
    chunk_overlap: int
    use_reranking: bool
    description: str

    @property
    def collection_suffix(self) -> str:
        """Suffix added to Qdrant collection names (e.g., '_256')."""
        return f"_{self.chunk_size}"


# =============================================================================
# AVAILABLE RAG CONFIGURATIONS
# =============================================================================

BASIC_256 = RAGConfig(
    name="basic-256",
    display_name="Basic 256",
    chunk_size=256,
    chunk_overlap=25,
    use_reranking=False,
    description="Small chunks (256 chars) — precise retrieval, less context",
)

BASIC_512 = RAGConfig(
    name="basic-512",
    display_name="Basic 512",
    chunk_size=512,
    chunk_overlap=50,
    use_reranking=False,
    description="Medium chunks (512 chars) — balanced precision and context",
)

BASIC_1024 = RAGConfig(
    name="basic-1024",
    display_name="Basic 1024",
    chunk_size=1024,
    chunk_overlap=100,
    use_reranking=False,
    description="Large chunks (1024 chars) — more context per chunk",
)

RERANKED_512 = RAGConfig(
    name="reranked-512",
    display_name="Reranked 512",
    chunk_size=512,
    chunk_overlap=50,
    use_reranking=True,
    description="Medium chunks + cross-encoder reranking for better relevance",
)

# --- Registry ---

RAG_CONFIGS = [BASIC_256, BASIC_512, BASIC_1024, RERANKED_512]

DEFAULT_RAG_CONFIG = BASIC_512

# All chunk sizes used (for ingestion — each needs a separate collection)
ALL_CHUNK_SIZES = sorted({c.chunk_size for c in RAG_CONFIGS})


def get_rag_config(name: str) -> RAGConfig:
    """Get a RAG config by name. Raises ValueError if not found."""
    for config in RAG_CONFIGS:
        if config.name == name:
            return config
    msg = f"RAG config '{name}' not found. Available: {[c.name for c in RAG_CONFIGS]}"
    raise ValueError(msg)


def get_collection_name(dataset: str, chunk_size: int) -> str:
    """
    Build Qdrant collection name from dataset + chunk_size.

    Examples:
        get_collection_name("wiki", 512) → "rag_wiki_512"
        get_collection_name("squad", 256) → "rag_squad_256"
    """
    return f"rag_{dataset}_{chunk_size}"
