"""
Retriever — searches Qdrant for relevant document chunks.

This is the "R" in RAG (Retrieval-Augmented Generation).
When a user asks a question:
  1. The question gets converted to a vector (embedding)
  2. Qdrant finds the most similar document chunks
  3. These chunks become the "context" for the LLM

Supports:
  - Dynamic collection switching (different datasets / chunk sizes)
  - Optional cross-encoder reranking for improved relevance
"""

import logging

from langchain_core.documents import Document as LCDocument
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from src.config.rag_config import DEFAULT_RAG_CONFIG, RAGConfig, get_collection_name
from src.config.settings import get_settings

logger = logging.getLogger(__name__)

# Cache for reranker model (loaded once)
_reranker = None


def _get_reranker():
    """Load cross-encoder reranker model (cached)."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        logger.info("Loading cross-encoder reranker...")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("Reranker ready")
    return _reranker


class Retriever:
    """
    Retrieves relevant document chunks from Qdrant.

    Supports dynamic collection switching and optional reranking.

    Usage:
        retriever = Retriever(embeddings)
        docs = retriever.search("What is the capital of France?", top_k=5)
    """

    def __init__(
        self,
        embeddings: Embeddings,
        dataset: str = "wiki",
        rag_config: RAGConfig = DEFAULT_RAG_CONFIG,
    ):
        self.embeddings = embeddings
        self.dataset = dataset
        self.rag_config = rag_config
        self.settings = get_settings()

        self.client = QdrantClient(
            host=self.settings.qdrant_host,
            port=self.settings.qdrant_port,
        )

        self.collection_name = get_collection_name(dataset, rag_config.chunk_size)

        # Check if collection exists, fallback to legacy name
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            # Fallback: try legacy collection name "rag_documents"
            if "rag_documents" in collections:
                logger.warning(
                    f"Collection '{self.collection_name}' not found, "
                    f"falling back to 'rag_documents'"
                )
                self.collection_name = "rag_documents"
            else:
                logger.warning(
                    f"Collection '{self.collection_name}' not found. Available: {collections}"
                )

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=embeddings,
        )

        logger.info(
            f"Retriever: collection={self.collection_name}, reranking={rag_config.use_reranking}"
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[LCDocument]:
        """
        Search for documents relevant to the query.

        If reranking is enabled, retrieves more candidates and
        reranks them with a cross-encoder.
        """
        if self.rag_config.use_reranking:
            return self._search_with_reranking(query, top_k)

        docs = self.vector_store.similarity_search(query, k=top_k)
        logger.debug(f"Retrieved {len(docs)} chunks for: '{query[:50]}...' (top_k={top_k})")
        return docs

    def _search_with_reranking(
        self,
        query: str,
        top_k: int,
    ) -> list[LCDocument]:
        """Retrieve candidates, then rerank with cross-encoder."""
        # Fetch more candidates for reranking
        candidates_k = min(top_k * 3, 30)
        candidates = self.vector_store.similarity_search(query, k=candidates_k)

        if not candidates:
            return []

        # Rerank with cross-encoder
        reranker = _get_reranker()
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = reranker.predict(pairs)

        # Sort by reranker score (descending) and take top_k
        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda x: x[1],
            reverse=True,
        )
        result = [doc for doc, _ in ranked[:top_k]]

        logger.debug(f"Reranked {len(candidates)} → {len(result)} chunks for: '{query[:50]}...'")
        return result

    def search_with_scores(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[LCDocument, float]]:
        """Search with similarity scores."""
        results = self.vector_store.similarity_search_with_score(query, k=top_k)
        return results

    def get_collection_info(self) -> dict:
        """Get info about the current Qdrant collection."""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status.value,
        }
