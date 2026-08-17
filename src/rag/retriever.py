"""
Retriever — searches Qdrant for relevant document chunks.

This is the "R" in RAG (Retrieval-Augmented Generation).
When a user asks a question:
  1. The question gets converted to a vector (embedding)
  2. Qdrant finds the most similar document chunks
  3. These chunks become the "context" for the LLM

Think of it like a smart librarian:
  - User: "What is the capital of France?"
  - Retriever searches the library (Qdrant) by MEANING
  - Returns the most relevant pages/paragraphs
"""

import logging

from langchain_core.documents import Document as LCDocument
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from src.config.settings import get_settings
from src.data.ingest import COLLECTION_NAME

logger = logging.getLogger(__name__)


class Retriever:
    """
    Retrieves relevant document chunks from Qdrant.

    Usage:
        retriever = Retriever(embeddings)
        docs = retriever.search("What is the capital of France?", top_k=5)
    """

    def __init__(
        self,
        embeddings: Embeddings,
        collection_name: str = COLLECTION_NAME,
    ):
        settings = get_settings()

        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=embeddings,
        )

        self.collection_name = collection_name
        logger.info(f"Retriever connected to Qdrant collection: {collection_name}")

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[LCDocument]:
        """
        Search for documents relevant to the query.

        Args:
            query: The user's question
            top_k: How many chunks to return (more = more context but slower)

        Returns:
            List of LangChain Document objects with content and metadata
        """
        docs = self.vector_store.similarity_search(query, k=top_k)

        logger.debug(f"Retrieved {len(docs)} chunks for query: '{query[:50]}...' (top_k={top_k})")

        return docs

    def search_with_scores(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[LCDocument, float]]:
        """
        Search with similarity scores (useful for debugging/evaluation).

        Returns:
            List of (document, score) tuples. Score is 0-1, higher = more similar.
        """
        results = self.vector_store.similarity_search_with_score(query, k=top_k)

        logger.debug(f"Retrieved {len(results)} chunks with scores for: '{query[:50]}...'")

        return results

    def get_collection_info(self) -> dict:
        """Get info about the Qdrant collection (useful for debugging)."""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status.value,
        }
