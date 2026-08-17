"""
Data ingestion pipeline — the "brain loader".

This module orchestrates the entire data pipeline:
  1. Load documents (Wikipedia articles)
  2. Chunk them into smaller pieces
  3. Embed them (convert text → vectors)
  4. Store in Qdrant (vector database)

After running this, Qdrant is loaded and ready for RAG queries.

Usage:
    uv run python -m src.data.ingest
"""

import logging
import time

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.config.settings import get_settings
from src.data.chunker import ChunkingStrategy, chunk_documents
from src.data.embeddings import EmbeddingProvider, get_embedding_model
from src.data.loader import (
    load_natural_questions,
    load_wikipedia_contexts,
    save_qa_pairs,
)

logger = logging.getLogger(__name__)

# Name of the Qdrant collection (like a "table" in a regular database)
COLLECTION_NAME = "rag_documents"


def ingest_data(
    n_articles: int = 500,
    n_qa_pairs: int = 2000,
    chunk_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    embedding_provider: EmbeddingProvider = EmbeddingProvider.LOCAL,
    recreate_collection: bool = True,
) -> dict:
    """
    Run the full ingestion pipeline.

    Args:
        n_articles: Number of Wikipedia articles to load
        n_qa_pairs: Number of QA pairs to save for evaluation
        chunk_strategy: How to split documents
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        embedding_provider: Which embedding model to use
        recreate_collection: If True, delete and recreate the Qdrant collection

    Returns:
        Dict with stats about the ingestion
    """
    settings = get_settings()
    start_time = time.time()

    # ---- Step 1: Load data ----
    print("📥 Step 1/4: Loading data...")
    qa_pairs, _ = load_natural_questions(subset_size=n_qa_pairs)
    documents = load_wikipedia_contexts(subset_size=n_articles)
    print(f"   ✓ {len(qa_pairs)} QA pairs, {len(documents)} articles")

    # Save QA pairs for later evaluation
    save_qa_pairs(qa_pairs)

    # ---- Step 2: Chunk documents ----
    print(f"✂️  Step 2/4: Chunking ({chunk_strategy}, size={chunk_size})...")
    chunks = chunk_documents(
        documents,
        strategy=chunk_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    print(f"   ✓ {len(chunks)} chunks created")

    # ---- Step 3: Initialize embedding model ----
    print(f"🧮 Step 3/4: Loading embedding model ({embedding_provider})...")
    embeddings = get_embedding_model(provider=embedding_provider)
    print("   ✓ Embedding model ready")

    # ---- Step 4: Store in Qdrant ----
    print("🗄️  Step 4/4: Storing in Qdrant...")
    _store_in_qdrant(
        chunks=chunks,
        embeddings=embeddings,
        settings=settings,
        recreate=recreate_collection,
    )

    elapsed = time.time() - start_time
    stats = {
        "qa_pairs": len(qa_pairs),
        "documents": len(documents),
        "chunks": len(chunks),
        "chunk_strategy": chunk_strategy.value,
        "chunk_size": chunk_size,
        "embedding_provider": embedding_provider.value,
        "elapsed_seconds": round(elapsed, 1),
    }

    print(f"\n🎉 Ingestion complete in {elapsed:.1f}s!")
    print(f"   📊 {stats['chunks']} chunks stored in Qdrant")
    print(f"   📝 {stats['qa_pairs']} QA pairs saved for evaluation")

    return stats


def _store_in_qdrant(
    chunks: list,
    embeddings,
    settings,
    recreate: bool = True,
) -> None:
    """Store chunks in Qdrant vector database."""
    client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )

    # Check if collection exists
    collections = [c.name for c in client.get_collections().collections]

    if recreate and COLLECTION_NAME in collections:
        logger.info(f"Deleting existing collection: {COLLECTION_NAME}")
        client.delete_collection(COLLECTION_NAME)

    if COLLECTION_NAME not in collections or recreate:
        # Get embedding dimension by embedding a test string
        test_embedding = embeddings.embed_query("test")
        vector_size = len(test_embedding)

        logger.info(f"Creating collection: {COLLECTION_NAME} (dim={vector_size})")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    # Prepare texts and metadata for Qdrant
    texts = [chunk.content for chunk in chunks]
    metadatas = [
        {
            "document_title": chunk.document_title,
            "chunk_index": chunk.chunk_index,
            "strategy": chunk.strategy,
            "language": chunk.language,
            "char_count": chunk.char_count,
            **chunk.metadata,
        }
        for chunk in chunks
    ]

    # Store in batches (Qdrant + LangChain integration)
    logger.info(f"Storing {len(texts)} chunks in Qdrant...")

    QdrantVectorStore.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        collection_name=COLLECTION_NAME,
        url=settings.qdrant_url,
    )

    logger.info("✓ All chunks stored successfully")


# ---------------------------------------------------------------------------
# CLI entry point: uv run python -m src.data.ingest
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    ingest_data()
