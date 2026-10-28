"""
Data ingestion pipeline — the "brain loader".

This module orchestrates the entire data pipeline:
  1. Load documents (Wikipedia / SQuAD / TriviaQA)
  2. Chunk them into smaller pieces
  3. Embed them (convert text → vectors)
  4. Store in Qdrant (vector database)

Supports multiple chunk sizes and datasets.

Usage:
    uv run python -m src.data.ingest                     # wiki, 512 only
    uv run python -m src.data.ingest --all-configs        # wiki, all chunk sizes
    uv run python -m src.data.ingest --dataset squad      # squad, 512 only
    uv run python -m src.data.ingest --dataset squad --all-configs
"""

import argparse
import logging
import time

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.config.rag_config import ALL_CHUNK_SIZES, get_collection_name
from src.config.settings import get_settings
from src.data.chunker import ChunkingStrategy, chunk_documents
from src.data.embeddings import EmbeddingProvider, get_embedding_model
from src.data.loader import (
    load_natural_questions,
    load_wikipedia_contexts,
    save_qa_pairs,
)

logger = logging.getLogger(__name__)

# Legacy collection name (kept for backwards compatibility)
COLLECTION_NAME = "rag_documents"


def ingest_dataset(
    dataset: str = "wiki",
    chunk_sizes: list[int] | None = None,
    n_articles: int = 500,
    n_qa_pairs: int = 2000,
    chunk_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
    embedding_provider: EmbeddingProvider = EmbeddingProvider.LOCAL,
) -> dict:
    """
    Run ingestion for a dataset with one or more chunk sizes.

    Args:
        dataset: Dataset name ("wiki", "squad", "trivia")
        chunk_sizes: List of chunk sizes to ingest (default: [512])
        n_articles: Number of documents to load
        n_qa_pairs: Number of QA pairs to save
        chunk_strategy: How to split documents
        embedding_provider: Which embedding model to use

    Returns:
        Dict with stats about the ingestion
    """
    if chunk_sizes is None:
        chunk_sizes = [512]

    settings = get_settings()
    start_time = time.time()

    # ---- Step 1: Load data ----
    print(f"📥 Step 1: Loading data for dataset='{dataset}'...")
    documents = _load_documents(dataset, n_articles, n_qa_pairs)
    print(f"   ✓ {len(documents)} documents loaded")

    # ---- Step 2: Initialize embedding model ----
    print(f"🧮 Step 2: Loading embedding model ({embedding_provider})...")
    embeddings = get_embedding_model(provider=embedding_provider)
    print("   ✓ Embedding model ready")

    # ---- Step 3: Chunk + store for each chunk size ----
    total_chunks = 0
    for chunk_size in chunk_sizes:
        overlap = max(chunk_size // 10, 25)
        collection = get_collection_name(dataset, chunk_size)

        print(
            f"\n✂️  Chunking (size={chunk_size}, overlap={overlap}) → collection '{collection}'..."
        )

        chunks = chunk_documents(
            documents,
            strategy=chunk_strategy,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )
        print(f"   ✓ {len(chunks)} chunks created")

        print(f"🗄️  Storing in Qdrant collection '{collection}'...")
        _store_in_qdrant(
            chunks=chunks,
            embeddings=embeddings,
            settings=settings,
            collection_name=collection,
        )
        total_chunks += len(chunks)

    elapsed = time.time() - start_time
    stats = {
        "dataset": dataset,
        "documents": len(documents),
        "chunk_sizes": chunk_sizes,
        "total_chunks": total_chunks,
        "elapsed_seconds": round(elapsed, 1),
    }

    print(f"\n🎉 Ingestion complete in {elapsed:.1f}s!")
    print(f"   📊 {total_chunks} total chunks across {len(chunk_sizes)} collections")

    return stats


def _load_documents(dataset: str, n_articles: int, n_qa_pairs: int) -> list:
    """Load documents for the given dataset."""
    if dataset == "wiki":
        qa_pairs, _ = load_natural_questions(subset_size=n_qa_pairs)
        save_qa_pairs(qa_pairs)
        return load_wikipedia_contexts(subset_size=n_articles)

    if dataset == "squad":
        return _load_squad_documents(n_articles)

    if dataset == "trivia":
        return _load_trivia_documents(n_articles)

    msg = f"Unknown dataset: {dataset}"
    raise ValueError(msg)


def _load_squad_documents(n_articles: int) -> list:
    """Load SQuAD 2.0 contexts from HuggingFace."""
    from src.data.loader import Document

    try:
        from datasets import load_dataset

        ds = load_dataset("rajpurkar/squad_v2", split="train", trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load SQuAD dataset: {e}")
        return []

    # Extract unique contexts (SQuAD has many questions per context)
    seen = set()
    documents = []
    for item in ds:
        ctx = item["context"]
        title = item["title"]
        key = ctx[:100]
        if key not in seen and len(ctx) > 100:
            seen.add(key)
            documents.append(
                Document(
                    content=ctx,
                    title=title,
                    source="squad_v2",
                    language="en",
                )
            )
            if len(documents) >= n_articles:
                break

    print(f"   ✓ Loaded {len(documents)} SQuAD contexts")
    return documents


def _load_trivia_documents(n_articles: int) -> list:
    """Load TriviaQA evidence documents from HuggingFace."""
    from src.data.loader import Document

    try:
        from datasets import load_dataset

        ds = load_dataset("trivia_qa", "rc", split="train", trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load TriviaQA dataset: {e}")
        return []

    # Extract unique search results / evidence documents
    seen = set()
    documents = []
    for item in ds:
        for ctx in item.get("search_results", {}).get("search_context", []):
            if not ctx or len(ctx) < 100:
                continue
            key = ctx[:100]
            if key not in seen:
                seen.add(key)
                documents.append(
                    Document(
                        content=ctx,
                        title=item.get("question", "TriviaQA"),
                        source="trivia_qa",
                        language="en",
                    )
                )
                if len(documents) >= n_articles:
                    break
        if len(documents) >= n_articles:
            break

    print(f"   ✓ Loaded {len(documents)} TriviaQA contexts")
    return documents


def _store_in_qdrant(
    chunks: list,
    embeddings,
    settings,
    collection_name: str,
) -> None:
    """Store chunks in Qdrant vector database."""
    client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )

    # Check if collection exists
    collections = [c.name for c in client.get_collections().collections]

    if collection_name in collections:
        logger.info(f"Deleting existing collection: {collection_name}")
        client.delete_collection(collection_name)

    # Get embedding dimension
    test_embedding = embeddings.embed_query("test")
    vector_size = len(test_embedding)

    logger.info(f"Creating collection: {collection_name} (dim={vector_size})")
    client.create_collection(
        collection_name=collection_name,
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

    # Store in batches
    logger.info(f"Storing {len(texts)} chunks in '{collection_name}'...")

    QdrantVectorStore.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        collection_name=collection_name,
        url=settings.qdrant_url,
    )

    logger.info("✓ All chunks stored successfully")


# Legacy wrapper for backwards compatibility
def ingest_data(**kwargs) -> dict:
    """Legacy wrapper — ingests wiki dataset with default settings."""
    return ingest_dataset(dataset="wiki", **kwargs)


# ---------------------------------------------------------------------------
# CLI entry point: uv run python -m src.data.ingest
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Ingest data into Qdrant vector database")
    parser.add_argument(
        "--dataset",
        default="wiki",
        choices=["wiki", "squad", "trivia"],
        help="Dataset to ingest (default: wiki)",
    )
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="Ingest with all chunk sizes (256, 512, 1024)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Single chunk size to ingest (default: 512)",
    )
    parser.add_argument(
        "--n-articles",
        type=int,
        default=500,
        help="Number of documents to load (default: 500)",
    )
    args = parser.parse_args()

    if args.all_configs:
        sizes = ALL_CHUNK_SIZES
    elif args.chunk_size:
        sizes = [args.chunk_size]
    else:
        sizes = [512]

    ingest_dataset(
        dataset=args.dataset,
        chunk_sizes=sizes,
        n_articles=args.n_articles,
    )
