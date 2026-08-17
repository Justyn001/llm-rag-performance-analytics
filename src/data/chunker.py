"""
Text chunking strategies for RAG.

Chunking = splitting long documents into smaller pieces.
WHY? Because:
  1. LLMs have limited context windows
  2. Smaller chunks = more precise retrieval
  3. Embeddings work better on focused text

We implement 3 strategies to compare in experiments:
  - Fixed-size: simple, split every N characters
  - Recursive: smart, splits on natural boundaries (paragraphs, sentences)
  - Semantic: advanced, splits based on meaning changes
"""

import logging
from dataclasses import dataclass, field
from enum import StrEnum

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

logger = logging.getLogger(__name__)


class ChunkingStrategy(StrEnum):
    """Available chunking strategies for experiments."""

    FIXED = "fixed"
    RECURSIVE = "recursive"


@dataclass
class Chunk:
    """A single chunk of text, ready for embedding."""

    content: str
    document_title: str
    chunk_index: int
    strategy: str
    language: str = "en"
    metadata: dict = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.content)

    @property
    def word_count(self) -> int:
        return len(self.content.split())


def chunk_documents(
    documents: list,
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """
    Split documents into chunks using the specified strategy.

    Args:
        documents: List of Document objects from loader.py
        strategy: Which chunking strategy to use
        chunk_size: Target size of each chunk (in characters)
        chunk_overlap: How much overlap between chunks (prevents cutting sentences)

    Returns:
        List of Chunk objects ready for embedding
    """
    logger.info(
        f"Chunking {len(documents)} documents "
        f"(strategy={strategy}, size={chunk_size}, overlap={chunk_overlap})"
    )

    # Pick the right splitter based on strategy
    splitter = _create_splitter(strategy, chunk_size, chunk_overlap)

    chunks: list[Chunk] = []
    for doc in documents:
        # Split the document text
        text_chunks = splitter.split_text(doc.content)

        for i, text in enumerate(text_chunks):
            # Skip very short chunks (noise)
            if len(text.strip()) < 50:
                continue

            chunk = Chunk(
                content=text.strip(),
                document_title=doc.title,
                chunk_index=i,
                strategy=strategy.value,
                language=doc.language,
                metadata={
                    "source": doc.source,
                    "chunk_size_setting": chunk_size,
                    **doc.metadata,
                },
            )
            chunks.append(chunk)

    logger.info(
        f"Created {len(chunks)} chunks "
        f"(avg {sum(c.char_count for c in chunks) // max(len(chunks), 1)} chars/chunk)"
    )

    return chunks


def _create_splitter(
    strategy: ChunkingStrategy,
    chunk_size: int,
    chunk_overlap: int,
) -> CharacterTextSplitter | RecursiveCharacterTextSplitter:
    """Create the appropriate text splitter for the given strategy."""

    if strategy == ChunkingStrategy.FIXED:
        # Simple: cut every N characters
        # Fast but can cut mid-sentence
        return CharacterTextSplitter(
            separator="\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    if strategy == ChunkingStrategy.RECURSIVE:
        # Smart: tries to split on paragraph -> sentence -> word boundaries
        # Best balance of quality and speed
        return RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    msg = f"Unknown chunking strategy: {strategy}"
    raise ValueError(msg)


def chunk_with_multiple_strategies(
    documents: list,
    strategies: list[ChunkingStrategy] | None = None,
    chunk_sizes: list[int] | None = None,
) -> dict[str, list[Chunk]]:
    """
    Chunk documents with multiple strategies for comparison experiments.

    Returns a dict: {strategy_name: [chunks]}

    Example:
        results = chunk_with_multiple_strategies(docs)
        for strategy, chunks in results.items():
            print(f"{strategy}: {len(chunks)} chunks")
    """
    if strategies is None:
        strategies = list(ChunkingStrategy)
    if chunk_sizes is None:
        chunk_sizes = [512, 1024]

    results: dict[str, list[Chunk]] = {}

    for strategy in strategies:
        for size in chunk_sizes:
            key = f"{strategy.value}_{size}"
            chunks = chunk_documents(documents, strategy=strategy, chunk_size=size)
            results[key] = chunks
            logger.info(f"  {key}: {len(chunks)} chunks")

    return results
