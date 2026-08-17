"""
Tests for the data pipeline (loader, chunker, embeddings).
"""

from src.data.chunker import Chunk, ChunkingStrategy, chunk_documents
from src.data.embeddings import EmbeddingProvider
from src.data.loader import Document, QAPair


class TestQAPair:
    """Test QA pair data structure."""

    def test_create_qa_pair(self):
        qa = QAPair(
            question="What is the capital of France?",
            short_answer="Paris",
            long_answer="Paris is the capital of France.",
            context="France is a country in Europe. Paris is its capital.",
            document_title="France",
        )
        assert qa.question == "What is the capital of France?"
        assert qa.short_answer == "Paris"
        assert qa.language == "en"


class TestDocument:
    """Test Document data structure."""

    def test_create_document(self):
        doc = Document(
            content="France is a country in Western Europe.",
            title="France",
        )
        assert doc.title == "France"
        assert doc.source == "natural_questions"
        assert doc.language == "en"


class TestChunker:
    """Test chunking strategies."""

    def _make_docs(self, n: int = 3) -> list[Document]:
        """Create dummy documents for testing."""
        return [
            Document(
                content=f"This is a test document number {i}. " * 50,
                title=f"Test Doc {i}",
            )
            for i in range(n)
        ]

    def test_fixed_chunking_creates_chunks(self):
        docs = self._make_docs()
        chunks = chunk_documents(
            docs,
            strategy=ChunkingStrategy.FIXED,
            chunk_size=200,
        )
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_recursive_chunking_creates_chunks(self):
        docs = self._make_docs()
        chunks = chunk_documents(
            docs,
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=200,
        )
        assert len(chunks) > 0

    def test_chunks_have_metadata(self):
        docs = self._make_docs(1)
        chunks = chunk_documents(docs, chunk_size=200)
        chunk = chunks[0]
        assert chunk.document_title == "Test Doc 0"
        assert chunk.chunk_index == 0
        assert chunk.char_count > 0
        assert chunk.word_count > 0

    def test_smaller_chunks_produce_more_results(self):
        docs = self._make_docs()
        small = chunk_documents(docs, chunk_size=100)
        large = chunk_documents(docs, chunk_size=500)
        assert len(small) > len(large)


class TestEmbeddingProvider:
    """Test embedding provider enum."""

    def test_providers_exist(self):
        assert EmbeddingProvider.LOCAL == "local"
        assert EmbeddingProvider.OPENAI == "openai"
        assert EmbeddingProvider.GOOGLE == "google"
