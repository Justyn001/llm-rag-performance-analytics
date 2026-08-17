"""
RAG Pipeline — the main brain of the application.

This ties everything together:
  1. User asks a question
  2. Retriever finds relevant chunks from Qdrant
  3. Chunks are formatted into context
  4. LLM generates an answer based on the context

This is what makes it "Retrieval-Augmented Generation":
  - Retrieval: finding relevant documents (Retriever)
  - Augmented: adding them to the prompt (context)
  - Generation: LLM writes the answer
"""

import logging
import time
from dataclasses import dataclass, field

from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage

from src.config.models import ModelConfig
from src.rag.llm_provider import get_llm
from src.rag.prompts import get_prompts
from src.rag.retriever import Retriever

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Complete response from the RAG pipeline, including metadata."""

    answer: str
    question: str
    model_name: str
    contexts: list[str]  # The retrieved chunks used to generate the answer
    context_metadata: list[dict]  # Metadata of retrieved chunks
    language: str = "en"
    retrieval_time_ms: float = 0.0  # How long retrieval took
    generation_time_ms: float = 0.0  # How long LLM generation took
    total_time_ms: float = 0.0  # Total end-to-end time
    top_k: int = 5
    metadata: dict = field(default_factory=dict)


class RAGPipeline:
    """
    Main RAG pipeline — ask questions, get answers.

    Usage:
        pipeline = RAGPipeline(model_config=LLAMA_3_1_8B, embeddings=embeddings)
        response = pipeline.ask("What is the capital of France?")
        print(response.answer)
    """

    def __init__(
        self,
        model_config: ModelConfig,
        embeddings: Embeddings,
        language: str = "en",
    ):
        self.model_config = model_config
        self.language = language

        # Initialize components
        self.llm = get_llm(model_config)
        self.retriever = Retriever(embeddings)
        self.system_prompt, self.user_prompt_template = get_prompts(language)

        logger.info(f"RAG Pipeline ready: model={model_config.name}, lang={language}")

    def ask(
        self,
        question: str,
        top_k: int = 5,
        language: str | None = None,
    ) -> RAGResponse:
        """
        Ask a question and get an answer from the RAG system.

        Args:
            question: The user's question
            top_k: Number of context chunks to retrieve
            language: Override the pipeline's default language

        Returns:
            RAGResponse with answer, timing, and metadata
        """
        total_start = time.perf_counter()

        # Use override language or default
        lang = language or self.language
        system_prompt, user_prompt_template = get_prompts(lang)

        # ---- Step 1: Retrieve relevant chunks ----
        retrieval_start = time.perf_counter()
        docs = self.retriever.search(question, top_k=top_k)
        retrieval_time = (time.perf_counter() - retrieval_start) * 1000

        # Format context from retrieved documents
        contexts = [doc.page_content for doc in docs]
        context_metadata = [doc.metadata for doc in docs]
        context_text = "\n\n---\n\n".join(contexts)

        # ---- Step 2: Generate answer with LLM ----
        generation_start = time.perf_counter()

        user_prompt = user_prompt_template.format(
            context=context_text,
            question=question,
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = self.llm.invoke(messages)
        generation_time = (time.perf_counter() - generation_start) * 1000

        total_time = (time.perf_counter() - total_start) * 1000

        # ---- Build response ----
        answer = response.content if hasattr(response, "content") else str(response)

        rag_response = RAGResponse(
            answer=answer,
            question=question,
            model_name=self.model_config.name,
            contexts=contexts,
            context_metadata=context_metadata,
            language=lang,
            retrieval_time_ms=round(retrieval_time, 1),
            generation_time_ms=round(generation_time, 1),
            total_time_ms=round(total_time, 1),
            top_k=top_k,
        )

        logger.info(
            f"[{self.model_config.name}] "
            f"Q: '{question[:60]}...' → "
            f"retrieval={retrieval_time:.0f}ms, "
            f"generation={generation_time:.0f}ms, "
            f"total={total_time:.0f}ms"
        )

        return rag_response


def create_pipeline(
    model_config: ModelConfig,
    embeddings: Embeddings | None = None,
    language: str = "en",
) -> RAGPipeline:
    """
    Convenience function to create a RAG pipeline.

    If embeddings are not provided, uses default local embeddings.
    """
    if embeddings is None:
        from src.data.embeddings import EmbeddingProvider, get_embedding_model

        embeddings = get_embedding_model(provider=EmbeddingProvider.LOCAL)

    return RAGPipeline(
        model_config=model_config,
        embeddings=embeddings,
        language=language,
    )


# ---------------------------------------------------------------------------
# CLI: Quick test — uv run python -m src.rag.pipeline
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    from src.rag.llm_provider import list_available_models

    # Show available models
    available = list_available_models()
    print(f"\n📋 Available models: {[m.name for m in available]}")

    if not available:
        print("❌ No models available! Pull a model: ollama pull llama3.1:8b")
        raise SystemExit(1)

    # Use first available model
    model = available[0]
    print(f"🤖 Using: {model.name}")

    # Create pipeline and ask a test question
    pipeline = create_pipeline(model)

    test_questions = [
        "What is the largest country in the world?",
        "Who wrote Romeo and Juliet?",
        "What is photosynthesis?",
    ]

    for q in test_questions:
        print(f"\n{'=' * 60}")
        print(f"❓ {q}")
        response = pipeline.ask(q)
        print(f"💬 {response.answer}")
        print(
            f"⏱️  retrieval={response.retrieval_time_ms}ms, "
            f"generation={response.generation_time_ms}ms"
        )
