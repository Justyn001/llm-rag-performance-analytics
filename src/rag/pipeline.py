"""
RAG Pipeline — the main brain of the application.

This ties everything together:
  1. User asks a question
  2. Retriever finds relevant chunks from Qdrant
  3. Chunks are formatted into context
  4. LLM generates an answer based on the context

Supports configurable RAG strategies (chunk sizes, reranking)
and multiple datasets.
"""

import logging
import time
from dataclasses import dataclass, field

from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage

from src.config.models import ModelConfig
from src.config.rag_config import DEFAULT_RAG_CONFIG, RAGConfig
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
    rag_config: str = ""  # Name of the RAG config used
    dataset: str = ""  # Name of the dataset used
    tokens_in: int = 0  # Input tokens (prompt)
    tokens_out: int = 0  # Output tokens (completion)
    tokens_per_second: float = 0.0  # Generation speed (tokens_out / gen_time_s)
    metadata: dict = field(default_factory=dict)


class RAGPipeline:
    """
    Main RAG pipeline — ask questions, get answers.

    Usage:
        pipeline = RAGPipeline(
            model_config=LLAMA_3_1_8B,
            embeddings=embeddings,
            rag_config=BASIC_512,
            dataset="wiki",
        )
        response = pipeline.ask("What is the capital of France?")
        print(response.answer)
    """

    def __init__(
        self,
        model_config: ModelConfig,
        embeddings: Embeddings,
        language: str = "en",
        rag_config: RAGConfig = DEFAULT_RAG_CONFIG,
        dataset: str = "wiki",
    ):
        self.model_config = model_config
        self.language = language
        self.rag_config = rag_config
        self.dataset = dataset

        # Initialize components
        self.llm = get_llm(model_config)
        self.retriever = Retriever(embeddings, dataset=dataset, rag_config=rag_config)
        self.system_prompt, self.user_prompt_template = get_prompts(language)

        logger.info(
            f"RAG Pipeline ready: model={model_config.name}, "
            f"lang={language}, rag={rag_config.name}, dataset={dataset}"
        )

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

        # Retry with backoff for rate limits (429)
        max_retries = 3
        response = None
        for attempt in range(max_retries + 1):
            try:
                response = self.llm.invoke(messages)
                break
            except Exception as exc:
                exc_str = str(exc)
                is_rate_limit = (
                    "429" in exc_str or "rate" in exc_str.lower() or "Rate limit" in exc_str
                )
                if is_rate_limit and attempt < max_retries:
                    wait = 2 ** (attempt + 1)
                    logger.warning(
                        f"Rate limited (attempt {attempt + 1}/"
                        f"{max_retries}), retrying in {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    raise

        generation_time = (time.perf_counter() - generation_start) * 1000

        total_time = (time.perf_counter() - total_start) * 1000

        # ---- Extract token usage ----
        tokens_in = 0
        tokens_out = 0
        tokens_per_second = 0.0

        # LangChain provides usage_metadata on AIMessage
        usage = getattr(response, "usage_metadata", None)
        if usage:
            tokens_in = usage.get("input_tokens", 0) or 0
            tokens_out = usage.get("output_tokens", 0) or 0

        # Fallback: try response_metadata (some providers put it there)
        if not tokens_out:
            resp_meta = getattr(response, "response_metadata", {})
            if "token_usage" in resp_meta:
                tokens_in = resp_meta["token_usage"].get("prompt_tokens", 0)
                tokens_out = resp_meta["token_usage"].get("completion_tokens", 0)

        # Calculate generation speed
        if tokens_out and generation_time > 0:
            tokens_per_second = round(tokens_out / (generation_time / 1000), 1)

        # ---- Build response ----
        answer = response.content if hasattr(response, "content") else str(response)

        # Strip chain-of-thought from models that include it
        if "</think>" in answer:
            answer = answer.split("</think>")[-1].strip()
        elif "<think>" in answer:
            answer = answer.split("<think>")[0].strip()

        # Fallback if answer is empty
        if not answer:
            answer = "I don't have enough information to answer this question."

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
            rag_config=self.rag_config.name,
            dataset=self.dataset,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            tokens_per_second=tokens_per_second,
        )

        logger.info(
            f"[{self.model_config.name}] "
            f"Q: '{question[:60]}...' → "
            f"retrieval={retrieval_time:.0f}ms, "
            f"generation={generation_time:.0f}ms, "
            f"total={total_time:.0f}ms, "
            f"tokens={tokens_in}→{tokens_out} ({tokens_per_second} tok/s)"
        )

        return rag_response


def create_pipeline(
    model_config: ModelConfig,
    embeddings: Embeddings | None = None,
    language: str = "en",
    rag_config: RAGConfig = DEFAULT_RAG_CONFIG,
    dataset: str = "wiki",
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
        rag_config=rag_config,
        dataset=dataset,
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
