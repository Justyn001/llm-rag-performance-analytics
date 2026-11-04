"""
Tests for the RAG pipeline components.
"""

from types import SimpleNamespace

import pytest

from src.config.models import (
    ALL_MODELS,
    GEMINI_2_5_FLASH,
    GEMMA_3_4B,
    GROQ_GPT_OSS_20B,
    GROQ_GPT_OSS_120B,
    GROQ_QWEN_3_8,
    LLAMA_4_SCOUT,
    MISTRAL_NEMO,
    QWEN_3_8B,
    ModelProvider,
)
from src.config.rag_config import BASIC_512
from src.rag.pipeline import RAGPipeline
from src.rag.prompts import get_prompts


class TestPrompts:
    """Test prompt templates."""

    def test_english_prompts_exist(self):
        system, user = get_prompts("en")
        assert "context" in user.lower()
        assert "question" in user.lower()
        assert len(system) > 50

    def test_polish_prompts_exist(self):
        system, user = get_prompts("pl")
        assert "kontekst" in user.lower()
        assert "pytanie" in user.lower()
        assert len(system) > 50

    def test_user_prompt_has_placeholders(self):
        _, user_en = get_prompts("en")
        _, user_pl = get_prompts("pl")
        assert "{context}" in user_en
        assert "{question}" in user_en
        assert "{context}" in user_pl
        assert "{question}" in user_pl

    def test_default_is_english(self):
        system_default, _ = get_prompts()
        system_en, _ = get_prompts("en")
        assert system_default == system_en


class TestLLMProviderConfig:
    """Test that model configs are wired to correct providers."""

    def test_llama4_uses_ollama(self):
        assert LLAMA_4_SCOUT.provider == ModelProvider.OLLAMA
        assert LLAMA_4_SCOUT.model_id == "llama4"

    def test_mistral_nemo_uses_ollama(self):
        assert MISTRAL_NEMO.provider == ModelProvider.OLLAMA
        assert MISTRAL_NEMO.parameters == 12_000_000_000

    def test_qwen3_uses_ollama(self):
        assert QWEN_3_8B.provider == ModelProvider.OLLAMA

    def test_gemma3_uses_ollama(self):
        assert GEMMA_3_4B.provider == ModelProvider.OLLAMA

    def test_groq_qwen38_uses_groq(self):
        assert GROQ_QWEN_3_8.provider == ModelProvider.GROQ
        assert GROQ_QWEN_3_8.model_id == "qwen/qwen3.8-27b"

    def test_groq_gpt_oss_20b_uses_groq(self):
        assert GROQ_GPT_OSS_20B.provider == ModelProvider.GROQ
        assert GROQ_GPT_OSS_20B.model_id == "openai/gpt-oss-20b"

    def test_gemini25_uses_google(self):
        assert GEMINI_2_5_FLASH.provider == ModelProvider.GOOGLE

    def test_groq_gpt_oss_120b_uses_groq(self):
        assert GROQ_GPT_OSS_120B.provider == ModelProvider.GROQ
        assert GROQ_GPT_OSS_120B.model_id == "openai/gpt-oss-120b"

    def test_all_models_has_all(self):
        assert len(ALL_MODELS) == 8

    def test_local_models_are_local(self):
        for m in ALL_MODELS:
            if m.provider == ModelProvider.OLLAMA:
                assert m.is_local is True

    def test_cloud_models_are_not_local(self):
        for m in ALL_MODELS:
            if m.provider != ModelProvider.OLLAMA:
                assert m.is_local is False

    def test_quantization_metadata_is_present(self):
        for model in ALL_MODELS:
            assert model.quantization
        for model in ALL_MODELS:
            if model.provider == ModelProvider.OLLAMA:
                assert model.quantization != "provider-managed"


class _FakeRetriever:
    def search(self, question: str, top_k: int = 5):
        return [SimpleNamespace(page_content="Paris is the capital.", metadata={"title": "France"})]


class _FlakyLLM:
    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("429 rate limit")
        return SimpleNamespace(
            content="Paris",
            usage_metadata={"input_tokens": 12, "output_tokens": 3},
        )


def _pipeline_with_fakes(llm) -> RAGPipeline:
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.model_config = GEMMA_3_4B
    pipeline.language = "en"
    pipeline.rag_config = BASIC_512
    pipeline.dataset = "wiki"
    pipeline.max_tokens = 1024
    pipeline.num_ctx = 8192
    pipeline.llm = llm
    pipeline.retriever = _FakeRetriever()
    pipeline.system_prompt, pipeline.user_prompt_template = get_prompts("en")
    return pipeline


def test_generation_time_excludes_retry_backoff(monkeypatch):
    # total_start, retrieval_start, retrieval_end,
    # failed_start, failed_end, success_start, success_end, total_end
    ticks = iter([0.0, 0.100, 0.120, 0.200, 0.300, 0.400, 0.500, 0.900])
    sleeps: list[float] = []

    monkeypatch.setattr("src.rag.pipeline.time.perf_counter", lambda: next(ticks))
    monkeypatch.setattr("src.rag.pipeline.time.sleep", sleeps.append)

    response = _pipeline_with_fakes(_FlakyLLM()).ask("What is the capital of France?")

    assert response.retrieval_time_ms == 20.0
    assert response.generation_time_ms == 100.0
    assert response.failed_attempt_time_ms == 100.0
    assert response.retry_backoff_time_ms == 2000.0
    assert response.retry_count == 1
    assert response.total_time_ms == 900.0
    assert response.tokens_per_second == 30.0
    assert response.max_tokens == 1024
    assert response.num_ctx == 8192
    assert sleeps == [2]


def test_pipeline_propagates_non_rate_limit_errors(monkeypatch):
    class _BrokenLLM:
        def invoke(self, messages):
            raise RuntimeError("model crashed")

    monkeypatch.setattr("src.rag.pipeline.time.perf_counter", lambda: 0.0)
    pipeline = _pipeline_with_fakes(_BrokenLLM())

    with pytest.raises(RuntimeError, match="model crashed"):
        pipeline.ask("question")
