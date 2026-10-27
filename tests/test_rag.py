"""
Tests for the RAG pipeline components.
"""

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
