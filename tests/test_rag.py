"""
Tests for the RAG pipeline components.
"""

from src.config.models import (
    GEMINI_FLASH,
    GROQ_LLAMA,
    LLAMA_3_1_8B,
    MISTRAL_CLOUD,
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

    def test_llama_uses_ollama(self):
        assert LLAMA_3_1_8B.provider == ModelProvider.OLLAMA
        assert LLAMA_3_1_8B.model_id == "llama3.1:8b"

    def test_groq_uses_groq(self):
        assert GROQ_LLAMA.provider == ModelProvider.GROQ
        assert GROQ_LLAMA.model_id == "llama-3.3-70b-versatile"

    def test_gemini_uses_google(self):
        assert GEMINI_FLASH.provider == ModelProvider.GOOGLE

    def test_mistral_cloud_uses_mistral(self):
        assert MISTRAL_CLOUD.provider == ModelProvider.MISTRAL
