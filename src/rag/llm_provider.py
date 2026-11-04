"""
LLM Provider — unified interface for local and cloud models.

This is the ABSTRACTION LAYER that lets us swap models easily:
  - Same code works with Ollama (local) and OpenAI/Gemini/Mistral (cloud)
  - Each model returns a LangChain LLM object
  - Used by the RAG pipeline to generate answers

Example:
    llm = get_llm(LLAMA_3_1_8B)        # local model
    llm = get_llm(GPT_4O_MINI)         # cloud model
    # Both have the same interface!
"""

import logging

from langchain_core.language_models import BaseChatModel

from src.config.models import ModelConfig, ModelProvider
from src.config.settings import get_settings

logger = logging.getLogger(__name__)


def get_llm(
    model: ModelConfig,
    *,
    max_tokens: int | None = None,
    num_ctx: int | None = None,
) -> BaseChatModel:
    """
    Get a LangChain LLM instance for the given model config.

    Args:
        model: A ModelConfig from src.config.models (e.g. LLAMA_3_1_8B, GPT_4O_MINI)
        max_tokens: Protocol-wide output limit; overrides ``model.max_tokens``.
        num_ctx: Ollama context window. Cloud providers keep their own fixed
            context handling and ignore this value.

    Returns:
        LangChain chat model — same interface regardless of provider

    Example:
        from src.config.models import LLAMA_3_1_8B, GPT_4O_MINI
        local_llm = get_llm(LLAMA_3_1_8B, max_tokens=1024, num_ctx=8192)
        cloud_llm = get_llm(GPT_4O_MINI, max_tokens=1024)
    """
    effective_max_tokens = max_tokens if max_tokens is not None else model.max_tokens
    logger.info(
        "Loading LLM: %s (provider=%s, max_tokens=%s, num_ctx=%s)",
        model.name,
        model.provider,
        effective_max_tokens,
        num_ctx,
    )

    if model.provider == ModelProvider.OLLAMA:
        return _get_ollama_llm(model, max_tokens=effective_max_tokens, num_ctx=num_ctx)
    if model.provider == ModelProvider.GROQ:
        return _get_groq_llm(model, max_tokens=effective_max_tokens)
    if model.provider == ModelProvider.GOOGLE:
        return _get_google_llm(model, max_tokens=effective_max_tokens)
    if model.provider == ModelProvider.MISTRAL:
        return _get_mistral_llm(model, max_tokens=effective_max_tokens)

    msg = f"Unknown provider: {model.provider}"
    raise ValueError(msg)


def supports_num_ctx(model: ModelConfig) -> bool:
    """Whether the provider exposes a context-window setting we can control."""
    return model.provider == ModelProvider.OLLAMA


def _get_ollama_llm(
    model: ModelConfig,
    *,
    max_tokens: int,
    num_ctx: int | None,
) -> BaseChatModel:
    """
    Local LLM via Ollama (runs on your GPU).

    Ollama must be running: ollama serve
    Model must be pulled: ollama pull llama3.1:8b
    """
    from langchain_ollama import ChatOllama

    settings = get_settings()

    return ChatOllama(
        model=model.model_id,
        base_url=settings.ollama_host,
        temperature=model.temperature,
        num_predict=max_tokens,
        num_ctx=num_ctx,
    )


def _get_groq_llm(model: ModelConfig, *, max_tokens: int) -> BaseChatModel:
    """Cloud LLM via Groq API (free, ultra-fast inference)."""
    from langchain_groq import ChatGroq

    settings = get_settings()
    if not settings.has_groq():
        msg = "Groq API key not configured. Set GROQ_API_KEY in .env"
        raise ValueError(msg)

    return ChatGroq(
        model=model.model_id,
        api_key=settings.groq_api_key,
        temperature=model.temperature,
        max_tokens=max_tokens,
        reasoning_format="hidden",
    )


def _get_google_llm(model: ModelConfig, *, max_tokens: int) -> BaseChatModel:
    """Cloud LLM via Google Gemini API (free tier)."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    settings = get_settings()
    if not settings.has_google():
        msg = "Google API key not configured. Set GOOGLE_API_KEY in .env"
        raise ValueError(msg)

    return ChatGoogleGenerativeAI(
        model=model.model_id,
        google_api_key=settings.google_api_key,
        temperature=model.temperature,
        max_output_tokens=max_tokens,
    )


def _get_mistral_llm(model: ModelConfig, *, max_tokens: int) -> BaseChatModel:
    """Cloud LLM via Mistral AI API (free tier)."""
    from langchain_mistralai import ChatMistralAI

    settings = get_settings()
    if not settings.has_mistral():
        msg = "Mistral API key not configured. Set MISTRAL_API_KEY in .env"
        raise ValueError(msg)

    return ChatMistralAI(
        model=model.model_id,
        api_key=settings.mistral_api_key,
        temperature=model.temperature,
        max_tokens=max_tokens,
    )


def list_available_models() -> list[ModelConfig]:
    """
    List models that are currently available (have API keys or Ollama running).

    Returns:
        List of ModelConfig objects that can be used right now
    """
    from src.config.models import ALL_MODELS

    settings = get_settings()
    available = []

    for model in ALL_MODELS:
        if model.provider == ModelProvider.OLLAMA:
            # Check if Ollama is reachable
            try:
                import httpx

                resp = httpx.get(f"{settings.ollama_host}/api/tags", timeout=2.0)
                if resp.status_code == 200:
                    # Check if this specific model is pulled
                    pulled = [m["name"] for m in resp.json().get("models", [])]
                    # Ollama model names can have :latest suffix
                    model_names = [model.model_id, f"{model.model_id}:latest"]
                    if any(name in pulled for name in model_names):
                        available.append(model)
            except Exception:
                pass

        else:
            # Cloud models — check if API key is configured
            provider_checks = {
                ModelProvider.GROQ: settings.has_groq,
                ModelProvider.GOOGLE: settings.has_google,
                ModelProvider.MISTRAL: settings.has_mistral,
            }
            checker = provider_checks.get(model.provider)
            if checker and checker():
                available.append(model)

    return available
