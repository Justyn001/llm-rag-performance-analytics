"""
Model configuration — defines all LLM models used in experiments.

This file is the single source of truth for:
- Which models we test (local and cloud)
- Their parameters (temperature, max_tokens, etc.)
- How to identify them in benchmarks
"""

from enum import StrEnum


class ModelProvider(StrEnum):
    """Where the model runs."""

    OLLAMA = "ollama"  # Local models via Ollama
    OPENAI = "openai"  # OpenAI API (GPT-4o-mini)
    GOOGLE = "google"  # Google Gemini API
    MISTRAL = "mistral"  # Mistral AI API


class ModelConfig:
    """Configuration for a single LLM model."""

    def __init__(
        self,
        name: str,
        provider: ModelProvider,
        model_id: str,
        description: str,
        is_local: bool,
        parameters: int | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        self.name = name
        self.provider = provider
        self.model_id = model_id
        self.description = description
        self.is_local = is_local
        self.parameters = parameters
        self.temperature = temperature
        self.max_tokens = max_tokens

    def __repr__(self) -> str:
        location = "local" if self.is_local else "cloud"
        return f"ModelConfig({self.name}, {location}, {self.provider.value})"


# =============================================================================
# LOCAL MODELS (via Ollama — runs on your RTX 5070 Ti, 16GB VRAM)
# =============================================================================

LLAMA_3_1_8B = ModelConfig(
    name="Llama 3.1 8B",
    provider=ModelProvider.OLLAMA,
    model_id="llama3.1:8b",
    description="Meta's Llama 3.1 — best quality in the 8B class",
    is_local=True,
    parameters=8_000_000_000,
)

MISTRAL_7B = ModelConfig(
    name="Mistral 7B",
    provider=ModelProvider.OLLAMA,
    model_id="mistral:7b",
    description="Mistral AI's 7B model — fast and capable",
    is_local=True,
    parameters=7_000_000_000,
)

GEMMA_2_9B = ModelConfig(
    name="Gemma 2 9B",
    provider=ModelProvider.OLLAMA,
    model_id="gemma2:9b",
    description="Google's Gemma 2 — strong reasoning",
    is_local=True,
    parameters=9_000_000_000,
)

QWEN_3_8B = ModelConfig(
    name="Qwen 3 8B",
    provider=ModelProvider.OLLAMA,
    model_id="qwen3:8b",
    description="Alibaba's Qwen 3 — rising star, great multilingual",
    is_local=True,
    parameters=8_000_000_000,
)

# =============================================================================
# CLOUD MODELS (via API)
# =============================================================================

GEMINI_FLASH = ModelConfig(
    name="Gemini 2.5 Flash",
    provider=ModelProvider.GOOGLE,
    model_id="gemini-2.5-flash",
    description="Google's Gemini Flash — free tier, production-grade",
    is_local=False,
)

MISTRAL_CLOUD = ModelConfig(
    name="Mistral Small",
    provider=ModelProvider.MISTRAL,
    model_id="mistral-small-latest",
    description="Mistral AI cloud — free tier, EU-based",
    is_local=False,
)

GPT_4O_MINI = ModelConfig(
    name="GPT-4o-mini",
    provider=ModelProvider.OPENAI,
    model_id="gpt-4o-mini",
    description="OpenAI's cost-effective model — industry standard",
    is_local=False,
)

# =============================================================================
# ALL MODELS — use this in experiments
# =============================================================================

LOCAL_MODELS = [LLAMA_3_1_8B, MISTRAL_7B, GEMMA_2_9B, QWEN_3_8B]
CLOUD_MODELS = [GEMINI_FLASH, MISTRAL_CLOUD, GPT_4O_MINI]
ALL_MODELS = LOCAL_MODELS + CLOUD_MODELS
