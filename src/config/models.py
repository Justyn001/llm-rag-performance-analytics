"""
Model configuration — defines all LLM models used in experiments.

This file is the single source of truth for:
- Which models we test (local and cloud)
- Their parameters (temperature, max_tokens, etc.)
- How to identify them in benchmarks

Final selection: 4 local + 4 cloud = 8 models
- Size range: 4B → 109B MoE (local), cloud APIs (Google, Mistral, Groq)
- Llama 4 Scout runs both locally and on Groq → direct local vs cloud comparison
"""

from enum import StrEnum


class ModelProvider(StrEnum):
    """Where the model runs."""

    OLLAMA = "ollama"  # Local models via Ollama
    GROQ = "groq"  # Groq API (free, ultra-fast inference)
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
# LOCAL MODELS (via Ollama — runs on RTX 5070 Ti, 16GB VRAM + 32GB RAM)
# =============================================================================

GEMMA_3_4B = ModelConfig(
    name="Gemma 3 4B",
    provider=ModelProvider.OLLAMA,
    model_id="gemma3:4b",
    description="Google Gemma 3 — newest, lightweight 4B",
    is_local=True,
    parameters=4_000_000_000,
)

QWEN_3_8B = ModelConfig(
    name="Qwen 3 8B",
    provider=ModelProvider.OLLAMA,
    model_id="qwen3:8b",
    description="Alibaba Qwen 3 — great multilingual, reasoning",
    is_local=True,
    parameters=8_000_000_000,
    max_tokens=2048,
)

MISTRAL_NEMO = ModelConfig(
    name="Mistral NeMo 12B",
    provider=ModelProvider.OLLAMA,
    model_id="mistral-nemo",
    description="Mistral AI + NVIDIA — 12B, 128k context",
    is_local=True,
    parameters=12_000_000_000,
)

LLAMA_4_SCOUT = ModelConfig(
    name="Llama 4 Scout",
    provider=ModelProvider.OLLAMA,
    model_id="llama4",
    description="Meta Llama 4 Scout — 17B active MoE (109B total)",
    is_local=True,
    parameters=109_000_000_000,
)

# =============================================================================
# CLOUD MODELS (via API — free tier)
# =============================================================================

GEMINI_2_5_FLASH = ModelConfig(
    name="Gemini 2.5 Flash",
    provider=ModelProvider.GOOGLE,
    model_id="gemini-2.5-flash",
    description="Google Gemini 2.5 Flash — free tier, latest",
    is_local=False,
)

GROQ_GPT_OSS_20B = ModelConfig(
    name="Groq GPT-OSS 20B",
    provider=ModelProvider.GROQ,
    model_id="openai/gpt-oss-20b",
    description="OpenAI GPT-OSS 20B on Groq — free, smaller open-source",
    is_local=False,
)

GROQ_QWEN_3_8 = ModelConfig(
    name="Groq Qwen 3.8 27B",
    provider=ModelProvider.GROQ,
    model_id="qwen/qwen3.8-27b",
    description="Qwen 3.8 27B on Groq — free, newest Qwen",
    is_local=False,
    max_tokens=2048,
)

GROQ_GPT_OSS_120B = ModelConfig(
    name="Groq GPT-OSS 120B",
    provider=ModelProvider.GROQ,
    model_id="openai/gpt-oss-120b",
    description="OpenAI GPT-OSS 120B on Groq — free, large open-source",
    is_local=False,
)

# =============================================================================
# ALL MODELS — use this in experiments
# =============================================================================

LOCAL_MODELS = [
    GEMMA_3_4B,
    QWEN_3_8B,
    MISTRAL_NEMO,
    LLAMA_4_SCOUT,
]

CLOUD_MODELS = [
    GEMINI_2_5_FLASH,
    GROQ_GPT_OSS_20B,
    GROQ_QWEN_3_8,
    GROQ_GPT_OSS_120B,
]

ALL_MODELS = LOCAL_MODELS + CLOUD_MODELS
