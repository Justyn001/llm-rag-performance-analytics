"""
Application settings loaded from environment variables.

HOW THIS WORKS:
- Pydantic Settings automatically reads from .env file
- Each field = one environment variable (see .env.example)
- Type validation happens automatically (wrong type = clear error)
- You use it like: settings = get_settings(); print(settings.openai_api_key)
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central configuration for the entire application.

    All values come from environment variables (or .env file).
    Field names are CASE-INSENSITIVE and map to env var names.
    Example: openai_api_key -> OPENAI_API_KEY
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unknown env vars
    )

    # --- API Keys ---
    openai_api_key: str = ""
    google_api_key: str = ""
    mistral_api_key: str = ""

    # --- Qdrant (Vector Database) ---
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # --- Ollama (Local LLMs) ---
    ollama_host: str = "http://localhost:11434"

    # --- Application ---
    log_level: str = "INFO"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    @property
    def qdrant_url(self) -> str:
        """Full Qdrant connection URL."""
        return f"http://{self.qdrant_host}:{self.qdrant_port}"

    def has_openai(self) -> bool:
        """Check if OpenAI API key is configured."""
        return bool(self.openai_api_key and self.openai_api_key != "sk-your-openai-key-here")

    def has_google(self) -> bool:
        """Check if Google Gemini API key is configured."""
        return bool(self.google_api_key and self.google_api_key != "your-google-api-key-here")

    def has_mistral(self) -> bool:
        """Check if Mistral API key is configured."""
        return bool(self.mistral_api_key and self.mistral_api_key != "your-mistral-api-key-here")


@lru_cache
def get_settings() -> Settings:
    """
    Get application settings (cached — loaded once, reused everywhere).

    Usage:
        from src.config.settings import get_settings
        settings = get_settings()
        print(settings.openai_api_key)
    """
    return Settings()
