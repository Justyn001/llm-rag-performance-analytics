"""
Smoke test — verifies that the project is set up correctly.

Run with: make test  (or: uv run pytest tests/ -v)
"""

from src.config.models import ALL_MODELS, CLOUD_MODELS, LOCAL_MODELS, ModelProvider
from src.config.settings import Settings


class TestSettings:
    """Test that settings load correctly."""

    def test_settings_can_be_created(self):
        """Settings should initialize with defaults even without .env file."""
        settings = Settings()
        assert settings.qdrant_host == "localhost"
        assert settings.qdrant_port == 6333
        assert settings.log_level == "INFO"

    def test_qdrant_url_property(self):
        """Qdrant URL should be constructed from host and port."""
        settings = Settings()
        assert settings.qdrant_url == "http://localhost:6333"


class TestModelConfig:
    """Test that model configurations are correct."""

    def test_has_four_local_models(self):
        """We should have exactly 4 local models."""
        assert len(LOCAL_MODELS) == 4

    def test_has_three_cloud_models(self):
        """We should have exactly 3 cloud models."""
        assert len(CLOUD_MODELS) == 3

    def test_all_models_total(self):
        """Total should be 7 models (4 local + 3 cloud)."""
        assert len(ALL_MODELS) == 7

    def test_local_models_use_ollama(self):
        """All local models should use Ollama provider."""
        for model in LOCAL_MODELS:
            assert model.provider == ModelProvider.OLLAMA
            assert model.is_local is True

    def test_cloud_models_are_not_local(self):
        """All cloud models should not be marked as local."""
        for model in CLOUD_MODELS:
            assert model.is_local is False
