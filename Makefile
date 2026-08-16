# =============================================================================
# Makefile — Common project commands
# =============================================================================
# Instead of remembering long commands, you just type:
#   make setup    — first-time setup
#   make dev      — start the app
#   make test     — run tests
#   make lint     — check code quality
# =============================================================================

.PHONY: setup dev test lint format docker-up docker-down pull-models clean help

# --- First-time setup ---
setup:
	uv sync --all-groups
	cp -n .env.example .env 2>/dev/null || true
	@echo "✅ Setup complete! Edit .env with your API keys, then run: make docker-up"

# --- Development ---
dev:
	uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# --- Docker services (Qdrant + Ollama) ---
docker-up:
	docker compose up -d
	@echo "✅ Qdrant: http://localhost:6333/dashboard"
	@echo "✅ Ollama: http://localhost:11434"

docker-down:
	docker compose down

# --- Pull local LLM models (run after docker-up) ---
pull-models:
	docker compose exec ollama ollama pull llama3.1:8b
	docker compose exec ollama ollama pull mistral:7b
	docker compose exec ollama ollama pull gemma2:9b
	docker compose exec ollama ollama pull qwen3:8b
	@echo "✅ All models pulled!"

# --- Code quality ---
lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

# --- Tests ---
test:
	uv run pytest tests/ -v

# --- Cleanup ---
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true

# --- Help ---
help:
	@echo "Available commands:"
	@echo "  make setup        — Install dependencies + create .env"
	@echo "  make dev          — Start FastAPI dev server"
	@echo "  make docker-up    — Start Qdrant + Ollama containers"
	@echo "  make docker-down  — Stop Docker containers"
	@echo "  make pull-models  — Download all local LLM models"
	@echo "  make lint         — Check code style"
	@echo "  make format       — Auto-format code"
	@echo "  make test         — Run tests"
	@echo "  make clean        — Remove cache files"
