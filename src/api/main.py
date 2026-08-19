"""
FastAPI application — the web server.

This is the entry point for the web application.
Start with: make dev (or: uv run uvicorn src.api.main:app --reload)

Endpoints:
  GET  /                    → Frontend UI
  POST /api/query           → Ask a question to the RAG system
  GET  /api/models          → List available models
  GET  /api/health          → Health check
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.routes.models import router as models_router
from src.api.routes.query import router as query_router

logger = logging.getLogger(__name__)

# Path to frontend files
WEB_DIR = Path(__file__).parent.parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger.info("🚀 RAG Performance Analytics starting up...")
    yield
    logger.info("👋 Shutting down...")


app = FastAPI(
    title="LLM RAG Performance Analytics",
    description="Performance benchmark and evaluation of local vs. cloud LLMs using RAG",
    version="0.1.0",
    lifespan=lifespan,
)

# Allow frontend to call API (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Routes ---
app.include_router(query_router, prefix="/api")
app.include_router(models_router, prefix="/api")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "rag-analytics"}


# --- Serve Frontend ---
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="frontend")


def start():
    """Entry point for `uv run rag-analytics`."""
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
