"""
Test sets endpoint — curated question sets for benchmarking.

GET /api/test-sets          → List available test sets
GET /api/test-sets/{name}   → Get questions from a specific test set
"""

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()

# Project root → data/benchmarks/
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
BENCHMARKS_DIR = PROJECT_ROOT / "data" / "benchmarks"


class TestQuestion(BaseModel):
    """A single test question with its gold evidence and provenance."""

    question_id: str
    question: str
    expected_answer: str | None = None
    gold_context: str | None = None
    gold_span: str | None = None
    dataset: str
    split: str
    set_type: str
    source: str
    gold_title: str | None = None
    gold_url: str | None = None
    answer_aliases: list[str] = Field(default_factory=list)
    answerable: bool = True


class TestSet(BaseModel):
    """A set of test questions."""

    name: str
    display_name: str
    description: str
    total_available: int
    dataset: str
    questions: list[TestQuestion]


class TestSetSummary(BaseModel):
    """Summary of a test set (without questions)."""

    name: str
    display_name: str
    description: str
    total_available: int
    dataset: str


class TestSetsResponse(BaseModel):
    """List of available test sets."""

    test_sets: list[TestSetSummary]


# ---- Registry of benchmark files ----
# The wiki set is a custom benchmark whose gold evidence is extracted from
# the ingested Simple Wikipedia collection.  SQuAD and TriviaQA are built
# from official validation splits.  All three are enabled; the frontend only
# offers a set when its dataset has an ingested Qdrant collection (see
# /api/datasets), so a missing collection is hidden instead of producing
# misleading benchmark results.
TEST_SET_REGISTRY = {
    "wikipedia-simple": {
        "display_name": "Simple Wikipedia (custom)",
        "description": (
            "40 custom questions grounded in the ingested Simple Wikipedia corpus "
            "(30 answerable + 10 unanswerable controls)"
        ),
        "file": "wikipedia_simple_benchmark.json",
        "dataset": "wiki",
        "enabled": True,
    },
    "squad": {
        "display_name": "SQuAD 2.0",
        "description": (
            "50 questions from the official SQuAD 2.0 validation split "
            "(40 answerable + 10 unanswerable)"
        ),
        "file": "squad_benchmark.json",
        "dataset": "squad",
        "enabled": True,
    },
    "trivia": {
        "display_name": "TriviaQA",
        "description": (
            "50 answerable questions from the official TriviaQA rc.wikipedia validation split"
        ),
        "file": "trivia_benchmark.json",
        "dataset": "trivia",
        "enabled": True,
    },
}


def _load_benchmark(filename: str) -> list[dict]:
    """Load a benchmark JSON file from data/benchmarks/."""
    path = BENCHMARKS_DIR / filename
    if not path.exists():
        logger.warning(f"Benchmark file not found: {path}")
        return []
    with open(path) as f:
        return json.load(f)


@router.get("/test-sets", response_model=TestSetsResponse)
async def list_test_sets():
    """List all available test sets."""
    summaries = []
    for name, info in TEST_SET_REGISTRY.items():
        if not info.get("enabled", True):
            continue
        data = _load_benchmark(info["file"])
        summaries.append(
            TestSetSummary(
                name=name,
                display_name=info["display_name"],
                description=info["description"],
                total_available=len(data),
                dataset=info["dataset"],
            )
        )
    return TestSetsResponse(test_sets=summaries)


@router.get("/test-sets/{name}", response_model=TestSet)
async def get_test_set(name: str, count: int = 10):
    """
    Get questions from a test set.

    The set is always returned in the same order for
    reproducibility. Use `count` to limit how many.

    Args:
        name: Test set name (e.g., 'wikipedia-simple')
        count: Number of questions to return (default 10, max 50)
    """
    info = TEST_SET_REGISTRY.get(name)
    if info is None:
        raise HTTPException(
            status_code=404,
            detail=f"Test set '{name}' not found",
        )
    if not info.get("enabled", True):
        raise HTTPException(
            status_code=404,
            detail=f"Test set '{name}' is not enabled: {info['disabled_reason']}",
        )

    data = _load_benchmark(info["file"])
    if not data:
        raise HTTPException(
            status_code=404,
            detail="Benchmark file not found.",
        )

    count = max(1, min(count, len(data)))

    # Fixed order — always the same questions for reproducibility
    questions = [TestQuestion.model_validate(question) for question in data[:count]]

    return TestSet(
        name=name,
        display_name=info["display_name"],
        description=info["description"],
        total_available=len(data),
        dataset=info["dataset"],
        questions=questions,
    )
