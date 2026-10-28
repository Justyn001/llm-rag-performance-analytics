"""
Test sets endpoint — curated question sets for benchmarking.

GET /api/test-sets          → List available test sets
GET /api/test-sets/{name}   → Get questions from a specific test set
"""

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

# Project root → data/benchmarks/
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
BENCHMARKS_DIR = PROJECT_ROOT / "data" / "benchmarks"


class TestQuestion(BaseModel):
    """A single test question with expected answer."""

    question: str
    expected_answer: str
    source: str
    answerable: bool = True


class TestSet(BaseModel):
    """A set of test questions."""

    name: str
    display_name: str
    description: str
    total_available: int
    questions: list[TestQuestion]


class TestSetSummary(BaseModel):
    """Summary of a test set (without questions)."""

    name: str
    display_name: str
    description: str
    total_available: int


class TestSetsResponse(BaseModel):
    """List of available test sets."""

    test_sets: list[TestSetSummary]


# ---- Registry of benchmark files ----
TEST_SET_REGISTRY = {
    "wikipedia-nq": {
        "display_name": "Wikipedia + NQ",
        "description": (
            "50 curated questions from Simple Wikipedia KB (40 answerable + 10 control)"
        ),
        "file": "wikipedia_nq_benchmark.json",
        "dataset": "wiki",
    },
    "squad": {
        "display_name": "SQuAD Science",
        "description": (
            "50 curated science & history questions from "
            "SQuAD 2.0 dataset (40 answerable + 10 control)"
        ),
        "file": "squad_benchmark.json",
        "dataset": "squad",
    },
    "trivia": {
        "display_name": "TriviaQA",
        "description": (
            "50 curated trivia questions from TriviaQA dataset (40 answerable + 10 control)"
        ),
        "file": "trivia_benchmark.json",
        "dataset": "trivia",
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
        data = _load_benchmark(info["file"])
        summaries.append(
            TestSetSummary(
                name=name,
                display_name=info["display_name"],
                description=info["description"],
                total_available=len(data),
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
        name: Test set name (e.g., 'wikipedia-nq')
        count: Number of questions to return (default 10, max 50)
    """
    if name not in TEST_SET_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Test set '{name}' not found",
        )

    info = TEST_SET_REGISTRY[name]
    data = _load_benchmark(info["file"])
    if not data:
        raise HTTPException(
            status_code=404,
            detail="Benchmark file not found.",
        )

    count = min(count, len(data))

    # Fixed order — always the same questions for reproducibility
    questions = [
        TestQuestion(
            question=q["question"],
            expected_answer=q["expected_answer"],
            source=q["source"],
            answerable=q.get("answerable", True),
        )
        for q in data[:count]
    ]

    return TestSet(
        name=name,
        display_name=info["display_name"],
        description=info["description"],
        total_available=len(data),
        questions=questions,
    )
