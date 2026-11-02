"""Shared fixtures/helpers for evaluation tests (not collected by pytest)."""

from __future__ import annotations

from typing import Any

from src.evaluation.storage import EvaluationStore


def make_metadata(run_id: str = "run-1") -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": "2026-01-01T00:00:00+00:00",
        "benchmark": "test_benchmark.json",
        "dataset": "wiki",
        "model_name": "Test Model",
        "model_provider": "ollama",
        "model_id": "test:1b",
        "rag_config": "basic-512",
        "top_k": 5,
        "language": "en",
        "judge_model": "Test Judge",
        "environment": {"python": "3.12"},
    }


def make_record(
    *,
    run_id: str = "run-1",
    question_id: str = "q-1",
    answerable: bool = True,
    answer: str = "Paris",
    error: str | None = None,
    judge: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expected = "Paris" if answerable else None
    return {
        "run_id": run_id,
        "question_id": question_id,
        "dataset": "wiki",
        "split": "custom",
        "set_type": "custom",
        "source": "test",
        "question": f"Question {question_id}?",
        "answerable": answerable,
        "expected_answer": expected,
        "gold_context": "Paris is the capital of France." if answerable else None,
        "gold_span": "Paris" if answerable else None,
        "gold_title": "France" if answerable else None,
        "gold_url": None,
        "answer_aliases": ["Paris"] if answerable else [],
        "answer": answer,
        "contexts": ["Paris is the capital of France."] if answerable else ["No evidence."],
        "context_titles": ["France"] if answerable else ["Nowhere"],
        "retrieval_time_ms": 1.0,
        "generation_time_ms": 2.0,
        "total_time_ms": 3.0,
        "tokens_in": 10,
        "tokens_out": 2,
        "tokens_per_second": 1.0,
        "exact_match": (1.0 if answer == "Paris" else 0.0) if answerable else None,
        "f1": (1.0 if answer == "Paris" else 0.0) if answerable else None,
        "is_refusal": answer.startswith("I don't"),
        "refusal_correct": answer.startswith("I don't") if not answerable else None,
        "automatic_correct": ((answer == "Paris") if answerable else answer.startswith("I don't")),
        "refusal_pattern": None,
        "gold_title_rank": 1 if answerable else None,
        "retrieval_hit": True if answerable else None,
        "gold_span_in_context": True if answerable else None,
        "judge": judge
        if judge is not None
        else {
            "correct": 1,
            "faithful": 1,
            "context_relevant": 1,
            "refusal_appropriate": 1,
            "reason": "test",
            "error": None,
        },
        "error": error,
    }


def seed_run(
    store: EvaluationStore,
    *,
    run_id: str = "run-1",
    records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    store.start_run(make_metadata(run_id))
    seeded = records or [make_record(run_id=run_id)]
    for record in seeded:
        store.append(record)
    return seeded
