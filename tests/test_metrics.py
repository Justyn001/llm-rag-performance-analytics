"""Tests for deterministic answer/retrieval metrics."""

from __future__ import annotations

import pytest

from src.evaluation.metrics import (
    best_answer_scores,
    compute_answer_metrics,
    compute_retrieval_metrics,
    detect_refusal,
    exact_match,
    normalize_answer,
    token_f1,
)


def test_normalize_answer_removes_case_punctuation_and_articles():
    assert normalize_answer("The  Paris!") == "paris"
    assert normalize_answer("A don't-care answer.") == "dontcare answer"
    assert normalize_answer("13 million") == "13 million"
    assert normalize_answer("") == ""


def test_exact_match_uses_normalization():
    assert exact_match("The Paris!", "paris") == 1.0
    assert exact_match("Paris", "London") == 0.0
    assert exact_match("", "") == 1.0


def test_token_f1_is_squad_style():
    assert token_f1("Paris", "Paris") == 1.0
    assert token_f1("Paris France", "Paris") == pytest.approx(2 / 3)
    assert token_f1("London", "Paris") == 0.0
    assert token_f1("", "") == 1.0
    assert token_f1("", "Paris") == 0.0


def test_best_answer_scores_uses_best_alias():
    em, f1 = best_answer_scores("Paris, France", ["Paris", "Paris, France"])
    assert em == 1.0
    assert f1 == 1.0


def test_refusal_detection_english_and_polish():
    assert detect_refusal("I don't have enough information to answer this question.").is_refusal
    assert detect_refusal("I cannot answer that.").is_refusal
    assert detect_refusal("Nie mam wystarczających informacji.").is_refusal
    assert detect_refusal("Nie wiem.").is_refusal
    assert detect_refusal("").is_refusal
    assert not detect_refusal("Paris is the capital of France.").is_refusal


def test_answer_metrics_for_answerable_question():
    metrics = compute_answer_metrics(
        "Paris",
        answerable=True,
        references=["Paris", "Paris, France"],
    )

    assert metrics.exact_match == 1.0
    assert metrics.f1 == 1.0
    assert metrics.is_refusal is False
    assert metrics.refusal_correct is None
    assert metrics.automatic_correct is True


def test_answer_metrics_for_unanswerable_control():
    metrics = compute_answer_metrics(
        "I don't have enough information to answer this question.",
        answerable=False,
        references=[],
    )

    assert metrics.exact_match is None
    assert metrics.f1 is None
    assert metrics.is_refusal is True
    assert metrics.refusal_correct is True
    assert metrics.automatic_correct is True


def test_unanswerable_control_that_answers_is_wrong():
    metrics = compute_answer_metrics("Paris", answerable=False, references=[])

    assert metrics.is_refusal is False
    assert metrics.refusal_correct is False
    assert metrics.automatic_correct is False


def test_answerable_question_requires_references():
    with pytest.raises(ValueError, match="reference"):
        compute_answer_metrics("Paris", answerable=True, references=[])


def test_retrieval_metrics_find_title_and_span():
    metrics = compute_retrieval_metrics(
        ["Paris is the capital of France.", "France is in Europe."],
        ["France", "Europe"],
        gold_title="France",
        gold_span="capital of France",
    )

    assert metrics.gold_title_rank == 1
    assert metrics.retrieval_hit is True
    assert metrics.gold_span_in_context is True


def test_retrieval_metrics_use_word_boundaries_for_short_spans():
    false_positive = compute_retrieval_metrics(
        ["People are diverse."],
        ["Cuba"],
        gold_title="Cuba",
        gold_span="O",
    )
    true_positive = compute_retrieval_metrics(
        ["Cuba has blood type O in this example."],
        ["Cuba"],
        gold_title="Cuba",
        gold_span="O",
    )

    assert false_positive.gold_span_in_context is False
    assert true_positive.gold_span_in_context is True


def test_retrieval_metrics_report_missing_evidence():
    metrics = compute_retrieval_metrics(
        ["Berlin is the capital of Germany."],
        ["Germany"],
        gold_title="France",
        gold_span="capital of France",
    )

    assert metrics.gold_title_rank is None
    assert metrics.retrieval_hit is False
    assert metrics.gold_span_in_context is False


def test_retrieval_metrics_are_unavailable_for_controls():
    metrics = compute_retrieval_metrics(
        ["Anything"],
        ["Nowhere"],
        gold_title=None,
        gold_span=None,
    )

    assert metrics.gold_title_rank is None
    assert metrics.retrieval_hit is None
    assert metrics.gold_span_in_context is None
