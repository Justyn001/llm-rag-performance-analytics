"""
Tests for benchmark schema, gold evidence and provenance.

These tests do not require Qdrant or network access.  Corpus alignment is
checked separately with:
    uv run python -m src.evaluation.benchmarks --validate --check-corpus
"""

import pytest

from src.evaluation.benchmarks import (
    BENCHMARKS_DIR,
    BenchmarkQuestion,
    extract_gold_context,
    load_benchmark,
    validate_all,
)
from src.evaluation.build_benchmarks import _question_matches_evidence


@pytest.fixture(scope="module")
def benchmarks():
    return validate_all()


def test_every_benchmark_file_validates(benchmarks):
    assert {path.name for path in benchmarks} == {
        "squad_benchmark.json",
        "trivia_benchmark.json",
        "wikipedia_simple_benchmark.json",
    }


def test_expected_set_sizes(benchmarks):
    wiki = benchmarks[BENCHMARKS_DIR / "wikipedia_simple_benchmark.json"]
    squad = benchmarks[BENCHMARKS_DIR / "squad_benchmark.json"]
    trivia = benchmarks[BENCHMARKS_DIR / "trivia_benchmark.json"]

    assert len(wiki) == 40
    assert sum(question.answerable for question in wiki) == 30

    assert len(squad) == 50
    assert sum(question.answerable for question in squad) == 40

    assert len(trivia) == 50
    assert all(question.answerable for question in trivia)


def test_official_sets_use_validation_split(benchmarks):
    for filename in ("squad_benchmark.json", "trivia_benchmark.json"):
        questions = benchmarks[BENCHMARKS_DIR / filename]
        assert all(question.split == "validation" for question in questions)
        assert all(question.set_type == "official" for question in questions)
        assert all(question.question_id for question in questions)


def test_answerable_questions_have_exact_gold_span(benchmarks):
    for questions in benchmarks.values():
        for question in questions:
            if not question.answerable:
                continue
            assert question.expected_answer
            assert question.gold_context
            assert question.gold_span
            assert question.gold_title
            assert question.gold_span in question.gold_context
            assert len(question.gold_context) >= 20


def test_unanswerable_questions_have_no_gold_evidence(benchmarks):
    for questions in benchmarks.values():
        for question in questions:
            if question.answerable:
                continue
            assert question.expected_answer is None
            assert question.gold_context is None
            assert question.gold_span is None
            assert question.gold_title is None
            assert question.answer_aliases == []


def test_wiki_set_is_labelled_custom(benchmarks):
    questions = benchmarks[BENCHMARKS_DIR / "wikipedia_simple_benchmark.json"]
    assert all(question.dataset == "wiki" for question in questions)
    assert all(question.split == "custom" for question in questions)
    assert all(question.set_type == "custom" for question in questions)


def test_question_ids_are_unique_within_each_file(benchmarks):
    for questions in benchmarks.values():
        ids = [question.question_id for question in questions]
        assert len(ids) == len(set(ids))


def test_answerable_question_requires_gold_context():
    with pytest.raises(ValueError, match="missing"):
        BenchmarkQuestion(
            question_id="broken-1",
            question="What is the capital of France?",
            answerable=True,
            dataset="wiki",
            split="custom",
            set_type="custom",
            source="test",
        )


def test_unanswerable_question_rejects_gold_context():
    with pytest.raises(ValueError, match="gold fields"):
        BenchmarkQuestion(
            question_id="broken-2",
            question="Who is the current king of France?",
            answerable=False,
            gold_context="France is a country in Europe.",
            dataset="wiki",
            split="custom",
            set_type="custom",
            source="test",
        )


def test_gold_span_must_be_inside_context():
    with pytest.raises(ValueError, match="exact substring"):
        BenchmarkQuestion(
            question_id="broken-3",
            question="What is the population of Cuba?",
            answerable=True,
            expected_answer="13 million",
            gold_context="Cuba is an island in the Caribbean.",
            gold_span="13 million",
            dataset="wiki",
            split="custom",
            set_type="custom",
            source="test",
            gold_title="Cuba",
        )


def test_extract_gold_context_uses_whole_words():
    text = "A blood type is a classification. People with type O are common donors."
    context, span = extract_gold_context(text, "O")

    assert span == "O"
    assert span in context
    assert "blood type" in context


def test_extract_gold_context_preserves_exact_text():
    text = "The population of Cuba is close to 13 million. The people are diverse."
    context, span = extract_gold_context(text, "13 million")

    assert span == "13 million"
    assert context in text
    assert span in context


def test_question_matches_evidence_rejects_unrelated_article():
    assert not _question_matches_evidence(
        "What colour is the cross on the national flag of Switzerland?",
        "Colours, standards and guidons",
        "Brazil carries a standard that is white with the Army coat of arms.",
    )


def test_question_matches_evidence_accepts_named_subject():
    assert _question_matches_evidence(
        "Who was Luigina Giavotti?",
        "Luigina Giavotti",
        "Luigina Giavotti was an Italian gymnast.",
    )


def test_benchmark_files_are_reloadable():
    path = BENCHMARKS_DIR / "wikipedia_simple_benchmark.json"
    first = load_benchmark(path)
    second = load_benchmark(path)

    assert [question.question_id for question in first] == [
        question.question_id for question in second
    ]
