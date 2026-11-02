"""
Automatic answer and retrieval metrics.

These metrics are deliberately deterministic and offline:

- exact match / token F1 use the SQuAD normalization rules,
- refusal detection covers the English and Polish prompts used by the app,
- retrieval checks use the gold title/span that every benchmark question
  already carries.

The LLM judge lives in ``src.evaluation.judge`` and adds semantic correctness,
faithfulness and context relevance on top of these rule-based metrics.
"""

from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter
from dataclasses import dataclass


def normalize_answer(text: str) -> str:
    """Lower-case, remove punctuation/articles and collapse whitespace."""
    if not text:
        return ""

    text = text.lower()
    text = "".join(
        char
        for char in text
        if char not in string.punctuation and not unicodedata.category(char).startswith("P")
    )
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> float:
    """Return 1.0 when the normalized prediction equals the reference."""
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style token F1 between one prediction and one reference."""
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()

    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return (2 * precision * recall) / (precision + recall)


def best_answer_scores(prediction: str, references: list[str]) -> tuple[float, float]:
    """Return the best exact match and F1 over all accepted references."""
    refs = [reference for reference in references if reference and reference.strip()]
    if not refs:
        return 0.0, 0.0

    return (
        max(exact_match(prediction, reference) for reference in refs),
        max(token_f1(prediction, reference) for reference in refs),
    )


_REFUSAL_PATTERNS = [
    re.compile(r"\bi\s+(?:do\s+not|don'?t)\s+have\s+enough\s+information\b", re.IGNORECASE),
    re.compile(r"\bnot\s+enough\s+information\b", re.IGNORECASE),
    re.compile(r"\binsufficient\s+information\b", re.IGNORECASE),
    re.compile(r"\b(?:cannot|can'?t|unable\s+to)\s+answer\b", re.IGNORECASE),
    re.compile(r"\bi\s+(?:do\s+not|don'?t)\s+know\b", re.IGNORECASE),
    re.compile(r"\bno\s+information\b", re.IGNORECASE),
    re.compile(r"\bnot\s+(?:mentioned|provided|specified)\b", re.IGNORECASE),
    re.compile(r"\bnie\s+mam\s+wystarczających\s+informacji\b", re.IGNORECASE),
    re.compile(r"\bbrak\s+(?:wystarczających\s+)?informacji\b", re.IGNORECASE),
    re.compile(r"\bnie\s+wiem\b", re.IGNORECASE),
    re.compile(r"\bnie\s+można\s+odpowiedzieć\b", re.IGNORECASE),
    re.compile(r"\bnie\s+podano\b", re.IGNORECASE),
]


@dataclass(frozen=True)
class RefusalDetection:
    """Result of checking whether an answer is an abstention."""

    is_refusal: bool
    matched_pattern: str | None = None


def detect_refusal(answer: str) -> RefusalDetection:
    """Detect an explicit refusal in English or Polish."""
    if not answer or not answer.strip():
        return RefusalDetection(is_refusal=True, matched_pattern="empty answer")

    for pattern in _REFUSAL_PATTERNS:
        match = pattern.search(answer)
        if match:
            return RefusalDetection(is_refusal=True, matched_pattern=match.group(0))

    return RefusalDetection(is_refusal=False)


@dataclass(frozen=True)
class AnswerMetrics:
    """Rule-based metrics for one answer."""

    exact_match: float | None
    f1: float | None
    is_refusal: bool
    refusal_correct: bool | None
    automatic_correct: bool | None
    refusal_pattern: str | None = None


def compute_answer_metrics(
    answer: str,
    *,
    answerable: bool,
    references: list[str],
) -> AnswerMetrics:
    """
    Compute EM/F1 for answerable questions and refusal accuracy for controls.

    ``automatic_correct`` is exact match for answerable questions and a correct
    refusal for unanswerable ones.  It is intentionally strict; the LLM judge
    and the manual rubric capture semantic equivalence.
    """
    refusal = detect_refusal(answer)

    if not answerable:
        return AnswerMetrics(
            exact_match=None,
            f1=None,
            is_refusal=refusal.is_refusal,
            refusal_correct=refusal.is_refusal,
            automatic_correct=refusal.is_refusal,
            refusal_pattern=refusal.matched_pattern,
        )

    if not references:
        msg = "answerable question must have at least one reference answer"
        raise ValueError(msg)

    em, f1 = best_answer_scores(answer, references)
    return AnswerMetrics(
        exact_match=em,
        f1=f1,
        is_refusal=refusal.is_refusal,
        refusal_correct=None,
        automatic_correct=em == 1.0,
        refusal_pattern=refusal.matched_pattern,
    )


@dataclass(frozen=True)
class RetrievalMetrics:
    """Cheap retrieval diagnostics based on gold evidence."""

    gold_title_rank: int | None
    retrieval_hit: bool | None
    gold_span_in_context: bool | None


def compute_retrieval_metrics(
    contexts: list[str],
    context_titles: list[str],
    *,
    gold_title: str | None,
    gold_span: str | None,
) -> RetrievalMetrics:
    """Check whether the retrieved chunks contain the gold document/span."""
    if not gold_title:
        return RetrievalMetrics(
            gold_title_rank=None,
            retrieval_hit=None,
            gold_span_in_context=None,
        )

    normalized_title = normalize_answer(gold_title)
    rank = next(
        (
            index
            for index, title in enumerate(context_titles, start=1)
            if normalize_answer(title) == normalized_title
        ),
        None,
    )

    joined_context = normalize_answer(" ".join(contexts))
    normalized_span = normalize_answer(gold_span or "")
    span_present = bool(normalized_span) and bool(
        re.search(rf"(?<!\w){re.escape(normalized_span)}(?!\w)", joined_context)
    )

    return RetrievalMetrics(
        gold_title_rank=rank,
        retrieval_hit=rank is not None,
        gold_span_in_context=span_present,
    )
