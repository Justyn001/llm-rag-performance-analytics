"""
Benchmark schema and validation.

A benchmark question is only useful if its gold evidence is present in the
knowledge base that the RAG pipeline actually searches.  This module defines
that contract and validates every file in ``data/benchmarks`` against it.

Usage:
    uv run python -m src.evaluation.benchmarks --validate
    uv run python -m src.evaluation.benchmarks --validate --check-corpus
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROJECT_ROOT = Path(__file__).parent.parent.parent
BENCHMARKS_DIR = PROJECT_ROOT / "data" / "benchmarks"

DatasetName = Literal["wiki", "squad", "trivia"]
SetType = Literal["official", "custom"]

# Qdrant collection used by default for each dataset.  Benchmarks and the
# ingestion pipeline must agree on this suffix.
DEFAULT_COLLECTION_SUFFIX = 512


class BenchmarkValidationError(ValueError):
    """Raised when one or more benchmark files violate the schema."""


def _answer_pattern(answer: str) -> str:
    """Match a whole answer, not a substring such as ``o`` inside ``blood``."""
    pattern = re.escape(answer)
    if answer[:1].isalnum() or answer[:1] == "_":
        pattern = rf"(?<!\w){pattern}"
    if answer[-1:].isalnum() or answer[-1:] == "_":
        pattern = rf"{pattern}(?!\w)"
    return pattern


def extract_gold_context(text: str, answer: str, max_chars: int = 1200) -> tuple[str, str]:
    """
    Return an exact excerpt around ``answer`` and the exact answer span.

    The returned context is a slice of ``text`` (never a reconstructed string),
    so ``gold_span`` is guaranteed to be a substring of ``gold_context``.  One
    sentence of context on each side is included; if that is still longer than
    ``max_chars``, only the sentence containing the answer is used.

    Raises:
        ValueError: if the answer does not occur in ``text``.
    """
    match = re.search(_answer_pattern(answer), text, flags=re.IGNORECASE)
    if match is None:
        msg = f"Answer {answer!r} not found in source text"
        raise ValueError(msg)

    sentence_spans = [
        (sentence.start(), sentence.end())
        for sentence in re.finditer(r"[^.!?]*[.!?]+(?=\s|$)|[^.!?]+$", text)
        if sentence.group().strip()
    ]
    index = next(
        (i for i, (start, end) in enumerate(sentence_spans) if start <= match.start() < end),
        None,
    )
    if index is None:
        context = text.strip()
    else:
        start = sentence_spans[max(0, index - 1)][0]
        end = sentence_spans[min(len(sentence_spans) - 1, index + 1)][1]
        context = text[start:end].strip()
        if len(context) > max_chars:
            start, end = sentence_spans[index]
            context = text[start:end].strip()

    context_match = re.search(_answer_pattern(answer), context, flags=re.IGNORECASE)
    if context_match is None:
        msg = f"Answer {answer!r} lost while extracting context"
        raise ValueError(msg)

    return context, context[context_match.start() : context_match.end()]


class BenchmarkQuestion(BaseModel):
    """One question with its gold evidence and provenance."""

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    question: str = Field(min_length=3)
    answerable: bool
    expected_answer: str | None = None
    gold_context: str | None = None
    gold_span: str | None = None
    dataset: DatasetName
    split: str = Field(min_length=1)
    set_type: SetType
    source: str = Field(min_length=1)
    gold_title: str | None = None
    gold_url: str | None = None
    answer_aliases: list[str] = Field(default_factory=list)

    @field_validator("question_id")
    @classmethod
    def _validate_question_id(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9._:-]+", value):
            msg = "question_id may only contain letters, digits, '.', '_', ':' and '-'"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _validate_answer_fields(self) -> BenchmarkQuestion:
        if self.answerable:
            missing = [
                name
                for name, value in (
                    ("expected_answer", self.expected_answer),
                    ("gold_context", self.gold_context),
                    ("gold_span", self.gold_span),
                    ("gold_title", self.gold_title),
                )
                if not value
            ]
            if missing:
                msg = f"answerable question is missing: {', '.join(missing)}"
                raise ValueError(msg)
            if len(self.gold_context) < 20:
                msg = "gold_context is suspiciously short"
                raise ValueError(msg)
            if self.gold_span not in self.gold_context:
                msg = "gold_span is not an exact substring of gold_context"
                raise ValueError(msg)
        else:
            unexpected = [
                name
                for name, value in (
                    ("expected_answer", self.expected_answer),
                    ("gold_context", self.gold_context),
                    ("gold_span", self.gold_span),
                    ("gold_title", self.gold_title),
                    ("gold_url", self.gold_url),
                )
                if value is not None
            ]
            if unexpected:
                msg = f"unanswerable question has gold fields: {', '.join(unexpected)}"
                raise ValueError(msg)
            if self.answer_aliases:
                msg = "unanswerable question must not define answer_aliases"
                raise ValueError(msg)

        return self


def load_benchmark(path: Path) -> list[BenchmarkQuestion]:
    """Load and validate a single benchmark JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        msg = f"{path}: expected a JSON list, got {type(data).__name__}"
        raise BenchmarkValidationError(msg)

    questions = [BenchmarkQuestion.model_validate(item) for item in data]
    ids = [question.question_id for question in questions]
    duplicates = sorted({question_id for question_id in ids if ids.count(question_id) > 1})
    if duplicates:
        msg = f"{path}: duplicate question_id values: {duplicates}"
        raise BenchmarkValidationError(msg)
    return questions


def validate_all(benchmarks_dir: Path = BENCHMARKS_DIR) -> dict[Path, list[BenchmarkQuestion]]:
    """Validate every top-level benchmark file and return it by path."""
    paths = sorted(path for path in benchmarks_dir.glob("*.json") if path.is_file())
    if not paths:
        msg = f"No benchmark files found in {benchmarks_dir}"
        raise BenchmarkValidationError(msg)

    validated: dict[Path, list[BenchmarkQuestion]] = {}
    errors: list[str] = []
    for path in paths:
        try:
            validated[path] = load_benchmark(path)
        except Exception as exc:  # noqa: BLE001 - collect all file errors in one report
            errors.append(f"{path.name}: {exc}")

    if errors:
        raise BenchmarkValidationError("\n".join(errors))
    return validated


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def check_corpus(
    validated: dict[Path, list[BenchmarkQuestion]],
    suffix: int = DEFAULT_COLLECTION_SUFFIX,
    strict: bool = False,
) -> list[str]:
    """
    Check that gold evidence is reachable in the ingested Qdrant collections.

    For every answerable question the collection must contain a document whose
    ``document_title`` equals ``gold_title`` and whose text contains
    ``gold_span``.  Missing collections are reported and, unless ``strict`` is
    set, do not fail the run (their datasets may not be ingested yet).
    """
    from qdrant_client import QdrantClient

    from src.config.settings import get_settings

    client = QdrantClient(url=get_settings().qdrant_url)
    available = {collection.name for collection in client.get_collections().collections}
    report: list[str] = []
    problems: list[str] = []

    by_dataset: dict[str, list[BenchmarkQuestion]] = defaultdict(list)
    for questions in validated.values():
        for question in questions:
            by_dataset[question.dataset].append(question)

    for dataset, questions in sorted(by_dataset.items()):
        collection = f"rag_{dataset}_{suffix}"
        answerable = [question for question in questions if question.answerable]
        if collection not in available:
            message = f"{collection}: skipped (collection not found)"
            report.append(message)
            if strict:
                problems.append(message)
            continue

        documents: dict[str, list[str]] = defaultdict(list)
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                metadata = payload.get("metadata") or {}
                title = metadata.get("document_title")
                text = payload.get("page_content") or ""
                if title and text:
                    documents[title].append(text)
            if offset is None:
                break

        missing = 0
        for question in answerable:
            assert question.gold_title is not None
            assert question.gold_span is not None
            text = _normalise(" ".join(documents.get(question.gold_title, [])))
            if not text or _normalise(question.gold_span) not in text:
                missing += 1
                problems.append(
                    f"{collection}: {question.question_id} gold span not found "
                    f"in document {question.gold_title!r}"
                )

        report.append(
            f"{collection}: {len(answerable) - missing}/{len(answerable)} gold spans found"
        )

    if problems and strict:
        raise BenchmarkValidationError("\n".join(problems))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate benchmark JSON files")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate schema and gold evidence (required)",
    )
    parser.add_argument(
        "--check-corpus",
        action="store_true",
        help="Also check gold spans against the ingested Qdrant collections",
    )
    parser.add_argument(
        "--strict-corpus",
        action="store_true",
        help="Fail when a collection is missing or a gold span cannot be found",
    )
    parser.add_argument(
        "--suffix",
        type=int,
        default=DEFAULT_COLLECTION_SUFFIX,
        help="Chunk-size suffix used in collection names (default: 512)",
    )
    args = parser.parse_args()

    if not args.validate:
        parser.error("--validate is required")

    validated = validate_all()
    total = sum(len(questions) for questions in validated.values())
    print(f"Validated {total} questions in {len(validated)} file(s)")
    for path, questions in sorted(validated.items()):
        answerable = sum(1 for question in questions if question.answerable)
        print(f"  {path.name}: {len(questions)} questions ({answerable} answerable)")

    if args.check_corpus:
        report = check_corpus(validated, suffix=args.suffix, strict=args.strict_corpus)
        print("\nCorpus alignment:")
        for line in report:
            print(f"  {line}")


if __name__ == "__main__":
    main()
