"""
Build benchmark files from official dataset splits or the ingested wiki corpus.

The generated files are the source of truth for evaluation.  Every answerable
question carries an exact gold context and gold span, plus a ``gold_title``
that can be checked against the Qdrant collection.

Usage:
    uv run python -m src.evaluation.build_benchmarks --dataset wiki
    uv run python -m src.evaluation.build_benchmarks --dataset squad
    uv run python -m src.evaluation.build_benchmarks --dataset trivia
    uv run python -m src.evaluation.build_benchmarks --dataset all
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from qdrant_client import QdrantClient

from src.config.settings import get_settings
from src.data.loader import (
    DEFAULT_CORPUS_SIZE,
    load_squad_validation_examples,
    load_trivia_validation_examples,
)
from src.evaluation.benchmarks import (
    BENCHMARKS_DIR,
    BenchmarkQuestion,
    extract_gold_context,
    load_benchmark,
)

SOURCES_DIR = BENCHMARKS_DIR / "sources"
WIKI_SOURCE_FILE = SOURCES_DIR / "wiki_custom_qa.json"
WIKI_OUTPUT_FILE = BENCHMARKS_DIR / "wikipedia_simple_benchmark.json"
SQUAD_OUTPUT_FILE = BENCHMARKS_DIR / "squad_benchmark.json"
TRIVIA_OUTPUT_FILE = BENCHMARKS_DIR / "trivia_benchmark.json"

DEFAULT_SEED = 42


def _clean_question(question: str) -> str:
    return re.sub(r"\s+", " ", question).strip()


def _unique_aliases(*values: str, limit: int = 20) -> list[str]:
    aliases: list[str] = []
    for value in values:
        value = value.strip()
        if value and value not in aliases:
            aliases.append(value)
    return aliases[:limit]


def _write(path: Path, questions: list[BenchmarkQuestion]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [question.model_dump(mode="json") for question in questions]
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    # Fail immediately if the generated file violates the schema.
    load_benchmark(path)


def build_squad_benchmark(
    max_contexts: int = DEFAULT_CORPUS_SIZE,
    n_answerable: int = 40,
    n_unanswerable: int = 10,
    seed: int = DEFAULT_SEED,
) -> list[BenchmarkQuestion]:
    """Build a deterministic sample from the official SQuAD 2.0 validation split."""
    examples = load_squad_validation_examples(max_contexts=max_contexts)

    answerable: list[tuple[dict, list[str], str, str]] = []
    unanswerable: list[dict] = []
    for example in examples:
        answers = [answer.strip() for answer in example["answers"]["text"] if answer.strip()]
        if not answers:
            unanswerable.append(example)
            continue
        try:
            gold_context, gold_span = extract_gold_context(example["context"], answers[0])
        except ValueError:
            continue
        answerable.append((example, answers, gold_context, gold_span))

    if len(answerable) < n_answerable:
        msg = f"Only {len(answerable)} answerable SQuAD questions found"
        raise ValueError(msg)
    if len(unanswerable) < n_unanswerable:
        msg = f"Only {len(unanswerable)} unanswerable SQuAD questions found"
        raise ValueError(msg)

    rng = random.Random(seed)
    selected_answerable = rng.sample(answerable, n_answerable)
    selected_unanswerable = rng.sample(unanswerable, n_unanswerable)

    questions: list[BenchmarkQuestion] = []
    for example, answers, gold_context, gold_span in selected_answerable:
        questions.append(
            BenchmarkQuestion(
                question_id=f"squad-{example['id']}",
                question=_clean_question(example["question"]),
                answerable=True,
                expected_answer=answers[0],
                gold_context=gold_context,
                gold_span=gold_span,
                dataset="squad",
                split="validation",
                set_type="official",
                source=f"SQuAD 2.0 validation — {example['title']}",
                gold_title=example["title"],
                answer_aliases=_unique_aliases(*answers, gold_span),
            )
        )
    for example in selected_unanswerable:
        questions.append(
            BenchmarkQuestion(
                question_id=f"squad-{example['id']}",
                question=_clean_question(example["question"]),
                answerable=False,
                dataset="squad",
                split="validation",
                set_type="official",
                source=f"SQuAD 2.0 validation (unanswerable) — {example['title']}",
            )
        )

    questions.sort(key=lambda question: question.question_id)
    return questions


_STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "been",
    "before",
    "but",
    "called",
    "did",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "into",
    "its",
    "more",
    "most",
    "not",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def _question_matches_evidence(question: str, title: str, context: str) -> bool:
    """
    Reject evidence that only happens to contain the answer string.

    When the question contains a proper noun, that name must occur in the
    evidence title or text.  Otherwise at least one content word from the
    question must occur in the title.
    """
    proper_nouns = [
        word
        for word in re.findall(r"\b[A-Z][A-Za-z'’-]+\b", question)
        if word.lower() not in _STOPWORDS
    ]
    haystack = f"{title} {context}".lower()
    if proper_nouns:
        return any(noun.lower() in haystack for noun in proper_nouns)

    title_words = set(re.findall(r"[a-z]{3,}", title.lower()))
    question_words = {
        word for word in re.findall(r"[a-z]{3,}", question.lower()) if word not in _STOPWORDS
    }
    return bool(title_words & question_words)


def _trivia_evidence(example: dict) -> list[tuple[str, str]]:
    entity_pages = example.get("entity_pages") or {}
    titles = entity_pages.get("title") or []
    contexts = entity_pages.get("wiki_context") or []
    return [
        (title, context)
        for title, context in zip(titles, contexts, strict=False)
        if context and len(context) > 100
    ]


def build_trivia_benchmark(
    max_contexts: int = DEFAULT_CORPUS_SIZE,
    n_questions: int = 50,
    seed: int = DEFAULT_SEED,
) -> list[BenchmarkQuestion]:
    """Build a deterministic sample from official TriviaQA rc.wikipedia validation."""
    examples = load_trivia_validation_examples(max_contexts=max_contexts)

    candidates: list[tuple[dict, str, list[str], str, str, str]] = []
    for example in examples:
        answer = example.get("answer") or {}
        value = (answer.get("value") or "").strip()
        aliases = [alias.strip() for alias in answer.get("aliases") or [] if alias.strip()]
        if not value:
            continue

        # The first entity page is the evidence TriviaQA considers most
        # relevant.  Falling back to other pages makes ambiguous questions
        # (for example a surname that occurs in an unrelated article) look
        # valid, so only the first page is used.
        found: tuple[str, str, str] | None = None
        for title, context in _trivia_evidence(example)[:1]:
            question = _clean_question(example["question"])
            if not _question_matches_evidence(question, title, context):
                continue
            # Use the official answer value, not a broad alias.  Otherwise a
            # question about a "bear cub" can be grounded on the word "bears"
            # and pass validation while no longer answering the question.
            try:
                gold_context, gold_span = extract_gold_context(context, value)
            except ValueError:
                continue
            found = (title, gold_context, gold_span)

        if found is not None:
            title, gold_context, gold_span = found
            candidates.append((example, value, aliases, title, gold_context, gold_span))

    if len(candidates) < n_questions:
        msg = f"Only {len(candidates)} TriviaQA questions with evidence found"
        raise ValueError(msg)

    selected = random.Random(seed).sample(candidates, n_questions)
    questions: list[BenchmarkQuestion] = []
    for example, value, aliases, title, gold_context, gold_span in selected:
        questions.append(
            BenchmarkQuestion(
                question_id=f"trivia-{example['question_id']}",
                question=_clean_question(example["question"]),
                answerable=True,
                expected_answer=value,
                gold_context=gold_context,
                gold_span=gold_span,
                dataset="trivia",
                split="validation",
                set_type="official",
                source=f"TriviaQA rc.wikipedia validation — {title}",
                gold_title=title,
                answer_aliases=_unique_aliases(value, *aliases, gold_span),
            )
        )

    questions.sort(key=lambda question: question.question_id)
    return questions


def _load_qdrant_articles(collection: str) -> tuple[dict[str, str], dict[str, str]]:
    """Reconstruct article texts and URLs from a Qdrant collection."""
    client = QdrantClient(url=get_settings().qdrant_url)
    if collection not in {item.name for item in client.get_collections().collections}:
        msg = f"Qdrant collection {collection!r} not found; ingest the wiki corpus first"
        raise ValueError(msg)

    parts: dict[str, dict[int, str]] = defaultdict(dict)
    urls: dict[str, str] = {}
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
            chunk_index = metadata.get("chunk_index")
            text = payload.get("page_content") or ""
            if title and isinstance(chunk_index, int) and text:
                parts[title][chunk_index] = text
                if metadata.get("url"):
                    urls.setdefault(title, metadata["url"])
        if offset is None:
            break

    articles = {
        title: " ".join(chunks[index] for index in sorted(chunks))
        for title, chunks in parts.items()
    }
    return articles, urls


def build_wiki_benchmark(
    source_file: Path = WIKI_SOURCE_FILE,
    collection: str = "rag_wiki_512",
) -> list[BenchmarkQuestion]:
    """
    Build the custom Simple Wikipedia benchmark from the ingested corpus.

    The source file only stores question, answer and article title.  Gold
    contexts and spans are extracted from the exact text stored in Qdrant, so
    they cannot drift from the knowledge base.
    """
    source = json.loads(source_file.read_text(encoding="utf-8"))
    articles, urls = _load_qdrant_articles(collection)

    questions: list[BenchmarkQuestion] = []
    for item in source["questions"]:
        article = item.get("article")
        answerable = item.get("answerable", True)
        if not answerable:
            questions.append(
                BenchmarkQuestion(
                    question_id=item["question_id"],
                    question=_clean_question(item["question"]),
                    answerable=False,
                    dataset="wiki",
                    split="custom",
                    set_type="custom",
                    source=item.get(
                        "source",
                        "Custom control — not answerable from the ingested Simple Wikipedia corpus",
                    ),
                )
            )
            continue

        if not article:
            msg = f"{item['question_id']}: answerable question has no article"
            raise ValueError(msg)
        text = articles.get(article)
        if text is None:
            msg = f"{item['question_id']}: article {article!r} is not in {collection}"
            raise ValueError(msg)

        gold_context, gold_span = extract_gold_context(text, item["answer"])
        questions.append(
            BenchmarkQuestion(
                question_id=item["question_id"],
                question=_clean_question(item["question"]),
                answerable=True,
                expected_answer=item["answer"],
                gold_context=gold_context,
                gold_span=gold_span,
                dataset="wiki",
                split="custom",
                set_type="custom",
                source=item.get(
                    "source",
                    f"Custom question grounded in Simple Wikipedia article {article!r}",
                ),
                gold_title=article,
                gold_url=urls.get(article),
                answer_aliases=_unique_aliases(item["answer"], *item.get("aliases", []), gold_span),
            )
        )

    return questions


def main() -> None:
    parser = argparse.ArgumentParser(description="Build benchmark JSON files")
    parser.add_argument(
        "--dataset",
        choices=["wiki", "squad", "trivia", "all"],
        default="all",
        help="Which benchmark to rebuild",
    )
    parser.add_argument(
        "--max-contexts",
        type=int,
        default=DEFAULT_CORPUS_SIZE,
        help="Corpus slice size used by ingestion (default: 500)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--collection",
        default="rag_wiki_512",
        help="Qdrant collection used to extract wiki gold contexts",
    )
    args = parser.parse_args()

    if args.dataset in {"wiki", "all"}:
        output = WIKI_OUTPUT_FILE
        _write(output, build_wiki_benchmark(collection=args.collection))
        print(f"Wrote {output}")
    if args.dataset in {"squad", "all"}:
        output = SQUAD_OUTPUT_FILE
        _write(
            output,
            build_squad_benchmark(max_contexts=args.max_contexts, seed=args.seed),
        )
        print(f"Wrote {output}")
    if args.dataset in {"trivia", "all"}:
        output = TRIVIA_OUTPUT_FILE
        _write(
            output,
            build_trivia_benchmark(max_contexts=args.max_contexts, seed=args.seed),
        )
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
