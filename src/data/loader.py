"""
Dataset loaders for the RAG knowledge bases and benchmarks.

The wiki knowledge base is a slice of Simple English Wikipedia.  SQuAD and
TriviaQA loaders use the official validation split, because that is the split
the benchmark files are built from.  Keeping the corpus and the benchmark on
the same split is what makes retrieval results comparable.

Natural Questions (``nq_open``) is kept as a helper for future experiments,
but it is not the source of the Simple Wikipedia knowledge base.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

# Where processed data gets saved locally
DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Official dataset identifiers used by both ingestion and benchmark building.
SQUAD_DATASET_ID = "rajpurkar/squad_v2"
SQUAD_SPLIT = "validation"
TRIVIA_DATASET_ID = "mandarjoshi/trivia_qa"
TRIVIA_CONFIG = "rc.wikipedia"
TRIVIA_SPLIT = "validation"
TRIVIA_PARQUET_FILE = "rc.wikipedia/validation-00000-of-00001.parquet"

# Default corpus size.  The benchmark builders use the same value so that
# every gold document is present in the collection created by ingestion.
DEFAULT_CORPUS_SIZE = 500


@dataclass
class QAPair:
    """A single question-answer pair from the dataset."""

    question: str
    short_answer: str
    long_answer: str
    context: str  # The Wikipedia article/paragraph
    document_title: str
    language: str = "en"  # "en" or "pl"
    metadata: dict = field(default_factory=dict)


@dataclass
class Document:
    """A Wikipedia document (or chunk of it) for the vector database."""

    content: str
    title: str
    source: str = "natural_questions"
    language: str = "en"
    metadata: dict = field(default_factory=dict)


def load_natural_questions(
    subset_size: int = 2000,
    split: str = "validation",
) -> tuple[list[QAPair], list[Document]]:
    """
    Load Natural Questions dataset from HuggingFace.

    We use the 'nq_open' version which is cleaner and lighter:
    - Only questions with short answers (factual questions)
    - Already extracted answers (no HTML parsing needed)

    Args:
        subset_size: How many QA pairs to load (default: 2000)
        split: Which split to use ("train" or "validation")

    Returns:
        Tuple of (qa_pairs, documents)
        - qa_pairs: for evaluation later
        - documents: for embedding and storing in Qdrant
    """
    logger.info(f"Loading Natural Questions ({split}, n={subset_size})...")

    # nq_open is a cleaned version of NQ focused on open-domain QA
    # It has: question (str), answer (list[str])
    dataset = load_dataset(
        "google-research-datasets/nq_open",
        split=f"{split}[:{subset_size}]",
    )

    logger.info(f"Downloaded {len(dataset)} examples")

    qa_pairs: list[QAPair] = []
    documents: list[Document] = []

    for idx, example in enumerate(dataset):
        question = example["question"]
        # nq_open has answers as a list — take the first one
        answers = example["answer"]
        short_answer = answers[0] if answers else ""

        # Create QA pair for evaluation
        qa_pair = QAPair(
            question=question,
            short_answer=short_answer,
            long_answer="",  # nq_open doesn't have long answers
            context="",  # will be filled by retriever during evaluation
            document_title="",
            metadata={"index": idx, "all_answers": answers},
        )
        qa_pairs.append(qa_pair)

    logger.info(f"Created {len(qa_pairs)} QA pairs")
    logger.info(f"Created {len(documents)} unique documents")

    return qa_pairs, documents


def load_wikipedia_contexts(
    subset_size: int = 500,
) -> list[Document]:
    """
    Load Wikipedia articles to build the knowledge base.

    We use the 'wikipedia' dataset from HuggingFace as our document corpus.
    These documents get chunked, embedded, and stored in Qdrant.

    Args:
        subset_size: Number of Wikipedia articles to load

    Returns:
        List of Document objects ready for chunking
    """
    logger.info(f"Loading Wikipedia articles (n={subset_size})...")

    # Load a subset of Simple English Wikipedia (modern Parquet format)
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.simple",
        split=f"train[:{subset_size}]",
    )

    documents: list[Document] = []
    for example in dataset:
        title = example["title"]
        text = example["text"]

        # Skip very short articles (< 200 chars — probably stubs)
        if len(text) < 200:
            continue

        doc = Document(
            content=text,
            title=title,
            source="wikipedia_simple",
            language="en",
            metadata={"url": example.get("url", "")},
        )
        documents.append(doc)

    logger.info(f"Loaded {len(documents)} Wikipedia articles")
    return documents


def load_squad_validation_examples(
    max_contexts: int = DEFAULT_CORPUS_SIZE,
) -> list[dict]:
    """
    Load SQuAD 2.0 validation examples whose contexts fit in the corpus slice.

    The first ``max_contexts`` unique contexts are kept, in dataset order.
    Every question attached to those contexts is returned, so the benchmark
    builder can sample questions without leaving the ingested corpus.
    """
    logger.info(f"Loading SQuAD 2.0 validation examples (max {max_contexts} contexts)...")
    dataset = load_dataset(SQUAD_DATASET_ID, split=SQUAD_SPLIT)

    examples: list[dict] = []
    seen: set[str] = set()
    for example in dataset:
        context = example["context"]
        if len(context) <= 100:
            continue
        key = context[:100]
        if key not in seen:
            if len(seen) >= max_contexts:
                break
            seen.add(key)
        examples.append(example)

    logger.info(f"Loaded {len(examples)} SQuAD examples from {len(seen)} unique contexts")
    return examples


def load_squad_validation_documents(
    max_contexts: int = DEFAULT_CORPUS_SIZE,
) -> list[Document]:
    """Load SQuAD validation contexts as documents for ingestion."""
    documents: list[Document] = []
    seen: set[str] = set()
    for example in load_squad_validation_examples(max_contexts=max_contexts):
        context = example["context"]
        key = context[:100]
        if key in seen:
            continue
        seen.add(key)
        documents.append(
            Document(
                content=context,
                title=example["title"],
                source="squad_v2",
                language="en",
            )
        )
    return documents


def load_trivia_validation_examples(
    max_contexts: int = DEFAULT_CORPUS_SIZE,
) -> list[dict]:
    """
    Load TriviaQA ``rc.wikipedia`` validation examples with entity evidence.

    The streaming ``datasets`` path currently crashes on interpreter shutdown
    with the pinned HuggingFace stack, so the validation Parquet file is read
    directly.  Only examples that have Wikipedia entity pages are returned;
    those are the documents the ingestion pipeline stores.
    """
    logger.info(f"Loading TriviaQA rc.wikipedia validation (max {max_contexts} contexts)...")
    parquet_path = hf_hub_download(
        repo_id=TRIVIA_DATASET_ID,
        filename=TRIVIA_PARQUET_FILE,
        repo_type="dataset",
    )

    parquet = pq.ParquetFile(parquet_path)
    examples: list[dict] = []
    seen: set[str] = set()
    corpus_full = False

    for batch in parquet.iter_batches(
        batch_size=64,
        columns=["question", "question_id", "entity_pages", "answer"],
    ):
        for row in batch.to_pylist():
            entity_pages = row.get("entity_pages") or {}
            titles = entity_pages.get("title") or []
            contexts = entity_pages.get("wiki_context") or []
            pairs = [
                (title, context)
                for title, context in zip(titles, contexts, strict=False)
                if context and len(context) > 100
            ]
            if not pairs:
                continue

            for _, context in pairs:
                key = context[:100]
                if key not in seen:
                    if len(seen) >= max_contexts:
                        corpus_full = True
                        break
                    seen.add(key)

            if corpus_full:
                break
            examples.append(row)

        if corpus_full:
            break

    logger.info(f"Loaded {len(examples)} TriviaQA examples from {len(seen)} unique contexts")
    return examples


def load_trivia_validation_documents(
    max_contexts: int = DEFAULT_CORPUS_SIZE,
) -> list[Document]:
    """Load TriviaQA entity-page evidence as documents for ingestion."""
    documents: list[Document] = []
    seen: set[str] = set()
    for example in load_trivia_validation_examples(max_contexts=max_contexts):
        entity_pages = example.get("entity_pages") or {}
        titles = entity_pages.get("title") or []
        contexts = entity_pages.get("wiki_context") or []
        for title, context in zip(titles, contexts, strict=False):
            if not context or len(context) <= 100:
                continue
            key = context[:100]
            if key in seen:
                continue
            seen.add(key)
            documents.append(
                Document(
                    content=context,
                    title=title or example["question"],
                    source="trivia_qa",
                    language="en",
                )
            )
    return documents


def save_qa_pairs(qa_pairs: list[QAPair], filepath: Path | None = None) -> Path:
    """Save QA pairs to JSON for later evaluation."""
    import json

    if filepath is None:
        filepath = PROCESSED_DIR / "qa_pairs.json"

    filepath.parent.mkdir(parents=True, exist_ok=True)

    data = [
        {
            "question": qa.question,
            "short_answer": qa.short_answer,
            "long_answer": qa.long_answer,
            "document_title": qa.document_title,
            "language": qa.language,
            "metadata": qa.metadata,
        }
        for qa in qa_pairs
    ]

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(qa_pairs)} QA pairs to {filepath}")
    return filepath


def load_qa_pairs_from_file(filepath: Path | None = None) -> list[QAPair]:
    """Load previously saved QA pairs from JSON."""
    import json

    if filepath is None:
        filepath = PROCESSED_DIR / "qa_pairs.json"

    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    qa_pairs = [
        QAPair(
            question=item["question"],
            short_answer=item["short_answer"],
            long_answer=item["long_answer"],
            context="",
            document_title=item["document_title"],
            language=item["language"],
            metadata=item.get("metadata", {}),
        )
        for item in data
    ]

    logger.info(f"Loaded {len(qa_pairs)} QA pairs from {filepath}")
    return qa_pairs
