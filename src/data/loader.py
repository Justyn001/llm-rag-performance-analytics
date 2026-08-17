"""
Natural Questions dataset loader.

Downloads Google's Natural Questions dataset from HuggingFace
and prepares it for the RAG pipeline.

Natural Questions = real Google Search queries + Wikipedia answers.
We use it because:
  - It's an industry-standard QA benchmark
  - Questions are REAL (not synthetic)
  - Answers come from Wikipedia articles (perfect for RAG)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from datasets import load_dataset

logger = logging.getLogger(__name__)

# Where processed data gets saved locally
DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


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
