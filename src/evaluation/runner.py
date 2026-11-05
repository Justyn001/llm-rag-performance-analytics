"""
Evaluation runner: benchmark questions -> RAG answers -> metrics -> storage.

The runner is intentionally single-threaded and sequential.  That makes each
run reproducible, keeps local GPU/API contention out of the measurements and
leaves warm-up/repetition handling to the dedicated measurement protocol.
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings

from src.config.models import ModelConfig, ModelProvider
from src.config.rag_config import RAGConfig
from src.evaluation.benchmarks import BenchmarkQuestion, load_benchmark
from src.evaluation.judge import JudgeVerdict
from src.evaluation.metrics import (
    compute_answer_metrics,
    compute_retrieval_metrics,
)
from src.evaluation.protocol import DEFAULT_PROTOCOL, MeasurementProtocol
from src.evaluation.stats import flatten_performance, performance_summary
from src.evaluation.storage import EvaluationStore
from src.rag.llm_provider import supports_num_ctx

DEFAULT_OUTPUT_DIR = Path("data/evaluation")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunResult:
    """Paths and aggregate summary produced by one evaluation run."""

    run_id: str
    records: list[dict[str, Any]]
    summary: dict[str, Any]
    jsonl_path: Path
    csv_path: Path
    summary_path: Path


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "run"


def generate_run_id(model_config: ModelConfig, rag_config: RAGConfig) -> str:
    """Create a filesystem-safe, sortable and collision-resistant run id."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    slug = _slugify(f"{model_config.name}-{rag_config.name}")
    return f"{timestamp}_{slug}_{uuid.uuid4().hex[:4]}"


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _gpu_details() -> dict[str, Any]:
    if shutil.which("nvidia-smi") is None:
        return {}
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if result.returncode != 0:
        return {}
    first_line = result.stdout.strip().splitlines()
    if not first_line:
        return {}
    parts = [part.strip() for part in first_line[0].split(",")]
    details: dict[str, Any] = {"name": parts[0] or None}
    if len(parts) > 1:
        try:
            details["memory_mb"] = int(float(parts[1]))
        except ValueError:
            details["memory_mb"] = None
    if len(parts) > 2:
        details["driver"] = parts[2] or None
    return details


def _cpu_name() -> str | None:
    if platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or None


def _ram_gb() -> float | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    return round(page_size * pages / (1024**3), 1)


def collect_environment_metadata() -> dict[str, Any]:
    """Collect reproducibility metadata without importing heavy ML libraries."""
    packages = {
        name: _package_version(name)
        for name in (
            "langchain",
            "langchain-core",
            "langchain-community",
            "langchain-ollama",
            "langchain-groq",
            "langchain-google-genai",
            "langchain-mistralai",
            "langchain-qdrant",
            "qdrant-client",
            "sentence-transformers",
            "torch",
            "datasets",
            "ragas",
            "pydantic",
        )
    }
    gpu = _gpu_details()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": _cpu_name(),
        "cpu_count": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "gpu": gpu.get("name"),
        "gpu_memory_mb": gpu.get("memory_mb"),
        "gpu_driver": gpu.get("driver"),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
        "packages": packages,
    }


def _ollama_model_details(model_id: str) -> dict[str, Any] | None:
    """Read model details from a running Ollama server, if available."""
    try:
        import httpx

        from src.config.settings import get_settings

        settings = get_settings()
        response = httpx.post(
            f"{settings.ollama_host}/api/show",
            json={"model": model_id},
            timeout=3.0,
        )
        if response.status_code != 200:
            return None
        payload = response.json()
    except Exception:  # noqa: BLE001 - metadata collection must never break a run
        return None
    details = payload.get("details")
    return details if isinstance(details, dict) else None


def collect_model_metadata(model_config: ModelConfig) -> dict[str, Any]:
    """Collect model identity, provider and quantisation information."""
    detected = (
        _ollama_model_details(model_config.model_id)
        if model_config.provider == ModelProvider.OLLAMA
        else None
    )
    quantization = model_config.quantization
    quantization_source = "config" if quantization else None
    if detected and detected.get("quantization_level"):
        quantization = detected["quantization_level"]
        quantization_source = "ollama"
    if not quantization:
        quantization = "provider-managed" if not model_config.is_local else "unknown"
        quantization_source = "fallback"

    metadata: dict[str, Any] = {
        "name": model_config.name,
        "provider": model_config.provider.value,
        "model_id": model_config.model_id,
        "is_local": model_config.is_local,
        "parameters": model_config.parameters,
        "temperature": model_config.temperature,
        "configured_max_tokens": model_config.max_tokens,
        "quantization": quantization,
        "quantization_source": quantization_source,
    }
    if detected:
        for key in ("family", "parameter_size", "context_length", "format"):
            if key in detected:
                metadata[key] = detected[key]
    return metadata


def _references(question: BenchmarkQuestion) -> list[str]:
    references: list[str] = []
    if question.expected_answer:
        references.append(question.expected_answer)
    for alias in question.answer_aliases:
        if alias and alias not in references:
            references.append(alias)
    return references


def _title(metadata: dict[str, Any] | None) -> str:
    if not metadata:
        return ""
    return str(metadata.get("document_title") or metadata.get("title") or "")


def _short_error(exc: Exception) -> str:
    message = f"{type(exc).__name__}: {exc}"
    return message[:2000]


def _judge_dict(verdict: JudgeVerdict | dict[str, Any] | None) -> dict[str, Any] | None:
    if verdict is None:
        return None
    if isinstance(verdict, JudgeVerdict):
        return asdict(verdict)
    if isinstance(verdict, dict):
        return dict(verdict)
    msg = f"unsupported judge verdict type: {type(verdict).__name__}"
    raise TypeError(msg)


def _empty_record(
    *,
    run_id: str,
    question: BenchmarkQuestion,
    repetition: int,
    error: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "question_id": question.question_id,
        "repetition": repetition,
        "dataset": question.dataset,
        "split": question.split,
        "set_type": question.set_type,
        "source": question.source,
        "question": question.question,
        "answerable": question.answerable,
        "expected_answer": question.expected_answer,
        "gold_context": question.gold_context,
        "gold_span": question.gold_span,
        "gold_title": question.gold_title,
        "gold_url": question.gold_url,
        "answer_aliases": list(question.answer_aliases),
        "answer": "",
        "contexts": [],
        "context_titles": [],
        "retrieval_time_ms": 0.0,
        "generation_time_ms": 0.0,
        "retry_backoff_time_ms": 0.0,
        "failed_attempt_time_ms": 0.0,
        "retry_count": 0,
        "total_time_ms": 0.0,
        "max_tokens": 0,
        "num_ctx": None,
        "tokens_in": 0,
        "tokens_out": 0,
        "tokens_per_second": 0.0,
        "exact_match": None,
        "f1": None,
        "is_refusal": False,
        "refusal_correct": None,
        "automatic_correct": None,
        "refusal_pattern": None,
        "gold_title_rank": None,
        "retrieval_hit": None,
        "gold_span_in_context": None,
        "judge": None,
        "error": error,
    }


def _evaluate_one(
    *,
    run_id: str,
    question: BenchmarkQuestion,
    response: Any,
    judge: Any | None,
    repetition: int,
) -> dict[str, Any]:
    contexts = list(getattr(response, "contexts", []) or [])
    context_metadata = list(getattr(response, "context_metadata", []) or [])
    context_titles = [_title(metadata) for metadata in context_metadata]
    answer = str(getattr(response, "answer", "") or "")

    answer_metrics = compute_answer_metrics(
        answer,
        answerable=question.answerable,
        references=_references(question),
    )
    retrieval_metrics = compute_retrieval_metrics(
        contexts,
        context_titles,
        gold_title=question.gold_title,
        gold_span=question.gold_span,
    )

    judge_data = None
    if judge is not None:
        try:
            verdict = judge.evaluate(
                question=question.question,
                answer=answer,
                contexts=contexts,
                answerable=question.answerable,
                references=_references(question),
                gold_context=question.gold_context,
            )
            judge_data = _judge_dict(verdict)
        except Exception as exc:  # noqa: BLE001 - a judge failure must not lose the answer
            judge_data = JudgeVerdict(error=_short_error(exc)).__dict__.copy()

    return {
        "run_id": run_id,
        "question_id": question.question_id,
        "repetition": repetition,
        "dataset": question.dataset,
        "split": question.split,
        "set_type": question.set_type,
        "source": question.source,
        "question": question.question,
        "answerable": question.answerable,
        "expected_answer": question.expected_answer,
        "gold_context": question.gold_context,
        "gold_span": question.gold_span,
        "gold_title": question.gold_title,
        "gold_url": question.gold_url,
        "answer_aliases": list(question.answer_aliases),
        "answer": answer,
        "contexts": contexts,
        "context_titles": context_titles,
        "retrieval_time_ms": float(getattr(response, "retrieval_time_ms", 0.0) or 0.0),
        "generation_time_ms": float(getattr(response, "generation_time_ms", 0.0) or 0.0),
        "retry_backoff_time_ms": float(getattr(response, "retry_backoff_time_ms", 0.0) or 0.0),
        "failed_attempt_time_ms": float(getattr(response, "failed_attempt_time_ms", 0.0) or 0.0),
        "retry_count": int(getattr(response, "retry_count", 0) or 0),
        "total_time_ms": float(getattr(response, "total_time_ms", 0.0) or 0.0),
        "max_tokens": int(getattr(response, "max_tokens", 0) or 0),
        "num_ctx": getattr(response, "num_ctx", None),
        "tokens_in": int(getattr(response, "tokens_in", 0) or 0),
        "tokens_out": int(getattr(response, "tokens_out", 0) or 0),
        "tokens_per_second": float(getattr(response, "tokens_per_second", 0.0) or 0.0),
        "exact_match": answer_metrics.exact_match,
        "f1": answer_metrics.f1,
        "is_refusal": answer_metrics.is_refusal,
        "refusal_correct": answer_metrics.refusal_correct,
        "automatic_correct": answer_metrics.automatic_correct,
        "refusal_pattern": answer_metrics.refusal_pattern,
        "gold_title_rank": retrieval_metrics.gold_title_rank,
        "retrieval_hit": retrieval_metrics.retrieval_hit,
        "gold_span_in_context": retrieval_metrics.gold_span_in_context,
        "judge": judge_data,
        "error": None,
    }


def _mean(values: list[float | int | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a run without hiding errors or missing judge verdicts."""
    completed = [record for record in records if not record.get("error")]
    answerable = [record for record in completed if record.get("answerable")]
    unanswerable = [record for record in completed if not record.get("answerable")]
    judged = [
        record
        for record in completed
        if isinstance(record.get("judge"), dict) and not record["judge"].get("error")
    ]

    def judge_rate(key: str) -> float | None:
        return _mean([record["judge"].get(key) for record in judged])

    return {
        "total": len(records),
        "completed": len(completed),
        "errors": len(records) - len(completed),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "exact_match": _mean([record.get("exact_match") for record in answerable]),
        "f1": _mean([record.get("f1") for record in answerable]),
        "automatic_accuracy": _mean([record.get("automatic_correct") for record in completed]),
        "refusal_accuracy": _mean([record.get("refusal_correct") for record in unanswerable]),
        "answerable_refusal_rate": _mean(
            [1 if record.get("is_refusal") else 0 for record in answerable]
        ),
        "retrieval_hit_rate": _mean([record.get("retrieval_hit") for record in answerable]),
        "gold_span_in_context_rate": _mean(
            [record.get("gold_span_in_context") for record in answerable]
        ),
        "judged_count": len(judged),
        "judge_errors": sum(
            1
            for record in completed
            if isinstance(record.get("judge"), dict) and record["judge"].get("error")
        ),
        "judge_correct_rate": judge_rate("correct"),
        "judge_faithful_rate": judge_rate("faithful"),
        "judge_context_relevant_rate": judge_rate("context_relevant"),
        "judge_refusal_appropriate_rate": judge_rate("refusal_appropriate"),
        "mean_retrieval_time_ms": _mean([record.get("retrieval_time_ms") for record in completed]),
        "mean_generation_time_ms": _mean(
            [record.get("generation_time_ms") for record in completed]
        ),
        "mean_total_time_ms": _mean([record.get("total_time_ms") for record in completed]),
        "mean_tokens_per_second": _mean([record.get("tokens_per_second") for record in completed]),
        "mean_retry_backoff_time_ms": _mean(
            [record.get("retry_backoff_time_ms") for record in completed]
        ),
        "mean_failed_attempt_time_ms": _mean(
            [record.get("failed_attempt_time_ms") for record in completed]
        ),
        "mean_retry_count": _mean([record.get("retry_count") for record in completed]),
    }


def _warmup_summary(
    responses: list[Any],
    question_ids: list[str],
    protocol: MeasurementProtocol,
) -> dict[str, Any]:
    timing_records = [
        {
            "retrieval_time_ms": float(getattr(response, "retrieval_time_ms", 0.0) or 0.0),
            "generation_time_ms": float(getattr(response, "generation_time_ms", 0.0) or 0.0),
            "total_time_ms": float(getattr(response, "total_time_ms", 0.0) or 0.0),
            "tokens_out": int(getattr(response, "tokens_out", 0) or 0),
            "tokens_per_second": float(getattr(response, "tokens_per_second", 0.0) or 0.0),
            "retry_backoff_time_ms": float(getattr(response, "retry_backoff_time_ms", 0.0) or 0.0),
        }
        for response in responses
    ]
    return {
        "count": len(timing_records),
        "question_ids": list(question_ids),
        "performance": performance_summary(
            timing_records,
            confidence_level=protocol.confidence_level,
        ),
    }


def _question_statistics(
    records: list[dict[str, Any]],
    protocol: MeasurementProtocol,
) -> list[dict[str, Any]]:
    order: list[str] = []
    for record in records:
        question_id = str(record.get("question_id", ""))
        if question_id and question_id not in order:
            order.append(question_id)

    statistics: list[dict[str, Any]] = []
    for question_id in order:
        subset = [record for record in records if str(record.get("question_id", "")) == question_id]
        completed = [record for record in subset if not record.get("error")]
        statistics.append(
            {
                "question_id": question_id,
                "answerable": bool(subset[0].get("answerable")) if subset else None,
                "measurements": len(subset),
                "completed": len(completed),
                "errors": len(subset) - len(completed),
                "performance": performance_summary(
                    subset,
                    confidence_level=protocol.confidence_level,
                ),
            }
        )
    return statistics


def _repetition_statistics(
    records: list[dict[str, Any]],
    protocol: MeasurementProtocol,
) -> list[dict[str, Any]]:
    statistics: list[dict[str, Any]] = []
    for repetition in range(1, protocol.repetitions + 1):
        subset = [record for record in records if int(record.get("repetition") or 1) == repetition]
        statistics.append(
            {
                "repetition": repetition,
                "quality": summarize_records(subset),
                "performance": performance_summary(
                    subset,
                    confidence_level=protocol.confidence_level,
                ),
            }
        )
    return statistics


def _add_performance_statistics(
    summary: dict[str, Any],
    records: list[dict[str, Any]],
    protocol: MeasurementProtocol,
) -> None:
    performance = performance_summary(
        records,
        confidence_level=protocol.confidence_level,
    )
    summary["confidence_level"] = protocol.confidence_level
    summary["performance"] = performance
    summary.update(flatten_performance(performance))


def run_evaluation(
    *,
    benchmark_path: Path | str,
    model_config: ModelConfig,
    rag_config: RAGConfig,
    top_k: int = 5,
    language: str = "en",
    judge: Any | None = None,
    limit: int | None = None,
    run_id: str | None = None,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    pipeline: Any | None = None,
    store: EvaluationStore | None = None,
    embeddings: Embeddings | None = None,
    protocol: MeasurementProtocol = DEFAULT_PROTOCOL,
) -> RunResult:
    """
    Run one benchmark through one RAG configuration and persist raw results.

    The run is strictly sequential: first the configured number of discarded
    warm-up requests, then ``protocol.repetitions`` measured passes over the
    benchmark.  Every measured pass is stored with its repetition number, so
    statistics can be recomputed from the raw JSONL/CSV/SQLite records.

    ``pipeline`` and ``judge`` are injectable so tests and offline analyses can
    run without Ollama, cloud keys or Qdrant.
    """
    path = Path(benchmark_path)
    questions = load_benchmark(path)
    if not questions:
        msg = f"benchmark {path} contains no questions"
        raise ValueError(msg)

    datasets = {question.dataset for question in questions}
    if len(datasets) != 1:
        msg = f"benchmark {path} mixes datasets: {sorted(datasets)}"
        raise ValueError(msg)
    dataset = datasets.pop()

    if limit is not None:
        if limit < 1:
            msg = "limit must be at least 1"
            raise ValueError(msg)
        questions = questions[:limit]

    owns_store = store is None
    store = store or EvaluationStore(output_dir)
    run_id = run_id or generate_run_id(model_config, rag_config)
    judge_model = getattr(judge, "model_config", None)
    judge_model_name = getattr(judge_model, "name", None)
    num_ctx_supported = supports_num_ctx(model_config)
    effective_num_ctx = protocol.num_ctx if num_ctx_supported else None
    model_metadata = collect_model_metadata(model_config)

    if pipeline is None:
        from src.data.embeddings import EmbeddingProvider, get_embedding_model
        from src.rag.pipeline import create_pipeline

        embeddings = embeddings or get_embedding_model(provider=EmbeddingProvider.LOCAL)
        pipeline = create_pipeline(
            model_config,
            embeddings=embeddings,
            language=language,
            rag_config=rag_config,
            dataset=dataset,
            max_tokens=protocol.max_tokens,
            num_ctx=effective_num_ctx,
        )

    try:
        # Warm-up requests are intentionally discarded: they load the model,
        # initialise caches and stabilise GPU clocks/API connections.
        warmup_responses: list[Any] = []
        warmup_question_ids: list[str] = []
        for index in range(protocol.warmup_runs):
            question = questions[index % len(questions)]
            logger.info(
                "[%s] warm-up %d/%d %s",
                run_id,
                index + 1,
                protocol.warmup_runs,
                question.question_id,
            )
            warmup_responses.append(
                pipeline.ask(
                    question.question,
                    top_k=top_k,
                    language=language,
                )
            )
            warmup_question_ids.append(question.question_id)

        metadata = {
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "benchmark": path.name,
            "benchmark_path": str(path),
            "dataset": dataset,
            "model_name": model_config.name,
            "model_provider": model_config.provider.value,
            "model_id": model_config.model_id,
            "model_is_local": model_config.is_local,
            "model_parameters": model_config.parameters,
            "temperature": model_config.temperature,
            "max_tokens": protocol.max_tokens,
            "configured_max_tokens": model_config.max_tokens,
            "num_ctx": protocol.num_ctx,
            "num_ctx_supported": num_ctx_supported,
            "num_ctx_effective": effective_num_ctx,
            "model_quantization": model_metadata["quantization"],
            "model": model_metadata,
            "rag_config": rag_config.name,
            "chunk_size": rag_config.chunk_size,
            "chunk_overlap": rag_config.chunk_overlap,
            "use_reranking": rag_config.use_reranking,
            "top_k": top_k,
            "language": language,
            "judge_model": judge_model_name,
            "question_count": len(questions),
            "limit": limit,
            "protocol": protocol.to_dict(),
            "planned_measurements": len(questions) * protocol.repetitions,
            "warmup": _warmup_summary(warmup_responses, warmup_question_ids, protocol),
            "environment": collect_environment_metadata(),
        }

        store.start_run(metadata)
        records: list[dict[str, Any]] = []
        total = len(questions) * protocol.repetitions
        completed_count = 0
        for repetition in range(1, protocol.repetitions + 1):
            for question in questions:
                completed_count += 1
                try:
                    response = pipeline.ask(
                        question.question,
                        top_k=top_k,
                        language=language,
                    )
                    record = _evaluate_one(
                        run_id=run_id,
                        question=question,
                        response=response,
                        judge=judge,
                        repetition=repetition,
                    )
                except Exception as exc:  # noqa: BLE001 - keep the run and the raw error
                    record = _empty_record(
                        run_id=run_id,
                        question=question,
                        repetition=repetition,
                        error=_short_error(exc),
                    )
                records.append(record)
                store.append(record)
                logger.info(
                    "[%s] rep=%d %d/%d %s%s",
                    run_id,
                    repetition,
                    completed_count,
                    total,
                    question.question_id,
                    " (error)" if record.get("error") else "",
                )
    finally:
        if owns_store:
            store.close()

    summary = summarize_records(records)
    _add_performance_statistics(summary, records, protocol)
    summary.update(
        {
            "run_id": run_id,
            "benchmark": path.name,
            "dataset": dataset,
            "model_name": model_config.name,
            "model_provider": model_config.provider.value,
            "rag_config": rag_config.name,
            "top_k": top_k,
            "language": language,
            "judge_model": judge_model_name,
            "unique_questions": len(questions),
            "repetitions": protocol.repetitions,
            "warmup_runs": protocol.warmup_runs,
            "max_tokens": protocol.max_tokens,
            "num_ctx": protocol.num_ctx,
            "num_ctx_supported": num_ctx_supported,
            "num_ctx_effective": effective_num_ctx,
            "model_quantization": model_metadata["quantization"],
            "protocol": protocol.to_dict(),
            "warmup": metadata["warmup"],
            "question_statistics": _question_statistics(records, protocol),
            "repetition_statistics": _repetition_statistics(records, protocol),
        }
    )
    summary_path = store.runs_dir / f"{run_id}_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return RunResult(
        run_id=run_id,
        records=records,
        summary=summary,
        jsonl_path=store.runs_dir / f"{run_id}.jsonl",
        csv_path=store.runs_dir / f"{run_id}.csv",
        summary_path=summary_path,
    )
