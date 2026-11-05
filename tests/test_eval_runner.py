"""Tests for the end-to-end evaluation runner (with fake model/judge)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.evaluation.runner as runner_module
from src.config.models import GEMMA_3_4B
from src.config.rag_config import BASIC_512
from src.evaluation.benchmarks import BenchmarkQuestion
from src.evaluation.judge import JudgeVerdict
from src.evaluation.protocol import MeasurementProtocol
from src.evaluation.runner import run_evaluation
from src.evaluation.storage import EvaluationStore


class FakePipeline:
    def __init__(self, fail_on: str | None = None):
        self.fail_on = fail_on
        self.calls: list[str] = []

    def ask(self, question: str, top_k: int = 5, language: str | None = None):
        self.calls.append(question)
        if self.fail_on and self.fail_on in question:
            raise RuntimeError("boom")

        if "king" in question:
            answer = "I don't have enough information to answer this question."
            contexts = ["France is a republic."]
            titles = ["France"]
        else:
            answer = "Paris"
            contexts = ["Paris is the capital of France."]
            titles = ["France"]

        return SimpleNamespace(
            answer=answer,
            contexts=contexts,
            context_metadata=[{"document_title": title} for title in titles],
            retrieval_time_ms=1.5,
            generation_time_ms=2.5,
            retry_backoff_time_ms=0.0,
            failed_attempt_time_ms=0.0,
            retry_count=0,
            total_time_ms=4.0,
            max_tokens=1024,
            num_ctx=8192,
            tokens_in=12,
            tokens_out=3,
            tokens_per_second=1.2,
        )


class FakeJudge:
    model_config = SimpleNamespace(name="Fake Judge")

    def evaluate(self, **kwargs):
        return JudgeVerdict(
            correct=1,
            faithful=1,
            context_relevant=1,
            refusal_appropriate=1,
            reason="ok",
        )


class BrokenJudge:
    model_config = SimpleNamespace(name="Broken Judge")

    def evaluate(self, **kwargs):
        raise RuntimeError("judge unavailable")


@pytest.fixture(autouse=True)
def _offline_model_metadata(monkeypatch):
    """Keep metadata collection deterministic in tests (no Ollama probe)."""
    monkeypatch.setattr(runner_module, "_ollama_model_details", lambda _model_id: None)


@pytest.fixture()
def single_pass() -> MeasurementProtocol:
    """P0.4-compatible protocol: no warm-up, one measured pass."""
    return MeasurementProtocol(warmup_runs=0, repetitions=1)


@pytest.fixture()
def benchmark_path(tmp_path: Path) -> Path:
    questions = [
        BenchmarkQuestion(
            question_id="q-1",
            question="What is the capital of France?",
            answerable=True,
            expected_answer="Paris",
            gold_context="Paris is the capital of France and its largest city.",
            gold_span="Paris",
            dataset="wiki",
            split="custom",
            set_type="custom",
            source="test",
            gold_title="France",
            answer_aliases=["Paris, France"],
        ),
        BenchmarkQuestion(
            question_id="q-2",
            question="Who is the current king of France?",
            answerable=False,
            dataset="wiki",
            split="custom",
            set_type="custom",
            source="test",
        ),
    ]
    path = tmp_path / "benchmark.json"
    path.write_text(
        json.dumps([question.model_dump() for question in questions]),
        encoding="utf-8",
    )
    return path


def test_runner_records_metrics_and_summary(tmp_path, benchmark_path, single_pass):
    with EvaluationStore(tmp_path) as store:
        result = run_evaluation(
            benchmark_path=benchmark_path,
            model_config=GEMMA_3_4B,
            rag_config=BASIC_512,
            judge=FakeJudge(),
            run_id="run-1",
            output_dir=tmp_path,
            pipeline=FakePipeline(),
            store=store,
            protocol=single_pass,
        )

        records = store.get_records("run-1")
        assert result.records == records
        assert len(records) == 2
        assert records[0]["exact_match"] == 1.0
        assert records[0]["retrieval_hit"] is True
        assert records[0]["gold_span_in_context"] is True
        assert records[1]["answerable"] is False
        assert records[1]["refusal_correct"] is True
        assert records[1]["judge"]["correct"] == 1

        assert result.summary["exact_match"] == 1.0
        assert result.summary["refusal_accuracy"] == 1.0
        assert result.summary["judge_correct_rate"] == 1.0
        assert result.summary["errors"] == 0

    assert result.jsonl_path.exists()
    assert result.csv_path.exists()
    assert result.summary_path.exists()
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["run_id"] == "run-1"


def test_runner_keeps_going_after_pipeline_error(tmp_path, benchmark_path, single_pass):
    with EvaluationStore(tmp_path) as store:
        result = run_evaluation(
            benchmark_path=benchmark_path,
            model_config=GEMMA_3_4B,
            rag_config=BASIC_512,
            judge=FakeJudge(),
            run_id="run-error",
            output_dir=tmp_path,
            pipeline=FakePipeline(fail_on="capital"),
            store=store,
            protocol=single_pass,
        )

        records = store.get_records("run-error")
        assert len(records) == 2
        assert "RuntimeError: boom" in records[0]["error"]
        assert records[0]["exact_match"] is None
        assert records[1]["error"] is None
        assert result.summary["errors"] == 1
        assert result.summary["completed"] == 1


def test_runner_records_judge_error_without_losing_answer(tmp_path, benchmark_path, single_pass):
    with EvaluationStore(tmp_path) as store:
        result = run_evaluation(
            benchmark_path=benchmark_path,
            model_config=GEMMA_3_4B,
            rag_config=BASIC_512,
            judge=BrokenJudge(),
            run_id="run-judge-error",
            output_dir=tmp_path,
            pipeline=FakePipeline(),
            store=store,
            protocol=single_pass,
        )

        records = store.get_records("run-judge-error")
        assert records[0]["answer"] == "Paris"
        assert "judge unavailable" in records[0]["judge"]["error"]
        assert result.summary["judge_errors"] == 2


def test_runner_limit_and_no_judge(tmp_path, benchmark_path, single_pass):
    with EvaluationStore(tmp_path) as store:
        result = run_evaluation(
            benchmark_path=benchmark_path,
            model_config=GEMMA_3_4B,
            rag_config=BASIC_512,
            judge=None,
            limit=1,
            run_id="run-limit",
            output_dir=tmp_path,
            pipeline=FakePipeline(),
            store=store,
            protocol=single_pass,
        )

        if len(result.records) != 1:
            pytest.fail("limit should produce exactly one record")
        assert result.records[0]["question_id"] == "q-1"
        assert result.records[0]["judge"] is None
        assert result.summary["judged_count"] == 0
        assert result.summary["judge_errors"] == 0


def test_runner_owns_and_closes_its_store(tmp_path, benchmark_path, single_pass):
    result = run_evaluation(
        benchmark_path=benchmark_path,
        model_config=GEMMA_3_4B,
        rag_config=BASIC_512,
        judge=None,
        limit=1,
        run_id="run-owned",
        output_dir=tmp_path,
        pipeline=FakePipeline(),
        protocol=single_pass,
    )

    assert result.jsonl_path.exists()
    assert result.csv_path.exists()
    assert result.summary_path.exists()
    with EvaluationStore(tmp_path) as store:
        assert [record["question_id"] for record in store.get_records("run-owned")] == ["q-1"]


def test_runner_runs_warmup_then_repeated_sequential_passes(tmp_path, benchmark_path):
    pipeline = FakePipeline()
    protocol = MeasurementProtocol(warmup_runs=2, repetitions=3)

    with EvaluationStore(tmp_path) as store:
        result = run_evaluation(
            benchmark_path=benchmark_path,
            model_config=GEMMA_3_4B,
            rag_config=BASIC_512,
            judge=None,
            limit=2,
            run_id="run-repeated",
            output_dir=tmp_path,
            pipeline=pipeline,
            store=store,
            protocol=protocol,
        )

        records = store.get_records("run-repeated")
        assert len(records) == 6
        assert [record["repetition"] for record in records] == [1, 1, 2, 2, 3, 3]
        assert [record["question_id"] for record in records] == [
            "q-1",
            "q-2",
            "q-1",
            "q-2",
            "q-1",
            "q-2",
        ]

        # Two discarded warm-ups, then three full measured passes.
        assert pipeline.calls == [
            "What is the capital of France?",
            "Who is the current king of France?",
            "What is the capital of France?",
            "Who is the current king of France?",
            "What is the capital of France?",
            "Who is the current king of France?",
            "What is the capital of France?",
            "Who is the current king of France?",
        ]

        run = store.get_run("run-repeated")
        assert run is not None
        metadata = run["metadata"]
        assert metadata["protocol"]["warmup_runs"] == 2
        assert metadata["protocol"]["repetitions"] == 3
        assert metadata["protocol"]["execution_mode"] == "sequential"
        assert metadata["warmup"]["count"] == 2
        assert metadata["max_tokens"] == 1024
        assert metadata["num_ctx_effective"] == 8192
        assert metadata["model_quantization"] == "Q4_K_M"

        assert result.summary["unique_questions"] == 2
        assert result.summary["repetitions"] == 3
        assert result.summary["warmup_runs"] == 2
        assert result.summary["warmup"]["count"] == 2
        assert result.summary["performance"]["generation_time_ms"]["n"] == 6
        assert result.summary["performance"]["generation_time_ms"]["mean"] == 2.5
        assert result.summary["performance"]["generation_time_ms"]["stdev"] == 0.0
        assert len(result.summary["repetition_statistics"]) == 3
        assert len(result.summary["question_statistics"]) == 2
        assert result.summary["repetition_statistics"][0]["quality"]["exact_match"] == 1.0
