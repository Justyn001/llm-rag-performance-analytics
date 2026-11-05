"""
CLI for running automatic RAG evaluations.

Examples:
    uv run python -m src.evaluation.run_eval --list-models
    uv run python -m src.evaluation.run_eval --list-benchmarks

    uv run python -m src.evaluation.run_eval \
        --benchmark wikipedia-simple \
        --models "Gemma 3 4B" \
        --rag-config basic-512 \
        --judge-model "Gemma 3 4B" \
        --limit 5

    uv run python -m src.evaluation.run_eval \
        --benchmark squad \
        --all-models \
        --rag-config basic-256 \
        --rag-config basic-512

Raw outputs are written under ``data/evaluation`` (ignored by git).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.config.models import ALL_MODELS, ModelConfig
from src.config.rag_config import RAG_CONFIGS, get_rag_config
from src.evaluation.benchmarks import BENCHMARKS_DIR
from src.evaluation.judge import make_judge
from src.evaluation.protocol import (
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_NUM_CTX,
    DEFAULT_REPETITIONS,
    DEFAULT_WARMUP_RUNS,
    MeasurementProtocol,
)
from src.evaluation.runner import (
    DEFAULT_OUTPUT_DIR,
    RunResult,
    _slugify,
    generate_run_id,
    run_evaluation,
)
from src.rag.llm_provider import list_available_models

BENCHMARK_FILES = {
    "wikipedia-simple": BENCHMARKS_DIR / "wikipedia_simple_benchmark.json",
    "squad": BENCHMARKS_DIR / "squad_benchmark.json",
    "trivia": BENCHMARKS_DIR / "trivia_benchmark.json",
}

RATE_KEYS = {
    "exact_match",
    "f1",
    "automatic_accuracy",
    "refusal_accuracy",
    "answerable_refusal_rate",
    "retrieval_hit_rate",
    "gold_span_in_context_rate",
    "judge_correct_rate",
    "judge_faithful_rate",
    "judge_context_relevant_rate",
    "judge_refusal_appropriate_rate",
}


def _resolve_model(name: str) -> ModelConfig:
    normalized = name.strip().lower()
    for model in ALL_MODELS:
        if normalized in {model.name.lower(), model.model_id.lower()}:
            return model
    available = ", ".join(model.name for model in ALL_MODELS)
    msg = f"unknown model {name!r}. Available: {available}"
    raise ValueError(msg)


def _split_model_names(values: list[str] | None) -> list[str]:
    names: list[str] = []
    for value in values or []:
        for name in value.split(","):
            name = name.strip()
            if name and name not in names:
                names.append(name)
    return names


def _select_models(args: argparse.Namespace) -> list[ModelConfig]:
    if args.all_models:
        if args.models:
            raise ValueError("use either --all-models or --models, not both")
        models = list_available_models()
        if not models:
            raise ValueError("no models are currently available (Ollama/API keys)")
        return models

    names = _split_model_names(args.models)
    if not names:
        available = ", ".join(model.name for model in ALL_MODELS)
        raise ValueError(
            f"no model selected. Use --models 'Model Name' or --all-models. Available: {available}"
        )
    return [_resolve_model(name) for name in names]


def _select_rag_configs(values: list[str] | None) -> list:
    names = values or ["basic-512"]
    return [get_rag_config(name) for name in names]


def _run_id_for(args: argparse.Namespace, model: ModelConfig, rag_config, index: int, total: int):
    if not args.run_id:
        return None
    if total == 1:
        return args.run_id
    return f"{args.run_id}_{_slugify(model.name)}_{rag_config.name}_{index + 1}"


def _format_value(key: str, value: object) -> str:
    if value is None:
        return "n/a"
    if key in RATE_KEYS and isinstance(value, int | float):
        return f"{value:.1%}"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def _print_performance(summary: dict) -> None:
    performance = summary.get("performance") or {}
    if not performance:
        return
    print(f"\n  performance (n per metric, CI {summary.get('confidence_level', 0.95):.0%}):")
    print(
        f"  {'metric':24s} {'n':>4s} {'mean':>10s} {'median':>10s} "
        f"{'stdev':>10s} {'ci_low':>10s} {'ci_high':>10s}"
    )
    for metric in (
        "retrieval_time_ms",
        "generation_time_ms",
        "total_time_ms",
        "tokens_per_second",
        "retry_backoff_time_ms",
    ):
        stats = performance.get(metric) or {}
        values = [stats.get(key) for key in ("mean", "median", "stdev", "ci_low", "ci_high")]

        def fmt(value):
            return "n/a" if value is None else f"{value:.2f}"

        print(
            f"  {metric:24s} {stats.get('n', 0):4d} "
            f"{fmt(values[0]):>10s} {fmt(values[1]):>10s} {fmt(values[2]):>10s} "
            f"{fmt(values[3]):>10s} {fmt(values[4]):>10s}"
        )


def _print_summary(result: RunResult) -> None:
    summary = result.summary
    print(f"\n=== Run {result.run_id} ===")
    print(
        f"  protocol: warm-up={summary.get('warmup_runs')}, "
        f"repetitions={summary.get('repetitions')}, "
        f"max_tokens={summary.get('max_tokens')}, "
        f"num_ctx={summary.get('num_ctx_effective')} "
        f"(requested={summary.get('num_ctx')})"
    )
    for key in (
        "total",
        "completed",
        "errors",
        "unique_questions",
        "answerable",
        "unanswerable",
        "exact_match",
        "f1",
        "automatic_accuracy",
        "refusal_accuracy",
        "answerable_refusal_rate",
        "retrieval_hit_rate",
        "gold_span_in_context_rate",
        "judged_count",
        "judge_errors",
        "judge_correct_rate",
        "judge_faithful_rate",
        "judge_context_relevant_rate",
        "judge_refusal_appropriate_rate",
        "mean_retrieval_time_ms",
        "mean_generation_time_ms",
        "mean_total_time_ms",
        "mean_tokens_per_second",
        "mean_retry_count",
        "mean_retry_backoff_time_ms",
    ):
        print(f"  {key:32s} {_format_value(key, summary.get(key))}")
    _print_performance(summary)
    if summary.get("errors"):
        print("  ! some questions failed; inspect the raw JSONL/CSV records")
    print(f"\n  JSONL:   {result.jsonl_path}")
    print(f"  CSV:     {result.csv_path}")
    print(f"  Summary: {result.summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run automatic RAG evaluation and save JSONL/CSV/SQLite results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--benchmark",
        choices=sorted(BENCHMARK_FILES),
        default="wikipedia-simple",
        help="benchmark set to run",
    )
    parser.add_argument(
        "--models",
        action="append",
        metavar="NAME",
        help="model name or model_id; repeat or comma-separate multiple values",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="run every model currently available via Ollama/API keys",
    )
    parser.add_argument(
        "--rag-config",
        action="append",
        choices=[config.name for config in RAG_CONFIGS],
        metavar="NAME",
        help="RAG configuration; repeat for multiple configs",
    )
    parser.add_argument(
        "--judge-model",
        metavar="NAME",
        help="model used as LLM-as-judge (omit to collect only lexical metrics)",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="disable the LLM judge even if --judge-model is set",
    )
    parser.add_argument("--limit", type=int, help="evaluate only the first N questions")
    parser.add_argument("--top-k", type=int, default=5, help="retrieved chunks per question")
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP_RUNS,
        help="discarded warm-up requests before measurement",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=DEFAULT_REPETITIONS,
        help="measured sequential passes over the benchmark",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="fixed output limit for every model",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=DEFAULT_NUM_CTX,
        help="fixed Ollama context window (cloud providers are provider-managed)",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=DEFAULT_CONFIDENCE_LEVEL,
        help="confidence level for interval estimates",
    )
    parser.add_argument("--language", choices=["en", "pl"], default="en")
    parser.add_argument(
        "--run-id",
        help="base run id; with multiple runs a model/config suffix is appended",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for JSONL/CSV/SQLite outputs",
    )
    parser.add_argument("--list-models", action="store_true", help="list configured models")
    parser.add_argument(
        "--list-benchmarks",
        action="store_true",
        help="list benchmark names and files",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if args.list_benchmarks:
        for name, path in BENCHMARK_FILES.items():
            print(f"{name:20s} {path}")
        return 0

    if args.list_models:
        for model in ALL_MODELS:
            location = "local" if model.is_local else "cloud"
            quantization = model.quantization or "unknown"
            print(
                f"{model.name:28s} {model.provider.value:8s} {location:5s} "
                f"{quantization:16s} {model.model_id}"
            )
        available = list_available_models()
        print("\nCurrently available:")
        if available:
            for model in available:
                print(f"  - {model.name}")
        else:
            print("  (none — start Ollama or configure an API key)")
        return 0

    try:
        models = _select_models(args)
        rag_configs = _select_rag_configs(args.rag_config)
    except ValueError as exc:
        parser.error(str(exc))

    judge = None
    if args.no_judge and args.judge_model:
        parser.error("use either --judge-model or --no-judge, not both")
    if args.judge_model and not args.no_judge:
        try:
            judge = make_judge(_resolve_model(args.judge_model))
        except ValueError as exc:
            parser.error(str(exc))

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    try:
        protocol = MeasurementProtocol(
            warmup_runs=args.warmup,
            repetitions=args.repetitions,
            max_tokens=args.max_tokens,
            num_ctx=args.num_ctx,
            confidence_level=args.confidence_level,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if not judge:
        print("No LLM judge configured; collecting exact match/F1/refusal metrics only.")

    benchmark_path = BENCHMARK_FILES[args.benchmark]
    jobs = [(model, rag_config) for model in models for rag_config in rag_configs]
    exit_code = 0

    try:
        from src.data.embeddings import EmbeddingProvider, get_embedding_model

        embeddings = get_embedding_model(provider=EmbeddingProvider.LOCAL)
    except Exception as exc:  # noqa: BLE001 - give a clear CLI message
        print(f"Could not load embeddings: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    for index, (model, rag_config) in enumerate(jobs):
        run_id = _run_id_for(args, model, rag_config, index, len(jobs))
        print(
            f"\nRunning {model.name} / {rag_config.name} on {args.benchmark}"
            + (f" (limit={args.limit})" if args.limit else "")
            + (
                f" [warm-up={protocol.warmup_runs}, repetitions={protocol.repetitions}, "
                f"max_tokens={protocol.max_tokens}, num_ctx={protocol.num_ctx}]"
            )
        )
        try:
            result = run_evaluation(
                benchmark_path=benchmark_path,
                model_config=model,
                rag_config=rag_config,
                top_k=args.top_k,
                language=args.language,
                judge=judge,
                limit=args.limit,
                run_id=run_id or generate_run_id(model, rag_config),
                output_dir=args.output_dir,
                embeddings=embeddings,
                protocol=protocol,
            )
        except Exception as exc:  # noqa: BLE001 - one failed model must not stop the batch
            print(
                f"Run failed for {model.name} / {rag_config.name}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            exit_code = 1
            continue
        _print_summary(result)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
