# llm-rag-performance-analytics
Performance benchmark and evaluation of local vs. cloud LLMs using Retrieval-Augmented Generation.

## Evaluation

P0.4 adds automatic evaluation (exact match/F1, refusal detection, retrieval
diagnostics and an LLM-as-judge) plus a two-rater manual rubric. P0.5 adds a
fixed measurement protocol: warm-up requests, repeated sequential passes,
common `max_tokens`/`num_ctx`, retry-excluded generation timing, model
quantisation/hardware metadata and per-question/repetition statistics with
confidence intervals. Raw results are stored as JSONL, CSV and SQLite under
`data/evaluation/`.

See [`docs/evaluation.md`](docs/evaluation.md) for commands, the measurement
protocol, output schema and the manual evaluation workflow.
