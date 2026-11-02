# llm-rag-performance-analytics
Performance benchmark and evaluation of local vs. cloud LLMs using Retrieval-Augmented Generation.

## Evaluation

P0.4 adds automatic evaluation (exact match/F1, refusal detection, retrieval
diagnostics and an LLM-as-judge) plus a two-rater manual rubric. Raw results
are stored as JSONL, CSV and SQLite under `data/evaluation/`.

See [`docs/evaluation.md`](docs/evaluation.md) for commands, output schema and
the manual evaluation workflow.
