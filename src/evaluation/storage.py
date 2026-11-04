"""
Persistent storage for evaluation runs.

Every run is written to three places:

- ``runs/<run_id>.jsonl`` — append-only raw records (one JSON object per line),
- ``runs/<run_id>.csv`` — flat table for spreadsheets / quick analysis,
- ``evaluation.sqlite`` — queryable tables for runs, results and human ratings.

JSONL is the source of truth.  The CSV and SQLite copies are derived from the
same record so they cannot silently disagree.
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")

HUMAN_RATING_FIELDS = [
    "run_id",
    "question_id",
    "rater_id",
    "correct",
    "faithful",
    "context_relevant",
    "refusal_appropriate",
    "notes",
    "updated_at",
]

CSV_FIELDS = [
    "run_id",
    "question_id",
    "repetition",
    "dataset",
    "split",
    "set_type",
    "source",
    "question",
    "answerable",
    "expected_answer",
    "gold_context",
    "gold_span",
    "gold_title",
    "gold_url",
    "answer_aliases_json",
    "answer",
    "contexts_json",
    "context_titles_json",
    "retrieval_time_ms",
    "generation_time_ms",
    "retry_backoff_time_ms",
    "failed_attempt_time_ms",
    "retry_count",
    "total_time_ms",
    "max_tokens",
    "num_ctx",
    "tokens_in",
    "tokens_out",
    "tokens_per_second",
    "exact_match",
    "f1",
    "is_refusal",
    "refusal_correct",
    "automatic_correct",
    "refusal_pattern",
    "gold_title_rank",
    "retrieval_hit",
    "gold_span_in_context",
    "judge_correct",
    "judge_faithful",
    "judge_context_relevant",
    "judge_refusal_appropriate",
    "judge_reason",
    "error",
]

RESULT_COLUMNS = [
    "run_id",
    "question_id",
    "repetition",
    "question",
    "answerable",
    "expected_answer",
    "gold_context",
    "gold_span",
    "gold_title",
    "answer",
    "contexts_json",
    "context_titles_json",
    "retrieval_time_ms",
    "generation_time_ms",
    "retry_backoff_time_ms",
    "failed_attempt_time_ms",
    "retry_count",
    "total_time_ms",
    "max_tokens",
    "num_ctx",
    "tokens_in",
    "tokens_out",
    "tokens_per_second",
    "exact_match",
    "f1",
    "is_refusal",
    "refusal_correct",
    "automatic_correct",
    "refusal_pattern",
    "gold_title_rank",
    "retrieval_hit",
    "gold_span_in_context",
    "judge_correct",
    "judge_faithful",
    "judge_context_relevant",
    "judge_refusal_appropriate",
    "judge_reason",
    "judged",
    "error",
    "raw_json",
]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    benchmark TEXT NOT NULL,
    dataset TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_provider TEXT NOT NULL,
    model_id TEXT NOT NULL,
    rag_config TEXT NOT NULL,
    top_k INTEGER NOT NULL,
    language TEXT NOT NULL,
    judge_model TEXT,
    metadata_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    question_id TEXT NOT NULL,
    repetition INTEGER NOT NULL DEFAULT 1,
    question TEXT NOT NULL,
    answerable INTEGER NOT NULL,
    expected_answer TEXT,
    gold_context TEXT,
    gold_span TEXT,
    gold_title TEXT,
    answer TEXT NOT NULL,
    contexts_json TEXT NOT NULL,
    context_titles_json TEXT NOT NULL,
    retrieval_time_ms REAL NOT NULL DEFAULT 0,
    generation_time_ms REAL NOT NULL DEFAULT 0,
    retry_backoff_time_ms REAL NOT NULL DEFAULT 0,
    failed_attempt_time_ms REAL NOT NULL DEFAULT 0,
    retry_count INTEGER NOT NULL DEFAULT 0,
    total_time_ms REAL NOT NULL DEFAULT 0,
    max_tokens INTEGER NOT NULL DEFAULT 0,
    num_ctx INTEGER,
    tokens_in INTEGER NOT NULL DEFAULT 0,
    tokens_out INTEGER NOT NULL DEFAULT 0,
    tokens_per_second REAL NOT NULL DEFAULT 0,
    exact_match REAL,
    f1 REAL,
    is_refusal INTEGER NOT NULL,
    refusal_correct INTEGER,
    automatic_correct INTEGER,
    refusal_pattern TEXT,
    gold_title_rank INTEGER,
    retrieval_hit INTEGER,
    gold_span_in_context INTEGER,
    judge_correct INTEGER,
    judge_faithful INTEGER,
    judge_context_relevant INTEGER,
    judge_refusal_appropriate INTEGER,
    judge_reason TEXT,
    judged INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    raw_json TEXT NOT NULL,
    UNIQUE(run_id, repetition, question_id)
);

CREATE INDEX IF NOT EXISTS idx_results_run_id ON results(run_id);

CREATE TABLE IF NOT EXISTS human_ratings (
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    question_id TEXT NOT NULL,
    rater_id TEXT NOT NULL,
    correct INTEGER,
    faithful INTEGER,
    context_relevant INTEGER,
    refusal_appropriate INTEGER,
    notes TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL,
    PRIMARY KEY (run_id, question_id, rater_id)
);
"""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _json_default(value: Any) -> str:
    return str(value)


class EvaluationStore:
    """JSONL + CSV + SQLite writer/reader for one or more evaluation runs."""

    def __init__(self, output_dir: Path | str = "data/evaluation"):
        self.output_dir = Path(output_dir)
        self.runs_dir = self.output_dir / "runs"
        self.manual_dir = self.output_dir / "manual"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.manual_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.output_dir / "evaluation.sqlite"

        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._migrate_results_schema()
        self._conn.commit()

        self._jsonl_handle: Any = None
        self._csv_handle: Any = None
        self._csv_writer: csv.DictWriter | None = None
        self._current_run_id: str | None = None

    def __enter__(self) -> EvaluationStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _migrate_results_schema(self) -> None:
        """Upgrade a P0.4 ``results`` table to the repetition-aware schema."""
        columns = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(results)").fetchall()
        }
        required = {
            "repetition",
            "retry_backoff_time_ms",
            "failed_attempt_time_ms",
            "retry_count",
            "max_tokens",
            "num_ctx",
        }
        if required <= columns and self._has_repetition_unique_index():
            return

        self._conn.execute("PRAGMA foreign_keys=OFF")
        self._conn.execute("ALTER TABLE results RENAME TO results_legacy")
        self._conn.execute("DROP INDEX IF EXISTS idx_results_run_id")
        # _SCHEMA uses IF NOT EXISTS, so this recreates only the results table.
        self._conn.executescript(_SCHEMA)

        legacy_columns = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(results_legacy)").fetchall()
        }
        defaults = {
            "repetition": "1",
            "retry_backoff_time_ms": "0",
            "failed_attempt_time_ms": "0",
            "retry_count": "0",
            "max_tokens": "0",
            "num_ctx": "NULL",
        }
        select_columns = ", ".join(
            column if column in legacy_columns else defaults.get(column, "NULL")
            for column in RESULT_COLUMNS
        )
        insert_columns = ", ".join(RESULT_COLUMNS)
        self._conn.execute(
            f"INSERT INTO results ({insert_columns}) "
            f"SELECT {select_columns} FROM results_legacy"
        )
        self._conn.execute("DROP TABLE results_legacy")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.commit()

    def _has_repetition_unique_index(self) -> bool:
        indexes = self._conn.execute("PRAGMA index_list(results)").fetchall()
        for index in indexes:
            if not index["unique"]:
                continue
            columns = [
                row["name"]
                for row in self._conn.execute(
                    f"PRAGMA index_info({index['name']})"
                ).fetchall()
            ]
            if columns == ["run_id", "repetition", "question_id"]:
                return True
        return False

    def _close_run_files(self) -> None:
        if self._jsonl_handle is not None:
            self._jsonl_handle.close()
            self._jsonl_handle = None
        if self._csv_handle is not None:
            self._csv_handle.close()
            self._csv_handle = None
        self._csv_writer = None
        self._current_run_id = None

    def close(self) -> None:
        """Flush and close open file handles and the SQLite connection."""
        self._close_run_files()
        if self._conn is not None:
            self._conn.close()
            self._conn = None  # type: ignore[assignment]

    def start_run(self, metadata: dict[str, Any]) -> None:
        """Register a run and create its JSONL/CSV output files."""
        run_id = str(metadata.get("run_id", ""))
        if not RUN_ID_RE.fullmatch(run_id):
            msg = f"invalid run_id {run_id!r}; use letters, digits, '.', '_' or '-'"
            raise ValueError(msg)

        existing = self._conn.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if existing:
            msg = f"run_id {run_id!r} already exists in {self.db_path}"
            raise ValueError(msg)

        self._close_run_files()
        self._conn.execute(
            """
            INSERT INTO runs (
                run_id, created_at, benchmark, dataset, model_name, model_provider,
                model_id, rag_config, top_k, language, judge_model, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                metadata.get("created_at") or _now_iso(),
                metadata.get("benchmark", ""),
                metadata.get("dataset", ""),
                metadata.get("model_name", ""),
                metadata.get("model_provider", ""),
                metadata.get("model_id", ""),
                metadata.get("rag_config", ""),
                int(metadata.get("top_k", 0)),
                metadata.get("language", "en"),
                metadata.get("judge_model"),
                json.dumps(metadata, ensure_ascii=False, default=_json_default),
            ),
        )
        self._conn.commit()

        jsonl_path = self.runs_dir / f"{run_id}.jsonl"
        csv_path = self.runs_dir / f"{run_id}.csv"
        self._jsonl_handle = jsonl_path.open("w", encoding="utf-8")
        self._csv_handle = csv_path.open("w", encoding="utf-8", newline="")
        self._csv_writer = csv.DictWriter(self._csv_handle, fieldnames=CSV_FIELDS)
        self._csv_writer.writeheader()
        self._csv_handle.flush()
        self._current_run_id = run_id

    def append(self, record: dict[str, Any]) -> None:
        """Append one fully evaluated question to all three stores."""
        run_id = str(record.get("run_id", ""))
        if not self._current_run_id or run_id != self._current_run_id:
            msg = f"no active run for record with run_id {run_id!r}"
            raise RuntimeError(msg)
        if self._jsonl_handle is None or self._csv_writer is None:
            msg = "start_run() must be called before append()"
            raise RuntimeError(msg)

        self._jsonl_handle.write(
            json.dumps(record, ensure_ascii=False, default=_json_default) + "\n"
        )
        self._jsonl_handle.flush()

        self._csv_writer.writerow(self._csv_row(record))
        self._csv_handle.flush()

        params = self._sql_params(record)
        placeholders = ", ".join("?" for _ in RESULT_COLUMNS)
        self._conn.execute(
            f"INSERT INTO results ({', '.join(RESULT_COLUMNS)}) VALUES ({placeholders})",
            params,
        )
        self._conn.commit()

    def get_records(
        self,
        run_id: str,
        *,
        repetition: int | None = None,
    ) -> list[dict[str, Any]]:
        """Read raw records for a run from the SQLite mirror."""
        query = "SELECT raw_json FROM results WHERE run_id = ?"
        params: list[Any] = [run_id]
        if repetition is not None:
            query += " AND repetition = ?"
            params.append(repetition)
        query += " ORDER BY id"
        rows = self._conn.execute(query, params).fetchall()
        return [json.loads(row["raw_json"]) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Return run metadata, with the nested metadata JSON expanded."""
        row = self._conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["metadata"] = json.loads(data.pop("metadata_json"))
        return data

    def list_runs(self) -> list[dict[str, Any]]:
        """Return run metadata, newest first."""
        rows = self._conn.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            data = dict(row)
            data["metadata"] = json.loads(data.pop("metadata_json"))
            result.append(data)
        return result

    def insert_human_rating(
        self,
        *,
        run_id: str,
        question_id: str,
        rater_id: str,
        correct: int | None = None,
        faithful: int | None = None,
        context_relevant: int | None = None,
        refusal_appropriate: int | None = None,
        notes: str = "",
    ) -> None:
        """Insert or update one rater's ratings for one question."""
        if not rater_id.strip():
            msg = "rater_id must not be empty"
            raise ValueError(msg)
        self._conn.execute(
            """
            INSERT INTO human_ratings (
                run_id, question_id, rater_id, correct, faithful,
                context_relevant, refusal_appropriate, notes, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, question_id, rater_id) DO UPDATE SET
                correct = excluded.correct,
                faithful = excluded.faithful,
                context_relevant = excluded.context_relevant,
                refusal_appropriate = excluded.refusal_appropriate,
                notes = excluded.notes,
                updated_at = excluded.updated_at
            """,
            (
                run_id,
                question_id,
                rater_id,
                correct,
                faithful,
                context_relevant,
                refusal_appropriate,
                notes,
                _now_iso(),
            ),
        )
        self._conn.commit()

    def get_human_ratings(self, run_id: str) -> list[dict[str, Any]]:
        """Return all human ratings for a run, grouped by question then rater."""
        rows = self._conn.execute(
            """
            SELECT run_id, question_id, rater_id, correct, faithful,
                   context_relevant, refusal_appropriate, notes, updated_at
            FROM human_ratings
            WHERE run_id = ?
            ORDER BY question_id, rater_id
            """,
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def export_human_ratings(self, run_id: str) -> tuple[Path, Path]:
        """Export human ratings to JSONL and CSV alongside the manual sheet."""
        ratings = self.get_human_ratings(run_id)
        jsonl_path = self.manual_dir / f"{run_id}_human_ratings.jsonl"
        csv_path = self.manual_dir / f"{run_id}_human_ratings.csv"

        with jsonl_path.open("w", encoding="utf-8") as handle:
            for rating in ratings:
                handle.write(json.dumps(rating, ensure_ascii=False, default=_json_default) + "\n")
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=HUMAN_RATING_FIELDS)
            writer.writeheader()
            writer.writerows(ratings)
        return jsonl_path, csv_path

    def _csv_row(self, record: dict[str, Any]) -> dict[str, Any]:
        judge = record.get("judge") or {}
        return {
            "run_id": record.get("run_id"),
            "question_id": record.get("question_id"),
            "repetition": record.get("repetition", 1),
            "dataset": record.get("dataset"),
            "split": record.get("split"),
            "set_type": record.get("set_type"),
            "source": record.get("source"),
            "question": record.get("question"),
            "answerable": record.get("answerable"),
            "expected_answer": record.get("expected_answer"),
            "gold_context": record.get("gold_context"),
            "gold_span": record.get("gold_span"),
            "gold_title": record.get("gold_title"),
            "gold_url": record.get("gold_url"),
            "answer_aliases_json": json.dumps(
                record.get("answer_aliases", []),
                ensure_ascii=False,
                default=_json_default,
            ),
            "answer": record.get("answer"),
            "contexts_json": json.dumps(
                record.get("contexts", []), ensure_ascii=False, default=_json_default
            ),
            "context_titles_json": json.dumps(
                record.get("context_titles", []), ensure_ascii=False, default=_json_default
            ),
            "retrieval_time_ms": record.get("retrieval_time_ms"),
            "generation_time_ms": record.get("generation_time_ms"),
            "retry_backoff_time_ms": record.get("retry_backoff_time_ms"),
            "failed_attempt_time_ms": record.get("failed_attempt_time_ms"),
            "retry_count": record.get("retry_count"),
            "total_time_ms": record.get("total_time_ms"),
            "max_tokens": record.get("max_tokens"),
            "num_ctx": record.get("num_ctx"),
            "tokens_in": record.get("tokens_in"),
            "tokens_out": record.get("tokens_out"),
            "tokens_per_second": record.get("tokens_per_second"),
            "exact_match": record.get("exact_match"),
            "f1": record.get("f1"),
            "is_refusal": record.get("is_refusal"),
            "refusal_correct": record.get("refusal_correct"),
            "automatic_correct": record.get("automatic_correct"),
            "refusal_pattern": record.get("refusal_pattern"),
            "gold_title_rank": record.get("gold_title_rank"),
            "retrieval_hit": record.get("retrieval_hit"),
            "gold_span_in_context": record.get("gold_span_in_context"),
            "judge_correct": judge.get("correct"),
            "judge_faithful": judge.get("faithful"),
            "judge_context_relevant": judge.get("context_relevant"),
            "judge_refusal_appropriate": judge.get("refusal_appropriate"),
            "judge_reason": judge.get("reason"),
            "error": record.get("error"),
        }

    def _sql_params(self, record: dict[str, Any]) -> tuple[Any, ...]:
        judge = record.get("judge") or {}
        judged = int(bool(judge) and not judge.get("error"))
        return (
            record.get("run_id"),
            record.get("question_id"),
            int(record.get("repetition") or 1),
            record.get("question"),
            int(bool(record.get("answerable"))),
            record.get("expected_answer"),
            record.get("gold_context"),
            record.get("gold_span"),
            record.get("gold_title"),
            record.get("answer", ""),
            json.dumps(record.get("contexts", []), ensure_ascii=False, default=_json_default),
            json.dumps(record.get("context_titles", []), ensure_ascii=False, default=_json_default),
            float(record.get("retrieval_time_ms") or 0),
            float(record.get("generation_time_ms") or 0),
            float(record.get("retry_backoff_time_ms") or 0),
            float(record.get("failed_attempt_time_ms") or 0),
            int(record.get("retry_count") or 0),
            float(record.get("total_time_ms") or 0),
            int(record.get("max_tokens") or 0),
            record.get("num_ctx"),
            int(record.get("tokens_in") or 0),
            int(record.get("tokens_out") or 0),
            float(record.get("tokens_per_second") or 0),
            record.get("exact_match"),
            record.get("f1"),
            int(bool(record.get("is_refusal"))),
            _nullable_int(record.get("refusal_correct")),
            _nullable_int(record.get("automatic_correct")),
            record.get("refusal_pattern"),
            record.get("gold_title_rank"),
            _nullable_int(record.get("retrieval_hit")),
            _nullable_int(record.get("gold_span_in_context")),
            _nullable_int(judge.get("correct")),
            _nullable_int(judge.get("faithful")),
            _nullable_int(judge.get("context_relevant")),
            _nullable_int(judge.get("refusal_appropriate")),
            judge.get("reason"),
            judged,
            record.get("error"),
            json.dumps(record, ensure_ascii=False, default=_json_default),
        )


def _nullable_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(bool(value))
