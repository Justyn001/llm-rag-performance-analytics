"""Tests for JSONL/CSV/SQLite evaluation storage."""

from __future__ import annotations

import csv
import json
import sqlite3

import pytest

from src.evaluation.storage import EvaluationStore
from tests.eval_helpers import make_metadata, make_record


def test_store_writes_jsonl_csv_and_sqlite(tmp_path):
    with EvaluationStore(tmp_path) as store:
        store.start_run(make_metadata("run-1"))
        store.append(make_record(run_id="run-1", question_id="q-1"))
        store.append(
            make_record(
                run_id="run-1",
                question_id="q-2",
                answerable=False,
                answer="I don't have enough information.",
            )
        )

        records = store.get_records("run-1")
        assert [record["question_id"] for record in records] == ["q-1", "q-2"]

        run = store.get_run("run-1")
        assert run is not None
        assert run["model_name"] == "Test Model"
        assert run["metadata"]["dataset"] == "wiki"

        runs = store.list_runs()
        assert len(runs) == 1
        assert runs[0]["run_id"] == "run-1"

    jsonl_path = tmp_path / "runs" / "run-1.jsonl"
    csv_path = tmp_path / "runs" / "run-1.csv"
    db_path = tmp_path / "evaluation.sqlite"
    assert jsonl_path.exists()
    assert csv_path.exists()
    assert db_path.exists()

    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["question_id"] == "q-1"

    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["dataset"] == "wiki"
    assert rows[0]["judge_correct"] == "1"
    assert rows[1]["answerable"] == "False"

    with sqlite3.connect(db_path) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM results WHERE run_id = ?", ("run-1",)
        ).fetchone()[0]
    assert count == 2


def test_store_reports_duplicate_run_and_wrong_active_run(tmp_path):
    with EvaluationStore(tmp_path) as store:
        store.start_run(make_metadata("run-1"))
        with pytest.raises(ValueError, match="already exists"):
            store.start_run(make_metadata("run-1"))

        with pytest.raises(RuntimeError, match="active run"):
            store.append(make_record(run_id="other", question_id="q-1"))


def test_store_upserts_human_ratings(tmp_path):
    with EvaluationStore(tmp_path) as store:
        store.start_run(make_metadata("run-1"))
        store.append(make_record(run_id="run-1", question_id="q-1"))
        store.insert_human_rating(
            run_id="run-1",
            question_id="q-1",
            rater_id="rater_1",
            correct=1,
            faithful=1,
            context_relevant=1,
            refusal_appropriate=1,
            notes="first",
        )
        store.insert_human_rating(
            run_id="run-1",
            question_id="q-1",
            rater_id="rater_1",
            correct=0,
            notes="changed",
        )

        ratings = store.get_human_ratings("run-1")
        assert len(ratings) == 1
        assert ratings[0]["correct"] == 0
        assert ratings[0]["faithful"] is None
        assert ratings[0]["notes"] == "changed"


def test_store_rejects_unsafe_run_id(tmp_path):
    with EvaluationStore(tmp_path) as store, pytest.raises(ValueError, match="invalid run_id"):
        store.start_run(make_metadata("../escape"))


def test_store_can_switch_to_a_second_run(tmp_path):
    with EvaluationStore(tmp_path) as store:
        store.start_run(make_metadata("run-1"))
        store.append(make_record(run_id="run-1", question_id="q-1"))

        store.start_run(make_metadata("run-2"))
        store.append(make_record(run_id="run-2", question_id="q-2"))

        assert [record["question_id"] for record in store.get_records("run-1")] == ["q-1"]
        assert [record["question_id"] for record in store.get_records("run-2")] == ["q-2"]
