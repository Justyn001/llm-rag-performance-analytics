"""Tests for manual sample export, rating import and agreement."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.evaluation.manual_eval import (
    SHEET_FIELDS,
    agreement_report,
    cohen_kappa,
    import_manual_ratings,
    percent_agreement,
    select_manual_sample,
    write_manual_sheet,
)
from src.evaluation.storage import EvaluationStore
from tests.eval_helpers import make_metadata, make_record


def _seed_manual_run(store: EvaluationStore) -> list[dict]:
    records = [make_record(run_id="run-1", question_id=f"a-{index}") for index in range(4)]
    records += [
        make_record(
            run_id="run-1",
            question_id=f"u-{index}",
            answerable=False,
            answer="I don't have enough information.",
        )
        for index in range(4)
    ]
    store.start_run(make_metadata("run-1"))
    for record in records:
        store.append(record)
    return records


def _fill_sheet(path, *, rater_1: list[int], rater_2: list[int]) -> None:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEET_FIELDS)
        writer.writeheader()
        for index, row in enumerate(rows):
            for rater_id, values in (("rater_1", rater_1), ("rater_2", rater_2)):
                row[f"{rater_id}_correct"] = str(values[index])
                row[f"{rater_id}_faithful"] = str(values[index])
                row[f"{rater_id}_context_relevant"] = "1"
                row[f"{rater_id}_refusal_appropriate"] = str(values[index])
                row[f"{rater_id}_notes"] = f"note {rater_id} {index}"
            writer.writerow(row)


def test_manual_sample_is_stratified_and_deterministic():
    records = [make_record(question_id=f"a-{index}") for index in range(10)] + [
        make_record(question_id=f"u-{index}", answerable=False, answer="I don't know.")
        for index in range(10)
    ]

    first = select_manual_sample(records, sample_size=8, seed=42)
    second = select_manual_sample(records, sample_size=8, seed=42)

    assert [record["question_id"] for record in first] == [
        record["question_id"] for record in second
    ]
    assert len(first) == 8
    assert sum(1 for record in first if record["answerable"]) == 6
    assert sum(1 for record in first if not record["answerable"]) == 2


def test_manual_sheet_and_import_end_to_end(tmp_path):
    with EvaluationStore(tmp_path) as store:
        _seed_manual_run(store)
        records = store.get_records("run-1")
        sample = select_manual_sample(records, sample_size=4, seed=42)
        sheet_path = write_manual_sheet(sample, tmp_path / "manual.csv")

        assert sheet_path.exists()
        with sheet_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 4
        assert set(SHEET_FIELDS) <= set(rows[0])
        assert {row["answerable"] for row in rows} == {"true", "false"}

        _fill_sheet(sheet_path, rater_1=[1, 1, 0, 0], rater_2=[1, 1, 0, 0])
        imported = import_manual_ratings(sheet_path, store)

        assert imported["run_id"] == "run-1"
        assert imported["questions"] == 4
        assert imported["ratings_imported"] == 8
        assert Path(imported["human_ratings_jsonl"]).exists()
        assert Path(imported["human_ratings_csv"]).exists()
        assert len(Path(imported["human_ratings_jsonl"]).read_text().splitlines()) == 8

        ratings = store.get_human_ratings("run-1")
        assert len(ratings) == 8
        report = agreement_report(store, "run-1")
        assert report["questions_with_both_raters"] == 4
        assert report["metrics"]["correct"]["n"] == 4
        assert report["metrics"]["correct"]["percent_agreement"] == 1.0
        assert report["metrics"]["correct"]["cohen_kappa"] == 1.0


def test_import_skips_blank_rater_columns(tmp_path):
    with EvaluationStore(tmp_path) as store:
        _seed_manual_run(store)
        records = store.get_records("run-1")
        sample = select_manual_sample(records, sample_size=2, seed=42)
        sheet_path = write_manual_sheet(sample, tmp_path / "manual.csv")

        with sheet_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        with sheet_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SHEET_FIELDS)
            writer.writeheader()
            for row in rows:
                row["rater_1_correct"] = "1"
                row["rater_1_faithful"] = "1"
                writer.writerow(row)

        imported = import_manual_ratings(sheet_path, store)
        assert imported["ratings_imported"] == 2
        assert {rating["rater_id"] for rating in store.get_human_ratings("run-1")} == {"rater_1"}


def test_import_rejects_wrong_expected_run_id(tmp_path):
    with EvaluationStore(tmp_path) as store:
        _seed_manual_run(store)
        records = store.get_records("run-1")
        sample = select_manual_sample(records, sample_size=2, seed=42)
        sheet_path = write_manual_sheet(sample, tmp_path / "manual.csv")
        _fill_sheet(sheet_path, rater_1=[1, 1], rater_2=[1, 1])

        with pytest.raises(ValueError, match="does not match"):
            import_manual_ratings(sheet_path, store, expected_run_id="other-run")


def test_percent_agreement_and_kappa():
    assert percent_agreement([]) is None
    assert percent_agreement([(1, 1), (0, 0)]) == 1.0
    assert percent_agreement([(1, 0), (0, 1)]) == 0.0
    assert cohen_kappa([(1, 1), (0, 0)]) == 1.0
    assert cohen_kappa([(1, 1), (1, 0), (0, 0), (0, 1)]) == pytest.approx(0.0)
    assert cohen_kappa([]) is None


def test_manual_sample_handles_empty_records():
    assert select_manual_sample([], sample_size=5) == []
