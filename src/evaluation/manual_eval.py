"""
Manual evaluation workflow: sample sheet, two independent raters, agreement.

The automatic judge and EM/F1 are useful for scale, but a thesis-quality
evaluation also needs a human check.  This module:

1. selects a deterministic, stratified sample from one stored run,
2. exports a CSV with the rubric and separate columns for two raters,
3. imports both raters' 0/1 ratings into SQLite,
4. reports percentage agreement and Cohen's kappa per rubric dimension.

Usage:
    uv run python -m src.evaluation.manual_eval rubric
    uv run python -m src.evaluation.manual_eval sheet --run-id RUN_ID --sample-size 20
    uv run python -m src.evaluation.manual_eval import --sheet SHEET.csv
    uv run python -m src.evaluation.manual_eval agreement --run-id RUN_ID
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

from src.evaluation.storage import EvaluationStore

METRICS = ("correct", "faithful", "context_relevant", "refusal_appropriate")
RATER_IDS = ("rater_1", "rater_2")
RATER_FIELDS = [f"{rater}_{metric}" for rater in RATER_IDS for metric in (*METRICS, "notes")]
SHEET_FIELDS = [
    "run_id",
    "question_id",
    "dataset",
    "split",
    "set_type",
    "source",
    "answerable",
    "question",
    "expected_answer",
    "gold_context",
    "gold_span",
    "gold_title",
    "retrieved_context_titles",
    "retrieved_contexts",
    "answer",
    *RATER_FIELDS,
]

RUBRIC_TEXT = """Rubryka oceny ręcznej (0/1)

correct — poprawność odpowiedzi
  1: odpowiedź jest semantycznie zgodna z expected_answer (lub aliasem).
  0: odpowiedź jest błędna, niepełna, albo pomija wymaganą odpowiedź.
  Dla pytania unanswerable: 1 tylko wtedy, gdy model wprost odmawia odpowiedzi.

faithful — wierność wobec kontekstu
  1: każde twierdzenie faktu w odpowiedzi jest poparte którymś z retrieved_contexts.
  0: odpowiedź zawiera twierdzenie, którego nie ma w kontekście (halucynacja).
  Czysta odmowa bez wymyślonych faktów jest faithful = 1.

context_relevant — trafność kontekstu
  1: co najmniej jeden retrieved_context zawiera informacje potrzebne do pytania.
  0: żaden retrieved_context nie jest istotny dla pytania.

refusal_appropriate — trafność odmowy
  1: model odmawia dokładnie wtedy, gdy pytanie jest unanswerable.
  0: model odmawia przy answerable albo odpowiada na unanswerable.

notes — krótki komentarz (opcjonalnie), np. dlaczego answerable uznano za błędne.

Zasady:
- Dwóch oceniających wypełnia swoje kolumny niezależnie.
- Wartości: 0, 1 lub puste (puste = brak oceny; nie zgadujemy).
- Nie zmieniaj question_id ani run_id.
- Dla unanswerable pola gold_* są puste — to oczekiwane.
"""


def _context_text(record: dict[str, Any]) -> str:
    contexts = record.get("contexts") or []
    return "\n\n---\n\n".join(str(context) for context in contexts)


def _titles_text(record: dict[str, Any]) -> str:
    titles = record.get("context_titles") or []
    return " | ".join(str(title) for title in titles if title)


def select_manual_sample(
    records: list[dict[str, Any]],
    *,
    sample_size: int = 20,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Select a deterministic sample with answerable and control questions.

    At least one unanswerable control is included when the run contains one and
    the sample has room; the rest is filled from answerable questions.  The
    returned list is sorted by ``question_id`` for reproducible sheets.
    """
    if sample_size < 1:
        msg = "sample_size must be at least 1"
        raise ValueError(msg)

    unique: dict[str, dict[str, Any]] = {}
    for record in records:
        question_id = str(record.get("question_id", ""))
        if question_id and question_id not in unique:
            unique[question_id] = record
    if not unique:
        return []

    target = min(sample_size, len(unique))
    answerable = sorted(
        (record for record in unique.values() if record.get("answerable")),
        key=lambda record: str(record["question_id"]),
    )
    unanswerable = sorted(
        (record for record in unique.values() if not record.get("answerable")),
        key=lambda record: str(record["question_id"]),
    )

    rng = random.Random(seed)
    n_unanswerable = min(len(unanswerable), max(1, target // 4)) if unanswerable else 0
    n_answerable = min(len(answerable), target - n_unanswerable)
    selected = rng.sample(unanswerable, n_unanswerable) if n_unanswerable else []
    selected += rng.sample(answerable, n_answerable) if n_answerable else []

    remaining = target - len(selected)
    if remaining > 0:
        selected_ids = {str(record["question_id"]) for record in selected}
        rest = [
            record
            for record in (*unanswerable, *answerable)
            if str(record["question_id"]) not in selected_ids
        ]
        selected += rng.sample(rest, min(remaining, len(rest)))

    return sorted(selected, key=lambda record: str(record["question_id"]))


def _sheet_row(record: dict[str, Any]) -> dict[str, Any]:
    row = {
        "run_id": record.get("run_id", ""),
        "question_id": record.get("question_id", ""),
        "dataset": record.get("dataset", ""),
        "split": record.get("split", ""),
        "set_type": record.get("set_type", ""),
        "source": record.get("source", ""),
        "answerable": str(bool(record.get("answerable"))).lower(),
        "question": record.get("question", ""),
        "expected_answer": record.get("expected_answer") or "",
        "gold_context": record.get("gold_context") or "",
        "gold_span": record.get("gold_span") or "",
        "gold_title": record.get("gold_title") or "",
        "retrieved_context_titles": _titles_text(record),
        "retrieved_contexts": _context_text(record),
        "answer": record.get("answer", ""),
    }
    for field in RATER_FIELDS:
        row[field] = ""
    return row


def write_manual_sheet(
    records: list[dict[str, Any]],
    path: Path | str,
) -> Path:
    """Write a wide CSV sheet for two raters. Existing files are overwritten."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEET_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(_sheet_row(record))
    return output


def _parse_flag(value: Any, *, row_number: int, column: str) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "na", "n/a", "none", "-"}:
        return None
    if text in {"1", "1.0", "true", "yes", "y", "tak"}:
        return 1
    if text in {"0", "0.0", "false", "no", "n", "nie"}:
        return 0
    msg = f"row {row_number}, column {column}: expected 0/1 or blank, got {value!r}"
    raise ValueError(msg)


def import_manual_ratings(
    sheet_path: Path | str,
    store: EvaluationStore,
    *,
    expected_run_id: str | None = None,
) -> dict[str, Any]:
    """Import both rater columns from a sheet into SQLite."""
    path = Path(sheet_path)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        msg = f"manual sheet {path} has no data rows"
        raise ValueError(msg)

    sheet_run_ids = {str(row.get("run_id", "")).strip() for row in rows}
    sheet_run_ids.discard("")
    if expected_run_id is not None:
        if sheet_run_ids and sheet_run_ids != {expected_run_id}:
            msg = (
                f"sheet run_id {sorted(sheet_run_ids)} does not match "
                f"expected run_id {expected_run_id!r}"
            )
            raise ValueError(msg)
        run_id = expected_run_id
    else:
        if len(sheet_run_ids) != 1:
            msg = f"manual sheet must contain exactly one run_id, found: {sorted(sheet_run_ids)}"
            raise ValueError(msg)
        run_id = sheet_run_ids.pop()

    imported = 0
    questions = 0
    for row_number, row in enumerate(rows, start=2):
        question_id = str(row.get("question_id", "")).strip()
        if not question_id:
            msg = f"row {row_number}: missing question_id"
            raise ValueError(msg)
        questions += 1
        for rater_id in RATER_IDS:
            values = {
                metric: _parse_flag(
                    row.get(f"{rater_id}_{metric}"),
                    row_number=row_number,
                    column=f"{rater_id}_{metric}",
                )
                for metric in METRICS
            }
            notes = str(row.get(f"{rater_id}_notes") or "").strip()
            if all(value is None for value in values.values()) and not notes:
                continue
            store.insert_human_rating(
                run_id=run_id,
                question_id=question_id,
                rater_id=rater_id,
                correct=values["correct"],
                faithful=values["faithful"],
                context_relevant=values["context_relevant"],
                refusal_appropriate=values["refusal_appropriate"],
                notes=notes,
            )
            imported += 1

    jsonl_path, csv_path = store.export_human_ratings(run_id)
    return {
        "run_id": run_id,
        "questions": questions,
        "ratings_imported": imported,
        "sheet": str(path),
        "human_ratings_jsonl": str(jsonl_path),
        "human_ratings_csv": str(csv_path),
    }


def percent_agreement(pairs: list[tuple[int, int]]) -> float | None:
    """Simple observed agreement for paired binary ratings."""
    if not pairs:
        return None
    return sum(1 for left, right in pairs if left == right) / len(pairs)


def cohen_kappa(pairs: list[tuple[int, int]]) -> float | None:
    """Cohen's kappa for two raters and binary labels."""
    if not pairs:
        return None

    n = len(pairs)
    observed = sum(1 for left, right in pairs if left == right) / n
    left_positive = sum(left for left, _ in pairs) / n
    right_positive = sum(right for _, right in pairs) / n
    expected = (left_positive * right_positive) + ((1 - left_positive) * (1 - right_positive))
    if expected == 1:
        return 1.0 if observed == 1 else None
    return (observed - expected) / (1 - expected)


def agreement_report(store: EvaluationStore, run_id: str) -> dict[str, Any]:
    """Compute agreement for both raters across the four rubric dimensions."""
    ratings = store.get_human_ratings(run_id)
    by_question: dict[str, dict[str, dict[str, Any]]] = {}
    for rating in ratings:
        by_question.setdefault(str(rating["question_id"]), {})[str(rating["rater_id"])] = rating

    report_metrics: dict[str, dict[str, Any]] = {}
    for metric in METRICS:
        pairs: list[tuple[int, int]] = []
        for per_rater in by_question.values():
            left = per_rater.get("rater_1", {}).get(metric)
            right = per_rater.get("rater_2", {}).get(metric)
            if left is not None and right is not None:
                pairs.append((int(left), int(right)))
        report_metrics[metric] = {
            "n": len(pairs),
            "percent_agreement": percent_agreement(pairs),
            "cohen_kappa": cohen_kappa(pairs),
        }

    record_map = {
        str(record["question_id"]): record for record in store.get_records(run_id, repetition=1)
    }
    rated_ids = set(by_question)
    answerable_count = sum(
        1 for question_id in rated_ids if record_map.get(question_id, {}).get("answerable")
    )
    return {
        "run_id": run_id,
        "raters": list(RATER_IDS),
        "questions_with_any_rating": len(rated_ids),
        "questions_with_both_raters": sum(
            1
            for per_rater in by_question.values()
            if "rater_1" in per_rater and "rater_2" in per_rater
        ),
        "answerable_rated": answerable_count,
        "unanswerable_rated": len(rated_ids) - answerable_count,
        "metrics": report_metrics,
    }


def _print_agreement(report: dict[str, Any]) -> None:
    print(
        f"Run {report['run_id']}: {report['questions_with_both_raters']} questions with both raters"
    )
    print(f"{'metric':24s} {'n':>4s} {'agreement':>10s} {'kappa':>8s}")
    for metric, values in report["metrics"].items():
        agreement = values["percent_agreement"]
        kappa = values["cohen_kappa"]
        agreement_text = "n/a" if agreement is None else f"{agreement:.1%}"
        kappa_text = "n/a" if kappa is None else f"{kappa:.3f}"
        print(f"{metric:24s} {values['n']:4d} {agreement_text:>10s} {kappa_text:>8s}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manual evaluation sheet/import/agreement workflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/evaluation"),
        help="evaluation directory containing evaluation.sqlite",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("rubric", help="print the manual rating rubric")

    sheet = subparsers.add_parser("sheet", help="export a stratified sample CSV")
    sheet.add_argument("--run-id", required=True)
    sheet.add_argument("--sample-size", type=int, default=20)
    sheet.add_argument("--seed", type=int, default=42)
    sheet.add_argument("--path", type=Path, help="override output CSV path")

    import_cmd = subparsers.add_parser("import", help="import a completed CSV sheet")
    import_cmd.add_argument("--sheet", type=Path, required=True)
    import_cmd.add_argument("--run-id", help="require this run id in the sheet")

    agreement = subparsers.add_parser("agreement", help="report inter-rater agreement")
    agreement.add_argument("--run-id", required=True)
    agreement.add_argument("--json", action="store_true", help="print JSON instead of a table")

    list_runs = subparsers.add_parser("list-runs", help="list stored evaluation runs")
    list_runs.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "rubric":
        print(RUBRIC_TEXT)
        return 0

    try:
        store = EvaluationStore(args.output_dir)
    except Exception as exc:  # noqa: BLE001 - CLI should explain, not dump a traceback
        print(f"Could not open evaluation store: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    try:
        if args.command == "list-runs":
            runs = store.list_runs()
            if args.json:
                print(json.dumps(runs, indent=2, ensure_ascii=False))
            else:
                print(f"{'run_id':48s} {'dataset':8s} {'model':28s} {'rag_config':16s}")
                for run in runs:
                    print(
                        f"{run['run_id']:48s} {run['dataset']:8s} "
                        f"{run['model_name']:28s} {run['rag_config']:16s}"
                    )
            return 0

        if args.command == "sheet":
            if args.sample_size < 1:
                parser.error("--sample-size must be at least 1")
            records = [
                record
                for record in store.get_records(args.run_id, repetition=1)
                if not record.get("error")
            ]
            if not records:
                print(f"No completed records for run {args.run_id!r}", file=sys.stderr)
                return 1
            sample = select_manual_sample(
                records,
                sample_size=args.sample_size,
                seed=args.seed,
            )
            path = args.path or (store.manual_dir / f"{args.run_id}_manual_sheet.csv")
            write_manual_sheet(sample, path)
            answerable = sum(1 for record in sample if record.get("answerable"))
            print(
                f"Wrote {len(sample)} questions ({answerable} answerable, "
                f"{len(sample) - answerable} unanswerable) to {path}"
            )
            return 0

        if args.command == "import":
            result = import_manual_ratings(
                args.sheet,
                store,
                expected_run_id=args.run_id,
            )
            print(
                f"Imported {result['ratings_imported']} ratings for "
                f"{result['questions']} questions in run {result['run_id']}"
            )
            return 0

        if args.command == "agreement":
            report = agreement_report(store, args.run_id)
            output_path = store.manual_dir / f"{args.run_id}_agreement.json"
            output_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            if args.json:
                print(json.dumps(report, indent=2, ensure_ascii=False))
            else:
                _print_agreement(report)
                print(f"\nAgreement report: {output_path}")
            return 0
    finally:
        store.close()

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
