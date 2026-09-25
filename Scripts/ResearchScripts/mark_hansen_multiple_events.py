from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from sqlalchemy import select


DRY_RUN = False
LIMIT_SAMPLES: int | None = None
APPEND_HANSEN_COMMENT = True
CREATE_MANUAL_VALIDATION_IF_MISSING = False
HANSEN_COMMENT_PREFIX = "[Hansen multi-event]"


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "Scripts"
RESEARCH_TOOL_ROOT = SCRIPTS_ROOT / "ResearchTool"
sys.path.insert(0, str(RESEARCH_TOOL_ROOT))

from backend.forest.database import SessionLocal, init_forest_database  # noqa: E402
from backend.forest.hansen_service import multiple_event_metrics  # noqa: E402
from backend.forest.models import HansenAnalysis, ManualValidation, Sample, utc_now  # noqa: E402


SCRIPT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_ROOT / "Outputs"


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_report(rows: list[dict[str, object]]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / f"hansen_multiple_events_{now_tag()}.csv"
    fieldnames = [
        "action",
        "sample_id",
        "analysis_id",
        "event_year",
        "status",
        "existing_manual_validation",
        "existing_notes",
        "old_has_multiple_events",
        "new_has_multiple_events",
        "multiple_event_year_count",
        "multiple_event_years",
        "error",
        "comment",
        "notes_after_hansen_comment",
        "comment_action",
    ]
    with report_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return report_path


def latest_analyses(db) -> list[HansenAnalysis]:
    analyses = list(
        db.scalars(
            select(HansenAnalysis).order_by(
                HansenAnalysis.sample_id,
                HansenAnalysis.created_at.desc(),
            )
        )
    )
    latest_by_sample: dict[str, HansenAnalysis] = {}
    for analysis in analyses:
        latest_by_sample.setdefault(analysis.sample_id, analysis)
    values = list(latest_by_sample.values())
    if LIMIT_SAMPLES is not None:
        values = values[:LIMIT_SAMPLES]
    return values


def hansen_comment(metrics: dict[str, object], analysis: HansenAnalysis) -> str:
    years = ", ".join(str(year) for year in metrics["multiple_event_years"])
    return (
        f"{HANSEN_COMMENT_PREFIX} detected {metrics['multiple_event_year_count']} loss years "
        f">= {metrics['multiple_event_cutoff_year']}: {years}. "
        f"Dominant event year: {analysis.event_year or 'n/a'}."
    )


def notes_with_hansen_comment(existing_notes: str | None, comment: str) -> str:
    notes = existing_notes or ""
    if not comment or HANSEN_COMMENT_PREFIX in notes:
        return notes
    separator = "\n" if notes.strip() else ""
    return f"{notes.rstrip()}{separator}{comment}"


def append_manual_note(db, sample_id: str, comment: str) -> str:
    manual = db.get(ManualValidation, sample_id)
    if manual is None:
        if not CREATE_MANUAL_VALIDATION_IF_MISSING:
            return "manual_validation_missing"
        if DRY_RUN:
            return "would_create_manual_validation_and_append_comment"
        manual = ManualValidation(sample_id=sample_id, validation="Unclear", notes=None)
        db.add(manual)

    existing_notes = manual.notes or ""
    if HANSEN_COMMENT_PREFIX in existing_notes:
        return "comment_already_present"

    if DRY_RUN:
        return "would_append_comment"

    manual.notes = notes_with_hansen_comment(existing_notes, comment)
    manual.updated_at = utc_now()
    return "comment_appended"


def main() -> None:
    init_forest_database()
    rows: list[dict[str, object]] = []

    with SessionLocal() as db:
        for analysis in latest_analyses(db):
            sample = db.get(Sample, analysis.sample_id)
            if sample is None:
                rows.append(
                    {
                        "action": "missing_sample",
                        "sample_id": analysis.sample_id,
                        "analysis_id": analysis.analysis_id,
                        "event_year": analysis.event_year,
                        "status": analysis.status,
                        "error": "Sample was not found.",
                        "comment": "",
                    }
                )
                continue

            try:
                histogram = json.loads(analysis.histogram_json)
                metrics = multiple_event_metrics(histogram)
            except Exception as exc:
                manual = db.get(ManualValidation, sample.sample_id)
                rows.append(
                    {
                        "action": "failed",
                        "sample_id": analysis.sample_id,
                        "analysis_id": analysis.analysis_id,
                        "event_year": analysis.event_year,
                        "status": analysis.status,
                        "existing_manual_validation": manual.validation if manual is not None else "",
                        "existing_notes": manual.notes if manual is not None and manual.notes else "",
                        "old_has_multiple_events": bool(sample.has_multiple_events),
                        "error": str(exc)[:500],
                    }
                )
                continue

            old_value = bool(sample.has_multiple_events)
            new_value = bool(metrics["has_multiple_events"])
            comment = hansen_comment(metrics, analysis) if new_value else ""
            manual = db.get(ManualValidation, sample.sample_id)
            existing_manual_validation = manual.validation if manual is not None else ""
            existing_notes = manual.notes if manual is not None and manual.notes else ""
            notes_after_hansen_comment = notes_with_hansen_comment(existing_notes, comment)
            comment_action = ""
            if old_value == new_value:
                action = "no_change"
            elif new_value:
                action = "would_mark" if DRY_RUN else "marked"
            else:
                action = "would_clear" if DRY_RUN else "cleared"

            if APPEND_HANSEN_COMMENT and new_value:
                comment_action = append_manual_note(db, sample.sample_id, comment)

            if not DRY_RUN and old_value != new_value:
                sample.has_multiple_events = new_value
                sample.updated_at = utc_now()

            rows.append(
                {
                    "action": action,
                    "sample_id": analysis.sample_id,
                    "analysis_id": analysis.analysis_id,
                    "event_year": analysis.event_year,
                    "status": analysis.status,
                    "existing_manual_validation": existing_manual_validation,
                    "existing_notes": existing_notes,
                    "old_has_multiple_events": old_value,
                    "new_has_multiple_events": new_value,
                    "multiple_event_year_count": metrics["multiple_event_year_count"],
                    "multiple_event_years": ",".join(str(year) for year in metrics["multiple_event_years"]),
                    "error": "",
                    "comment": comment,
                    "notes_after_hansen_comment": notes_after_hansen_comment,
                    "comment_action": comment_action,
                }
            )

        if DRY_RUN:
            db.rollback()
        else:
            db.commit()

    report_path = write_report(rows)
    print(f"DRY_RUN: {DRY_RUN}")
    print(f"LIMIT_SAMPLES: {LIMIT_SAMPLES}")
    print(f"APPEND_HANSEN_COMMENT: {APPEND_HANSEN_COMMENT}")
    print(f"CREATE_MANUAL_VALIDATION_IF_MISSING: {CREATE_MANUAL_VALIDATION_IF_MISSING}")
    print(f"Analyses checked: {len(rows)}")
    print(f"Actions: {dict(sorted(Counter(row['action'] for row in rows).items()))}")
    print(f"Comment actions: {dict(sorted(Counter(row.get('comment_action', '') for row in rows).items()))}")
    print(f"Rows with existing notes: {sum(1 for row in rows if row.get('existing_notes'))}")
    print(f"Report: {report_path}")
    if DRY_RUN:
        print("Dry run only. Set DRY_RUN = False to update sample has_multiple_events flags.")


if __name__ == "__main__":
    main()
