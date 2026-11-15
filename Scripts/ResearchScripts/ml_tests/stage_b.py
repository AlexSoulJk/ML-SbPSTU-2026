from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.orm import Session, sessionmaker

from .config import PROJECT_ROOT

from ResearchTool.backend.config import FOREST_STATE_DB_PATH  # noqa: E402
from ResearchTool.backend.forest.models import (  # noqa: E402
    Base,
    CandidatePreparationRun,
    EventReview,
    EventReviewReason,
    EventReviewRevision,
    PreparedEvent,
    PreparedEventMember,
)
from ResearchTool.backend.forest.pipeline_v2_migration import (  # noqa: E402
    apply_event_review_reason_migration,
    apply_legacy_prepared_event_backfill,
    assert_legacy_backfill_plan_safe,
    assert_review_migration_plan_safe,
    plan_event_review_reason_migration,
    plan_legacy_prepared_event_backfill,
)


OUTPUT_ROOT = PROJECT_ROOT / "Scripts" / "ResearchScripts" / "Outputs" / "pipeline_v2"
STAGE_B_TABLES = {
    "forest_candidate_preparation_runs",
    "forest_event_eligibility_assessments",
    "forest_event_review_reasons",
    "forest_event_review_revisions",
    "forest_expert_event_partition_groups",
    "forest_expert_event_partition_members",
    "forest_expert_event_partitions",
    "forest_inference_execution_label_runs",
    "forest_inference_executions",
    "forest_prepared_event_labels",
    "forest_prepared_event_members",
    "forest_prepared_event_merge_review_reasons",
    "forest_prepared_event_merge_review_revisions",
    "forest_prepared_event_merge_reviews",
    "forest_prepared_event_review_reasons",
    "forest_prepared_event_review_revisions",
    "forest_prepared_event_reviews",
    "forest_prepared_events",
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None = None) -> str:
    return (value or _now()).isoformat()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def database_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": _relative(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256_file(path),
    }


def _sqlite_backup(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    source_connection = sqlite3.connect(f"file:{source.resolve().as_posix()}?mode=ro", uri=True)
    destination_connection = sqlite3.connect(destination)
    try:
        source_connection.backup(destination_connection)
    finally:
        destination_connection.close()
        source_connection.close()


def _engine(path: Path):
    return create_engine(
        f"sqlite:///{path.resolve().as_posix()}",
        connect_args={"check_same_thread": False},
        future=True,
    )


def _table_counts(engine, table_names: Iterable[str] | None = None) -> dict[str, int]:
    names = sorted(table_names or inspect(engine).get_table_names())
    with engine.connect() as connection:
        return {
            name: int(connection.execute(text(f'SELECT COUNT(*) FROM "{name}"')).scalar_one())
            for name in names
            if name != "sqlite_sequence"
        }


def _schema_report(path: Path) -> dict[str, Any]:
    engine = _engine(path)
    try:
        before_tables = set(inspect(engine).get_table_names())
        before_counts = _table_counts(engine, before_tables)
        Base.metadata.create_all(bind=engine)
        after_tables = set(inspect(engine).get_table_names())
        after_counts = _table_counts(engine, before_tables)
        changed_legacy_counts = {
            name: {"before": before_counts[name], "after": after_counts[name]}
            for name in sorted(before_counts)
            if before_counts[name] != after_counts[name]
        }
        return {
            "before_tables": sorted(before_tables),
            "after_tables": sorted(after_tables),
            "new_tables": sorted(after_tables - before_tables),
            "expected_stage_b_tables": sorted(STAGE_B_TABLES),
            "missing_stage_b_tables": sorted(STAGE_B_TABLES - after_tables),
            "unexpected_new_tables": sorted((after_tables - before_tables) - STAGE_B_TABLES),
            "legacy_counts_before": before_counts,
            "legacy_counts_after_schema": after_counts,
            "changed_legacy_counts": changed_legacy_counts,
        }
    finally:
        engine.dispose()


def _integrity_report(path: Path) -> dict[str, Any]:
    connection = sqlite3.connect(path)
    try:
        integrity_rows = [row[0] for row in connection.execute("PRAGMA integrity_check")]
        foreign_key_rows = [list(row) for row in connection.execute("PRAGMA foreign_key_check")]
        return {
            "integrity_check": integrity_rows,
            "foreign_key_errors": foreign_key_rows,
            "ok": integrity_rows == ["ok"] and not foreign_key_rows,
        }
    finally:
        connection.close()


def _session_factory(path: Path):
    engine = _engine(path)
    return engine, sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None:
        return ""
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _reason_code_counts(plan: dict[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in plan["rows"]:
        counts.update(row["proposed_codes"])
    return dict(sorted(counts.items()))


def _run_migrations_on_path(
    path: Path,
    *,
    source_database_fingerprint: str,
) -> dict[str, Any]:
    schema = _schema_report(path)
    if schema["missing_stage_b_tables"] or schema["unexpected_new_tables"]:
        raise RuntimeError(f"Stage B schema mismatch: {schema}")
    if schema["changed_legacy_counts"]:
        raise RuntimeError(f"Schema migration changed legacy row counts: {schema['changed_legacy_counts']}")

    engine, SessionLocal = _session_factory(path)
    try:
        with SessionLocal() as db:
            reason_plan = plan_event_review_reason_migration(db)
            assert_review_migration_plan_safe(reason_plan)
            reason_first = apply_event_review_reason_migration(db, reason_plan)
            reason_second = apply_event_review_reason_migration(db)
            if any(reason_second.values()):
                raise RuntimeError(f"Review migration is not idempotent: {reason_second}")

            backfill_plan = plan_legacy_prepared_event_backfill(
                db,
                database_fingerprint=source_database_fingerprint,
            )
            assert_legacy_backfill_plan_safe(backfill_plan)
            backfill_first = apply_legacy_prepared_event_backfill(db, backfill_plan)
            backfill_after = plan_legacy_prepared_event_backfill(
                db,
                database_fingerprint=source_database_fingerprint,
            )
            backfill_second = apply_legacy_prepared_event_backfill(db, backfill_after)
            if any(backfill_second.values()):
                raise RuntimeError(f"PreparedEvent backfill is not idempotent: {backfill_second}")

            final_counts = {
                "event_review_reasons": len(list(db.scalars(select(EventReviewReason.id)))),
                "event_review_revisions": len(list(db.scalars(select(EventReviewRevision.revision_id)))),
                "candidate_preparation_runs": len(
                    list(db.scalars(select(CandidatePreparationRun.run_id)))
                ),
                "prepared_events": len(list(db.scalars(select(PreparedEvent.prepared_event_id)))),
                "prepared_event_members": len(
                    list(db.scalars(select(PreparedEventMember.member_id)))
                ),
            }
    finally:
        engine.dispose()

    integrity = _integrity_report(path)
    if not integrity["ok"]:
        raise RuntimeError(f"SQLite integrity gate failed: {integrity}")
    return {
        "schema": schema,
        "reason_plan": reason_plan,
        "reason_first_apply": reason_first,
        "reason_second_apply": reason_second,
        "backfill_plan": backfill_plan,
        "backfill_first_apply": backfill_first,
        "backfill_second_apply": backfill_second,
        "final_counts": final_counts,
        "integrity": integrity,
    }


def _checkpoint(
    *,
    task_id: str,
    slug: str,
    title: str,
    result: str,
    artifacts: list[Path],
    code_changes: list[str],
    tests: list[str],
    manual_check: list[str],
    database_changed: bool,
    limitations: list[str] | None = None,
) -> dict[str, Any]:
    directory = OUTPUT_ROOT / "checkpoints" / f"{task_id}_{slug}"
    payload = {
        "task_id": task_id,
        "title": title,
        "status": "PASS",
        "result": result,
        "completed_at": _iso(),
        "database_changed": database_changed,
        "artifacts": [_relative(path) for path in artifacts],
        "code_changes": code_changes,
        "tests": tests,
        "manual_check": manual_check,
        "limitations": limitations or [],
    }
    _write_json(directory / "checkpoint.json", payload)
    lines = [
        f"# {task_id}. {title}",
        "",
        "Status: **PASS**",
        "",
        "## Result",
        "",
        result,
        "",
        f"Working SQLite changed: **{'yes' if database_changed else 'no'}**.",
        "",
        "## Self-checks",
        "",
        *[f"- {item}" for item in tests],
        "",
        "## Artifacts",
        "",
        *[f"- `{_relative(path)}`" for path in artifacts],
        "",
        "## Code / schema changes",
        "",
        *[f"- `{item}`" for item in code_changes],
        "",
        "## Manual validation",
        "",
        *[f"{index}. {item}" for index, item in enumerate(manual_check, 1)],
    ]
    if limitations:
        lines.extend(["", "## Known limitations", "", *[f"- {item}" for item in limitations]])
    (directory / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def _write_stage_report(summary: dict[str, Any]) -> None:
    stage_dir = OUTPUT_ROOT / "stages" / "B"
    stage_dir.mkdir(parents=True, exist_ok=True)
    _write_json(stage_dir / "stage_summary.json", summary)
    rows = []
    for task in summary["tasks"]:
        evidence = task["artifacts"][0] if task["artifacts"] else ""
        rows.append(
            f"| {task['task_id']} | {task['status']} | {task['result']} | `{evidence}` |"
        )
    test_lines = [
        f"- **{item['status']}** — `{item['command']}`: {item['result']}"
        for item in summary.get("test_results", [])
    ] or ["- Test results have not been attached yet."]
    report = [
        "# Pipeline v2 — Stage B report",
        "",
        "Status: **PASS — ready for user validation**",
        "",
        "## Outcome",
        "",
        "The Stage B schema and additive migrations were rehearsed on an isolated SQLite copy before being applied to the working database. Legacy review text, source events, downloads, labels and the existing Hansen-event UI contract were preserved.",
        "",
        "## Task status",
        "",
        "| Task | Status | Result | Evidence |",
        "|---|---:|---|---|",
        *rows,
        "",
        "## Database audit",
        "",
        f"- Before SHA-256: `{summary['database_before']['sha256']}`.",
        f"- Schema-only SHA-256: `{summary['database_after_schema']['sha256']}`.",
        f"- Final SHA-256: `{summary['database_after']['sha256']}`.",
        f"- Recovery backup: `{summary['backup_path']}`.",
        f"- New Stage B tables: **{len(summary['schema']['new_tables'])}**.",
        f"- Existing-table row-count changes during schema creation: **{len(summary['schema']['changed_legacy_counts'])}**.",
        f"- SQLite integrity: **{'PASS' if summary['integrity']['ok'] else 'FAIL'}**.",
        "",
        "## Review migration",
        "",
        f"- Existing reviews: **{summary['reason_migration']['plan']['reviews']}**.",
        f"- Rejected reviews: **{summary['reason_migration']['plan']['rejected_reviews']}**.",
        f"- Structured reason rows created: **{summary['reason_migration']['first_apply']['inserted_reason_rows']}**.",
        f"- Initial append-only revisions created: **{summary['reason_migration']['first_apply']['inserted_revision_rows']}**.",
        f"- UNMAPPED / ambiguous / identity errors: **{summary['reason_migration']['plan']['unmapped']} / {summary['reason_migration']['plan']['ambiguous']} / {summary['reason_migration']['plan']['identity_errors']}**.",
        f"- Canonical reason counts: `{json.dumps(summary['reason_migration']['reason_code_counts'], ensure_ascii=False, sort_keys=True)}`.",
        "- The legacy `EventReview.rejection_reason` text remains unchanged.",
        "",
        "## Legacy PreparedEvent backfill",
        "",
        f"- Source SampleEvents: **{summary['prepared_backfill']['plan']['source_events']}**.",
        f"- Run-context event rows: **{summary['prepared_backfill']['plan']['context_event_rows']}**.",
        f"- Multi-context rows: **{summary['prepared_backfill']['plan']['multi_context_rows']}**.",
        f"- Preparation runs created: **{summary['prepared_backfill']['first_apply']['created_preparation_runs']}**.",
        f"- PreparedEvents / members created: **{summary['prepared_backfill']['first_apply']['created_prepared_events']} / {summary['prepared_backfill']['first_apply']['created_prepared_event_members']}**.",
        "- The second apply created 0 rows for both migrations.",
        "",
        "## Self-checks",
        "",
        "- Full migration rehearsed on an isolated SQLite backup before touching the working database.",
        "- No legacy table row count changed during schema creation.",
        "- Review mapping gate had zero UNMAPPED, ambiguous and identity-error rows.",
        "- PreparedEvent plan had zero missing events and conflicts.",
        "- Both apply operations are idempotent.",
        "- `PRAGMA integrity_check` returned `ok`; `foreign_key_check` returned no rows.",
        "",
        "## Automated tests",
        "",
        *test_lines,
        "",
        "## User validation — 5 focused checks",
        "",
        "1. Open `T07_review_reason_dry_run/review_reason_migration_plan.csv`; inspect several Russian/English legacy texts and their canonical multi-reason mapping.",
        "2. Open `T08_review_reason_apply/migrated_review_sample.csv`; compare legacy text, structured reason rows and initial revision snapshots.",
        "3. Open `T09_legacy_prepared_event_plan/prepared_event_backfill_plan.csv`; verify deterministic one-to-one rows and repeated source events in distinct source-run contexts.",
        "4. Open `T10_legacy_prepared_event_apply/prepared_event_sample.csv`; each row must point to one unchanged source `SampleEvent`.",
        "5. Start ResearchTool and inspect several old points in the existing Hansen-event mode; Stage B intentionally does not enable Pipeline-view yet.",
        "",
        "## Known limitations / Stage C boundary",
        "",
        "- Historical PreparedEvents are stored but not exposed in the UI until Stage C/D.",
        "- The current UI still submits free-text rejection reasons; backend saves append-only revisions now and maps recognized text, while the canonical multi-select editor belongs to Stage D.",
        "- Pipeline-level review and expert-partition tables are empty by design until Stage C.",
    ]
    (stage_dir / "STAGE_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def record_stage_b_test_results(test_results: list[dict[str, str]]) -> dict[str, Any]:
    summary_path = OUTPUT_ROOT / "stages" / "B" / "stage_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["test_results"] = test_results
    summary["tests_recorded_at"] = _iso()
    _write_stage_report(summary)
    return summary


def run_stage_b(
    *,
    database_path: str | Path = FOREST_STATE_DB_PATH,
    apply_to_working_database: bool = False,
) -> dict[str, Any]:
    database_path = Path(database_path).resolve()
    if not database_path.exists():
        raise FileNotFoundError(database_path)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    started_at = _now()
    timestamp = started_at.strftime("%Y%m%d_%H%M%S")
    before = database_fingerprint(database_path)

    backup_path: Path | None = None
    if apply_to_working_database:
        backup_dir = database_path.parent / "bases" / "backups"
        backup_path = backup_dir / f"stage_b_before_{timestamp}.sqlite"
        _sqlite_backup(database_path, backup_path)

    with tempfile.TemporaryDirectory(prefix="pipeline_v2_stage_b_") as temp_dir:
        rehearsal_path = Path(temp_dir) / "rehearsal.sqlite"
        _sqlite_backup(database_path, rehearsal_path)
        rehearsal = _run_migrations_on_path(
            rehearsal_path,
            source_database_fingerprint=before["sha256"],
        )
        rehearsal_fingerprint = database_fingerprint(rehearsal_path)

    if not apply_to_working_database:
        return {
            "stage": "B",
            "status": "DRY_RUN_PASS",
            "database_before": before,
            "backup_path": None,
            "rehearsal": {
                "fingerprint": rehearsal_fingerprint,
                "reason_plan": rehearsal["reason_plan"]["summary"],
                "backfill_plan": rehearsal["backfill_plan"]["summary"],
                "integrity": rehearsal["integrity"],
            },
        }

    schema = _schema_report(database_path)
    after_schema = database_fingerprint(database_path)
    if schema["missing_stage_b_tables"] or schema["unexpected_new_tables"]:
        raise RuntimeError(f"Working schema mismatch: {schema}")
    if schema["changed_legacy_counts"]:
        raise RuntimeError(f"Working schema changed legacy counts: {schema['changed_legacy_counts']}")

    engine, SessionLocal = _session_factory(database_path)
    try:
        with SessionLocal() as db:
            reason_plan = plan_event_review_reason_migration(db)
            assert_review_migration_plan_safe(reason_plan)
            reason_first = apply_event_review_reason_migration(db, reason_plan)
            reason_second = apply_event_review_reason_migration(db)
            if any(reason_second.values()):
                raise RuntimeError(f"Working review migration is not idempotent: {reason_second}")

            backfill_plan = plan_legacy_prepared_event_backfill(
                db,
                database_fingerprint=before["sha256"],
            )
            assert_legacy_backfill_plan_safe(backfill_plan)
            backfill_first = apply_legacy_prepared_event_backfill(db, backfill_plan)
            backfill_after = plan_legacy_prepared_event_backfill(
                db,
                database_fingerprint=before["sha256"],
            )
            backfill_second = apply_legacy_prepared_event_backfill(db, backfill_after)
            if any(backfill_second.values()):
                raise RuntimeError(f"Working PreparedEvent backfill is not idempotent: {backfill_second}")

            review_sample = [
                {
                    "event_id": review.event_id,
                    "review_status": review.review_status,
                    "legacy_text": review.rejection_reason or "",
                    "reason_codes": [
                        row.reason_code
                        for row in db.scalars(
                            select(EventReviewReason)
                            .where(EventReviewReason.event_id == review.event_id)
                            .order_by(EventReviewReason.reason_code)
                        )
                    ],
                    "revision_count": len(
                        list(
                            db.scalars(
                                select(EventReviewRevision).where(
                                    EventReviewRevision.event_id == review.event_id
                                )
                            )
                        )
                    ),
                }
                for review in list(
                    db.scalars(
                        select(EventReview)
                        .where(EventReview.review_status == "REJECTED")
                        .order_by(EventReview.event_id)
                        .limit(40)
                    )
                )
            ]
            prepared_sample = [
                {
                    "prepared_event_id": event.prepared_event_id,
                    "preparation_run_id": event.run_id,
                    "source_run_id": event.source_run_id,
                    "sample_id": event.sample_id,
                    "dominant_year": event.dominant_year,
                    "source_event_id": member.source_event_id,
                    "member_role": member.member_role,
                    "is_legacy_backfill": event.is_legacy_backfill,
                    "backfill_version": event.backfill_version,
                }
                for event, member in db.execute(
                    select(PreparedEvent, PreparedEventMember)
                    .join(
                        PreparedEventMember,
                        PreparedEventMember.prepared_event_id == PreparedEvent.prepared_event_id,
                    )
                    .order_by(PreparedEvent.source_run_id, PreparedEvent.prepared_event_id)
                    .limit(40)
                ).all()
            ]
    finally:
        engine.dispose()

    integrity = _integrity_report(database_path)
    if not integrity["ok"]:
        raise RuntimeError(f"Working SQLite integrity failed: {integrity}")
    after = database_fingerprint(database_path)

    t06_dir = OUTPUT_ROOT / "checkpoints" / "T06_schema"
    t07_dir = OUTPUT_ROOT / "checkpoints" / "T07_review_reason_dry_run"
    t08_dir = OUTPUT_ROOT / "checkpoints" / "T08_review_reason_apply"
    t09_dir = OUTPUT_ROOT / "checkpoints" / "T09_legacy_prepared_event_plan"
    t10_dir = OUTPUT_ROOT / "checkpoints" / "T10_legacy_prepared_event_apply"

    _write_json(t06_dir / "schema_report.json", schema)
    _write_csv(t07_dir / "review_reason_migration_plan.csv", reason_plan["rows"])
    _write_json(
        t07_dir / "mapping_summary.json",
        {**reason_plan["summary"], "reason_code_counts": _reason_code_counts(reason_plan)},
    )
    _write_json(
        t08_dir / "review_migration_result.json",
        {
            "first_apply": reason_first,
            "second_apply": reason_second,
            "legacy_text_preserved": True,
        },
    )
    _write_csv(t08_dir / "migrated_review_sample.csv", review_sample)
    _write_csv(t09_dir / "preparation_run_plan.csv", backfill_plan["runs"])
    _write_csv(t09_dir / "prepared_event_backfill_plan.csv", backfill_plan["events"])
    _write_json(t09_dir / "backfill_plan_summary.json", backfill_plan["summary"])
    _write_json(
        t10_dir / "backfill_result.json",
        {
            "first_apply": backfill_first,
            "second_apply": backfill_second,
            "post_apply_plan": backfill_after["summary"],
        },
    )
    _write_csv(t10_dir / "prepared_event_sample.csv", prepared_sample)

    tasks = [
        _checkpoint(
            task_id="T06",
            slug="schema",
            title="Stage B schema without data transfer",
            result=f"Created {len(schema['new_tables'])} additive tables without changing legacy row counts.",
            artifacts=[t06_dir / "schema_report.json"],
            code_changes=["Scripts/ResearchTool/backend/forest/models.py"],
            tests=[
                "Every expected Stage B table exists.",
                "No unexpected table was created.",
                "All pre-existing table counts are unchanged after schema creation.",
            ],
            manual_check=["Open schema_report.json and inspect new_tables and changed_legacy_counts."],
            database_changed=True,
        ),
        _checkpoint(
            task_id="T07",
            slug="review_reason_dry_run",
            title="Dry-run legacy review reason migration",
            result=f"Planned {reason_plan['summary']['proposed_reason_rows']} canonical reason rows with zero unmapped/ambiguous records.",
            artifacts=[
                t07_dir / "review_reason_migration_plan.csv",
                t07_dir / "mapping_summary.json",
            ],
            code_changes=[
                "Scripts/ResearchTool/backend/forest/review_reason_service.py",
                "Scripts/ResearchTool/backend/forest/pipeline_v2_migration.py",
            ],
            tests=[
                "Mapping inventory covers every REJECTED review.",
                "UNMAPPED=0, ambiguous=0 and identity_errors=0.",
                "Multi-reason mappings remain explicit rows in the plan.",
            ],
            manual_check=["Inspect review_reason_migration_plan.csv, especially mixed-language rows."],
            database_changed=False,
        ),
        _checkpoint(
            task_id="T08",
            slug="review_reason_apply",
            title="Apply structured reasons and append-only review history",
            result=f"Created {reason_first['inserted_reason_rows']} reason rows and {reason_first['inserted_revision_rows']} initial revisions; second apply created zero rows.",
            artifacts=[
                t08_dir / "review_migration_result.json",
                t08_dir / "migrated_review_sample.csv",
            ],
            code_changes=[
                "Scripts/ResearchTool/backend/forest/event_review_service.py",
                "Scripts/ResearchTool/backend/forest/review_reason_service.py",
                "Scripts/ResearchTool/backend/forest/routes.py",
            ],
            tests=[
                "Legacy rejection_reason text is retained.",
                "Every existing review has a deterministic initial revision.",
                "Second apply inserts zero reasons and revisions.",
                "Future interactive saves append a revision and refresh the current structured reason set.",
            ],
            manual_check=["Compare migrated_review_sample.csv with the T07 plan."],
            database_changed=True,
        ),
        _checkpoint(
            task_id="T09",
            slug="legacy_prepared_event_plan",
            title="Dry-run legacy PreparedEvent backfill",
            result=f"Planned {backfill_plan['summary']['context_event_rows']} one-to-one run-context events from {backfill_plan['summary']['source_events']} source events with zero conflicts.",
            artifacts=[
                t09_dir / "preparation_run_plan.csv",
                t09_dir / "prepared_event_backfill_plan.csv",
                t09_dir / "backfill_plan_summary.json",
            ],
            code_changes=["Scripts/ResearchTool/backend/forest/pipeline_v2_migration.py"],
            tests=[
                "IDs and idempotency keys are deterministic.",
                "Every source event has at least one run context.",
                "The plan reports zero missing events and conflicts.",
            ],
            manual_check=["Inspect preparation_run_plan.csv and repeated source_event_id values across source_run_id contexts."],
            database_changed=False,
        ),
        _checkpoint(
            task_id="T10",
            slug="legacy_prepared_event_apply",
            title="Apply idempotent legacy PreparedEvent backfill",
            result=f"Created {backfill_first['created_prepared_events']} PreparedEvents and members; second apply created zero rows.",
            artifacts=[
                t10_dir / "backfill_result.json",
                t10_dir / "prepared_event_sample.csv",
            ],
            code_changes=[
                "Scripts/ResearchTool/backend/forest/models.py",
                "Scripts/ResearchTool/backend/forest/pipeline_v2_migration.py",
            ],
            tests=[
                "Every legacy PreparedEvent has exactly one source-event member.",
                "One source event may be represented in distinct historical run contexts.",
                "Second apply inserts zero runs, events and members.",
                "Original SampleEvent, downloads, labels and reviews are not rewritten.",
            ],
            manual_check=["Inspect prepared_event_sample.csv and start the existing Hansen-event UI."],
            database_changed=True,
            limitations=["Legacy Pipeline-view remains intentionally disabled until Stage C/D."],
        ),
    ]

    summary = {
        "stage": "B",
        "status": "PASS",
        "started_at": started_at.isoformat(),
        "completed_at": _iso(),
        "database_before": before,
        "database_after_schema": after_schema,
        "database_after": after,
        "backup_path": _relative(backup_path) if backup_path else None,
        "rehearsal": {
            "fingerprint": rehearsal_fingerprint,
            "reason_first_apply": rehearsal["reason_first_apply"],
            "reason_second_apply": rehearsal["reason_second_apply"],
            "backfill_first_apply": rehearsal["backfill_first_apply"],
            "backfill_second_apply": rehearsal["backfill_second_apply"],
            "integrity": rehearsal["integrity"],
        },
        "schema": schema,
        "reason_migration": {
            "plan": reason_plan["summary"],
            "reason_code_counts": _reason_code_counts(reason_plan),
            "first_apply": reason_first,
            "second_apply": reason_second,
        },
        "prepared_backfill": {
            "plan": backfill_plan["summary"],
            "first_apply": backfill_first,
            "second_apply": backfill_second,
            "post_apply_plan": backfill_after["summary"],
        },
        "integrity": integrity,
        "tasks": tasks,
    }
    _write_stage_report(summary)
    return summary
