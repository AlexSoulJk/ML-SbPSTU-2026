from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import (
    CandidatePreparationRun,
    EventLabel,
    EventReview,
    EventReviewReason,
    EventReviewRevision,
    HansenCandidate,
    HansenCandidateRun,
    HansenCandidateRunItem,
    PreparedEvent,
    PreparedEventMember,
    SampleEvent,
)
from .review_reason_service import (
    append_event_review_revision,
    map_legacy_rejection_reason,
)


REVIEW_MIGRATION_VERSION = "event_review_reasons_v1"
LEGACY_BACKFILL_VERSION = "legacy_prepared_event_backfill_v1"
LEGACY_PIPELINE_NAME = "legacy_prepared_event_backfill"
LEGACY_PIPELINE_VERSION = "legacy_v1"
LEGACY_ALGORITHM_VERSION = "one_to_one_v1"
LEGACY_UNASSIGNED_SOURCE_RUN = "legacy_unassigned"


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha1(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _iso(value: datetime | None) -> str:
    return value.isoformat() if value is not None else ""


def _legacy_revision_id(review: EventReview) -> str:
    return _sha1(
        "|".join(
            (
                REVIEW_MIGRATION_VERSION,
                review.event_id,
                _iso(review.updated_at),
            )
        )
    )


def plan_event_review_reason_migration(db: Session) -> dict[str, Any]:
    reviews = list(db.scalars(select(EventReview).order_by(EventReview.event_id)))
    event_ids = set(db.scalars(select(SampleEvent.event_id)))
    existing_reasons: dict[str, list[str]] = defaultdict(list)
    for row in db.scalars(select(EventReviewReason).order_by(EventReviewReason.event_id)):
        existing_reasons[row.event_id].append(row.reason_code)
    existing_revisions = set(db.scalars(select(EventReviewRevision.revision_id)))
    legacy_revision_events = set(
        db.scalars(
            select(EventReviewRevision.event_id).where(
                EventReviewRevision.revision_source == "LEGACY_MIGRATION"
            )
        )
    )

    rows: list[dict[str, Any]] = []
    for review in reviews:
        already_migrated = review.event_id in legacy_revision_events
        if already_migrated:
            proposed = []
            mapping_status = "ALREADY_MIGRATED"
            mapping_rules = []
        elif review.review_status == "REJECTED":
            mapping = map_legacy_rejection_reason(review.rejection_reason)
            proposed = list(mapping.reason_codes)
            mapping_status = mapping.mapping_status
            mapping_rules = list(mapping.mapping_rules)
        else:
            proposed = []
            mapping_status = "NOT_APPLICABLE"
            mapping_rules = []
        revision_id = _legacy_revision_id(review)
        rows.append(
            {
                "review_id": review.event_id,
                "event_id": review.event_id,
                "review_status": review.review_status,
                "legacy_text": review.rejection_reason or "",
                "proposed_codes": proposed,
                "proposed_codes_text": "|".join(proposed),
                "mapping_status": mapping_status,
                "mapping_rules": mapping_rules,
                "mapping_rules_text": "|".join(mapping_rules),
                "existing_codes": sorted(existing_reasons.get(review.event_id, [])),
                "revision_id": revision_id,
                "revision_exists": (
                    revision_id in existing_revisions or already_migrated
                ),
                "identity_status": "OK" if review.event_id in event_ids else "MISSING_EVENT",
                "review_updated_at": _iso(review.updated_at),
            }
        )

    status_counts = Counter(row["mapping_status"] for row in rows)
    summary = {
        "migration_version": REVIEW_MIGRATION_VERSION,
        "reviews": len(rows),
        "rejected_reviews": sum(row["review_status"] == "REJECTED" for row in rows),
        "status_counts": dict(sorted(status_counts.items())),
        "proposed_reason_rows": sum(len(row["proposed_codes"]) for row in rows),
        "unmapped": sum(row["mapping_status"] == "UNMAPPED" for row in rows),
        "ambiguous": sum(row["mapping_status"] == "AMBIGUOUS" for row in rows),
        "identity_errors": sum(row["identity_status"] != "OK" for row in rows),
        "would_create_revisions": sum(not row["revision_exists"] for row in rows),
    }
    return {"summary": summary, "rows": rows}


def assert_review_migration_plan_safe(plan: dict[str, Any]) -> None:
    summary = plan["summary"]
    failures = []
    for key in ("unmapped", "ambiguous", "identity_errors"):
        if int(summary.get(key) or 0):
            failures.append(f"{key}={summary[key]}")
    if failures:
        raise ValueError("Review migration gate failed: " + ", ".join(failures))


def apply_event_review_reason_migration(
    db: Session,
    plan: dict[str, Any] | None = None,
) -> dict[str, int]:
    migration_plan = plan or plan_event_review_reason_migration(db)
    assert_review_migration_plan_safe(migration_plan)
    inserted_reasons = 0
    inserted_revisions = 0

    for row in migration_plan["rows"]:
        review = db.get(EventReview, row["event_id"])
        if review is None:
            raise ValueError(f"EventReview disappeared during migration: {row['event_id']}")
        existing_codes = set(
            db.scalars(
                select(EventReviewReason.reason_code).where(
                    EventReviewReason.event_id == review.event_id
                )
            )
        )
        for code in row["proposed_codes"]:
            if code in existing_codes:
                continue
            db.add(
                EventReviewReason(
                    event_id=review.event_id,
                    reason_code=code,
                    reason_text=(
                        review.rejection_reason if code == "OTHER_INVALID" else None
                    ),
                    created_at=review.updated_at,
                )
            )
            inserted_reasons += 1
        existing_legacy_revision = db.scalar(
            select(EventReviewRevision.revision_id).where(
                EventReviewRevision.event_id == review.event_id,
                EventReviewRevision.revision_source == "LEGACY_MIGRATION",
            )
        )
        if existing_legacy_revision is None:
            accepted = db.get(EventLabel, review.human_label_id) if review.human_label_id else None
            append_event_review_revision(
                db,
                review,
                reason_codes=row["proposed_codes"],
                revision_source="LEGACY_MIGRATION",
                revision_id=row["revision_id"],
                event_class=accepted.label if accepted is not None else None,
                created_at=review.updated_at,
            )
            inserted_revisions += 1
    db.commit()
    return {
        "inserted_reason_rows": inserted_reasons,
        "inserted_revision_rows": inserted_revisions,
    }


def _preparation_run_id(source_run_id: str) -> str:
    return _sha1(f"{LEGACY_BACKFILL_VERSION}|run|{source_run_id}")


def _prepared_event_id(source_run_id: str, source_event_id: str) -> str:
    return _sha1(f"{LEGACY_BACKFILL_VERSION}|event|{source_run_id}|{source_event_id}")


def _prepared_member_id(source_run_id: str, source_event_id: str) -> str:
    return _sha1(f"{LEGACY_BACKFILL_VERSION}|member|{source_run_id}|{source_event_id}")


def _idempotency_key(source_run_id: str, source_event_id: str) -> str:
    return _sha256(
        {
            "backfill_version": LEGACY_BACKFILL_VERSION,
            "source_event_id": source_event_id,
            "source_run_id": source_run_id,
        }
    )


def _source_contexts(db: Session) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    candidate_runs = list(
        db.scalars(select(HansenCandidateRun).order_by(HansenCandidateRun.run_id))
    )
    contexts: dict[str, set[str]] = {run.run_id: set() for run in candidate_runs}
    rows = db.execute(
        select(HansenCandidateRun.run_id, SampleEvent.event_id)
        .join(
            HansenCandidateRunItem,
            HansenCandidateRunItem.run_id == HansenCandidateRun.run_id,
        )
        .join(
            HansenCandidate,
            HansenCandidate.candidate_id == HansenCandidateRunItem.candidate_id,
        )
        .join(
            SampleEvent,
            SampleEvent.sample_id == HansenCandidate.created_sample_id,
        )
        .where(HansenCandidate.created_sample_id.is_not(None))
        .order_by(HansenCandidateRun.run_id, SampleEvent.event_id)
    ).all()
    for run_id, event_id in rows:
        contexts[str(run_id)].add(str(event_id))

    all_event_ids = set(db.scalars(select(SampleEvent.event_id)))
    assigned = set().union(*contexts.values()) if contexts else set()
    unassigned = all_event_ids - assigned
    if unassigned:
        contexts[LEGACY_UNASSIGNED_SOURCE_RUN] = unassigned

    run_rows = [
        {
            "source_run_id": run.run_id,
            "source_candidate_run_id": run.run_id,
            "source_status": run.status,
            "source_created_at": _iso(run.created_at),
        }
        for run in candidate_runs
    ]
    if unassigned:
        run_rows.append(
            {
                "source_run_id": LEGACY_UNASSIGNED_SOURCE_RUN,
                "source_candidate_run_id": None,
                "source_status": "SYNTHETIC",
                "source_created_at": "",
            }
        )
    return run_rows, contexts


def plan_legacy_prepared_event_backfill(
    db: Session,
    *,
    database_fingerprint: str,
) -> dict[str, Any]:
    source_run_rows, contexts = _source_contexts(db)
    events = {
        event.event_id: event
        for event in db.scalars(select(SampleEvent).order_by(SampleEvent.event_id))
    }
    existing_runs = {
        row.run_id: row for row in db.scalars(select(CandidatePreparationRun))
    }
    existing_events_by_id = {
        row.prepared_event_id: row for row in db.scalars(select(PreparedEvent))
    }
    existing_events_by_key = {
        row.idempotency_key: row for row in existing_events_by_id.values()
    }
    existing_members_by_id = {
        row.member_id: row for row in db.scalars(select(PreparedEventMember))
    }

    run_plan: list[dict[str, Any]] = []
    preparation_run_ids: dict[str, str] = {}
    for source in source_run_rows:
        source_run_id = str(source["source_run_id"])
        prep_run_id = _preparation_run_id(source_run_id)
        preparation_run_ids[source_run_id] = prep_run_id
        config = {
            "backfill_version": LEGACY_BACKFILL_VERSION,
            "mode": "ONE_TO_ONE",
            "source_run_id": source_run_id,
            "source_candidate_run_id": source["source_candidate_run_id"],
        }
        existing = existing_runs.get(prep_run_id)
        action = "create"
        conflict = ""
        if existing is not None:
            expected = (
                existing.pipeline_name == LEGACY_PIPELINE_NAME
                and existing.pipeline_version == LEGACY_PIPELINE_VERSION
                and existing.algorithm_version == LEGACY_ALGORITHM_VERSION
                and existing.source_run_id == source_run_id
                and existing.source_candidate_run_id == source["source_candidate_run_id"]
                and existing.config_hash == _sha256(config)
            )
            action = "reuse" if expected else "conflict"
            if not expected:
                conflict = "existing preparation run metadata differs"
        run_plan.append(
            {
                **source,
                "preparation_run_id": prep_run_id,
                "pipeline_name": LEGACY_PIPELINE_NAME,
                "pipeline_version": LEGACY_PIPELINE_VERSION,
                "algorithm_version": LEGACY_ALGORITHM_VERSION,
                "config": config,
                "config_json": _stable_json(config),
                "config_hash": _sha256(config),
                "database_fingerprint": database_fingerprint,
                "event_count": len(contexts.get(source_run_id, set())),
                "action": action,
                "conflict": conflict,
            }
        )

    event_plan: list[dict[str, Any]] = []
    missing_events = 0
    for source_run_id in sorted(contexts):
        prep_run_id = preparation_run_ids[source_run_id]
        for event_id in sorted(contexts[source_run_id]):
            event = events.get(event_id)
            if event is None:
                missing_events += 1
                continue
            prepared_id = _prepared_event_id(source_run_id, event_id)
            member_id = _prepared_member_id(source_run_id, event_id)
            key = _idempotency_key(source_run_id, event_id)
            existing_event = existing_events_by_id.get(prepared_id) or existing_events_by_key.get(key)
            event_action = "create"
            conflict = ""
            if existing_event is not None:
                expected = (
                    existing_event.prepared_event_id == prepared_id
                    and existing_event.run_id == prep_run_id
                    and existing_event.sample_id == event.sample_id
                    and existing_event.dominant_year == event.event_year
                    and existing_event.source_run_id == source_run_id
                    and existing_event.backfill_version == LEGACY_BACKFILL_VERSION
                    and existing_event.idempotency_key == key
                )
                event_action = "reuse" if expected else "conflict"
                if not expected:
                    conflict = "existing prepared event metadata differs"
            existing_member = existing_members_by_id.get(member_id)
            member_action = "create"
            if existing_member is not None:
                member_expected = (
                    existing_member.prepared_event_id == prepared_id
                    and existing_member.run_id == prep_run_id
                    and existing_member.source_event_id == event_id
                )
                member_action = "reuse" if member_expected else "conflict"
                if not member_expected:
                    conflict = conflict or "existing prepared member metadata differs"
            action = (
                "conflict"
                if "conflict" in {event_action, member_action}
                else "reuse"
                if event_action == member_action == "reuse"
                else "create"
                if event_action == member_action == "create"
                else "repair"
            )
            event_plan.append(
                {
                    "source_run_id": source_run_id,
                    "source_event_id": event_id,
                    "sample_id": event.sample_id,
                    "event_year": event.event_year,
                    "source": event.source,
                    "source_analysis_id": event.source_analysis_id or "",
                    "loss_area_ha": event.loss_area_ha,
                    "loss_share": event.loss_share,
                    "is_dominant": bool(event.is_dominant),
                    "preparation_run_id": prep_run_id,
                    "prepared_event_id": prepared_id,
                    "member_id": member_id,
                    "idempotency_key": key,
                    "event_action": event_action,
                    "member_action": member_action,
                    "action": action,
                    "conflict": conflict,
                }
            )

    unique_source_events = len({row["source_event_id"] for row in event_plan})
    summary = {
        "backfill_version": LEGACY_BACKFILL_VERSION,
        "source_runs": len(run_plan),
        "source_events": unique_source_events,
        "context_event_rows": len(event_plan),
        "multi_context_rows": len(event_plan) - unique_source_events,
        "missing_events": missing_events,
        "run_actions": dict(sorted(Counter(row["action"] for row in run_plan).items())),
        "event_actions": dict(sorted(Counter(row["action"] for row in event_plan).items())),
        "conflicts": sum(row["action"] == "conflict" for row in run_plan)
        + sum(row["action"] == "conflict" for row in event_plan),
    }
    return {"summary": summary, "runs": run_plan, "events": event_plan}


def assert_legacy_backfill_plan_safe(plan: dict[str, Any]) -> None:
    summary = plan["summary"]
    if int(summary.get("missing_events") or 0) or int(summary.get("conflicts") or 0):
        raise ValueError(
            "Legacy PreparedEvent backfill gate failed: "
            f"missing_events={summary.get('missing_events')}, conflicts={summary.get('conflicts')}"
        )
    if int(summary.get("source_events") or 0) <= 0:
        raise ValueError("Legacy PreparedEvent backfill gate failed: no source events")


def apply_legacy_prepared_event_backfill(
    db: Session,
    plan: dict[str, Any],
) -> dict[str, int]:
    assert_legacy_backfill_plan_safe(plan)
    created_runs = 0
    created_events = 0
    created_members = 0
    now = datetime.now(timezone.utc)

    for row in plan["runs"]:
        run_id = row["preparation_run_id"]
        if db.get(CandidatePreparationRun, run_id) is not None:
            continue
        db.add(
            CandidatePreparationRun(
                run_id=run_id,
                pipeline_name=row["pipeline_name"],
                pipeline_version=row["pipeline_version"],
                algorithm_version=row["algorithm_version"],
                config_json=row["config_json"],
                config_hash=row["config_hash"],
                database_fingerprint=row["database_fingerprint"],
                source_run_id=row["source_run_id"],
                source_candidate_run_id=row["source_candidate_run_id"],
                status="COMPLETED",
                completed_at=now,
            )
        )
        created_runs += 1
    db.flush()

    for row in plan["events"]:
        prepared_id = row["prepared_event_id"]
        if db.get(PreparedEvent, prepared_id) is None:
            aggregate_features = {
                "is_dominant": row["is_dominant"],
                "loss_area_ha": row["loss_area_ha"],
                "loss_share": row["loss_share"],
                "source": row["source"],
                "source_analysis_id": row["source_analysis_id"] or None,
            }
            db.add(
                PreparedEvent(
                    prepared_event_id=prepared_id,
                    run_id=row["preparation_run_id"],
                    sample_id=row["sample_id"],
                    dominant_year=row["event_year"],
                    hansen_years_json=json.dumps([row["event_year"]]),
                    aggregate_features_json=_stable_json(aggregate_features),
                    temporal_analysis_status="LEGACY_NOT_EVALUATED",
                    diagnostic_flags_json="[]",
                    source_run_id=row["source_run_id"],
                    is_legacy_backfill=True,
                    backfill_version=LEGACY_BACKFILL_VERSION,
                    idempotency_key=row["idempotency_key"],
                )
            )
            created_events += 1
        db.flush()
        if db.get(PreparedEventMember, row["member_id"]) is None:
            db.add(
                PreparedEventMember(
                    member_id=row["member_id"],
                    prepared_event_id=prepared_id,
                    run_id=row["preparation_run_id"],
                    source_event_id=row["source_event_id"],
                    event_year=row["event_year"],
                    year_offset=0,
                    member_role="SINGLE",
                    candidate_role="DOMINANT" if row["is_dominant"] else "CONTEXT",
                    relative_area=row["loss_share"],
                )
            )
            created_members += 1
    db.commit()
    return {
        "created_preparation_runs": created_runs,
        "created_prepared_events": created_events,
        "created_prepared_event_members": created_members,
    }
