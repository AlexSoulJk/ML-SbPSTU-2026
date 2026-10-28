from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections import Counter
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .models import (
    EventLabel,
    EventReview,
    LabelRun,
    ManualValidation,
    Sample,
    SampleEvent,
    SampleStatus,
    utc_now,
)


EVENT_REVIEW_STATUSES = {"DRAFT", "VALID", "UNCLEAR", "REJECTED"}
EVENT_LABELS = {"Logging", "Wildfire", "Other"}
MODEL_SOURCE_NAMES = {"REMOTECLIP_LOGREG", "XGBOOST"}


def configured_annotator_name() -> str | None:
    value = os.getenv("FOREST_ANNOTATOR_NAME", "").strip()
    return value or None


def require_actor_name(actor_name: str | None) -> str:
    value = (actor_name or configured_annotator_name() or "").strip()
    if not value:
        raise ValueError("actor_name is required; set FOREST_ANNOTATOR_NAME or enter a local name")
    return value


def _human_run_id(actor_name: str) -> str:
    digest = hashlib.sha1(actor_name.encode("utf-8")).hexdigest()
    return digest


def ensure_human_label_run(db: Session, actor_name: str) -> LabelRun:
    run_id = _human_run_id(actor_name)
    run = db.get(LabelRun, run_id)
    if run is None:
        run = LabelRun(
            label_run_id=run_id,
            producer_type="HUMAN",
            producer_name="HUMAN",
            producer_version="manual_v1",
            config_json=json.dumps({"actor_name": actor_name}, ensure_ascii=False, sort_keys=True),
            status="READY",
            finished_at=utc_now(),
        )
        db.add(run)
    return run


def create_draft_review(db: Session, event_id: str) -> EventReview:
    review = db.get(EventReview, event_id)
    if review is None:
        review = EventReview(event_id=event_id, review_status="DRAFT")
        db.add(review)
    return review


def _set_sample_manual_status(
    db: Session,
    sample_id: str,
    validation: str | None,
    notes: str | None,
) -> None:
    manual = db.get(ManualValidation, sample_id)
    status = db.scalar(
        select(SampleStatus).where(
            SampleStatus.sample_id == sample_id,
            SampleStatus.source == "manual",
        )
    )
    if validation is None:
        if manual is not None:
            db.delete(manual)
        if status is not None:
            db.delete(status)
        return

    if manual is None:
        manual = ManualValidation(sample_id=sample_id, validation=validation)
        db.add(manual)
    manual.validation = validation
    manual.notes = notes
    manual.updated_at = utc_now()

    if status is None:
        status = SampleStatus(sample_id=sample_id, source="manual", status=validation)
        db.add(status)
    status.status = validation
    status.reason = notes
    status.metrics_json = json.dumps(
        {"source": "event_review_rollup"},
        ensure_ascii=False,
        sort_keys=True,
    )
    status.updated_at = utc_now()


def review_completion(statuses: list[str]) -> str:
    if not statuses or all(status == "DRAFT" for status in statuses):
        return "NOT_STARTED"
    if any(status == "DRAFT" for status in statuses):
        return "PARTIAL"
    return "COMPLETE"


def roll_up_sample_reviews(db: Session, sample_id: str) -> dict[str, Any]:
    sample = db.get(Sample, sample_id)
    if sample is None:
        raise ValueError(f"Sample not found: {sample_id}")
    events = list(
        db.scalars(
            select(SampleEvent)
            .where(SampleEvent.sample_id == sample_id)
            .order_by(SampleEvent.event_year, SampleEvent.event_id)
        )
    )
    explicit = {
        review.event_id: review
        for review in db.scalars(
            select(EventReview).where(
                EventReview.event_id.in_([event.event_id for event in events])
            )
        )
    } if events else {}

    legacy = sample.manual_validation.validation if sample.manual_validation is not None else None
    statuses: list[str] = []
    for event in events:
        review = explicit.get(event.event_id)
        if review is not None:
            statuses.append(review.review_status)
        elif sample.source_format != "HANSEN_MINING" and legacy == "Valid":
            statuses.append("VALID")
        elif sample.source_format != "HANSEN_MINING" and legacy == "Unclear":
            statuses.append("UNCLEAR")
        else:
            statuses.append("DRAFT")

    validation: str | None
    if any(status == "VALID" for status in statuses):
        validation = "Valid"
    elif statuses and all(status == "REJECTED" for status in statuses):
        validation = "Wrong"
    elif statuses and "DRAFT" not in statuses and any(status == "UNCLEAR" for status in statuses):
        validation = "Unclear"
    else:
        validation = None

    _set_sample_manual_status(
        db,
        sample_id,
        validation,
        "Derived from event-level reviews" if validation else None,
    )
    sample.updated_at = utc_now()
    return {
        "sample_id": sample_id,
        "validation": validation,
        "completion": review_completion(statuses),
        "statuses": dict(Counter(statuses)),
    }


def save_event_review(
    db: Session,
    event_id: str,
    *,
    review_status: str,
    actor_name: str | None,
    label: str | None = None,
    based_on_label_id: str | None = None,
    rejection_reason: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    status = review_status.strip().upper()
    if status not in EVENT_REVIEW_STATUSES:
        raise ValueError(f"Unsupported review_status: {review_status}")
    actor = require_actor_name(actor_name)
    event = db.get(SampleEvent, event_id)
    if event is None:
        raise ValueError(f"Event not found: {event_id}")

    normalized_label = (label or "").strip() or None
    normalized_reason = (rejection_reason or "").strip() or None
    if status == "VALID" and normalized_label not in EVENT_LABELS:
        raise ValueError("VALID review requires label: Logging, Wildfire or Other")
    if status != "VALID" and normalized_label is not None:
        raise ValueError(f"{status} review cannot have a human label")
    if status == "REJECTED" and normalized_reason is None:
        raise ValueError("REJECTED review requires rejection_reason")
    if status != "REJECTED" and normalized_reason is not None:
        raise ValueError(f"{status} review cannot have rejection_reason")

    based_on = None
    if based_on_label_id:
        based_on = db.get(EventLabel, based_on_label_id)
        if based_on is None or based_on.event_id != event_id or based_on.source_type != "MODEL":
            raise ValueError("based_on_label_id must reference a model label for this event")
        if normalized_label is not None and based_on.label != normalized_label:
            raise ValueError("Human label must match the accepted model prediction")

    human_label: EventLabel | None = None
    if status == "VALID":
        run = ensure_human_label_run(db, actor)
        human_label = EventLabel(
            label_id=uuid.uuid4().hex,
            sample_id=event.sample_id,
            event_id=event.event_id,
            label_run_id=run.label_run_id,
            source_type="HUMAN",
            source_name="HUMAN",
            source_version="manual_v1",
            label=normalized_label,
            based_on_label_id=based_on.label_id if based_on else None,
            input_fingerprint=uuid.uuid4().hex,
            metadata_json=json.dumps({"actor_name": actor}, ensure_ascii=False, sort_keys=True),
        )
        db.add(human_label)
        db.flush()

    review = db.get(EventReview, event_id)
    if review is None:
        review = EventReview(event_id=event_id, review_status=status)
        db.add(review)
    review.review_status = status
    review.human_label_id = human_label.label_id if human_label else None
    review.rejection_reason = normalized_reason
    review.notes = (notes or "").strip() or None
    review.reviewer_name = actor
    review.updated_at = utc_now()
    db.flush()
    rollup = roll_up_sample_reviews(db, event.sample_id)
    db.commit()
    return {
        "event_id": event_id,
        "review": event_review_payload(db, event, include_labels=True),
        "sample_rollup": rollup,
    }


def _label_payload(label: EventLabel) -> dict[str, Any]:
    return {
        "label_id": label.label_id,
        "label_run_id": label.label_run_id,
        "source_type": label.source_type,
        "source_name": label.source_name,
        "source_version": label.source_version,
        "label": label.label,
        "confidence": label.confidence,
        "probabilities": json.loads(label.probabilities_json) if label.probabilities_json else None,
        "based_on_label_id": label.based_on_label_id,
        "input_fingerprint": label.input_fingerprint,
        "metadata": json.loads(label.metadata_json) if label.metadata_json else None,
        "created_at": label.created_at.isoformat() if label.created_at else None,
    }


def model_agreement(labels: list[EventLabel]) -> str:
    latest: dict[str, EventLabel] = {}
    for label in sorted(labels, key=lambda row: (row.created_at, row.label_id)):
        if label.source_type == "MODEL" and label.source_name in MODEL_SOURCE_NAMES:
            latest[label.source_name] = label
    if set(latest) != MODEL_SOURCE_NAMES:
        return "INCOMPLETE"
    remote = latest["REMOTECLIP_LOGREG"]
    xgboost = latest["XGBOOST"]
    if remote.confidence is not None and remote.confidence < 0.5:
        return "UNCERTAIN"
    if xgboost.confidence is not None and xgboost.confidence < 0.5:
        return "UNCERTAIN"
    if remote.label != xgboost.label:
        return "DISAGREE"
    return f"AGREE_{remote.label.upper()}"


def event_review_payload(
    db: Session,
    event: SampleEvent,
    *,
    include_labels: bool = True,
) -> dict[str, Any]:
    sample = db.get(Sample, event.sample_id)
    review = db.get(EventReview, event.event_id)
    labels = list(
        db.scalars(
            select(EventLabel)
            .where(EventLabel.event_id == event.event_id)
            .order_by(EventLabel.created_at, EventLabel.label_id)
        )
    ) if include_labels else []

    if review is not None:
        status = review.review_status
        human_label = db.get(EventLabel, review.human_label_id) if review.human_label_id else None
        inherited = False
        label_value = human_label.label if human_label is not None else None
        label_source = "HUMAN" if human_label is not None else None
    else:
        manual = sample.manual_validation.validation if sample and sample.manual_validation else None
        inherited = bool(sample and sample.source_format != "HANSEN_MINING")
        if inherited and manual == "Valid":
            status = "VALID"
            label_value = sample.driver_primary
            label_source = "LEGACY_SAMPLE_DRIVER"
        elif inherited and manual == "Unclear":
            status = "UNCLEAR"
            label_value = None
            label_source = "LEGACY_SAMPLE_VALIDATION"
        else:
            status = "DRAFT"
            label_value = None
            label_source = None

    return {
        "review_status": status,
        "human_label_id": review.human_label_id if review else None,
        "human_label": label_value,
        "label_source": label_source,
        "is_inherited": inherited,
        "rejection_reason": review.rejection_reason if review else None,
        "notes": review.notes if review else None,
        "reviewer_name": review.reviewer_name if review else None,
        "updated_at": review.updated_at.isoformat() if review and review.updated_at else None,
        "labels": [_label_payload(label) for label in labels],
        "model_agreement": model_agreement(labels),
    }


def sample_review_completion(db: Session, events: list[SampleEvent]) -> str:
    statuses = [event_review_payload(db, event, include_labels=False)["review_status"] for event in events]
    return review_completion(statuses)
