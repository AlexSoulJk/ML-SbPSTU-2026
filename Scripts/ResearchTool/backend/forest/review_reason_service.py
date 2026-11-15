from __future__ import annotations

import json
import re
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .models import (
    EventLabel,
    EventReview,
    EventReviewReason,
    EventReviewRevision,
)


EVENT_REVIEW_REASON_CODES = (
    "EVENT_TOO_SMALL",
    "YEAR_MISMATCH",
    "NO_PRE",
    "NO_POST",
    "QC_FAILED",
    "NO_VISIBLE_CHANGE",
    "OTHER_INVALID",
)
EVENT_REVIEW_REASON_CODE_SET = set(EVENT_REVIEW_REASON_CODES)


@dataclass(frozen=True)
class LegacyReasonMapping:
    legacy_text: str
    reason_codes: tuple[str, ...]
    mapping_status: str
    mapping_rules: tuple[str, ...]


def _normalized_text(value: str | None) -> str:
    normalized = unicodedata.normalize("NFKC", value or "").casefold().replace("ё", "е")
    return " ".join(normalized.replace("’", "'").split())


def normalize_reason_codes(reason_codes: Iterable[str] | None) -> list[str]:
    values = []
    for raw in reason_codes or ():
        code = str(raw or "").strip().upper()
        if not code or code in values:
            continue
        if code not in EVENT_REVIEW_REASON_CODE_SET:
            allowed = ", ".join(EVENT_REVIEW_REASON_CODES)
            raise ValueError(f"Unsupported event review reason code {code!r}; allowed: {allowed}")
        values.append(code)
    return [code for code in EVENT_REVIEW_REASON_CODES if code in values]


def map_legacy_rejection_reason(value: str | None) -> LegacyReasonMapping:
    original = (value or "").strip()
    text = _normalized_text(original)
    if not text:
        return LegacyReasonMapping(original, (), "UNMAPPED", ("empty",))

    codes: set[str] = set()
    rules: list[str] = []

    has_pre = bool(re.search(r"\bpre\b", text))
    has_post = bool(re.search(r"\bpost\b", text))
    says_missing = any(token in text for token in ("no ", "нет ", "отсутств"))
    says_invalid = any(token in text for token in ("valid", "валидн"))

    if has_pre and says_missing:
        if says_invalid:
            codes.add("QC_FAILED")
            rules.append("pre_present_but_no_valid_images")
        else:
            codes.add("NO_PRE")
            rules.append("missing_pre")
    if has_post and says_missing:
        if says_invalid:
            codes.add("QC_FAILED")
            rules.append("post_present_but_no_valid_images")
        else:
            codes.add("NO_POST")
            rules.append("missing_post")

    if any(
        token in text
        for token in (
            "small event",
            "small change",
            "small damage",
            "маленьк",
            "небольш",
            "слишком мал",
        )
    ):
        codes.add("EVENT_TOO_SMALL")
        rules.append("small_event_language")

    year_language = any(
        token in text
        for token in (
            "next yea",
            "last year",
            "other year",
            "different year",
            "year's",
            "year’s",
            "другого года",
            "прошлого года",
            "прошлогоо года",
            "следующего года",
            "неправильно определен год",
            "не правильно определен год",
            "неправильно определен год",
        )
    )
    year_post_context = bool(re.search(r"\b20\d{2}\b", text)) and (
        "post" in text or "consequence" in text or "последств" in text
    )
    if year_language or year_post_context:
        codes.add("YEAR_MISMATCH")
        rules.append("different_event_year_language")

    if (
        ("изменен" in text and "не видно" in text)
        or ("не видно" in text and any(token in text for token in ("разниц", "рвзниц")))
        or "no visible change" in text
    ):
        codes.add("NO_VISIBLE_CHANGE")
        rules.append("no_visible_change_language")

    if (
        "low vegetation" in text
        or (says_invalid and not has_pre and not has_post and "сним" in text)
    ):
        codes.add("QC_FAILED")
        rules.append("image_quality_language")

    explicit_other = {
        "both models made a mistake",
        "явно не пожар, но может и не вырубка - слишком маленькое событие",
    }
    if text in explicit_other:
        codes.add("OTHER_INVALID")
        rules.append("explicit_other_invalid")

    ordered = tuple(code for code in EVENT_REVIEW_REASON_CODES if code in codes)
    if not ordered:
        return LegacyReasonMapping(original, (), "UNMAPPED", tuple(rules))
    status = "MAPPED_MULTI" if len(ordered) > 1 else "MAPPED"
    return LegacyReasonMapping(original, ordered, status, tuple(rules))


def event_review_reason_codes(db: Session, event_id: str) -> list[str]:
    codes = list(
        db.scalars(
            select(EventReviewReason.reason_code)
            .where(EventReviewReason.event_id == event_id)
            .order_by(EventReviewReason.reason_code)
        )
    )
    return normalize_reason_codes(codes)


def replace_event_review_reasons(
    db: Session,
    event_id: str,
    reason_codes: Iterable[str] | None,
    *,
    reason_text: str | None = None,
) -> list[str]:
    normalized = normalize_reason_codes(reason_codes)
    db.execute(delete(EventReviewReason).where(EventReviewReason.event_id == event_id))
    for code in normalized:
        db.add(
            EventReviewReason(
                event_id=event_id,
                reason_code=code,
                reason_text=reason_text if code == "OTHER_INVALID" else None,
            )
        )
    return normalized


def append_event_review_revision(
    db: Session,
    review: EventReview,
    *,
    reason_codes: Iterable[str] | None,
    revision_source: str,
    revision_id: str | None = None,
    event_class: str | None = None,
    created_at: datetime | None = None,
) -> EventReviewRevision:
    normalized = normalize_reason_codes(reason_codes)
    if event_class is None and review.human_label_id:
        accepted = db.get(EventLabel, review.human_label_id)
        event_class = accepted.label if accepted is not None else None
    revision = EventReviewRevision(
        revision_id=revision_id or uuid.uuid4().hex,
        event_id=review.event_id,
        reviewer_name=review.reviewer_name,
        review_status=review.review_status,
        event_class=event_class,
        reason_codes_json=json.dumps(normalized, ensure_ascii=False),
        review_comment=review.notes,
        accepted_label_id=review.human_label_id,
        revision_source=revision_source,
        source_updated_at=review.updated_at,
        created_at=created_at or review.updated_at,
    )
    db.add(revision)
    return revision
