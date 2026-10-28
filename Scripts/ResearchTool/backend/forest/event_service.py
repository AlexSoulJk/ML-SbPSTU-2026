from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .event_review_service import event_review_payload, sample_review_completion
from .models import (
    HansenAnalysis,
    Sample,
    SampleEvent,
    SampleEventDownload,
    SentinelDownload,
    utc_now,
)


HANSEN_EVENT_SOURCE = "HANSEN"
HANSEN_EVENT_CUTOFF_YEAR = 2016
MULTIPLE_EVENT_MIN_COUNT = 3
EVENT_PERIODS = {"PRE", "EVENT", "POST"}

ProgressCallback = Callable[[int, int, str], None]
CancelCallback = Callable[[], bool]


def deterministic_id(*parts: object) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def event_id_for(sample_id: str, source: str, event_year: int) -> str:
    return deterministic_id("sample-event", sample_id, source.upper(), int(event_year))


def event_download_id_for(event_id: str, download_id: str, period: str) -> str:
    return deterministic_id("event-download", event_id, download_id, period.upper())


def decode_histogram(raw: str | None) -> list[dict[str, Any]]:
    try:
        value = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return []
    return value if isinstance(value, list) else []


def hansen_event_specs(analysis: HansenAnalysis) -> list[dict[str, Any]]:
    histogram = decode_histogram(analysis.histogram_json)
    rows: list[dict[str, Any]] = []
    total = sum(
        max(0.0, float(item.get("loss_area_ha") or 0.0))
        for item in histogram
        if int(item.get("year") or 0) >= HANSEN_EVENT_CUTOFF_YEAR
    )
    for item in histogram:
        year = int(item.get("year") or 0)
        area = max(0.0, float(item.get("loss_area_ha") or 0.0))
        if year < HANSEN_EVENT_CUTOFF_YEAR or area <= 0:
            continue
        rows.append(
            {
                "event_year": year,
                "loss_area_ha": round(area, 4),
                "loss_share": round(area / total, 6) if total > 0 else 0.0,
                "is_dominant": year == analysis.event_year,
                "metadata": {
                    "pixels": item.get("pixels"),
                    "hansen_analysis_status": analysis.status,
                },
            }
        )
    return sorted(rows, key=lambda row: int(row["event_year"]))


def upsert_event_download_link(
    db: Session,
    *,
    event_id: str,
    download_id: str,
    period: str,
    rank: int | None = None,
) -> SampleEventDownload:
    normalized_period = period.upper()
    if normalized_period not in EVENT_PERIODS:
        raise ValueError(f"Unsupported event period: {period}")
    link_id = event_download_id_for(event_id, download_id, normalized_period)
    link = db.get(SampleEventDownload, link_id)
    if link is None:
        link = SampleEventDownload(
            id=link_id,
            event_id=event_id,
            download_id=download_id,
            period=normalized_period,
            rank=rank,
        )
        db.add(link)
    elif rank is not None:
        link.rank = rank
    return link


def link_unambiguous_legacy_downloads(
    db: Session,
    sample_id: str,
    events: list[SampleEvent],
) -> int:
    if len(events) != 1:
        return 0
    event = events[0]
    added = 0
    downloads = list(
        db.scalars(
            select(SentinelDownload).where(SentinelDownload.sample_id == sample_id)
        )
    )
    for download in downloads:
        period = (download.period or "").upper()
        if period not in EVENT_PERIODS:
            continue
        link_id = event_download_id_for(event.event_id, download.download_id, period)
        if db.get(SampleEventDownload, link_id) is None:
            upsert_event_download_link(
                db,
                event_id=event.event_id,
                download_id=download.download_id,
                period=period,
            )
            added += 1
    return added


def sync_hansen_events(
    db: Session,
    sample: Sample,
    analysis: HansenAnalysis,
    *,
    link_legacy: bool = True,
) -> dict[str, Any]:
    specs = hansen_event_specs(analysis)
    existing = list(
        db.scalars(
            select(SampleEvent).where(
                SampleEvent.sample_id == sample.sample_id,
                SampleEvent.source == HANSEN_EVENT_SOURCE,
            )
        )
    )
    existing_by_year = {event.event_year: event for event in existing}
    expected_years = {int(spec["event_year"]) for spec in specs}
    created = 0
    updated = 0
    removed = 0

    for event in existing:
        if event.event_year in expected_years:
            continue
        db.execute(
            delete(SampleEventDownload).where(SampleEventDownload.event_id == event.event_id)
        )
        db.delete(event)
        removed += 1

    synced: list[SampleEvent] = []
    for spec in specs:
        year = int(spec["event_year"])
        event = existing_by_year.get(year)
        if event is None:
            event = SampleEvent(
                event_id=event_id_for(sample.sample_id, HANSEN_EVENT_SOURCE, year),
                sample_id=sample.sample_id,
                event_year=year,
                source=HANSEN_EVENT_SOURCE,
            )
            db.add(event)
            created += 1
        else:
            updated += 1
        event.source_analysis_id = analysis.analysis_id
        event.loss_area_ha = float(spec["loss_area_ha"])
        event.loss_share = float(spec["loss_share"])
        event.is_dominant = bool(spec["is_dominant"])
        event.metadata_json = json.dumps(spec["metadata"], ensure_ascii=False, sort_keys=True)
        event.updated_at = utc_now()
        synced.append(event)

    sample.has_multiple_events = len(synced) >= MULTIPLE_EVENT_MIN_COUNT
    sample.updated_at = utc_now()
    db.flush()
    linked_legacy = link_unambiguous_legacy_downloads(db, sample.sample_id, synced) if link_legacy else 0
    return {
        "sample_id": sample.sample_id,
        "analysis_id": analysis.analysis_id,
        "event_count": len(synced),
        "event_years": [event.event_year for event in synced],
        "has_multiple_events": bool(sample.has_multiple_events),
        "created": created,
        "updated": updated,
        "removed": removed,
        "legacy_links_created": linked_legacy,
    }


def sample_events(db: Session, sample_id: str) -> list[SampleEvent]:
    return list(
        db.scalars(
            select(SampleEvent)
            .where(SampleEvent.sample_id == sample_id)
            .order_by(SampleEvent.event_year, SampleEvent.event_id)
        )
    )


def sample_events_payload(db: Session, sample_id: str) -> list[dict[str, Any]]:
    events = sample_events(db, sample_id)
    if not events:
        return []
    event_ids = [event.event_id for event in events]
    links = list(
        db.scalars(
            select(SampleEventDownload).where(SampleEventDownload.event_id.in_(event_ids))
        )
    )
    counts: dict[str, Counter[str]] = {event_id: Counter() for event_id in event_ids}
    for link in links:
        counts[link.event_id][link.period] += 1
    completion = sample_review_completion(db, events)
    return [
        {
            "event_id": event.event_id,
            "event_year": event.event_year,
            "source": event.source,
            "source_analysis_id": event.source_analysis_id,
            "loss_area_ha": event.loss_area_ha,
            "loss_share": event.loss_share,
            "is_dominant": bool(event.is_dominant),
            "metadata": json.loads(event.metadata_json) if event.metadata_json else None,
            "download_counts": dict(counts[event.event_id]),
            "download_count": sum(counts[event.event_id].values()),
            "review": event_review_payload(db, event),
            "review_completion": completion,
            "created_at": event.created_at.isoformat() if event.created_at else None,
            "updated_at": event.updated_at.isoformat() if event.updated_at else None,
        }
        for event in events
    ]


def latest_analyses(db: Session, *, limit: int | None = None) -> list[HansenAnalysis]:
    rows = list(
        db.scalars(
            select(HansenAnalysis).order_by(
                HansenAnalysis.sample_id,
                HansenAnalysis.created_at.desc(),
                HansenAnalysis.analysis_id.desc(),
            )
        )
    )
    latest: dict[str, HansenAnalysis] = {}
    for analysis in rows:
        latest.setdefault(analysis.sample_id, analysis)
    values = list(latest.values())
    return values[:limit] if limit is not None else values


def backfill_hansen_events(
    db: Session,
    *,
    dry_run: bool,
    limit: int | None = None,
    progress: ProgressCallback | None = None,
    should_cancel: CancelCallback | None = None,
) -> dict[str, Any]:
    analyses = latest_analyses(db, limit=limit)
    rows: list[dict[str, Any]] = []
    failed = 0
    for index, analysis in enumerate(analyses, start=1):
        if should_cancel and should_cancel():
            break
        sample = db.get(Sample, analysis.sample_id)
        if sample is None:
            failed += 1
            rows.append({"sample_id": analysis.sample_id, "action": "missing_sample"})
            continue
        specs = hansen_event_specs(analysis)
        if dry_run:
            row = {
                "sample_id": sample.sample_id,
                "analysis_id": analysis.analysis_id,
                "action": "would_sync",
                "event_count": len(specs),
                "event_years": [spec["event_year"] for spec in specs],
                "has_multiple_events": len(specs) >= MULTIPLE_EVENT_MIN_COUNT,
            }
        else:
            try:
                result = sync_hansen_events(db, sample, analysis)
                db.commit()
                row = {**result, "action": "synced"}
            except Exception as exc:
                db.rollback()
                failed += 1
                row = {
                    "sample_id": sample.sample_id,
                    "analysis_id": analysis.analysis_id,
                    "action": "failed",
                    "error": str(exc)[:500],
                }
        rows.append(row)
        if progress:
            progress(index, len(analyses), f"{sample.sample_id[:8]}: {row['action']}")

    return {
        "dry_run": dry_run,
        "limit": limit,
        "event_year_cutoff": HANSEN_EVENT_CUTOFF_YEAR,
        "multiple_event_min_count": MULTIPLE_EVENT_MIN_COUNT,
        "total": len(analyses),
        "processed": len(rows),
        "failed": failed,
        "cancelled": bool(should_cancel and should_cancel()),
        "rows": rows,
        "event_rows": sum(int(row.get("event_count") or 0) for row in rows),
    }
