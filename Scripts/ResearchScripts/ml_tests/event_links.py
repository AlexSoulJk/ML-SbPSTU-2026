from __future__ import annotations

import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import PROJECT_ROOT

from ResearchTool.backend.forest.event_service import (  # noqa: E402
    EVENT_PERIODS,
    event_download_id_for,
)
from ResearchTool.backend.forest.models import (  # noqa: E402
    SampleEvent,
    SampleEventDownload,
    SentinelDownload,
)


BackfillRow = dict[str, Any]


def plan_dominant_event_backfill(db: Session) -> dict[str, Any]:
    events_by_sample: dict[str, list[SampleEvent]] = defaultdict(list)
    for event in db.scalars(select(SampleEvent).order_by(SampleEvent.sample_id, SampleEvent.event_year)):
        events_by_sample[event.sample_id].append(event)

    linked_download_ids = set(db.scalars(select(SampleEventDownload.download_id).distinct()))
    rows: list[BackfillRow] = []
    counts: Counter[str] = Counter()

    downloads = db.scalars(
        select(SentinelDownload).order_by(SentinelDownload.sample_id, SentinelDownload.download_id)
    )
    for download in downloads:
        period = (download.period or "").upper()
        base = {
            "sample_id": download.sample_id,
            "download_id": download.download_id,
            "scene_id": download.scene_id,
            "legacy_period": period,
            "event_id": "",
            "event_year": "",
            "link_id": "",
        }
        if download.download_id in linked_download_ids:
            action = "already_linked"
        elif period not in EVENT_PERIODS:
            action = "unknown_period"
        else:
            dominant = [event for event in events_by_sample[download.sample_id] if event.is_dominant]
            if not dominant:
                action = "missing_dominant_event"
            elif len(dominant) > 1:
                action = "multiple_dominant_events"
            else:
                event = dominant[0]
                base.update(
                    {
                        "event_id": event.event_id,
                        "event_year": event.event_year,
                        "link_id": event_download_id_for(
                            event.event_id,
                            download.download_id,
                            period,
                        ),
                    }
                )
                action = "would_link"
        base["action"] = action
        rows.append(base)
        counts[action] += 1

    return {
        "dry_run": True,
        "total_downloads": len(rows),
        "counts": dict(sorted(counts.items())),
        "rows": rows,
    }


def virtual_backfill_rows(plan: dict[str, Any]) -> list[BackfillRow]:
    return [dict(row) for row in plan.get("rows", []) if row.get("action") == "would_link"]


def apply_dominant_event_backfill(
    db: Session,
    rows: Iterable[BackfillRow],
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    results: list[BackfillRow] = []

    for raw in rows:
        row = dict(raw)
        if row.get("action") != "would_link":
            continue
        download_id = str(row["download_id"])
        event_id = str(row["event_id"])
        period = str(row["legacy_period"]).upper()

        existing = db.scalar(
            select(SampleEventDownload.id).where(
                SampleEventDownload.download_id == download_id
            ).limit(1)
        )
        if existing is not None:
            action = "skipped_already_linked"
        else:
            download = db.get(SentinelDownload, download_id)
            event = db.get(SampleEvent, event_id)
            if download is None:
                action = "skipped_missing_download"
            elif event is None:
                action = "skipped_missing_event"
            elif event.sample_id != download.sample_id:
                action = "skipped_sample_mismatch"
            elif not event.is_dominant:
                action = "skipped_event_not_dominant"
            elif period not in EVENT_PERIODS or (download.period or "").upper() != period:
                action = "skipped_period_changed"
            else:
                link_id = event_download_id_for(event_id, download_id, period)
                db.add(
                    SampleEventDownload(
                        id=link_id,
                        event_id=event_id,
                        download_id=download_id,
                        period=period,
                    )
                )
                action = "linked"

        row["action"] = action
        results.append(row)
        counts[action] += 1

    db.flush()
    return {
        "dry_run": False,
        "processed": len(results),
        "counts": dict(sorted(counts.items())),
        "rows": results,
    }


def backup_sqlite_database(
    database_path: str | Path,
    *,
    backup_dir: str | Path | None = None,
) -> Path:
    source = Path(database_path).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Database not found: {source}")
    destination_dir = Path(backup_dir) if backup_dir else source.parent / "backups"
    if not destination_dir.is_absolute():
        destination_dir = PROJECT_ROOT / destination_dir
    destination_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = destination_dir / f"{source.stem}_before_event_backfill_{timestamp}.sqlite"
    with sqlite3.connect(source) as source_db, sqlite3.connect(destination) as backup_db:
        source_db.backup(backup_db)
    return destination

