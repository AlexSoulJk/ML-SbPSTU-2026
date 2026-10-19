from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import PROJECT_ROOT

from ResearchTool.backend.forest.models import (  # noqa: E402
    DerivedPreview,
    ManualValidation,
    Sample,
    SampleEvent,
    SampleEventDownload,
    SentinelDownload,
    SentinelSceneReview,
    SentinelSceneReviewReason,
)
from ResearchTool.backend.forest.storage_paths import resolve_storage_path  # noqa: E402


COHORTS = {
    "valid": "MANUAL_VALID",
    "unclear": "MANUAL_UNCLEAR",
    "wrong": "MANUAL_WRONG",
    "too old": "MANUAL_TOO_OLD",
}


def validation_cohort(validation: str | None) -> str:
    if validation is None:
        return "NO_MANUAL_WEAK"
    normalized = " ".join(validation.strip().lower().replace("_", " ").split())
    return COHORTS.get(normalized, f"MANUAL_OTHER:{validation}")


def parse_scene_datetime(metadata_json: str | None, scene_id: str | None = None) -> datetime | None:
    try:
        metadata = json.loads(metadata_json or "{}")
    except json.JSONDecodeError:
        metadata = {}
    raw = (metadata.get("item") or {}).get("datetime") or metadata.get("datetime")
    if raw:
        try:
            return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except ValueError:
            pass
    match = re.search(r"(20\d{6}T\d{6})", scene_id or "")
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S")
        except ValueError:
            return None
    return None


def season_for_datetime(
    value: datetime | None,
    season_windows: dict[str, list[str] | tuple[str, str]],
) -> str | None:
    if value is None:
        return None
    mmdd = value.strftime("%m-%d")
    for season, bounds in season_windows.items():
        start, end = str(bounds[0]), str(bounds[1])
        if start <= end and start <= mmdd <= end:
            return season
        if start > end and (mmdd >= start or mmdd <= end):
            return season
    return "outside_defined_seasons"


def project_relative_path(path: str | Path | None) -> str:
    if not path:
        return ""
    resolved = resolve_storage_path(path).resolve(strict=False)
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def resolve_manifest_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def database_fingerprint(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {
        "path": project_relative_path(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": file_sha256(resolved),
    }


def _preview_map(db: Session, view: str) -> dict[str, DerivedPreview]:
    previews: dict[str, DerivedPreview] = {}
    rows = db.scalars(
        select(DerivedPreview)
        .where(DerivedPreview.view == view)
        .order_by(DerivedPreview.download_id, DerivedPreview.created_at.desc())
    )
    for preview in rows:
        previews.setdefault(preview.download_id, preview)
    return previews


def _reason_map(db: Session) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    for reason in db.scalars(
        select(SentinelSceneReviewReason).order_by(
            SentinelSceneReviewReason.download_id,
            SentinelSceneReviewReason.id,
        )
    ):
        if reason.reason_code not in result[reason.download_id]:
            result[reason.download_id].append(reason.reason_code)
    return result


def build_event_manifest(
    db: Session,
    *,
    dataset_config: dict[str, Any],
    virtual_links: Iterable[dict[str, Any]] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_classes = set(dataset_config.get("target_classes", ["Logging", "Wildfire"]))
    view = str(dataset_config.get("preview_view") or "rgb")
    allowed_periods = {str(period).upper() for period in dataset_config.get("periods", [])}
    season_windows = dict(dataset_config.get("season_windows") or {})

    samples = {
        sample.sample_id: sample
        for sample in db.scalars(
            select(Sample)
            .where(Sample.driver_primary.in_(sorted(target_classes)))
            .order_by(Sample.sample_id)
        )
    }
    sample_ids = list(samples)
    if not sample_ids:
        return [], {"rows": 0, "eligible_rows": 0, "reason_counts": {}}

    validations = {
        row.sample_id: row.validation
        for row in db.scalars(
            select(ManualValidation).where(ManualValidation.sample_id.in_(sample_ids))
        )
    }
    events = {
        event.event_id: event
        for event in db.scalars(
            select(SampleEvent).where(SampleEvent.sample_id.in_(sample_ids))
        )
    }
    downloads = {
        download.download_id: download
        for download in db.scalars(
            select(SentinelDownload).where(SentinelDownload.sample_id.in_(sample_ids))
        )
    }
    previews = _preview_map(db, view)
    reviews = {
        review.download_id: review
        for review in db.scalars(
            select(SentinelSceneReview).where(
                SentinelSceneReview.download_id.in_(list(downloads))
            )
        )
    }
    reasons = _reason_map(db)

    link_specs: list[dict[str, Any]] = []
    explicitly_linked_downloads: set[str] = set()
    if events:
        for link in db.scalars(
            select(SampleEventDownload).where(
                SampleEventDownload.event_id.in_(list(events))
            )
        ):
            link_specs.append(
                {
                    "event_id": link.event_id,
                    "download_id": link.download_id,
                    "period": link.period,
                    "event_assignment": "EXPLICIT",
                }
            )
            explicitly_linked_downloads.add(link.download_id)

    for link in virtual_links:
        download_id = str(link.get("download_id") or "")
        if not download_id or download_id in explicitly_linked_downloads:
            continue
        link_specs.append(
            {
                "event_id": str(link.get("event_id") or ""),
                "download_id": download_id,
                "period": str(link.get("legacy_period") or ""),
                "event_assignment": "LEGACY_DOMINANT_BACKFILL",
            }
        )

    manifest: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for link in link_specs:
        event = events.get(link["event_id"])
        download = downloads.get(link["download_id"])
        if event is None or download is None:
            continue
        sample = samples.get(event.sample_id)
        if sample is None or download.sample_id != sample.sample_id:
            continue

        period = str(link["period"] or "").upper()
        preview = previews.get(download.download_id)
        review = reviews.get(download.download_id)
        preview_path = resolve_storage_path(preview.png_path) if preview else None
        exclusions: list[str] = []
        if period not in allowed_periods:
            exclusions.append("UNSUPPORTED_PERIOD")
        if preview is None:
            exclusions.append("MISSING_RGB_PREVIEW")
        elif preview_path is None or not preview_path.exists():
            exclusions.append("MISSING_RGB_FILE")
        if review is not None and review.is_excluded:
            exclusions.append("SCENE_REVIEW_EXCLUDED")
        if download.is_bad_cloud:
            exclusions.append("BAD_CLOUD")
        if download.is_bad_quality:
            exclusions.append("BAD_QUALITY")

        for reason in exclusions:
            reason_counts[reason] += 1

        acquired = parse_scene_datetime(download.metadata_json, download.scene_id)
        validation = validations.get(sample.sample_id)
        row = {
            "sample_id": sample.sample_id,
            "event_id": event.event_id,
            "event_year": event.event_year,
            "event_loss_area_ha": event.loss_area_ha,
            "event_loss_share": event.loss_share,
            "event_is_dominant": bool(event.is_dominant),
            "is_multi_event_area": bool(sample.has_multiple_events),
            "label": sample.driver_primary,
            "manual_validation": validation or "",
            "validation_cohort": validation_cohort(validation),
            "download_id": download.download_id,
            "scene_id": download.scene_id,
            "period": period,
            "view": view,
            "scene_datetime": acquired.isoformat() if acquired else "",
            "scene_date": acquired.date().isoformat() if acquired else "",
            "day_of_year": acquired.timetuple().tm_yday if acquired else "",
            "season": season_for_datetime(acquired, season_windows) or "",
            "png_path": project_relative_path(preview.png_path if preview else None),
            "metadata_path": project_relative_path(preview.metadata_path if preview else None),
            "geotiff_path": project_relative_path(download.local_path),
            "cloud_fraction": download.cloud_fraction,
            "shadow_fraction": download.shadow_fraction,
            "nodata_fraction": download.nodata_fraction,
            "dark_fraction": download.dark_fraction,
            "is_bad_cloud": bool(download.is_bad_cloud),
            "is_bad_quality": bool(download.is_bad_quality),
            "review_is_excluded": bool(review.is_excluded) if review else False,
            "review_reason_codes": reasons.get(download.download_id, []),
            "eligible": not exclusions,
            "exclusion_reasons": exclusions,
            "event_assignment": link["event_assignment"],
        }
        manifest.append(row)

    manifest.sort(
        key=lambda row: (
            row["sample_id"],
            int(row["event_year"]),
            row["period"],
            row["scene_datetime"],
            row["download_id"],
        )
    )

    eligible_events_by_sample: dict[str, set[str]] = defaultdict(set)
    for row in manifest:
        if row["eligible"]:
            eligible_events_by_sample[row["sample_id"]].add(row["event_id"])
    for row in manifest:
        count = len(eligible_events_by_sample[row["sample_id"]])
        row["eligible_events_in_sample"] = count
        row["base_event_weight"] = (1.0 / count) if count else 0.0

    eligible = [row for row in manifest if row["eligible"]]
    cohorts = sorted({row["validation_cohort"] for row in eligible})
    sample_counts_by_cohort = {
        cohort: len({row["sample_id"] for row in eligible if row["validation_cohort"] == cohort})
        for cohort in cohorts
    }
    event_counts_by_cohort = {
        cohort: len({row["event_id"] for row in eligible if row["validation_cohort"] == cohort})
        for cohort in cohorts
    }
    image_counts_by_cohort = {
        cohort: sum(row["validation_cohort"] == cohort for row in eligible)
        for cohort in cohorts
    }
    audit = {
        "rows": len(manifest),
        "eligible_rows": len(eligible),
        "excluded_rows": len(manifest) - len(eligible),
        "unique_samples": len({row["sample_id"] for row in manifest}),
        "eligible_samples": len({row["sample_id"] for row in eligible}),
        "unique_events": len({row["event_id"] for row in manifest}),
        "eligible_events": len({row["event_id"] for row in eligible}),
        "cohort_counts": image_counts_by_cohort,
        "sample_counts_by_cohort": sample_counts_by_cohort,
        "event_counts_by_cohort": event_counts_by_cohort,
        "class_counts": dict(sorted(Counter(row["label"] for row in eligible).items())),
        "period_counts": dict(sorted(Counter(row["period"] for row in eligible).items())),
        "event_assignment_counts": dict(
            sorted(Counter(row["event_assignment"] for row in manifest).items())
        ),
        "reason_counts": dict(sorted(reason_counts.items())),
    }
    return manifest, audit

