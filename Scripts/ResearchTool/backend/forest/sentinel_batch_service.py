from __future__ import annotations

import csv
import json
import uuid
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..cdse_client import parse_datetime, stac_search_s2
from ..config import PROJECT_ROOT
from .event_service import sample_events, upsert_event_download_link
from .geometry import sample_geometries
from .models import (
    Sample,
    SampleEvent,
    SampleStatus,
    SentinelDownload,
    SampleEventDownload,
    SentinelSceneReview,
    SentinelSceneReviewReason,
    SentinelSearch,
    utc_now,
)
from .sentinel_service import (
    SEASON_PRESETS,
    SearchPeriod,
    download_sentinel_for_sample,
    rank_item,
    restore_sentinel_download_cache,
)
from .storage_paths import resolve_storage_path


DEFAULT_SEASONS = ("spring", "early_summer", "late_summer", "autumn")
DEFAULT_TARGET_VALID = 10
DEFAULT_MAX_QC_SCENES = 50
DEFAULT_MIN_DAYS = 7
DEFAULT_STAC_MAX_CLOUD = 30.0
DEFAULT_LOCAL_CLOUD_THRESHOLD = 0.10
DEFAULT_SEARCH_TOP_N = 40
DATA_OPERATIONS_OUTPUT_DIR = (
    PROJECT_ROOT / "Scripts" / "ResearchScripts" / "Outputs" / "data_operations"
)

ProgressCallback = Callable[[str, str, int, int], None]
CancelCallback = Callable[[], bool]


@dataclass(frozen=True)
class BatchOptions:
    mode: str = "empty_samples"
    seasons: tuple[str, ...] = DEFAULT_SEASONS
    target_valid: int = DEFAULT_TARGET_VALID
    max_qc_scenes: int = DEFAULT_MAX_QC_SCENES
    min_days: int = DEFAULT_MIN_DAYS
    stac_max_cloud: float = DEFAULT_STAC_MAX_CLOUD
    local_cloud_threshold: float = DEFAULT_LOCAL_CLOUD_THRESHOLD
    search_top_n: int = DEFAULT_SEARCH_TOP_N

    def validate(self) -> None:
        if self.mode not in {"empty_samples", "multi_event_enrichment", "hansen_mining_run"}:
            raise ValueError(f"Unsupported Sentinel batch mode: {self.mode}")
        if not self.seasons:
            raise ValueError("At least one season must be selected.")
        unknown = sorted(set(self.seasons) - set(SEASON_PRESETS))
        if unknown:
            raise ValueError(f"Unknown season preset(s): {', '.join(unknown)}")
        if self.target_valid < 1:
            raise ValueError("target_valid must be positive.")
        if self.max_qc_scenes < self.target_valid:
            raise ValueError("max_qc_scenes must be greater than or equal to target_valid.")
        if self.min_days < 0:
            raise ValueError("min_days cannot be negative.")


def operation_key(event_id: str, period: str, season: str) -> str:
    return f"{event_id}:{period}:{season}"


def period_for_year(event_year: int, calendar_year: int) -> str | None:
    if calendar_year == event_year - 1:
        return "PRE"
    if calendar_year == event_year:
        return "EVENT"
    if calendar_year == event_year + 1:
        return "POST"
    return None


def calendar_year_for_period(event_year: int, period: str) -> int:
    return event_year + {"PRE": -1, "EVENT": 0, "POST": 1}[period]


def decode_json(raw: str | None, fallback: Any) -> Any:
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return fallback


def item_datetime(item: dict[str, Any]) -> datetime | None:
    raw = item.get("datetime")
    if not raw:
        return None
    try:
        return parse_datetime(str(raw))
    except ValueError:
        return None


def download_datetime(download: SentinelDownload) -> datetime | None:
    metadata = decode_json(download.metadata_json, {})
    item = metadata.get("item") if isinstance(metadata, dict) else None
    return item_datetime(item) if isinstance(item, dict) else None


def download_sort_key(download: SentinelDownload) -> float:
    value = download_datetime(download)
    return value.timestamp() if value is not None else float("-inf")


def date_gap_ok(candidate: datetime | None, selected: list[datetime], min_days: int) -> bool:
    if candidate is None:
        return True
    return all(abs((candidate.date() - existing.date()).days) >= min_days for existing in selected)


def reviews_by_download(db: Session, download_ids: Iterable[str]) -> dict[str, SentinelSceneReview]:
    ids = list(download_ids)
    if not ids:
        return {}
    rows = db.scalars(
        select(SentinelSceneReview).where(SentinelSceneReview.download_id.in_(ids))
    )
    return {row.download_id: row for row in rows}


def passed_local_qc(download: SentinelDownload) -> bool:
    return any(
        value is not None
        for value in (
            download.cloud_fraction,
            download.shadow_fraction,
            download.nodata_fraction,
            download.dark_fraction,
            download.quality_flags_json,
        )
    )


def download_is_valid(
    download: SentinelDownload,
    review: SentinelSceneReview | None,
    local_cloud_threshold: float,
) -> bool:
    if review is not None and review.is_excluded:
        return False
    if download.is_bad_quality or download.is_bad_cloud:
        return False
    if download.cloud_fraction is not None and float(download.cloud_fraction) > local_cloud_threshold:
        return False
    return passed_local_qc(download)


def sample_is_too_old(sample: Sample) -> bool:
    return any(status.source == "hansen" and status.status == "TOO_OLD" for status in sample.statuses)


def build_operation_rows(events: list[SampleEvent], seasons: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events:
        for period in ("PRE", "EVENT", "POST"):
            year = calendar_year_for_period(event.event_year, period)
            for season in seasons:
                start_mmdd, end_mmdd = SEASON_PRESETS[season]
                rows.append(
                    {
                        "operation_key": operation_key(event.event_id, period, season),
                        "event_id": event.event_id,
                        "event_year": event.event_year,
                        "is_dominant": bool(event.is_dominant),
                        "period": period,
                        "calendar_year": year,
                        "season": season,
                        "start": f"{year}-{start_mmdd}",
                        "end": f"{year}-{end_mmdd}",
                    }
                )
    return rows


def sample_download_counts(
    db: Session,
    sample_id: str,
    options: BatchOptions,
) -> dict[str, int]:
    downloads = list(
        db.scalars(select(SentinelDownload).where(SentinelDownload.sample_id == sample_id))
    )
    reviews = reviews_by_download(db, (download.download_id for download in downloads))
    qc_checked = sum(1 for download in downloads if passed_local_qc(download))
    valid = sum(
        1
        for download in downloads
        if download_is_valid(download, reviews.get(download.download_id), options.local_cloud_threshold)
    )
    return {"downloads": len(downloads), "qc_checked": qc_checked, "valid": valid}


def plan_sentinel_batch(
    db: Session,
    samples: list[Sample],
    options: BatchOptions,
) -> dict[str, Any]:
    options.validate()
    planned: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for sample in samples:
        counts = sample_download_counts(db, sample.sample_id, options)
        if sample_is_too_old(sample):
            skipped.append({"sample_id": sample.sample_id, "reason": "too_old"})
            continue
        if options.mode == "empty_samples":
            if sample.manual_validation is not None:
                skipped.append({"sample_id": sample.sample_id, "reason": "has_manual_validation"})
                continue
            if counts["downloads"]:
                skipped.append({"sample_id": sample.sample_id, "reason": "already_has_downloads"})
                continue
        events = sample_events(db, sample.sample_id)
        if not events:
            skipped.append({"sample_id": sample.sample_id, "reason": "no_sample_events"})
            continue
        operations = build_operation_rows(events, options.seasons)
        planned.append(
            {
                "sample_id": sample.sample_id,
                "short_id": sample.sample_id[:8],
                "display_name": sample.display_name or sample.source_id or sample.sample_id,
                "driver": sample.driver_primary or sample.driver_secondary,
                "event_count": len(events),
                "valid_count": counts["valid"],
                "qc_checked_count": counts["qc_checked"],
                "target_valid": options.target_valid,
                "max_qc_scenes": options.max_qc_scenes,
                "deficit": max(0, options.target_valid - counts["valid"]),
                "stop_reason": (
                    "target_reached"
                    if counts["valid"] >= options.target_valid
                    else "qc_cap_reached"
                    if counts["qc_checked"] >= options.max_qc_scenes
                    else None
                ),
                "operations": operations,
            }
        )

    active = [row for row in planned if row["stop_reason"] is None]
    return {
        "mode": options.mode,
        "options": {
            "seasons": list(options.seasons),
            "target_valid": options.target_valid,
            "max_qc_scenes": options.max_qc_scenes,
            "min_days": options.min_days,
            "stac_max_cloud": options.stac_max_cloud,
            "local_cloud_threshold": options.local_cloud_threshold,
            "search_top_n": options.search_top_n,
        },
        "summary": {
            "samples_considered": len(samples),
            "samples_planned": len(planned),
            "samples_actionable": len(active),
            "samples_skipped": len(skipped),
            "operation_rows": sum(len(row["operations"]) for row in planned),
            "valid_existing": sum(int(row["valid_count"]) for row in planned),
            "qc_checked_existing": sum(int(row["qc_checked_count"]) for row in planned),
        },
        "samples": planned,
        "skipped": skipped,
    }


def ensure_cloud_review(db: Session, download: SentinelDownload, threshold: float) -> None:
    if download.cloud_fraction is None or float(download.cloud_fraction) <= threshold:
        return
    review = db.scalar(
        select(SentinelSceneReview).where(SentinelSceneReview.download_id == download.download_id)
    )
    if review is None:
        review = SentinelSceneReview(download_id=download.download_id)
        db.add(review)
    review.is_excluded = True
    review.reason_code = review.reason_code or "CLOUDS"
    review.notes = review.notes or (
        f"Auto-excluded by Data Operations: local cloud_fraction "
        f"{float(download.cloud_fraction) * 100:.1f}% > {threshold * 100:.1f}%."
    )
    review.updated_at = utc_now()
    reason = db.scalar(
        select(SentinelSceneReviewReason).where(
            SentinelSceneReviewReason.download_id == download.download_id,
            SentinelSceneReviewReason.reason_code == "CLOUDS",
        )
    )
    if reason is None:
        db.add(
            SentinelSceneReviewReason(
                download_id=download.download_id,
                reason_code="CLOUDS",
                reason_text=None,
            )
        )


def matching_relations(events: list[SampleEvent], calendar_year: int) -> list[tuple[SampleEvent, str]]:
    relations: list[tuple[SampleEvent, str]] = []
    for event in events:
        period = period_for_year(event.event_year, calendar_year)
        if period:
            relations.append((event, period))
    return relations


def save_search_rows(
    db: Session,
    *,
    sample: Sample,
    period: SearchPeriod,
    relations: list[tuple[SampleEvent, str]],
    result: dict[str, Any],
    ranked: list[dict[str, Any]],
) -> None:
    for event, relation_period in relations:
        db.add(
            SentinelSearch(
                search_id=uuid.uuid4().hex,
                sample_id=sample.sample_id,
                event_id=event.event_id,
                period=relation_period,
                event_year=event.event_year,
                query_json=json.dumps(result.get("request"), ensure_ascii=False, sort_keys=True),
                results_json=json.dumps(ranked, ensure_ascii=False, sort_keys=True),
                cache_path=None,
            )
        )


def selected_relations_for_bucket(
    events: list[SampleEvent],
    calendar_year: int,
    season: str,
    selected_operation_keys: set[str] | None,
) -> list[tuple[SampleEvent, str]]:
    relations = matching_relations(events, calendar_year)
    if selected_operation_keys is None:
        return relations
    return [
        (event, period)
        for event, period in relations
        if operation_key(event.event_id, period, season) in selected_operation_keys
    ]


def round_robin_buckets(
    events: list[SampleEvent],
    seasons: tuple[str, ...],
    selected_operation_keys: set[str] | None,
) -> list[tuple[int, str, list[tuple[SampleEvent, str]]]]:
    years = sorted(
        {
            calendar_year_for_period(event.event_year, period)
            for event in events
            for period in ("PRE", "EVENT", "POST")
        }
    )
    buckets: list[tuple[int, str, list[tuple[SampleEvent, str]]]] = []
    for season in seasons:
        for year in years:
            relations = selected_relations_for_bucket(
                events,
                year,
                season,
                selected_operation_keys,
            )
            if relations:
                buckets.append((year, season, relations))
    return buckets


def link_download(
    db: Session,
    download: SentinelDownload,
    relations: list[tuple[SampleEvent, str]],
    rank: int | None,
) -> None:
    for event, period in relations:
        upsert_event_download_link(
            db,
            event_id=event.event_id,
            download_id=download.download_id,
            period=period,
            rank=rank,
        )


def output_dir(operation: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = DATA_OPERATIONS_OUTPUT_DIR / f"{operation}_{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys or ["action"])
        writer.writeheader()
        writer.writerows(rows)


def execute_sentinel_batch(
    db: Session,
    samples: list[Sample],
    options: BatchOptions,
    *,
    selected_operation_keys: set[str] | None = None,
    progress: ProgressCallback | None = None,
    should_cancel: CancelCallback | None = None,
) -> dict[str, Any]:
    options.validate()
    report_rows: list[dict[str, Any]] = []
    bundle_download_ids: list[str] = []
    seen_bundle_ids: set[str] = set()
    total = len(samples)

    for sample_index, sample in enumerate(samples, start=1):
        if should_cancel and should_cancel():
            break
        events = sample_events(db, sample.sample_id)
        if not events:
            report_rows.append({"sample_id": sample.sample_id, "action": "skip_no_sample_events"})
            continue

        downloads = list(
            db.scalars(select(SentinelDownload).where(SentinelDownload.sample_id == sample.sample_id))
        )
        reviews = reviews_by_download(db, (download.download_id for download in downloads))
        existing_by_scene = {download.scene_id: download for download in downloads}
        selected_dates: list[datetime] = []
        valid_ids: set[str] = set()
        qc_ids = {download.download_id for download in downloads if passed_local_qc(download)}
        for download in sorted(downloads, key=download_sort_key):
            dt = download_datetime(download)
            if download_is_valid(download, reviews.get(download.download_id), options.local_cloud_threshold):
                if date_gap_ok(dt, selected_dates, options.min_days):
                    valid_ids.add(download.download_id)
                    if dt is not None:
                        selected_dates.append(dt)

        if len(valid_ids) >= options.target_valid:
            report_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "action": "skip_target_reached",
                    "valid": len(valid_ids),
                    "qc_checked": len(qc_ids),
                }
            )
            continue

        buckets = round_robin_buckets(events, options.seasons, selected_operation_keys)
        bbox = sample_geometries(sample.lon, sample.lat)["viewer_aoi_bbox"]
        seen_scene_ids = set(existing_by_scene)
        stop_reason = "windows_exhausted"
        for calendar_year, season, relations in buckets:
            if should_cancel and should_cancel():
                stop_reason = "cancelled"
                break
            if len(valid_ids) >= options.target_valid:
                stop_reason = "target_reached"
                break
            if len(qc_ids) >= options.max_qc_scenes:
                stop_reason = "qc_cap_reached"
                break

            start_mmdd, end_mmdd = SEASON_PRESETS[season]
            search_period = SearchPeriod("WINDOW", calendar_year, start_mmdd, end_mmdd)
            if progress:
                progress(
                    "searching",
                    f"{sample.sample_id[:8]} {calendar_year} {season}: searching",
                    sample_index - 1,
                    total,
                )
            try:
                result = stac_search_s2(
                    bbox=tuple(bbox),
                    period=search_period,
                    max_cloud=options.stac_max_cloud,
                    limit=max(options.search_top_n, options.target_valid),
                )
                ranked = sorted(
                    (rank_item(item, search_period) for item in result["items"]),
                    key=lambda item: item["combined_score"],
                )[: options.search_top_n]
                save_search_rows(
                    db,
                    sample=sample,
                    period=search_period,
                    relations=relations,
                    result=result,
                    ranked=ranked,
                )
                db.commit()
            except Exception as exc:
                db.rollback()
                report_rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "calendar_year": calendar_year,
                        "season": season,
                        "action": "failed_search",
                        "error": str(exc)[:500],
                    }
                )
                continue

            for rank, item in enumerate(ranked, start=1):
                if len(valid_ids) >= options.target_valid or len(qc_ids) >= options.max_qc_scenes:
                    break
                if should_cancel and should_cancel():
                    stop_reason = "cancelled"
                    break
                scene_id = str(item.get("id") or "")
                if not scene_id:
                    continue
                existing = existing_by_scene.get(scene_id)
                dt = item_datetime(item)
                if existing is not None:
                    try:
                        cache_missing = not resolve_storage_path(existing.local_path).exists()
                        qc_missing = not passed_local_qc(existing)
                        if cache_missing or qc_missing:
                            if progress:
                                progress(
                                    "restoring",
                                    f"{sample.sample_id[:8]}: restoring {scene_id[:64]}",
                                    sample_index - 1,
                                    total,
                                )
                            restore_sentinel_download_cache(
                                db,
                                existing,
                                max_cloud=options.stac_max_cloud,
                            )
                            ensure_cloud_review(db, existing, options.local_cloud_threshold)
                            qc_ids.add(existing.download_id)
                            review = db.scalar(
                                select(SentinelSceneReview).where(
                                    SentinelSceneReview.download_id == existing.download_id
                                )
                            )
                            if (
                                download_is_valid(existing, review, options.local_cloud_threshold)
                                and date_gap_ok(dt, selected_dates, options.min_days)
                            ):
                                valid_ids.add(existing.download_id)
                                if dt is not None:
                                    selected_dates.append(dt)
                        link_download(db, existing, relations, rank)
                        db.commit()
                        if existing.download_id not in seen_bundle_ids:
                            seen_bundle_ids.add(existing.download_id)
                            bundle_download_ids.append(existing.download_id)
                        report_rows.append(
                            {
                                "sample_id": sample.sample_id,
                                "scene_id": scene_id,
                                "download_id": existing.download_id,
                                "calendar_year": calendar_year,
                                "season": season,
                                "action": "restored_existing" if cache_missing or qc_missing else "linked_existing",
                            }
                        )
                    except Exception as exc:
                        db.rollback()
                        report_rows.append(
                            {
                                "sample_id": sample.sample_id,
                                "scene_id": scene_id,
                                "download_id": existing.download_id,
                                "calendar_year": calendar_year,
                                "season": season,
                                "action": "failed_existing_scene",
                                "error": str(exc)[:500],
                            }
                        )
                    continue
                if scene_id in seen_scene_ids:
                    continue
                seen_scene_ids.add(scene_id)
                if not date_gap_ok(dt, selected_dates, options.min_days):
                    report_rows.append(
                        {
                            "sample_id": sample.sample_id,
                            "scene_id": scene_id,
                            "calendar_year": calendar_year,
                            "season": season,
                            "action": "skip_min_spacing",
                        }
                    )
                    continue

                primary_period = relations[0][1]
                if progress:
                    progress(
                        "downloading",
                        f"{sample.sample_id[:8]} {calendar_year} {season}: {scene_id[:64]}",
                        sample_index - 1,
                        total,
                    )
                try:
                    payload = download_sentinel_for_sample(
                        db,
                        sample,
                        item=item,
                        period=primary_period,
                        max_cloud=options.stac_max_cloud,
                    )
                    if progress:
                        progress(
                            "local_qc",
                            f"{sample.sample_id[:8]}: quality check {scene_id[:64]}",
                            sample_index - 1,
                            total,
                        )
                    download = db.get(SentinelDownload, payload["download_id"])
                    if download is None:
                        report_rows.append(
                            {
                                "sample_id": sample.sample_id,
                                "scene_id": scene_id,
                                "action": "failed_missing_download_row",
                            }
                        )
                        continue
                    ensure_cloud_review(db, download, options.local_cloud_threshold)
                    link_download(db, download, relations, rank)
                    db.commit()
                except Exception as exc:
                    db.rollback()
                    report_rows.append(
                        {
                            "sample_id": sample.sample_id,
                            "scene_id": scene_id,
                            "calendar_year": calendar_year,
                            "season": season,
                            "action": "failed_download",
                            "error": str(exc)[:500],
                        }
                    )
                    continue
                existing_by_scene[scene_id] = download
                qc_ids.add(download.download_id)
                review = db.scalar(
                    select(SentinelSceneReview).where(
                        SentinelSceneReview.download_id == download.download_id
                    )
                )
                valid = download_is_valid(download, review, options.local_cloud_threshold)
                if valid and date_gap_ok(dt, selected_dates, options.min_days):
                    valid_ids.add(download.download_id)
                    if dt is not None:
                        selected_dates.append(dt)
                if download.download_id not in seen_bundle_ids:
                    seen_bundle_ids.add(download.download_id)
                    bundle_download_ids.append(download.download_id)
                report_rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "scene_id": scene_id,
                        "download_id": download.download_id,
                        "calendar_year": calendar_year,
                        "season": season,
                        "periods": ",".join(sorted({period for _, period in relations})),
                        "cloud_fraction": download.cloud_fraction,
                        "action": "downloaded_valid" if valid else "downloaded_excluded",
                    }
                )

        report_rows.append(
            {
                "sample_id": sample.sample_id,
                "action": "sample_done",
                "valid": len(valid_ids),
                "qc_checked": len(qc_ids),
                "stop_reason": stop_reason,
            }
        )
        if progress:
            progress(
                "sample_done",
                f"{sample.sample_id[:8]}: {len(valid_ids)}/{options.target_valid} valid, "
                f"{len(qc_ids)}/{options.max_qc_scenes} checked",
                sample_index,
                total,
            )

    artifact_dir = output_dir("sentinel_download")
    operations_path = artifact_dir / "operations.csv"
    errors_path = artifact_dir / "errors.csv"
    selection_path = artifact_dir / "bundle_selection.json"
    summary_path = artifact_dir / "summary.json"
    write_rows_csv(operations_path, report_rows)
    error_rows = [row for row in report_rows if str(row.get("action", "")).startswith("failed")]
    write_rows_csv(errors_path, error_rows)
    selected_download_by_id = {
        download.download_id: download
        for download in db.scalars(
            select(SentinelDownload).where(
                SentinelDownload.download_id.in_(bundle_download_ids)
            )
        )
    } if bundle_download_ids else {}
    selected_download_rows = [
        selected_download_by_id[download_id]
        for download_id in bundle_download_ids
        if download_id in selected_download_by_id
    ]
    selected_reviews = reviews_by_download(
        db,
        (download.download_id for download in selected_download_rows),
    )
    selected_links = list(
        db.scalars(
            select(SampleEventDownload).where(
                SampleEventDownload.download_id.in_(bundle_download_ids)
            ).order_by(
                SampleEventDownload.download_id,
                SampleEventDownload.event_id,
                SampleEventDownload.period,
            )
        )
    ) if bundle_download_ids else []
    selection = {
        "manifest_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "download_ids": bundle_download_ids,
        "downloads": [
            {
                "download_id": download.download_id,
                "sample_id": download.sample_id,
                "scene_id": download.scene_id,
                "cloud_fraction": download.cloud_fraction,
                "is_bad_quality": bool(download.is_bad_quality),
                "is_excluded": bool(
                    selected_reviews.get(download.download_id)
                    and selected_reviews[download.download_id].is_excluded
                ),
                "local_qc_complete": passed_local_qc(download),
            }
            for download in selected_download_rows
        ],
        "event_links": [
            {
                "event_id": link.event_id,
                "download_id": link.download_id,
                "period": link.period,
                "rank": link.rank,
            }
            for link in selected_links
        ],
        "options": {
            "mode": options.mode,
            "seasons": list(options.seasons),
            "target_valid": options.target_valid,
            "max_qc_scenes": options.max_qc_scenes,
        },
    }
    selection_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    action_counts = Counter(str(row.get("action") or "") for row in report_rows)
    summary = {
        "status": "cancelled" if should_cancel and should_cancel() else "completed",
        "samples": len(samples),
        "downloads_created": len(bundle_download_ids),
        "options": {
            "mode": options.mode,
            "seasons": list(options.seasons),
            "target_valid": options.target_valid,
            "max_qc_scenes": options.max_qc_scenes,
            "min_days": options.min_days,
            "stac_max_cloud": options.stac_max_cloud,
            "local_cloud_threshold": options.local_cloud_threshold,
            "search_top_n": options.search_top_n,
            "selected_operation_keys": sorted(selected_operation_keys)
            if selected_operation_keys is not None
            else None,
        },
        "actions": dict(sorted(action_counts.items())),
        "artifact_dir": str(artifact_dir),
        "operations_csv": str(operations_path),
        "errors_csv": str(errors_path),
        "bundle_selection": str(selection_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
