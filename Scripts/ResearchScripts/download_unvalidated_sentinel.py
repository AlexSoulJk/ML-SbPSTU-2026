from __future__ import annotations

import csv
import json
import sys
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import or_, select


DRY_RUN = False
ENSURE_SCHEMA = False

LIMIT_SAMPLES = 100
DRIVERS = ("Logging", "Wildfire")
PERIODS = ("PRE", "POST")
SEASON_PRESET = "full_snow_free"

# STAC cloud filter is scene-level and optimistic; keep it wide, then filter by local AOI cloud_fraction.
STAC_MAX_CLOUD = 30.0
SEARCH_TOP_N_PER_PERIOD = 40
MAX_ACCEPTED_PER_PERIOD = 10
MIN_DAYS_BETWEEN_ACCEPTED = 7
LOCAL_CLOUD_THRESHOLD = 0.30

AUTO_EXCLUDE_CLOUDY = True
CLOUD_REVIEW_REASON = "CLOUDS"

# Leave empty for all selected samples, or set prefixes for a targeted run.
SAMPLE_ID_PREFIXES: tuple[str, ...] = ()

# Dry-run defaults to DB-only planning. Set True if you want dry-run to call STAC search without downloads.
SEARCH_IN_DRY_RUN = False

# Save STAC search rows so UI can show the candidates found by this script.
SAVE_SEARCH_RESULTS = True


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOL_ROOT = PROJECT_ROOT / "Scripts" / "ResearchTool"
sys.path.insert(0, str(TOOL_ROOT))

from backend.cdse_client import parse_datetime, stac_search_s2  # noqa: E402
from backend.config import ensure_directories, load_dotenv  # noqa: E402
from backend.forest.database import SessionLocal, init_forest_database  # noqa: E402
from backend.forest.geometry import sample_geometries  # noqa: E402
from backend.forest.models import (  # noqa: E402
    HansenAnalysis,
    ManualValidation,
    Sample,
    SampleStatus,
    SentinelDownload,
    SentinelSceneReview,
    SentinelSearch,
    utc_now,
)
from backend.forest.sentinel_service import (  # noqa: E402
    download_sentinel_for_sample,
    periods_for_event_year,
    rank_item,
)


SCRIPT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_ROOT / "Outputs"


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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
    item = metadata.get("item") if isinstance(metadata, dict) else {}
    return item_datetime(item or {})


def date_gap_ok(candidate: datetime | None, selected: list[datetime]) -> bool:
    if candidate is None:
        return True
    return all(abs((candidate.date() - existing.date()).days) >= MIN_DAYS_BETWEEN_ACCEPTED for existing in selected)


def latest_hansen_event_year(db, sample_id: str) -> int | None:
    analysis = db.scalar(
        select(HansenAnalysis)
        .where(HansenAnalysis.sample_id == sample_id)
        .order_by(HansenAnalysis.created_at.desc())
        .limit(1)
    )
    return analysis.event_year if analysis is not None else None


def event_year_for_sample(db, sample: Sample) -> int | None:
    return latest_hansen_event_year(db, sample.sample_id) or sample.source_event_year


def select_target_samples(db) -> list[Sample]:
    hansen_too_old = (
        select(SampleStatus.sample_id)
        .where(
            SampleStatus.sample_id == Sample.sample_id,
            SampleStatus.source == "hansen",
            SampleStatus.status == "TOO_OLD",
        )
        .exists()
    )
    stmt = (
        select(Sample)
        .outerjoin(ManualValidation, ManualValidation.sample_id == Sample.sample_id)
        .where(
            ManualValidation.sample_id.is_(None),
            or_(
                Sample.driver_primary.in_(DRIVERS),
                Sample.driver_secondary.in_(DRIVERS),
            ),
            ~hansen_too_old,
        )
        .order_by(Sample.source_file, Sample.source_id, Sample.sample_id)
    )
    samples = list(db.scalars(stmt))
    if SAMPLE_ID_PREFIXES:
        samples = [
            sample for sample in samples
            if any(sample.sample_id.startswith(prefix) for prefix in SAMPLE_ID_PREFIXES)
        ]
    return samples[:LIMIT_SAMPLES] if LIMIT_SAMPLES is not None else samples


def review_for_download(db, download_id: str) -> SentinelSceneReview | None:
    return db.scalar(
        select(SentinelSceneReview).where(SentinelSceneReview.download_id == download_id)
    )


def is_excluded(db, download_id: str) -> bool:
    review = review_for_download(db, download_id)
    return bool(review and review.is_excluded)


def upsert_cloud_review(db, download: SentinelDownload, cloud_fraction: float | None) -> str:
    if not AUTO_EXCLUDE_CLOUDY:
        return "cloud_review_disabled"
    if DRY_RUN:
        return "would_save_cloud_review"
    review = review_for_download(db, download.download_id)
    if review is not None and review.is_excluded and review.reason_code and review.reason_code != CLOUD_REVIEW_REASON:
        return "kept_existing_review"
    if review is None:
        review = SentinelSceneReview(download_id=download.download_id)
        db.add(review)

    percent = 100.0 * float(cloud_fraction or 0.0)
    threshold = 100.0 * LOCAL_CLOUD_THRESHOLD
    review.is_excluded = True
    review.reason_code = CLOUD_REVIEW_REASON
    review.reason_text = None
    review.notes = (
        f"Auto-excluded by bulk Sentinel downloader: local cloud_fraction "
        f"{percent:.1f}% > {threshold:.1f}%."
    )
    review.updated_at = utc_now()
    db.commit()
    return "cloud_review_saved"


def existing_period_downloads(db, sample_id: str, period: str) -> list[SentinelDownload]:
    return list(
        db.scalars(
            select(SentinelDownload)
            .where(
                SentinelDownload.sample_id == sample_id,
                SentinelDownload.period == period,
            )
            .order_by(SentinelDownload.created_at)
        )
    )


def selected_existing_dates(db, downloads: list[SentinelDownload]) -> tuple[list[datetime], int]:
    selected_dates: list[datetime] = []
    accepted = 0
    def sort_key(item: SentinelDownload) -> float:
        dt = download_datetime(item)
        return dt.timestamp() if dt is not None else 0.0

    for download in sorted(downloads, key=sort_key):
        if is_excluded(db, download.download_id):
            continue
        if download.cloud_fraction is not None and float(download.cloud_fraction) > LOCAL_CLOUD_THRESHOLD:
            upsert_cloud_review(db, download, float(download.cloud_fraction))
            continue
        dt = download_datetime(download)
        if not date_gap_ok(dt, selected_dates):
            continue
        if dt is not None:
            selected_dates.append(dt)
        accepted += 1
        if accepted >= MAX_ACCEPTED_PER_PERIOD:
            break
    return selected_dates, accepted


def save_search_result(
    db,
    *,
    sample: Sample,
    event_year: int,
    period: Any,
    result: dict[str, Any],
    ranked: list[dict[str, Any]],
) -> None:
    if DRY_RUN or not SAVE_SEARCH_RESULTS:
        return
    db.add(
        SentinelSearch(
            search_id=uuid.uuid4().hex,
            sample_id=sample.sample_id,
            period=period.key,
            event_year=event_year,
            query_json=json.dumps(result.get("request"), ensure_ascii=False, sort_keys=True),
            results_json=json.dumps(ranked, ensure_ascii=False, sort_keys=True),
            cache_path=None,
        )
    )
    db.flush()


def report_row(
    rows: list[dict[str, Any]],
    *,
    sample: Sample,
    period: str,
    event_year: int | None,
    scene_id: str = "",
    scene_datetime: str = "",
    action: str,
    cloud_fraction: float | None = None,
    download_id: str = "",
    reason: str = "",
    error: str = "",
) -> None:
    rows.append(
        {
            "sample_id": sample.sample_id,
            "source_id": sample.source_id or "",
            "display_name": sample.display_name or "",
            "driver_primary": sample.driver_primary or "",
            "driver_secondary": sample.driver_secondary or "",
            "event_year": event_year or "",
            "period": period,
            "scene_id": scene_id,
            "scene_datetime": scene_datetime,
            "download_id": download_id,
            "cloud_fraction": "" if cloud_fraction is None else round(float(cloud_fraction), 4),
            "action": action,
            "reason": reason,
            "error": error[:500],
        }
    )


def process_period(db, sample: Sample, event_year: int, period: Any, rows: list[dict[str, Any]]) -> None:
    existing_downloads = existing_period_downloads(db, sample.sample_id, period.key)
    existing_by_scene = {download.scene_id: download for download in existing_downloads}
    selected_dates, accepted_count = selected_existing_dates(db, existing_downloads)

    if accepted_count >= MAX_ACCEPTED_PER_PERIOD:
        report_row(
            rows,
            sample=sample,
            period=period.key,
            event_year=event_year,
            action="skip_period_limit_already_met",
            reason=f"accepted_existing={accepted_count}",
        )
        return

    if DRY_RUN and not SEARCH_IN_DRY_RUN:
        report_row(
            rows,
            sample=sample,
            period=period.key,
            event_year=event_year,
            action="would_search_period",
            reason=f"accepted_existing={accepted_count}",
        )
        return

    bbox = sample_geometries(sample.lon, sample.lat)["viewer_aoi_bbox"]
    result = stac_search_s2(
        bbox=tuple(bbox),
        period=period,
        max_cloud=STAC_MAX_CLOUD,
        limit=max(SEARCH_TOP_N_PER_PERIOD, MAX_ACCEPTED_PER_PERIOD),
    )
    ranked = sorted(
        (rank_item(item, period) for item in result["items"]),
        key=lambda item: item["combined_score"],
    )[:SEARCH_TOP_N_PER_PERIOD]
    save_search_result(db, sample=sample, event_year=event_year, period=period, result=result, ranked=ranked)

    for item in ranked:
        if accepted_count >= MAX_ACCEPTED_PER_PERIOD:
            break

        scene_id = item.get("id") or ""
        scene_datetime = item.get("datetime") or ""
        dt = item_datetime(item)
        existing = existing_by_scene.get(scene_id)

        if not date_gap_ok(dt, selected_dates):
            report_row(
                rows,
                sample=sample,
                period=period.key,
                event_year=event_year,
                scene_id=scene_id,
                scene_datetime=scene_datetime,
                action="skip_min_spacing",
                reason=f"min_days={MIN_DAYS_BETWEEN_ACCEPTED}",
            )
            continue

        if existing is not None:
            if is_excluded(db, existing.download_id):
                report_row(
                    rows,
                    sample=sample,
                    period=period.key,
                    event_year=event_year,
                    scene_id=scene_id,
                    scene_datetime=scene_datetime,
                    download_id=existing.download_id,
                    cloud_fraction=existing.cloud_fraction,
                    action="skip_existing_excluded",
                )
                continue
            if existing.cloud_fraction is not None and float(existing.cloud_fraction) > LOCAL_CLOUD_THRESHOLD:
                review_action = upsert_cloud_review(db, existing, float(existing.cloud_fraction))
                report_row(
                    rows,
                    sample=sample,
                    period=period.key,
                    event_year=event_year,
                    scene_id=scene_id,
                    scene_datetime=scene_datetime,
                    download_id=existing.download_id,
                    cloud_fraction=existing.cloud_fraction,
                    action="skip_existing_cloudy",
                    reason=review_action,
                )
                continue
            if dt is not None:
                selected_dates.append(dt)
            accepted_count += 1
            report_row(
                rows,
                sample=sample,
                period=period.key,
                event_year=event_year,
                scene_id=scene_id,
                scene_datetime=scene_datetime,
                download_id=existing.download_id,
                cloud_fraction=existing.cloud_fraction,
                action="count_existing_accepted",
            )
            continue

        if DRY_RUN:
            if dt is not None:
                selected_dates.append(dt)
            accepted_count += 1
            report_row(
                rows,
                sample=sample,
                period=period.key,
                event_year=event_year,
                scene_id=scene_id,
                scene_datetime=scene_datetime,
                action="would_download_candidate",
            )
            continue

        result_payload = download_sentinel_for_sample(
            db,
            sample,
            item=item,
            period=period.key,
            max_cloud=STAC_MAX_CLOUD,
        )
        download = db.get(SentinelDownload, result_payload["download_id"])
        cloud_fraction = float(result_payload["cloud_fraction"])
        if download is not None and cloud_fraction > LOCAL_CLOUD_THRESHOLD:
            review_action = upsert_cloud_review(db, download, cloud_fraction)
            report_row(
                rows,
                sample=sample,
                period=period.key,
                event_year=event_year,
                scene_id=scene_id,
                scene_datetime=scene_datetime,
                download_id=result_payload["download_id"],
                cloud_fraction=cloud_fraction,
                action="downloaded_auto_excluded_cloudy",
                reason=review_action,
            )
            continue

        if dt is not None:
            selected_dates.append(dt)
        accepted_count += 1
        report_row(
            rows,
            sample=sample,
            period=period.key,
            event_year=event_year,
            scene_id=scene_id,
            scene_datetime=scene_datetime,
            download_id=result_payload["download_id"],
            cloud_fraction=cloud_fraction,
            action="downloaded_accepted",
        )


def write_report(rows: list[dict[str, Any]]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / f"sentinel_unvalidated_download_{now_tag()}.csv"
    fieldnames = [
        "sample_id",
        "source_id",
        "display_name",
        "driver_primary",
        "driver_secondary",
        "event_year",
        "period",
        "scene_id",
        "scene_datetime",
        "download_id",
        "cloud_fraction",
        "action",
        "reason",
        "error",
    ]
    with report_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return report_path


def main() -> None:
    load_dotenv()
    ensure_directories()
    if ENSURE_SCHEMA:
        init_forest_database()

    rows: list[dict[str, Any]] = []
    with SessionLocal() as db:
        samples = select_target_samples(db)
        print(f"Selected samples: {len(samples)}")
        print(f"LIMIT_SAMPLES: {LIMIT_SAMPLES}")
        print(f"Drivers: {', '.join(DRIVERS)}")
        print(f"Periods: {', '.join(PERIODS)}")
        print(f"MAX_ACCEPTED_PER_PERIOD: {MAX_ACCEPTED_PER_PERIOD}")
        print(f"MIN_DAYS_BETWEEN_ACCEPTED: {MIN_DAYS_BETWEEN_ACCEPTED}")
        print(f"LOCAL_CLOUD_THRESHOLD: {LOCAL_CLOUD_THRESHOLD}")
        print(f"STAC_MAX_CLOUD: {STAC_MAX_CLOUD}")
        print(f"DRY_RUN: {DRY_RUN}")
        print(f"SEARCH_IN_DRY_RUN: {SEARCH_IN_DRY_RUN}")

        for index, sample in enumerate(samples, start=1):
            event_year = event_year_for_sample(db, sample)
            if event_year is None:
                report_row(
                    rows,
                    sample=sample,
                    period="",
                    event_year=None,
                    action="skip_no_event_year",
                )
                continue

            periods = [
                period
                for period in periods_for_event_year(int(event_year), SEASON_PRESET)
                if period.key in PERIODS
            ]
            print(f"[{index}/{len(samples)}] {sample.sample_id[:8]} event={event_year}")
            for period in periods:
                try:
                    process_period(db, sample, int(event_year), period, rows)
                except Exception as exc:
                    db.rollback()
                    report_row(
                        rows,
                        sample=sample,
                        period=period.key,
                        event_year=int(event_year),
                        action="failed_period",
                        error=str(exc),
                    )
                    print(f"  {period.key} failed: {exc}")

    report_path = write_report(rows)
    action_counts = Counter(row["action"] for row in rows)
    print(f"Actions: {dict(sorted(action_counts.items()))}")
    print(f"Report: {report_path}")
    if DRY_RUN:
        print("Dry run only. Set DRY_RUN = False to search/download Sentinel imagery.")


if __name__ == "__main__":
    main()
