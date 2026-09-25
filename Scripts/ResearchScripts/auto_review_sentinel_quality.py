from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from sqlalchemy import select


DRY_RUN = True
LIMIT_DOWNLOADS: int | None = None
OVERWRITE_EXISTING_REVIEWS = False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "Scripts"
RESEARCH_TOOL_ROOT = SCRIPTS_ROOT / "ResearchTool"
sys.path.insert(0, str(RESEARCH_TOOL_ROOT))

from backend.forest.database import SessionLocal, init_forest_database  # noqa: E402
from backend.forest.models import SentinelDownload, SentinelSceneReview, utc_now  # noqa: E402
from backend.forest.sentinel_service import AUTO_REVIEW_NOTES, auto_review_reason, cloud_metrics  # noqa: E402
from backend.forest.storage_paths import resolve_storage_path  # noqa: E402


SCRIPT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_ROOT / "Outputs"


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_report(rows: list[dict[str, object]]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / f"sentinel_quality_auto_review_{now_tag()}.csv"
    fieldnames = [
        "action",
        "download_id",
        "sample_id",
        "period",
        "scene_id",
        "local_path",
        "reason_code",
        "cloud_fraction",
        "shadow_fraction",
        "nodata_fraction",
        "dark_fraction",
        "brightness_p50",
        "brightness_p95",
        "quality_flags",
        "error",
    ]
    with report_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return report_path


def row_base(download: SentinelDownload) -> dict[str, object]:
    return {
        "download_id": download.download_id,
        "sample_id": download.sample_id,
        "period": download.period or "",
        "scene_id": download.scene_id,
        "local_path": download.local_path,
    }


def main() -> None:
    init_forest_database()
    rows: list[dict[str, object]] = []

    with SessionLocal() as db:
        stmt = select(SentinelDownload).order_by(
            SentinelDownload.sample_id,
            SentinelDownload.period,
            SentinelDownload.scene_id,
            SentinelDownload.download_id,
        )
        if LIMIT_DOWNLOADS is not None:
            stmt = stmt.limit(LIMIT_DOWNLOADS)
        downloads = list(db.scalars(stmt))

        for download in downloads:
            base = row_base(download)
            path = resolve_storage_path(download.local_path)
            if not path.exists():
                rows.append({**base, "action": "missing_tif", "error": str(path)})
                continue

            try:
                metrics = cloud_metrics(path)
            except Exception as exc:
                rows.append({**base, "action": "failed", "error": str(exc)[:500]})
                continue

            reason_code = auto_review_reason(metrics)
            flags_json = json.dumps(metrics["quality_flags"], ensure_ascii=False, sort_keys=True)
            review = db.scalar(
                select(SentinelSceneReview).where(
                    SentinelSceneReview.download_id == download.download_id
                )
            )

            action = "ok"
            if reason_code:
                if review is None:
                    action = "would_auto_exclude" if DRY_RUN else "auto_excluded"
                    if not DRY_RUN:
                        db.add(
                            SentinelSceneReview(
                                download_id=download.download_id,
                                is_excluded=True,
                                reason_code=reason_code,
                                reason_text=None,
                                notes=AUTO_REVIEW_NOTES,
                            )
                        )
                elif OVERWRITE_EXISTING_REVIEWS:
                    action = "would_update_review" if DRY_RUN else "updated_review"
                    if not DRY_RUN:
                        review.is_excluded = True
                        review.reason_code = reason_code
                        review.reason_text = None
                        review.notes = AUTO_REVIEW_NOTES
                        review.updated_at = utc_now()
                else:
                    action = "skip_existing_review"

            if not DRY_RUN:
                download.cloud_fraction = float(metrics["cloud_fraction"])
                download.shadow_fraction = float(metrics["shadow_fraction"])
                download.nodata_fraction = float(metrics["nodata_fraction"])
                download.dark_fraction = float(metrics["dark_fraction"])
                download.is_bad_cloud = bool(metrics["is_bad_cloud"])
                download.is_bad_quality = bool(metrics["is_bad_quality"])
                download.quality_flags_json = flags_json

            rows.append(
                {
                    **base,
                    "action": action,
                    "reason_code": reason_code or "",
                    "cloud_fraction": metrics["cloud_fraction"],
                    "shadow_fraction": metrics["shadow_fraction"],
                    "nodata_fraction": metrics["nodata_fraction"],
                    "dark_fraction": metrics["dark_fraction"],
                    "brightness_p50": metrics["brightness_p50"],
                    "brightness_p95": metrics["brightness_p95"],
                    "quality_flags": flags_json,
                    "error": "",
                }
            )

        if DRY_RUN:
            db.rollback()
        else:
            db.commit()

    report_path = write_report(rows)
    print(f"DRY_RUN: {DRY_RUN}")
    print(f"LIMIT_DOWNLOADS: {LIMIT_DOWNLOADS}")
    print(f"OVERWRITE_EXISTING_REVIEWS: {OVERWRITE_EXISTING_REVIEWS}")
    print(f"Downloads checked: {len(rows)}")
    print(f"Actions: {dict(sorted(Counter(row['action'] for row in rows).items()))}")
    print(f"Report: {report_path}")
    if DRY_RUN:
        print("Dry run only. Set DRY_RUN = False to write metrics and auto-exclusions.")


if __name__ == "__main__":
    main()
