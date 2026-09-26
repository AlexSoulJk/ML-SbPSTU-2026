from __future__ import annotations

import csv
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from sqlalchemy import select


LIMIT_DOWNLOADS: int | None = None
ONLY_MISSING = True


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "Scripts"
RESEARCH_TOOL_ROOT = SCRIPTS_ROOT / "ResearchTool"
sys.path.insert(0, str(RESEARCH_TOOL_ROOT))

from backend.forest.database import SessionLocal, init_forest_database  # noqa: E402
from backend.forest.models import DerivedPreview, SentinelDownload  # noqa: E402
from backend.forest.storage_paths import forest_cache_url, resolve_storage_path  # noqa: E402


SCRIPT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_ROOT / "Outputs"
EXPECTED_VIEWS = ("rgb", "false_color", "ndvi", "nbr", "ndmi", "scl")


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_report(rows: list[dict[str, object]]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / f"sentinel_preview_cache_check_{now_tag()}.csv"
    fieldnames = [
        "status",
        "download_id",
        "sample_id",
        "period",
        "scene_id",
        "view",
        "local_path",
        "png_path",
        "metadata_path",
        "png_exists",
        "metadata_exists",
        "png_url",
        "error",
    ]
    with report_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return report_path


def preview_rows(db, download: SentinelDownload) -> list[dict[str, object]]:
    previews = {
        preview.view: preview
        for preview in db.scalars(
            select(DerivedPreview).where(DerivedPreview.download_id == download.download_id)
        )
    }
    rows = []
    for view in EXPECTED_VIEWS:
        preview = previews.get(view)
        if preview is None:
            rows.append(
                {
                    "status": "missing_db_preview",
                    "download_id": download.download_id,
                    "sample_id": download.sample_id,
                    "period": download.period or "",
                    "scene_id": download.scene_id,
                    "view": view,
                    "local_path": download.local_path,
                    "error": "No DerivedPreview row for this view.",
                }
            )
            continue

        png_path = resolve_storage_path(preview.png_path)
        metadata_path = resolve_storage_path(preview.metadata_path)
        png_exists = png_path.exists()
        metadata_exists = metadata_path.exists()
        status = "ok" if png_exists and metadata_exists else "missing_file"
        rows.append(
            {
                "status": status,
                "download_id": download.download_id,
                "sample_id": download.sample_id,
                "period": download.period or "",
                "scene_id": download.scene_id,
                "view": view,
                "local_path": download.local_path,
                "png_path": preview.png_path,
                "metadata_path": preview.metadata_path,
                "png_exists": png_exists,
                "metadata_exists": metadata_exists,
                "png_url": forest_cache_url(preview.png_path) or "",
                "error": "",
            }
        )
    return rows


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
            rows.extend(preview_rows(db, download))

    report_rows = [row for row in rows if row["status"] != "ok"] if ONLY_MISSING else rows
    report_path = write_report(report_rows)
    print(f"LIMIT_DOWNLOADS: {LIMIT_DOWNLOADS}")
    print(f"ONLY_MISSING: {ONLY_MISSING}")
    print(f"Downloads checked: {len({row['download_id'] for row in rows})}")
    print(f"Preview rows checked: {len(rows)}")
    print(f"Statuses: {dict(sorted(Counter(row['status'] for row in rows).items()))}")
    print(f"Missing by view: {dict(sorted(Counter(row['view'] for row in rows if row['status'] != 'ok').items()))}")
    print(f"Rows written to report: {len(report_rows)}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
