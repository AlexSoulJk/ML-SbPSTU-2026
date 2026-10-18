from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import select


# Edit these values in the file and run it again. No CLI arguments are required.
DRY_RUN = False

PERIODS: tuple[str, ...] | None = None
SAMPLE_ID_PREFIXES: tuple[str, ...] = ()
DOWNLOAD_ID_PREFIXES: tuple[str, ...] = ()
LIMIT_DOWNLOADS: int | None = None

BUNDLE_PROFILE = "geotiff_only"  # geotiff_only | preview_only | full
EXCLUDE_REVIEWED_BAD = False
CREATE_ZIP = True


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOL_ROOT = PROJECT_ROOT / "Scripts" / "ResearchTool"
sys.path.insert(0, str(TOOL_ROOT))

from backend.forest.bundle_service import bundle_plan, export_bundle  # noqa: E402
from backend.forest.database import SessionLocal, init_forest_database  # noqa: E402
from backend.forest.models import SentinelDownload  # noqa: E402


def selected_download_ids(db) -> list[str]:
    stmt = select(SentinelDownload).order_by(
        SentinelDownload.sample_id,
        SentinelDownload.period,
        SentinelDownload.scene_id,
        SentinelDownload.download_id,
    )
    if PERIODS:
        stmt = stmt.where(SentinelDownload.period.in_(PERIODS))
    downloads = list(db.scalars(stmt))
    if SAMPLE_ID_PREFIXES:
        downloads = [
            row
            for row in downloads
            if any(row.sample_id.startswith(prefix) for prefix in SAMPLE_ID_PREFIXES)
        ]
    if DOWNLOAD_ID_PREFIXES:
        downloads = [
            row
            for row in downloads
            if any(row.download_id.startswith(prefix) for prefix in DOWNLOAD_ID_PREFIXES)
        ]
    if LIMIT_DOWNLOADS is not None:
        downloads = downloads[:LIMIT_DOWNLOADS]
    return [row.download_id for row in downloads]


def main() -> None:
    init_forest_database()
    with SessionLocal() as db:
        download_ids = selected_download_ids(db)
        plan = bundle_plan(
            db,
            download_ids,
            profile=BUNDLE_PROFILE,
            exclude_reviewed_bad=EXCLUDE_REVIEWED_BAD,
            source="research_script",
        )
        print(f"DRY_RUN: {DRY_RUN}")
        print(f"Bundle profile: {BUNDLE_PROFILE}")
        print(f"Downloads: {plan['summary']['unique_downloads']}")
        print(f"Files: {plan['summary']['unique_files']}")
        print(f"Existing bytes: {plan['summary']['total_bytes'] / 1024 / 1024:.1f} MiB")
        print(f"Missing source files: {plan['summary']['missing_source_files']}")
        print(f"Event links: {plan['summary']['event_links']}")
        if DRY_RUN:
            print("Dry run only. Set DRY_RUN = False to create the bundle.")
            return

        result = export_bundle(
            db,
            plan["download_ids"],
            profile=BUNDLE_PROFILE,
            exclude_reviewed_bad=EXCLUDE_REVIEWED_BAD,
            create_zip=CREATE_ZIP,
            source="research_script",
            progress=lambda stage, message, done, total: print(
                f"[{done}/{total}] {stage}: {message}"
            ),
        )
        print(f"Copied: {result['copied']}")
        print(f"Failed: {result['failed']}")
        print(f"Bundle dir: {result['bundle_dir']}")
        print(f"Manifest: {result['manifest_path']}")
        if result.get("zip_path"):
            print(f"ZIP: {result['zip_path']}")


if __name__ == "__main__":
    main()
