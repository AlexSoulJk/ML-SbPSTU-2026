from __future__ import annotations

import csv
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from sqlalchemy import select


DRY_RUN = True

# Export all Sentinel downloads by default. Restrict these for smaller bundles.
PERIODS: tuple[str, ...] | None = None
SAMPLE_ID_PREFIXES: tuple[str, ...] = ()
DOWNLOAD_ID_PREFIXES: tuple[str, ...] = ()
LIMIT_DOWNLOADS = None

INCLUDE_RASTERS = True
INCLUDE_RASTER_METADATA = True
INCLUDE_PREVIEWS = True
CREATE_ZIP = True


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "Scripts"
RESEARCH_TOOL_ROOT = SCRIPTS_ROOT / "ResearchTool"
sys.path.insert(0, str(RESEARCH_TOOL_ROOT))

from backend.forest.database import SessionLocal  # noqa: E402
from backend.forest.models import DerivedPreview, SentinelDownload  # noqa: E402
from backend.forest.storage_paths import resolve_storage_path, to_storage_path  # noqa: E402


SCRIPT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_ROOT / "Outputs"


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def normalize_relative(path: str | Path) -> str:
    stored = to_storage_path(path)
    if stored is not None:
        return PurePosixPath(*Path(stored).parts).as_posix()
    resolved = Path(path)
    try:
        return PurePosixPath(*resolved.resolve(strict=False).relative_to(PROJECT_ROOT.resolve(strict=False)).parts).as_posix()
    except ValueError:
        return PurePosixPath(*resolved.parts).as_posix()


def display_path(path: str | Path | None) -> str:
    if path is None:
        return ""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    try:
        return str(resolved.resolve(strict=False).relative_to(PROJECT_ROOT.resolve(strict=False)))
    except ValueError:
        return str(resolved)


def selected_downloads(db) -> list[SentinelDownload]:
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
            download for download in downloads
            if any(download.sample_id.startswith(prefix) for prefix in SAMPLE_ID_PREFIXES)
        ]
    if DOWNLOAD_ID_PREFIXES:
        downloads = [
            download for download in downloads
            if any(download.download_id.startswith(prefix) for prefix in DOWNLOAD_ID_PREFIXES)
        ]
    if LIMIT_DOWNLOADS is not None:
        downloads = downloads[:LIMIT_DOWNLOADS]
    return downloads


def add_file(
    files: list[dict[str, Any]],
    *,
    kind: str,
    source_path: str | Path | None,
    download_id: str,
    sample_id: str,
    scene_id: str,
    period: str | None,
    preview_id: str | None = None,
    view: str | None = None,
) -> None:
    if source_path is None:
        return

    path = resolve_storage_path(source_path)
    relative_path = normalize_relative(path)
    files.append(
        {
            "kind": kind,
            "download_id": download_id,
            "sample_id": sample_id,
            "scene_id": scene_id,
            "period": period or "",
            "preview_id": preview_id or "",
            "view": view or "",
            "relative_path": relative_path,
            "source_path": display_path(path),
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
            "action": "pending",
            "error": "",
        }
    )


def collect_bundle_files(db, downloads: list[SentinelDownload]) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for download in downloads:
        if INCLUDE_RASTERS:
            raster_path = resolve_storage_path(download.local_path)
            add_file(
                files,
                kind="sentinel_raster_tif",
                source_path=raster_path,
                download_id=download.download_id,
                sample_id=download.sample_id,
                scene_id=download.scene_id,
                period=download.period,
            )
            if INCLUDE_RASTER_METADATA:
                add_file(
                    files,
                    kind="sentinel_raster_metadata",
                    source_path=raster_path.with_suffix(".json"),
                    download_id=download.download_id,
                    sample_id=download.sample_id,
                    scene_id=download.scene_id,
                    period=download.period,
                )

        if INCLUDE_PREVIEWS:
            previews = list(
                db.scalars(
                    select(DerivedPreview)
                    .where(DerivedPreview.download_id == download.download_id)
                    .order_by(DerivedPreview.view)
                )
            )
            for preview in previews:
                add_file(
                    files,
                    kind="sentinel_preview_png",
                    source_path=preview.png_path,
                    download_id=download.download_id,
                    sample_id=download.sample_id,
                    scene_id=download.scene_id,
                    period=download.period,
                    preview_id=preview.preview_id,
                    view=preview.view,
                )
                add_file(
                    files,
                    kind="sentinel_preview_metadata",
                    source_path=preview.metadata_path,
                    download_id=download.download_id,
                    sample_id=download.sample_id,
                    scene_id=download.scene_id,
                    period=download.period,
                    preview_id=preview.preview_id,
                    view=preview.view,
                )
    return files


def copy_file_to_bundle(bundle_dir: Path, row: dict[str, Any]) -> None:
    source = PROJECT_ROOT / row["relative_path"]
    if not source.exists():
        row["action"] = "missing_source"
        return

    target = bundle_dir / "files" / row["relative_path"]
    if DRY_RUN:
        row["action"] = "would_copy"
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    row["action"] = "copied"


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "kind",
        "download_id",
        "sample_id",
        "scene_id",
        "period",
        "preview_id",
        "view",
        "relative_path",
        "source_path",
        "exists",
        "bytes",
        "action",
        "error",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(bundle_dir: Path, downloads: list[SentinelDownload], files: list[dict[str, Any]]) -> Path:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "bundle_name": bundle_dir.name,
        "source_project_root": str(PROJECT_ROOT),
        "dry_run": DRY_RUN,
        "options": {
            "periods": list(PERIODS) if PERIODS else None,
            "sample_id_prefixes": list(SAMPLE_ID_PREFIXES),
            "download_id_prefixes": list(DOWNLOAD_ID_PREFIXES),
            "limit_downloads": LIMIT_DOWNLOADS,
            "include_rasters": INCLUDE_RASTERS,
            "include_raster_metadata": INCLUDE_RASTER_METADATA,
            "include_previews": INCLUDE_PREVIEWS,
        },
        "downloads": [
            {
                "download_id": download.download_id,
                "sample_id": download.sample_id,
                "scene_id": download.scene_id,
                "period": download.period,
                "raster_path": normalize_relative(resolve_storage_path(download.local_path)),
            }
            for download in downloads
        ],
        "files": files,
    }
    manifest_path = bundle_dir / "manifest.json"
    if not DRY_RUN:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle_dir = OUTPUT_DIR / f"sentinel_cache_bundle_{now_tag()}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    with SessionLocal() as db:
        downloads = selected_downloads(db)
        files = collect_bundle_files(db, downloads)

    for row in files:
        try:
            copy_file_to_bundle(bundle_dir, row)
        except Exception as exc:
            row["action"] = "failed"
            row["error"] = str(exc)[:500]

    report_path = bundle_dir / "files.csv"
    write_csv(files, report_path)
    manifest_path = write_manifest(bundle_dir, downloads, files)

    archive_path = None
    if CREATE_ZIP and not DRY_RUN:
        archive_path = Path(shutil.make_archive(str(bundle_dir), "zip", root_dir=bundle_dir))

    action_counts = Counter(row["action"] for row in files)
    kind_counts = Counter(row["kind"] for row in files)
    missing_source_by_kind = Counter(row["kind"] for row in files if row["action"] == "missing_source")
    failed_by_kind = Counter(row["kind"] for row in files if row["action"] == "failed")
    total_bytes = sum(int(row["bytes"]) for row in files if row["exists"])
    print(f"DRY_RUN: {DRY_RUN}")
    print(f"Downloads: {len(downloads)}")
    print(f"Files: {len(files)}")
    print(f"Existing bytes: {total_bytes / 1024 / 1024:.1f} MiB")
    print(f"File kinds: {dict(sorted(kind_counts.items()))}")
    print(f"Actions: {dict(sorted(action_counts.items()))}")
    if missing_source_by_kind:
        print(f"Missing source by kind: {dict(sorted(missing_source_by_kind.items()))}")
    if failed_by_kind:
        print(f"Failed by kind: {dict(sorted(failed_by_kind.items()))}")
    print(f"Bundle dir: {bundle_dir}")
    print(f"Report: {report_path}")
    if not DRY_RUN:
        print(f"Manifest: {manifest_path}")
        if archive_path:
            print(f"ZIP: {archive_path}")
    else:
        print("Dry run only. Set DRY_RUN = False to copy files and create manifest/zip.")


if __name__ == "__main__":
    main()
