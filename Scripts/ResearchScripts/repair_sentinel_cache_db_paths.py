from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select


DRY_RUN = True
LIMIT_DOWNLOADS: int | None = None

# Leave empty for all selected downloads, or set prefixes for a targeted repair.
SAMPLE_ID_PREFIXES: tuple[str, ...] = ()
DOWNLOAD_ID_PREFIXES: tuple[str, ...] = ()

REPAIR_RASTER_PATHS = True
DERIVE_PREVIEWS_FROM_RASTER = True
UPSERT_PREVIEW_ROWS_FROM_EXISTING_FILES = True
SYNC_QUALITY_METRICS = True


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOL_ROOT = PROJECT_ROOT / "Scripts" / "ResearchTool"
sys.path.insert(0, str(TOOL_ROOT))

from backend.config import (  # noqa: E402
    FOREST_DERIVED_CACHE_DIR,
    FOREST_SENTINEL_RASTER_CACHE_DIR,
    ensure_directories,
)
from backend.forest.database import SessionLocal, init_forest_database  # noqa: E402
from backend.forest.models import DerivedPreview, SentinelDownload  # noqa: E402
from backend.forest.sentinel_service import (  # noqa: E402
    S2_BANDS,
    SENTINEL_VIEWS,
    cloud_metrics,
    derive_previews,
    read_json,
    safe_name,
    stable_id,
)
from backend.forest.storage_paths import resolve_storage_path, to_storage_path  # noqa: E402


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


def index_rasters() -> dict[str, list[Path]]:
    indexed: dict[str, list[Path]] = defaultdict(list)
    if not FOREST_SENTINEL_RASTER_CACHE_DIR.exists():
        return indexed
    for path in FOREST_SENTINEL_RASTER_CACHE_DIR.rglob("*.tif"):
        download_id = path.name.split("_", 1)[0]
        if download_id:
            indexed[download_id].append(path)
    return indexed


def expected_raster_path(download: SentinelDownload) -> Path:
    scene_name = safe_name(download.scene_id)
    return FOREST_SENTINEL_RASTER_CACHE_DIR / download.sample_id / f"{download.download_id}_{scene_name}.tif"


def find_raster(download: SentinelDownload, indexed: dict[str, list[Path]]) -> Path | None:
    candidates = [
        resolve_storage_path(download.local_path),
        expected_raster_path(download),
        *indexed.get(download.download_id, []),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def bbox_for_download(download: SentinelDownload, tif_path: Path | None) -> list[float] | None:
    metadata = decode_json(download.metadata_json, {})
    sidecar = read_json(tif_path.with_suffix(".json")) if tif_path is not None else None
    if sidecar:
        metadata = {**metadata, **sidecar}
    bbox = metadata.get("bbox") or decode_json(download.bbox_json, [])
    if isinstance(bbox, list) and len(bbox) == 4:
        return [float(value) for value in bbox]
    return None


def preview_dir(download: SentinelDownload) -> Path:
    return FOREST_DERIVED_CACHE_DIR / "sentinel" / download.sample_id / download.download_id


def preview_file_status(download: SentinelDownload) -> dict[str, Any]:
    existing_files = []
    missing_files = []
    for view in SENTINEL_VIEWS:
        png_path = preview_dir(download) / f"{view}.png"
        json_path = preview_dir(download) / f"{view}.json"
        if png_path.exists() and json_path.exists():
            existing_files.append(view)
        else:
            missing_files.append(view)
    return {
        "existing_preview_files": existing_files,
        "missing_preview_files": missing_files,
    }


def db_preview_views(db, download: SentinelDownload) -> set[str]:
    return set(
        db.scalars(
            select(DerivedPreview.view).where(DerivedPreview.download_id == download.download_id)
        )
    )


def upsert_preview_rows_from_files(db, download: SentinelDownload) -> Counter:
    stats = Counter()
    if not UPSERT_PREVIEW_ROWS_FROM_EXISTING_FILES:
        return stats

    for view in SENTINEL_VIEWS:
        png_path = preview_dir(download) / f"{view}.png"
        json_path = preview_dir(download) / f"{view}.json"
        if not png_path.exists() or not json_path.exists():
            continue

        preview_id = stable_id({"download_id": download.download_id, "view": view})
        png_storage = to_storage_path(png_path)
        json_storage = to_storage_path(json_path)
        existing = db.get(DerivedPreview, preview_id)
        if existing is None:
            stats["preview_row_created"] += 1
            if not DRY_RUN:
                db.add(
                    DerivedPreview(
                        preview_id=preview_id,
                        download_id=download.download_id,
                        view=view,
                        png_path=png_storage,
                        metadata_path=json_storage,
                    )
                )
            continue

        if existing.png_path != png_storage or existing.metadata_path != json_storage:
            stats["preview_row_updated"] += 1
            if not DRY_RUN:
                existing.png_path = png_storage
                existing.metadata_path = json_storage
        else:
            stats["preview_row_ok"] += 1
    return stats


def selected_downloads(db) -> list[SentinelDownload]:
    downloads = list(
        db.scalars(
            select(SentinelDownload).order_by(
                SentinelDownload.sample_id,
                SentinelDownload.period,
                SentinelDownload.scene_id,
                SentinelDownload.download_id,
            )
        )
    )
    if SAMPLE_ID_PREFIXES:
        downloads = [
            download
            for download in downloads
            if any(download.sample_id.startswith(prefix) for prefix in SAMPLE_ID_PREFIXES)
        ]
    if DOWNLOAD_ID_PREFIXES:
        downloads = [
            download
            for download in downloads
            if any(download.download_id.startswith(prefix) for prefix in DOWNLOAD_ID_PREFIXES)
        ]
    if LIMIT_DOWNLOADS is not None:
        downloads = downloads[:LIMIT_DOWNLOADS]
    return downloads


def repair_download(db, download: SentinelDownload, indexed: dict[str, list[Path]]) -> dict[str, Any]:
    tif_path = find_raster(download, indexed)
    bbox = bbox_for_download(download, tif_path)
    previews_before = preview_file_status(download)
    db_views_before = db_preview_views(db, download)
    old_local_path = download.local_path
    storage_tif_path = to_storage_path(tif_path) if tif_path is not None else ""
    actions = Counter()
    errors = []

    if tif_path is None:
        actions["missing_raster"] += 1
    elif REPAIR_RASTER_PATHS and download.local_path != storage_tif_path:
        actions["raster_path_updated"] += 1
        if not DRY_RUN:
            download.local_path = storage_tif_path
    elif tif_path is not None:
        actions["raster_path_ok"] += 1

    if tif_path is not None and bbox is None:
        actions["missing_bbox"] += 1
        errors.append("Cannot derive previews without bbox.")

    if tif_path is not None and bbox is not None and DERIVE_PREVIEWS_FROM_RASTER:
        missing_files = previews_before["missing_preview_files"]
        missing_db_views = [view for view in SENTINEL_VIEWS if view not in db_views_before]
        if missing_files or missing_db_views:
            actions["derive_previews"] += 1
            if not DRY_RUN:
                derive_previews(
                    db,
                    sample_id=download.sample_id,
                    download_id=download.download_id,
                    tif_path=tif_path,
                    bbox=bbox,
                )
        else:
            actions["preview_files_and_rows_ok"] += 1

    preview_row_stats = upsert_preview_rows_from_files(db, download)
    actions.update(preview_row_stats)

    if tif_path is not None and SYNC_QUALITY_METRICS:
        actions["sync_quality_metrics"] += 1
        if not DRY_RUN:
            metrics = cloud_metrics(tif_path)
            download.cloud_fraction = float(metrics["cloud_fraction"])
            download.shadow_fraction = float(metrics["shadow_fraction"])
            download.nodata_fraction = float(metrics["nodata_fraction"])
            download.dark_fraction = float(metrics["dark_fraction"])
            download.is_bad_cloud = bool(metrics["is_bad_cloud"])
            download.is_bad_quality = bool(metrics["is_bad_quality"])
            download.quality_flags_json = json.dumps(metrics["quality_flags"], ensure_ascii=False, sort_keys=True)

    previews_after = preview_file_status(download)
    db_views_after = db_preview_views(db, download) if DRY_RUN else set(SENTINEL_VIEWS)
    return {
        "download_id": download.download_id,
        "sample_id": download.sample_id,
        "period": download.period or "",
        "scene_id": download.scene_id,
        "old_local_path": old_local_path,
        "found_raster_path": display_path(tif_path) if tif_path is not None else "",
        "new_local_path": storage_tif_path or "",
        "bbox_ok": bbox is not None,
        "preview_files_before": ",".join(previews_before["existing_preview_files"]),
        "missing_preview_files_before": ",".join(previews_before["missing_preview_files"]),
        "db_preview_views_before": ",".join(sorted(db_views_before)),
        "preview_files_after": ",".join(previews_after["existing_preview_files"]),
        "missing_preview_files_after": ",".join(previews_after["missing_preview_files"]),
        "db_preview_views_after": ",".join(sorted(db_views_after)),
        "actions": ",".join(f"{key}:{value}" for key, value in sorted(actions.items())),
        "error": "; ".join(errors),
    }


def write_report(rows: list[dict[str, Any]]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / f"sentinel_cache_db_path_repair_{now_tag()}.csv"
    fieldnames = [
        "download_id",
        "sample_id",
        "period",
        "scene_id",
        "old_local_path",
        "found_raster_path",
        "new_local_path",
        "bbox_ok",
        "preview_files_before",
        "missing_preview_files_before",
        "db_preview_views_before",
        "preview_files_after",
        "missing_preview_files_after",
        "db_preview_views_after",
        "actions",
        "error",
    ]
    with report_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return report_path


def action_counts(rows: list[dict[str, Any]]) -> Counter:
    counts = Counter()
    for row in rows:
        for item in str(row.get("actions") or "").split(","):
            if not item:
                continue
            key, _, raw_count = item.partition(":")
            counts[key] += int(raw_count or "1")
    return counts


def main() -> None:
    ensure_directories()
    init_forest_database()
    indexed = index_rasters()
    rows: list[dict[str, Any]] = []

    with SessionLocal() as db:
        downloads = selected_downloads(db)
        db_download_ids = {download.download_id for download in downloads}
        orphan_raster_ids = sorted(set(indexed) - db_download_ids)
        for download in downloads:
            rows.append(repair_download(db, download, indexed))
        if DRY_RUN:
            db.rollback()
        else:
            db.commit()

    report_path = write_report(rows)
    print(f"DRY_RUN: {DRY_RUN}")
    print(f"LIMIT_DOWNLOADS: {LIMIT_DOWNLOADS}")
    print(f"Downloads checked: {len(rows)}")
    print(f"Indexed raster download_ids: {len(indexed)}")
    print(f"Orphan raster download_ids without DB download row: {len(orphan_raster_ids)}")
    print(f"Actions: {dict(sorted(action_counts(rows).items()))}")
    print(f"Rows with errors: {sum(1 for row in rows if row.get('error'))}")
    print(f"Report: {report_path}")
    if DRY_RUN:
        print("Dry run only. Set DRY_RUN = False to update DB paths and preview rows.")


if __name__ == "__main__":
    main()
