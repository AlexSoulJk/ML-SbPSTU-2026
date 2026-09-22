from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select


DRY_RUN = True
ENSURE_SCHEMA = False

# Set to None for all periods, or limit to a subset, for example ("PRE", "POST").
PERIODS: tuple[str, ...] | None = None

# Optional filters for small batches. Leave empty to process everything from DB.
SAMPLE_ID_PREFIXES: tuple[str, ...] = ()
DOWNLOAD_ID_PREFIXES: tuple[str, ...] = ()

# Set to an integer for a small test batch, or None for all missing/invalid cache rows.
LIMIT_DOWNLOADS = None

# Keep existing GeoTIFF files by default. Missing sidecar JSON and previews are still restored.
OVERWRITE_RASTERS = False
ENSURE_PREVIEWS = True
SYNC_EXISTING_RECORDS = True


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOL_ROOT = PROJECT_ROOT / "Scripts" / "ResearchTool"
sys.path.insert(0, str(TOOL_ROOT))

from backend.cdse_client import S2_RAW_EVALSCRIPT, process_request, run_process_api, tight_time_range  # noqa: E402
from backend.config import (  # noqa: E402
    DEFAULT_MAX_CLOUD,
    FOREST_DERIVED_CACHE_DIR,
    FOREST_SENTINEL_RASTER_CACHE_DIR,
    ensure_directories,
    load_dotenv,
)
from backend.forest.database import SessionLocal, init_forest_database  # noqa: E402
from backend.forest.models import DerivedPreview, SentinelDownload  # noqa: E402
from backend.forest.sentinel_service import (  # noqa: E402
    S2_BANDS,
    SENTINEL_VIEWS,
    cloud_metrics,
    derive_previews,
    pixel_size_for_bbox,
    read_json,
    safe_name,
    save_bytes,
    write_json,
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


def path_inside_project(path: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(PROJECT_ROOT.resolve(strict=False))
        return True
    except ValueError:
        return False


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


def target_raster_path(download: SentinelDownload) -> Path:
    stored = Path(download.local_path)
    resolved = resolve_storage_path(download.local_path)
    if stored.is_absolute() and not path_inside_project(resolved):
        scene_name = safe_name(download.scene_id)
        return FOREST_SENTINEL_RASTER_CACHE_DIR / download.sample_id / f"{download.download_id}_{scene_name}.tif"
    return resolved


def metadata_for_download(download: SentinelDownload) -> dict[str, Any]:
    metadata = decode_json(download.metadata_json, {})
    if not isinstance(metadata, dict):
        metadata = {}

    bbox = decode_json(download.bbox_json, metadata.get("bbox") or [])
    if bbox and "bbox" not in metadata:
        metadata["bbox"] = bbox

    bands = decode_json(download.bands_json, S2_BANDS)
    if bands and "bands" not in metadata:
        metadata["bands"] = bands

    metadata.setdefault("download_id", download.download_id)
    metadata.setdefault("sample_id", download.sample_id)
    metadata.setdefault("scene_id", download.scene_id)
    metadata.setdefault("period", download.period)
    return metadata


def expected_preview_dir(download: SentinelDownload) -> Path:
    return FOREST_DERIVED_CACHE_DIR / "sentinel" / download.sample_id / download.download_id


def preview_summary(db, download: SentinelDownload) -> dict[str, Any]:
    preview_dir = expected_preview_dir(download)
    db_views = set(
        db.scalars(
            select(DerivedPreview.view).where(DerivedPreview.download_id == download.download_id)
        )
    )
    existing_files = []
    missing_files = []
    for view in SENTINEL_VIEWS:
        png_path = preview_dir / f"{view}.png"
        json_path = preview_dir / f"{view}.json"
        if png_path.exists() and json_path.exists():
            existing_files.append(view)
        else:
            missing_files.append(view)

    missing_db_views = [view for view in SENTINEL_VIEWS if view not in db_views]
    return {
        "preview_dir": preview_dir,
        "existing_files": existing_files,
        "missing_files": missing_files,
        "db_views": sorted(db_views),
        "missing_db_views": missing_db_views,
        "needs_previews": bool(missing_files or missing_db_views),
    }


def can_download(metadata: dict[str, Any]) -> bool:
    item = metadata.get("item") or {}
    return bool(metadata.get("request") or (item.get("id") and item.get("datetime")))


def max_cloud_from_metadata(metadata: dict[str, Any]) -> float:
    try:
        return float(
            metadata["request"]["input"]["data"][0]["dataFilter"]["maxCloudCoverage"]
        )
    except (KeyError, TypeError, ValueError):
        return float(DEFAULT_MAX_CLOUD)


def build_process_request(metadata: dict[str, Any]) -> dict[str, Any]:
    request = metadata.get("request")
    if isinstance(request, dict):
        return request

    item = metadata.get("item") or {}
    if not item.get("datetime"):
        raise ValueError("Sentinel metadata has no item.datetime and no saved Process API request.")

    bbox = metadata.get("bbox") or []
    if len(bbox) != 4:
        raise ValueError("Sentinel metadata has no valid bbox.")

    width = int(metadata.get("width") or 0)
    height = int(metadata.get("height") or 0)
    if width <= 0 or height <= 0:
        width, height = pixel_size_for_bbox([float(value) for value in bbox])

    time_from, time_to = tight_time_range(str(item["datetime"]))
    return process_request(
        bbox=tuple(float(value) for value in bbox),
        time_from=time_from,
        time_to=time_to,
        evalscript=S2_RAW_EVALSCRIPT,
        width=width,
        height=height,
        output_type="image/tiff",
        max_cloud=max_cloud_from_metadata(metadata),
    )


def download_raster(download: SentinelDownload, tif_path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    request = build_process_request(metadata)
    content, content_type = run_process_api(request, accept="image/tiff")
    save_bytes(tif_path, content)

    metadata = {
        **metadata,
        "download_id": download.download_id,
        "sample_id": download.sample_id,
        "scene_id": download.scene_id,
        "period": download.period,
        "bbox": metadata.get("bbox") or decode_json(download.bbox_json, []),
        "bands": metadata.get("bands") or S2_BANDS,
        "path": to_storage_path(tif_path),
        "content_type": content_type,
        "bytes": len(content),
        "request": request,
    }
    if "width" not in metadata or "height" not in metadata:
        output = request.get("output") or {}
        metadata["width"] = output.get("width")
        metadata["height"] = output.get("height")
    return metadata


def write_sidecar(download: SentinelDownload, tif_path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        **metadata,
        "download_id": download.download_id,
        "sample_id": download.sample_id,
        "scene_id": download.scene_id,
        "period": download.period,
        "bbox": metadata.get("bbox") or decode_json(download.bbox_json, []),
        "bands": metadata.get("bands") or S2_BANDS,
        "path": to_storage_path(tif_path),
    }
    metadata_path = tif_path.with_suffix(".json")
    write_json(metadata_path, metadata)
    return metadata


def sync_download_record(
    db,
    download: SentinelDownload,
    tif_path: Path,
    metadata: dict[str, Any],
    *,
    ensure_previews: bool,
) -> None:
    bbox = metadata.get("bbox") or decode_json(download.bbox_json, [])
    if len(bbox) != 4:
        raise ValueError("Cannot derive previews: invalid bbox.")

    metrics = cloud_metrics(tif_path)
    if ensure_previews:
        derive_previews(
            db,
            sample_id=download.sample_id,
            download_id=download.download_id,
            tif_path=tif_path,
            bbox=[float(value) for value in bbox],
        )

    download.local_path = to_storage_path(tif_path) or str(tif_path)
    download.bbox_json = json.dumps([float(value) for value in bbox])
    download.bands_json = json.dumps(metadata.get("bands") or S2_BANDS)
    download.cloud_fraction = float(metrics["cloud_fraction"])
    download.shadow_fraction = float(metrics["shadow_fraction"])
    download.is_bad_cloud = bool(metrics["is_bad_cloud"])
    download.metadata_json = json.dumps(metadata, ensure_ascii=False, sort_keys=True)


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
    return downloads


def plan_row(db, download: SentinelDownload) -> dict[str, Any]:
    tif_path = target_raster_path(download)
    metadata_path = tif_path.with_suffix(".json")
    metadata = metadata_for_download(download)
    previews = preview_summary(db, download)
    storage_path = to_storage_path(tif_path) or str(tif_path)

    raster_exists = tif_path.exists()
    sidecar_exists = metadata_path.exists()
    needs_previews = ENSURE_PREVIEWS and previews["needs_previews"]
    needs_record_sync = SYNC_EXISTING_RECORDS and download.local_path != storage_path

    if (OVERWRITE_RASTERS or not raster_exists) and can_download(metadata):
        action = "redownload_raster" if raster_exists else "download_raster"
    elif not raster_exists:
        action = "cannot_download_missing_metadata"
    elif not sidecar_exists:
        action = "write_metadata"
    elif needs_previews:
        action = "derive_previews"
    elif needs_record_sync:
        action = "sync_record"
    else:
        action = "skip_cached"

    return {
        "download_id": download.download_id,
        "sample_id": download.sample_id,
        "period": download.period or "",
        "scene_id": download.scene_id,
        "raster_exists": raster_exists,
        "metadata_exists": sidecar_exists,
        "preview_files_existing": len(previews["existing_files"]),
        "preview_db_records": len(previews["db_views"]),
        "missing_preview_views": ",".join(previews["missing_files"]),
        "missing_preview_db_views": ",".join(previews["missing_db_views"]),
        "can_download": can_download(metadata),
        "local_path": download.local_path,
        "target_path": display_path(tif_path),
        "action": action if not DRY_RUN or action == "skip_cached" else f"would_{action}",
        "error": "",
    }


def process_download(db, download: SentinelDownload, row: dict[str, Any]) -> None:
    tif_path = target_raster_path(download)
    metadata_path = tif_path.with_suffix(".json")
    metadata = metadata_for_download(download)
    raster_exists = tif_path.exists()

    if OVERWRITE_RASTERS or not raster_exists:
        metadata = download_raster(download, tif_path, metadata)
        row["raster_exists"] = True

    if not metadata_path.exists():
        metadata = write_sidecar(download, tif_path, metadata)
        row["metadata_exists"] = True
    else:
        sidecar_metadata = read_json(metadata_path)
        if sidecar_metadata:
            metadata = {
                **metadata,
                **sidecar_metadata,
                "path": to_storage_path(tif_path),
            }

    sync_download_record(
        db,
        download,
        tif_path,
        metadata,
        ensure_previews=ENSURE_PREVIEWS,
    )
    db.commit()

    refreshed = plan_row(db, download)
    row.update(refreshed)
    row["action"] = "restored"


def write_report(rows: list[dict[str, Any]]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / f"sentinel_cache_prefetch_{now_tag()}.csv"
    fieldnames = [
        "download_id",
        "sample_id",
        "period",
        "scene_id",
        "raster_exists",
        "metadata_exists",
        "preview_files_existing",
        "preview_db_records",
        "missing_preview_views",
        "missing_preview_db_views",
        "can_download",
        "local_path",
        "target_path",
        "action",
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

    with SessionLocal() as db:
        downloads = selected_downloads(db)
        planned: list[tuple[SentinelDownload, dict[str, Any]]] = [
            (download, plan_row(db, download)) for download in downloads
        ]
        work_items = [
            item for item in planned
            if item[1]["action"] not in {"skip_cached", "cannot_download_missing_metadata"}
        ]
        if LIMIT_DOWNLOADS is not None:
            work_items = work_items[:LIMIT_DOWNLOADS]

        action_counts = Counter(row["action"] for _, row in planned)
        print(f"Sentinel downloads in DB: {len(planned)}")
        print(f"Work items: {len(work_items)}")
        print(f"DRY_RUN: {DRY_RUN}")
        print(f"Actions: {dict(sorted(action_counts.items()))}")

        if DRY_RUN:
            report_path = write_report([row for _, row in planned])
            print(f"Report: {report_path}")
            print("Dry run only. Set DRY_RUN = False to restore/download Sentinel cache.")
            return

        restored = 0
        failed = 0
        for download, row in work_items:
            try:
                process_download(db, download, row)
                restored += 1
                print(f"[restored] {download.period or 'n/a'} {download.download_id} {download.scene_id}")
            except Exception as exc:
                db.rollback()
                row["action"] = "failed"
                row["error"] = str(exc)[:500]
                failed += 1
                print(f"[failed] {download.download_id}: {exc}")

        report_path = write_report([row for _, row in planned])
        print(f"Restored: {restored}")
        print(f"Failed: {failed}")
        print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
