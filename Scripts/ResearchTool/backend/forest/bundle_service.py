from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sqlite3
import zipfile
from collections import Counter
from collections.abc import Callable
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import (
    FOREST_DATA_DIR,
    FOREST_IMPORTS_DIR,
    FOREST_SENTINEL_RASTER_CACHE_DIR,
    FOREST_STATE_DB_PATH,
    PROJECT_ROOT,
)
from .models import (
    DerivedPreview,
    ManualValidation,
    SampleEvent,
    SampleEventDownload,
    SentinelDownload,
    SentinelSceneReview,
)
from .sentinel_batch_service import DATA_OPERATIONS_OUTPUT_DIR, output_dir, write_rows_csv
from .sentinel_service import derive_previews
from .storage_paths import resolve_storage_path, to_storage_path


BUNDLE_PROFILES = {
    "geotiff_only": {
        "include_rasters": True,
        "include_raster_metadata": True,
        "include_previews": False,
    },
    "preview_only": {
        "include_rasters": False,
        "include_raster_metadata": False,
        "include_previews": True,
    },
    "full": {
        "include_rasters": True,
        "include_raster_metadata": True,
        "include_previews": True,
    },
}

ProgressCallback = Callable[[str, str, int, int], None]
CancelCallback = Callable[[], bool]
BUNDLE_BACKUP_DIR = FOREST_DATA_DIR / "backups"
EXPECTED_PREVIEW_VIEWS = {"rgb", "false_color", "ndvi", "nbr", "ndmi", "scl"}


def portable_path(path: str | Path) -> str:
    stored = to_storage_path(path)
    if stored is None:
        raise ValueError(f"Could not create a portable path for {path}")
    raw = Path(stored)
    if raw.is_absolute():
        try:
            raw = raw.resolve(strict=False).relative_to(PROJECT_ROOT.resolve(strict=False))
        except ValueError as exc:
            raise ValueError(f"Path is outside the project root: {path}") from exc
    return PurePosixPath(*raw.parts).as_posix()


def path_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def resolved_raster_path(download: SentinelDownload) -> tuple[Path, str | None]:
    path = resolve_storage_path(download.local_path)
    if path.exists():
        return path, None
    fallback = FOREST_SENTINEL_RASTER_CACHE_DIR / download.sample_id / path.name
    if fallback.exists():
        return fallback, "stored_path_mismatch"
    if not download.local_path:
        return fallback, "missing_path"
    if not path_inside(path, PROJECT_ROOT):
        return fallback, "path_outside_cache"
    return path, "missing_file"


def missing_geotiff_rows(db: Session) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    downloads = list(
        db.scalars(
            select(SentinelDownload).order_by(
                SentinelDownload.sample_id,
                SentinelDownload.scene_id,
                SentinelDownload.download_id,
            )
        )
    )
    for download in downloads:
        path, reason = resolved_raster_path(download)
        if reason is None or reason == "stored_path_mismatch":
            continue
        expected = FOREST_SENTINEL_RASTER_CACHE_DIR / download.sample_id / path.name
        rows.append(
            {
                "download_id": download.download_id,
                "sample_id": download.sample_id,
                "scene_id": download.scene_id,
                "period": download.period or "",
                "expected_relative_path": portable_path(expected),
                "reason": reason,
            }
        )
    return rows


def write_missing_geotiff_inventory(
    db: Session,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    if progress:
        progress("scanning", "Checking Sentinel GeoTIFF cache", 0, 1)
    rows = missing_geotiff_rows(db)
    artifact_dir = output_dir("missing_geotiffs")
    csv_path = artifact_dir / "missing_sentinel_geotiffs.csv"
    operations_path = artifact_dir / "operations.csv"
    errors_path = artifact_dir / "errors.csv"
    summary_path = artifact_dir / "summary.json"
    write_rows_csv(csv_path, rows)
    write_rows_csv(operations_path, rows)
    write_rows_csv(errors_path, [])
    summary = {
        "missing_geotiffs": len(rows),
        "reasons": dict(sorted(Counter(row["reason"] for row in rows).items())),
        "artifact_dir": str(artifact_dir),
        "csv_path": str(csv_path),
        "operations_csv": str(operations_path),
        "errors_csv": str(errors_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if progress:
        progress("done", f"Found {len(rows)} missing GeoTIFF(s)", 1, 1)
    return summary


def download_ids_from_inventory(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(f"Inventory CSV was not found: {path}")
    ids: list[str] = []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            download_id = str(row.get("download_id") or "").strip()
            rows.append(dict(row))
            if download_id and download_id not in seen:
                seen.add(download_id)
                ids.append(download_id)
    return ids, rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_bundle_file(
    rows: list[dict[str, Any]],
    *,
    kind: str,
    path: Path,
    download: SentinelDownload,
    preview: DerivedPreview | None = None,
) -> None:
    exists = path.exists()
    rows.append(
        {
            "kind": kind,
            "download_id": download.download_id,
            "sample_id": download.sample_id,
            "scene_id": download.scene_id,
            "period": download.period or "",
            "preview_id": preview.preview_id if preview else "",
            "view": preview.view if preview else "",
            "relative_path": portable_path(path),
            "exists": exists,
            "bytes": path.stat().st_size if exists else 0,
            "sha256": sha256_file(path) if exists else "",
            "action": "pending" if exists else "missing_source",
            "error": "",
        }
    )


def collect_bundle_files(
    db: Session,
    downloads: list[SentinelDownload],
    profile: str,
) -> list[dict[str, Any]]:
    if profile not in BUNDLE_PROFILES:
        raise ValueError(f"Unknown bundle profile: {profile}")
    options = BUNDLE_PROFILES[profile]
    rows: list[dict[str, Any]] = []
    for download in downloads:
        raster_path, _ = resolved_raster_path(download)
        if options["include_rasters"]:
            add_bundle_file(
                rows,
                kind="sentinel_raster_tif",
                path=raster_path,
                download=download,
            )
        if options["include_raster_metadata"]:
            add_bundle_file(
                rows,
                kind="sentinel_raster_metadata",
                path=raster_path.with_suffix(".json"),
                download=download,
            )
        if options["include_previews"]:
            previews = list(
                db.scalars(
                    select(DerivedPreview)
                    .where(DerivedPreview.download_id == download.download_id)
                    .order_by(DerivedPreview.view)
                )
            )
            for preview in previews:
                add_bundle_file(
                    rows,
                    kind="sentinel_preview_png",
                    path=resolve_storage_path(preview.png_path),
                    download=download,
                    preview=preview,
                )
                add_bundle_file(
                    rows,
                    kind="sentinel_preview_metadata",
                    path=resolve_storage_path(preview.metadata_path),
                    download=download,
                    preview=preview,
                )
    return rows


def selected_downloads(
    db: Session,
    download_ids: list[str],
    *,
    exclude_reviewed_bad: bool,
) -> tuple[list[SentinelDownload], list[str]]:
    unique_ids = list(dict.fromkeys(download_ids))
    if not unique_ids:
        return [], []
    rows = list(
        db.scalars(
            select(SentinelDownload).where(SentinelDownload.download_id.in_(unique_ids))
        )
    )
    by_id = {row.download_id: row for row in rows}
    unknown = [download_id for download_id in unique_ids if download_id not in by_id]
    ordered = [by_id[download_id] for download_id in unique_ids if download_id in by_id]
    if exclude_reviewed_bad and ordered:
        excluded_ids = set(
            db.scalars(
                select(SentinelSceneReview.download_id).where(
                    SentinelSceneReview.download_id.in_([row.download_id for row in ordered]),
                    SentinelSceneReview.is_excluded.is_(True),
                )
            )
        )
        ordered = [row for row in ordered if row.download_id not in excluded_ids]
    return ordered, unknown


def event_link_snapshot(db: Session, downloads: list[SentinelDownload]) -> list[dict[str, Any]]:
    ids = [download.download_id for download in downloads]
    if not ids:
        return []
    links = list(
        db.scalars(
            select(SampleEventDownload).where(SampleEventDownload.download_id.in_(ids))
        )
    )
    return [
        {
            "id": link.id,
            "event_id": link.event_id,
            "download_id": link.download_id,
            "period": link.period,
            "rank": link.rank,
        }
        for link in links
    ]


def event_snapshot(db: Session, downloads: list[SentinelDownload]) -> list[dict[str, Any]]:
    sample_ids = sorted({download.sample_id for download in downloads})
    if not sample_ids:
        return []
    events = list(
        db.scalars(
            select(SampleEvent)
            .where(SampleEvent.sample_id.in_(sample_ids))
            .order_by(SampleEvent.sample_id, SampleEvent.event_year, SampleEvent.event_id)
        )
    )
    return [
        {
            "event_id": event.event_id,
            "sample_id": event.sample_id,
            "event_year": event.event_year,
            "source": event.source,
            "loss_area_ha": event.loss_area_ha,
            "loss_share": event.loss_share,
            "is_dominant": bool(event.is_dominant),
        }
        for event in events
    ]


def bundle_plan(
    db: Session,
    download_ids: list[str],
    *,
    profile: str = "geotiff_only",
    exclude_reviewed_bad: bool = True,
    source: str = "selection",
) -> dict[str, Any]:
    downloads, unknown = selected_downloads(
        db,
        download_ids,
        exclude_reviewed_bad=exclude_reviewed_bad,
    )
    files = collect_bundle_files(db, downloads, profile)
    missing = [row for row in files if not row["exists"]]
    event_links = event_link_snapshot(db, downloads)
    events = event_snapshot(db, downloads)
    return {
        "source": source,
        "profile": profile,
        "exclude_reviewed_bad": exclude_reviewed_bad,
        "download_ids": [download.download_id for download in downloads],
        "unknown_download_ids": unknown,
        "summary": {
            "unique_downloads": len(downloads),
            "unique_files": len(files),
            "total_bytes": sum(int(row["bytes"]) for row in files if row["exists"]),
            "missing_source_files": len(missing),
            "unknown_download_ids": len(unknown),
            "previews_to_rebuild": len(downloads) if profile == "geotiff_only" else 0,
            "event_links": len(event_links),
        },
        "events": events,
        "event_links": event_links,
        "files": files,
    }


def export_bundle(
    db: Session,
    download_ids: list[str],
    *,
    profile: str = "geotiff_only",
    exclude_reviewed_bad: bool = True,
    create_zip: bool = True,
    source: str = "selection",
    progress: ProgressCallback | None = None,
    should_cancel: CancelCallback | None = None,
) -> dict[str, Any]:
    if progress:
        progress("planning", "Resolving files and checksums", 0, len(download_ids))
    plan = bundle_plan(
        db,
        download_ids,
        profile=profile,
        exclude_reviewed_bad=exclude_reviewed_bad,
        source=source,
    )
    downloads, _ = selected_downloads(
        db,
        plan["download_ids"],
        exclude_reviewed_bad=False,
    )
    bundle_dir = output_dir("sentinel_bundle")
    copied = 0
    failed = 0
    for index, row in enumerate(plan["files"], start=1):
        if should_cancel and should_cancel():
            row["action"] = "cancelled"
            continue
        if not row["exists"]:
            continue
        source_path = resolve_storage_path(row["relative_path"])
        target = bundle_dir / "files" / row["relative_path"]
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target)
            row["action"] = "copied"
            copied += 1
        except Exception as exc:
            row["action"] = "failed"
            row["error"] = str(exc)[:500]
            failed += 1
        if progress:
            progress("copying", f"{index}/{len(plan['files'])} {row['kind']}", index, len(plan["files"]))

    manifest = {
        "manifest_version": 2,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "bundle_name": bundle_dir.name,
        "source": source,
        "profile": profile,
        "options": {
            "exclude_reviewed_bad": exclude_reviewed_bad,
            "create_zip": create_zip,
        },
        "downloads": [
            {
                "download_id": download.download_id,
                "sample_id": download.sample_id,
                "scene_id": download.scene_id,
                "period": download.period,
                "raster_path": portable_path(resolved_raster_path(download)[0]),
            }
            for download in downloads
        ],
        "events": event_snapshot(db, downloads),
        "event_links": event_link_snapshot(db, downloads),
        "files": plan["files"],
    }
    manifest_path = bundle_dir / "manifest.json"
    files_path = bundle_dir / "files.csv"
    operations_path = bundle_dir / "operations.csv"
    summary_path = bundle_dir / "summary.json"
    errors_path = bundle_dir / "errors.csv"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_rows_csv(files_path, plan["files"])
    write_rows_csv(operations_path, plan["files"])
    write_rows_csv(errors_path, [row for row in plan["files"] if row["action"] == "failed"])
    archive_path = bundle_dir.with_suffix(".zip") if create_zip else None
    summary = {
        **plan["summary"],
        "status": "cancelled" if should_cancel and should_cancel() else "completed",
        "options": {
            "profile": profile,
            "exclude_reviewed_bad": exclude_reviewed_bad,
            "create_zip": create_zip,
            "source": source,
        },
        "copied": copied,
        "failed": failed,
        "bundle_dir": str(bundle_dir),
        "manifest_path": str(manifest_path),
        "zip_path": str(archive_path) if archive_path else None,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if create_zip and not (should_cancel and should_cancel()):
        if progress:
            progress("archiving", "Creating ZIP archive", len(plan["files"]), len(plan["files"]))
        archive_path = Path(shutil.make_archive(str(bundle_dir), "zip", root_dir=bundle_dir))
    elif create_zip:
        archive_path = None
        summary["zip_path"] = None
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def bundle_candidates() -> list[dict[str, Any]]:
    roots = [FOREST_IMPORTS_DIR, DATA_OPERATIONS_OUTPUT_DIR]
    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        candidates.extend(path for path in root.iterdir() if path.suffix.lower() == ".zip")
        candidates.extend(path for path in root.iterdir() if path.is_dir() and (path / "manifest.json").exists())
    return [
        {
            "path": str(path),
            "name": path.name,
            "kind": "zip" if path.is_file() else "directory",
            "bytes": path.stat().st_size if path.is_file() else None,
            "updated_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
        }
        for path in sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True)
    ]


def safe_bundle_path(raw_path: str) -> Path:
    path = Path(raw_path).resolve(strict=False)
    research_outputs = PROJECT_ROOT / "Scripts" / "ResearchScripts" / "Outputs"
    allowed_roots = [
        FOREST_IMPORTS_DIR.resolve(strict=False),
        DATA_OPERATIONS_OUTPUT_DIR.resolve(strict=False),
        research_outputs.resolve(strict=False),
    ]
    if not any(path_inside(path, root) or path == root for root in allowed_roots):
        raise ValueError("Bundle path is outside the allowed incoming/output directories.")
    if not path.exists():
        raise FileNotFoundError(f"Bundle was not found: {path}")
    return path


def validate_relative_path(raw: str) -> str:
    value = PurePosixPath(raw)
    if value.is_absolute() or ".." in value.parts or not value.parts:
        raise ValueError(f"Unsafe bundle relative path: {raw}")
    destination = PROJECT_ROOT.joinpath(*value.parts)
    if not path_inside(destination, PROJECT_ROOT):
        raise ValueError(f"Bundle file points outside the project: {raw}")
    return value.as_posix()


def read_bundle_manifest(path: Path) -> dict[str, Any]:
    if path.is_file():
        with zipfile.ZipFile(path) as archive:
            for item in archive.infolist():
                validate_relative_path(item.filename)
            try:
                return json.loads(archive.read("manifest.json").decode("utf-8"))
            except KeyError as exc:
                raise FileNotFoundError("manifest.json is missing from the bundle ZIP.") from exc
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json is missing: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def bundle_member_exists(path: Path, relative_path: str) -> bool:
    member = f"files/{validate_relative_path(relative_path)}"
    if path.is_file():
        with zipfile.ZipFile(path) as archive:
            return member in {item.filename for item in archive.infolist() if not item.is_dir()}
    return (path / member).exists()


def bundle_member_sha256(path: Path, relative_path: str) -> str:
    member = f"files/{validate_relative_path(relative_path)}"
    digest = hashlib.sha256()
    if path.is_file():
        with zipfile.ZipFile(path) as archive, archive.open(member) as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    with (path / member).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_bundle_plan(db: Session, raw_path: str) -> dict[str, Any]:
    path = safe_bundle_path(raw_path)
    manifest = read_bundle_manifest(path)
    download_rows = {row.get("download_id"): row for row in manifest.get("downloads", [])}
    identity_conflicts: list[dict[str, Any]] = []
    missing_db: list[str] = []
    for download_id, incoming in download_rows.items():
        record = db.get(SentinelDownload, download_id)
        if record is None:
            missing_db.append(str(download_id))
            continue
        if record.sample_id != incoming.get("sample_id") or record.scene_id != incoming.get("scene_id"):
            identity_conflicts.append(
                {
                    "download_id": download_id,
                    "local_sample_id": record.sample_id,
                    "incoming_sample_id": incoming.get("sample_id"),
                    "local_scene_id": record.scene_id,
                    "incoming_scene_id": incoming.get("scene_id"),
                }
            )

    file_rows: list[dict[str, Any]] = []
    for row in manifest.get("files", []):
        relative = validate_relative_path(str(row.get("relative_path") or ""))
        target = PROJECT_ROOT.joinpath(*PurePosixPath(relative).parts)
        source_exists = bundle_member_exists(path, relative)
        action = "missing_bundle_file" if not source_exists else "skip_exists" if target.exists() else "copy"
        expected_sha256 = str(row.get("sha256") or "").strip().lower()
        actual_sha256 = bundle_member_sha256(path, relative) if source_exists and expected_sha256 else ""
        if source_exists and expected_sha256 and actual_sha256 != expected_sha256:
            action = "checksum_mismatch"
        if row.get("download_id") in missing_db:
            action = "database_not_current"
        file_rows.append(
            {
                **row,
                "relative_path": relative,
                "source_exists": source_exists,
                "target_exists": target.exists(),
                "actual_sha256": actual_sha256,
                "action": action,
            }
        )

    return {
        "bundle_path": str(path),
        "manifest_version": manifest.get("manifest_version", 1),
        "profile": manifest.get("profile") or manifest.get("options", {}).get("bundle_profile"),
        "summary": {
            "files": len(file_rows),
            "copy": sum(row["action"] == "copy" for row in file_rows),
            "skip_exists": sum(row["action"] == "skip_exists" for row in file_rows),
            "missing_bundle_file": sum(row["action"] == "missing_bundle_file" for row in file_rows),
            "checksum_mismatches": sum(row["action"] == "checksum_mismatch" for row in file_rows),
            "missing_db_downloads": len(missing_db),
            "identity_conflicts": len(identity_conflicts),
            "bytes_to_copy": sum(
                int(row.get("bytes") or 0) for row in file_rows if row["action"] == "copy"
            ),
            "previews_to_rebuild": len(
                {
                    row.get("download_id")
                    for row in file_rows
                    if row.get("kind") == "sentinel_raster_tif"
                    and row["action"] in {"copy", "skip_exists"}
                }
            ),
        },
        "missing_download_ids": missing_db,
        "identity_conflicts": identity_conflicts,
        "files": file_rows,
        "can_apply": (
            not missing_db
            and not identity_conflicts
            and not any(row["action"] in {"missing_bundle_file", "checksum_mismatch"} for row in file_rows)
        ),
    }


def copy_stream(source: BinaryIO, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        shutil.copyfileobj(source, handle)


def copy_bundle_member(bundle_path: Path, relative_path: str, target: Path) -> None:
    member = f"files/{validate_relative_path(relative_path)}"
    if bundle_path.is_file():
        with zipfile.ZipFile(bundle_path) as archive, archive.open(member) as source:
            copy_stream(source, target)
        return
    source_path = bundle_path / member
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target)


def verify_bundle_import(db: Session, download_ids: list[str]) -> dict[str, Any]:
    unique_ids = sorted(set(download_id for download_id in download_ids if download_id))
    downloads = (
        list(
            db.scalars(
                select(SentinelDownload).where(SentinelDownload.download_id.in_(unique_ids))
            )
        )
        if unique_ids
        else []
    )
    sample_ids = sorted({download.sample_id for download in downloads})
    manual_sample_ids = (
        set(
            db.scalars(
                select(ManualValidation.sample_id).where(
                    ManualValidation.sample_id.in_(sample_ids)
                )
            )
        )
        if sample_ids
        else set()
    )
    previews = (
        list(
            db.scalars(
                select(DerivedPreview).where(DerivedPreview.download_id.in_(unique_ids))
            )
        )
        if unique_ids
        else []
    )
    preview_download_ids = {preview.download_id for preview in previews}
    missing_preview_rows = [
        preview for preview in previews if not resolve_storage_path(preview.png_path).exists()
    ]
    missing_preview_download_ids = sorted(
        {preview.download_id for preview in missing_preview_rows}
    )
    downloads_without_preview_rows = sorted(set(unique_ids) - preview_download_ids)
    preview_views_by_download: dict[str, set[str]] = {}
    for preview in previews:
        preview_views_by_download.setdefault(preview.download_id, set()).add(preview.view)
    incomplete_preview_download_ids = sorted(
        download_id
        for download_id in unique_ids
        if preview_views_by_download.get(download_id, set()) != EXPECTED_PREVIEW_VIEWS
    )
    return {
        "bundle_downloads": len(unique_ids),
        "database_downloads": len(downloads),
        "samples": len(sample_ids),
        "samples_without_manual_validation": len(set(sample_ids) - manual_sample_ids),
        "raster_files_ready": sum(
            resolve_storage_path(download.local_path).exists() for download in downloads
        ),
        "preview_rows": len(previews),
        "preview_files_ready": sum(
            resolve_storage_path(preview.png_path).exists() for preview in previews
        ),
        "missing_preview_files": len(missing_preview_rows),
        "downloads_without_preview_rows": len(downloads_without_preview_rows),
        "incomplete_preview_downloads": len(incomplete_preview_download_ids),
        "missing_preview_download_ids": missing_preview_download_ids[:100],
        "downloads_without_preview_row_ids": downloads_without_preview_rows[:100],
        "incomplete_preview_download_ids": incomplete_preview_download_ids[:100],
        "sample_ids": sample_ids,
    }


def import_bundle(
    db: Session,
    raw_path: str,
    *,
    overwrite_existing: bool = False,
    rebuild_previews: bool = True,
    progress: ProgressCallback | None = None,
    should_cancel: CancelCallback | None = None,
) -> dict[str, Any]:
    if progress:
        progress("validating", "Validating bundle manifest and checksums", 0, 1)
    plan = import_bundle_plan(db, raw_path)
    if not plan["can_apply"]:
        raise ValueError("Bundle cannot be imported until database_not_current conflicts are resolved.")
    bundle_path = Path(plan["bundle_path"])
    manifest = read_bundle_manifest(bundle_path)
    downloads = {row.get("download_id"): row for row in manifest.get("downloads", [])}
    backup_path: Path | None = None

    def ensure_database_backup() -> None:
        nonlocal backup_path
        if backup_path is not None or not FOREST_STATE_DB_PATH.exists():
            return
        BUNDLE_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        backup_path = BUNDLE_BACKUP_DIR / (
            f"{FOREST_STATE_DB_PATH.stem}_before_bundle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sqlite"
        )
        with sqlite3.connect(FOREST_STATE_DB_PATH) as source_db, sqlite3.connect(backup_path) as backup_db:
            source_db.backup(backup_db)
    operation_rows: list[dict[str, Any]] = []
    total = len(plan["files"])
    copied = 0
    failed = 0

    for index, row in enumerate(plan["files"], start=1):
        if should_cancel and should_cancel():
            break
        target = PROJECT_ROOT.joinpath(*PurePosixPath(row["relative_path"]).parts)
        action = row["action"]
        try:
            if action == "copy" or (action == "skip_exists" and overwrite_existing):
                copy_bundle_member(bundle_path, row["relative_path"], target)
                action = "copied"
                copied += 1
            elif action == "skip_exists":
                action = "skip_exists"
        except Exception as exc:
            action = "failed"
            row = {**row, "error": str(exc)[:500]}
            failed += 1
        operation_rows.append({**row, "action": action})
        if progress:
            progress("copying", f"{index}/{total} {row.get('kind')}", index, total)

    available = {
        row["relative_path"]
        for row in operation_rows
        if (PROJECT_ROOT.joinpath(*PurePosixPath(row["relative_path"]).parts)).exists()
    }
    if progress:
        progress("updating_paths", "Updating portable cache paths", total, total)
    for download_id, incoming in downloads.items():
        record = db.get(SentinelDownload, download_id)
        if record is None:
            continue
        raster_path = incoming.get("raster_path")
        if raster_path in available:
            portable_raster_path = validate_relative_path(raster_path)
            if record.local_path != portable_raster_path:
                ensure_database_backup()
                record.local_path = portable_raster_path
    for row in operation_rows:
        preview_id = row.get("preview_id")
        if not preview_id or row["relative_path"] not in available:
            continue
        preview = db.get(DerivedPreview, preview_id)
        if preview is None:
            continue
        if row.get("kind") == "sentinel_preview_png":
            if preview.png_path != row["relative_path"]:
                ensure_database_backup()
                preview.png_path = row["relative_path"]
        elif row.get("kind") == "sentinel_preview_metadata":
            if preview.metadata_path != row["relative_path"]:
                ensure_database_backup()
                preview.metadata_path = row["relative_path"]
    db.commit()

    rebuilt = 0
    if rebuild_previews:
        raster_download_ids = {
            row.get("download_id")
            for row in operation_rows
            if row.get("kind") == "sentinel_raster_tif"
            and row["action"] in {"copied", "skip_exists"}
        }
        raster_ids = sorted(download_id for download_id in raster_download_ids if download_id)
        for rebuild_index, download_id in enumerate(raster_ids, start=1):
            if should_cancel and should_cancel():
                break
            download = db.get(SentinelDownload, download_id)
            if download is None:
                continue
            tif_path = resolve_storage_path(download.local_path)
            if not tif_path.exists():
                continue
            bbox = json.loads(download.bbox_json)
            ensure_database_backup()
            existing_previews = list(
                db.scalars(
                    select(DerivedPreview).where(DerivedPreview.download_id == download.download_id)
                )
            )
            for preview in existing_previews:
                for stored_path in (preview.png_path, preview.metadata_path):
                    preview_path = resolve_storage_path(stored_path)
                    if preview_path.exists() and path_inside(preview_path, PROJECT_ROOT):
                        preview_path.unlink()
            derive_previews(
                db,
                sample_id=download.sample_id,
                download_id=download.download_id,
                tif_path=tif_path,
                bbox=bbox,
            )
            rebuilt += 1
            if progress:
                progress(
                    "rebuilding_previews",
                    f"{rebuild_index}/{len(raster_ids)} {download.download_id[:8]}",
                    rebuild_index,
                    len(raster_ids),
                )
        db.commit()

    verification = verify_bundle_import(db, list(downloads))
    artifact_dir = output_dir("sentinel_bundle_import")
    operations_path = artifact_dir / "operations.csv"
    errors_path = artifact_dir / "errors.csv"
    summary_path = artifact_dir / "summary.json"
    write_rows_csv(operations_path, operation_rows)
    write_rows_csv(errors_path, [row for row in operation_rows if row["action"] == "failed"])
    summary = {
        "status": "cancelled" if should_cancel and should_cancel() else "completed",
        "bundle_path": str(bundle_path),
        "options": {
            "overwrite_existing": overwrite_existing,
            "rebuild_previews": rebuild_previews,
        },
        "files_processed": len(operation_rows),
        "copied": copied,
        "failed": failed,
        "previews_rebuilt": rebuilt,
        "verification": verification,
        "backup_path": str(backup_path) if backup_path else None,
        "operations_csv": str(operations_path),
        "errors_csv": str(errors_path),
        "artifact_dir": str(artifact_dir),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
