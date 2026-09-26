from __future__ import annotations

import csv
import json
import shutil
import sys
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


DRY_RUN = True

# Set explicitly if needed. If None, the newest sentinel_cache_bundle_* folder/zip in Outputs is used.
BUNDLE_PATH: Path | None = None

OVERWRITE_EXISTING = False
SYNC_DB_RECORDS = True


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "Scripts"
RESEARCH_TOOL_ROOT = SCRIPTS_ROOT / "ResearchTool"
sys.path.insert(0, str(RESEARCH_TOOL_ROOT))

from backend.forest.database import SessionLocal, init_forest_database  # noqa: E402
from backend.forest.models import DerivedPreview, SentinelDownload  # noqa: E402


SCRIPT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_ROOT / "Outputs"


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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


def latest_bundle_path() -> Path:
    candidates = []
    if OUTPUT_DIR.exists():
        candidates.extend(
            path
            for path in OUTPUT_DIR.glob("sentinel_cache_bundle_*")
            if path.is_dir() and (path / "manifest.json").exists()
        )
        candidates.extend(path for path in OUTPUT_DIR.glob("sentinel_cache_bundle_*.zip") if path.is_file())
    if not candidates:
        raise FileNotFoundError(
            "No sentinel_cache_bundle_* folder or zip was found in Scripts/ResearchScripts/Outputs."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def is_zip_bundle(path: Path) -> bool:
    return path.suffix.lower() == ".zip"


def zip_member_name(*parts: str) -> str:
    return "/".join(part.strip("/\\") for part in parts if part)


def validate_zip_archive(archive: zipfile.ZipFile) -> None:
    for member in archive.infolist():
        member_path = Path(member.filename)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise RuntimeError(f"Unsafe zip member path: {member.filename}")


def read_manifest_from_path(path: Path, archive: zipfile.ZipFile | None = None) -> dict[str, Any]:
    if archive is not None:
        try:
            return json.loads(archive.read("manifest.json").decode("utf-8"))
        except KeyError as exc:
            raise FileNotFoundError(f"Manifest was not found in zip: {path}") from exc

    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest was not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def project_destination(relative_path: str) -> Path:
    destination = PROJECT_ROOT / relative_path
    project_root = PROJECT_ROOT.resolve(strict=False)
    try:
        destination.resolve(strict=False).relative_to(project_root)
    except ValueError as exc:
        raise RuntimeError(f"Bundle file points outside project: {relative_path}") from exc
    return destination


def copy_bundle_file_from_dir(bundle_dir: Path, row: dict[str, Any]) -> dict[str, Any]:
    relative_path = str(row.get("relative_path") or "")
    source = bundle_dir / "files" / relative_path
    target = project_destination(relative_path)
    result = {
        **row,
        "source_in_bundle": display_path(source),
        "target_path": display_path(target),
        "source_exists": source.exists(),
        "target_exists_before": target.exists(),
        "action": "",
        "error": "",
    }

    if not source.exists():
        if row.get("action") == "missing_source":
            result["action"] = "missing_source_at_export"
        else:
            result["action"] = "missing_bundle_file"
        return result
    if target.exists() and not OVERWRITE_EXISTING:
        result["action"] = "skip_exists"
        return result
    if DRY_RUN:
        result["action"] = "would_copy"
        return result

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    result["action"] = "copied"
    return result


def copy_bundle_file_from_zip(
    bundle_path: Path,
    archive: zipfile.ZipFile,
    zip_entries: set[str],
    row: dict[str, Any],
) -> dict[str, Any]:
    relative_path = str(row.get("relative_path") or "")
    member_name = zip_member_name("files", relative_path)
    target = project_destination(relative_path)
    source_exists = member_name in zip_entries
    result = {
        **row,
        "source_in_bundle": f"{bundle_path.name}!/{member_name}",
        "target_path": display_path(target),
        "source_exists": source_exists,
        "target_exists_before": target.exists(),
        "action": "",
        "error": "",
    }

    if not source_exists:
        if row.get("action") == "missing_source":
            result["action"] = "missing_source_at_export"
        else:
            result["action"] = "missing_bundle_file"
        return result
    if target.exists() and not OVERWRITE_EXISTING:
        result["action"] = "skip_exists"
        return result
    if DRY_RUN:
        result["action"] = "would_copy"
        return result

    target.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(member_name) as source_handle, target.open("wb") as target_handle:
        shutil.copyfileobj(source_handle, target_handle)
    result["action"] = "copied"
    return result


def load_bundle_rows(bundle_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if is_zip_bundle(bundle_path):
        with zipfile.ZipFile(bundle_path) as archive:
            validate_zip_archive(archive)
            zip_entries = {item.filename for item in archive.infolist() if not item.is_dir()}
            manifest = read_manifest_from_path(bundle_path, archive)
            rows = [
                copy_bundle_file_from_zip(bundle_path, archive, zip_entries, row)
                for row in manifest.get("files", [])
            ]
        return manifest, rows

    manifest = read_manifest_from_path(bundle_path)
    rows = [copy_bundle_file_from_dir(bundle_path, row) for row in manifest.get("files", [])]
    return manifest, rows


def sync_db_records(manifest: dict[str, Any]) -> dict[str, int]:
    stats = Counter()
    if not SYNC_DB_RECORDS:
        return {"disabled": 1}
    if DRY_RUN:
        return {"would_sync": 1}

    available_paths = {
        row["relative_path"]
        for row in manifest.get("files", [])
        if (PROJECT_ROOT / str(row.get("relative_path") or "")).exists()
    }

    with SessionLocal() as db:
        for download in manifest.get("downloads", []):
            download_id = download.get("download_id")
            record = db.get(SentinelDownload, download_id)
            if record is None:
                stats["missing_download_record"] += 1
                continue
            raster_path = download.get("raster_path")
            if raster_path and raster_path in available_paths:
                record.local_path = raster_path
            elif raster_path:
                stats["missing_download_file"] += 1
            stats["download_synced"] += 1

        preview_paths: dict[str, dict[str, str]] = {}
        for file_row in manifest.get("files", []):
            preview_id = file_row.get("preview_id")
            kind = file_row.get("kind")
            if not preview_id:
                continue
            preview_paths.setdefault(preview_id, {})
            if kind == "sentinel_preview_png":
                preview_paths[preview_id]["png_path"] = file_row["relative_path"]
            elif kind == "sentinel_preview_metadata":
                preview_paths[preview_id]["metadata_path"] = file_row["relative_path"]

        for preview_id, paths in preview_paths.items():
            record = db.get(DerivedPreview, preview_id)
            if record is None:
                stats["missing_preview_record"] += 1
                continue
            if "png_path" in paths and paths["png_path"] in available_paths:
                record.png_path = paths["png_path"]
            elif "png_path" in paths:
                stats["missing_preview_png_file"] += 1
            if "metadata_path" in paths and paths["metadata_path"] in available_paths:
                record.metadata_path = paths["metadata_path"]
            elif "metadata_path" in paths:
                stats["missing_preview_metadata_file"] += 1
            stats["preview_synced"] += 1

        db.commit()
    return dict(stats)


def write_report(rows: list[dict[str, Any]]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / f"sentinel_cache_import_{now_tag()}.csv"
    fieldnames = [
        "kind",
        "download_id",
        "sample_id",
        "scene_id",
        "period",
        "preview_id",
        "view",
        "relative_path",
        "source_in_bundle",
        "target_path",
        "source_exists",
        "target_exists_before",
        "bytes",
        "action",
        "error",
    ]
    with report_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return report_path


def main() -> None:
    init_forest_database()
    bundle_path = BUNDLE_PATH or latest_bundle_path()
    manifest, rows = load_bundle_rows(bundle_path)
    db_stats = sync_db_records(manifest)
    report_path = write_report(rows)

    action_counts = Counter(row["action"] for row in rows)
    kind_counts = Counter(row["kind"] for row in rows)
    unavailable_by_kind = Counter(
        row["kind"]
        for row in rows
        if row["action"] in {"missing_bundle_file", "missing_source_at_export"}
    )
    copied_bytes = sum(int(row.get("bytes") or 0) for row in rows if row["action"] in {"copied", "would_copy"})
    print(f"DRY_RUN: {DRY_RUN}")
    print(f"Bundle: {bundle_path}")
    print(f"Files: {len(rows)}")
    print(f"Copy bytes: {copied_bytes / 1024 / 1024:.1f} MiB")
    print(f"File kinds: {dict(sorted(kind_counts.items()))}")
    print(f"Actions: {dict(sorted(action_counts.items()))}")
    if unavailable_by_kind:
        print(f"Unavailable by kind: {dict(sorted(unavailable_by_kind.items()))}")
    print(f"DB sync: {db_stats}")
    print(f"Report: {report_path}")
    if DRY_RUN:
        print("Dry run only. Set DRY_RUN = False to copy files into the project cache.")


if __name__ == "__main__":
    main()
