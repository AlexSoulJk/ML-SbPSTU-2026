from __future__ import annotations

import csv
import json
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


# Set explicitly if needed. If None, the newest sentinel_cache_bundle_* folder/zip in Outputs is used.
BUNDLE_PATH: Path | None = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_ROOT / "Outputs"


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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


def read_manifest_from_folder(bundle_path: Path) -> dict[str, Any]:
    manifest_path = bundle_path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest was not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def read_manifest_from_zip(bundle_path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(bundle_path) as archive:
        return json.loads(archive.read("manifest.json").decode("utf-8"))


def validate_folder(bundle_path: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in manifest.get("files", []):
        relative_path = str(row.get("relative_path") or "")
        actual_path = bundle_path / "files" / relative_path
        exists = actual_path.exists()
        actual_bytes = actual_path.stat().st_size if exists else 0
        expected_bytes = int(row.get("bytes") or 0)
        rows.append(
            {
                "kind": row.get("kind", ""),
                "download_id": row.get("download_id", ""),
                "sample_id": row.get("sample_id", ""),
                "scene_id": row.get("scene_id", ""),
                "period": row.get("period", ""),
                "preview_id": row.get("preview_id", ""),
                "view": row.get("view", ""),
                "relative_path": relative_path,
                "manifest_action": row.get("action", ""),
                "expected_bytes": expected_bytes,
                "exists": exists,
                "actual_bytes": actual_bytes,
                "size_matches": exists and actual_bytes == expected_bytes,
            }
        )
    return rows


def validate_zip(bundle_path: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    with zipfile.ZipFile(bundle_path) as archive:
        zip_entries = {item.filename: item.file_size for item in archive.infolist() if not item.is_dir()}

    for row in manifest.get("files", []):
        relative_path = str(row.get("relative_path") or "")
        archive_path = f"files/{relative_path}"
        exists = archive_path in zip_entries
        actual_bytes = zip_entries.get(archive_path, 0)
        expected_bytes = int(row.get("bytes") or 0)
        rows.append(
            {
                "kind": row.get("kind", ""),
                "download_id": row.get("download_id", ""),
                "sample_id": row.get("sample_id", ""),
                "scene_id": row.get("scene_id", ""),
                "period": row.get("period", ""),
                "preview_id": row.get("preview_id", ""),
                "view": row.get("view", ""),
                "relative_path": relative_path,
                "manifest_action": row.get("action", ""),
                "expected_bytes": expected_bytes,
                "exists": exists,
                "actual_bytes": actual_bytes,
                "size_matches": exists and actual_bytes == expected_bytes,
            }
        )
    return rows


def write_report(rows: list[dict[str, Any]]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / f"sentinel_cache_bundle_validate_{now_tag()}.csv"
    fieldnames = [
        "kind",
        "download_id",
        "sample_id",
        "scene_id",
        "period",
        "preview_id",
        "view",
        "relative_path",
        "manifest_action",
        "expected_bytes",
        "exists",
        "actual_bytes",
        "size_matches",
    ]
    with report_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return report_path


def main() -> None:
    bundle_path = BUNDLE_PATH or latest_bundle_path()
    if bundle_path.suffix.lower() == ".zip":
        manifest = read_manifest_from_zip(bundle_path)
        rows = validate_zip(bundle_path, manifest)
    else:
        manifest = read_manifest_from_folder(bundle_path)
        rows = validate_folder(bundle_path, manifest)

    report_path = write_report(rows)
    present_rows = [row for row in rows if row["exists"]]
    missing_rows = [row for row in rows if not row["exists"]]
    size_mismatch_rows = [row for row in rows if row["exists"] and not row["size_matches"]]

    print(f"Bundle: {bundle_path}")
    print(f"Manifest files: {len(rows)}")
    print(f"Present files: {len(present_rows)}")
    print(f"Missing files: {len(missing_rows)}")
    print(f"Size mismatches: {len(size_mismatch_rows)}")
    print(f"Manifest actions: {dict(sorted(Counter(row['manifest_action'] for row in rows).items()))}")
    print(f"File kinds: {dict(sorted(Counter(row['kind'] for row in rows).items()))}")
    if missing_rows:
        print(f"Missing by kind: {dict(sorted(Counter(row['kind'] for row in missing_rows).items()))}")
    if size_mismatch_rows:
        print(f"Size mismatch by kind: {dict(sorted(Counter(row['kind'] for row in size_mismatch_rows).items()))}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
