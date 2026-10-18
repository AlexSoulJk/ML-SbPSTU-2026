from __future__ import annotations

import csv
import json
import shutil
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import FOREST_DATA_DIR, FOREST_HANSEN_CACHE_DIR, FOREST_STATE_DB_PATH, PROJECT_ROOT, TOOL_ROOT
from .models import (
    DerivedPreview,
    ManualValidation,
    Sample,
    SampleStatus,
    SentinelDownload,
    SentinelSceneReview,
    SentinelSceneReviewReason,
    utc_now,
)
from .storage_paths import to_storage_path


DB_SYNC_BASES_DIR = FOREST_DATA_DIR / "bases"
DB_SYNC_BACKUP_DIR = FOREST_DATA_DIR / "backups"
DB_SYNC_OUTPUTS_DIR = PROJECT_ROOT / "Scripts" / "ResearchScripts" / "Outputs"

REQUIRED_TABLES = {
    "forest_samples",
    "forest_manual_validations",
    "forest_sample_statuses",
    "forest_sentinel_downloads",
    "forest_derived_previews",
    "forest_sentinel_scene_reviews",
    "forest_sentinel_scene_review_reasons",
    "forest_hansen_analyses",
    "forest_hansen_tiles",
    "forest_hansen_masks",
}

DOWNLOAD_COLUMNS = [
    "download_id",
    "sample_id",
    "scene_id",
    "period",
    "local_path",
    "bbox_json",
    "bands_json",
    "cloud_fraction",
    "shadow_fraction",
    "nodata_fraction",
    "dark_fraction",
    "is_bad_cloud",
    "is_bad_quality",
    "quality_flags_json",
    "metadata_json",
]

PREVIEW_COLUMNS = [
    "preview_id",
    "download_id",
    "view",
    "png_path",
    "metadata_path",
]


class DbSyncError(ValueError):
    pass


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def json_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def values_equal(left: Any, right: Any) -> bool:
    if left is None and right in {None, ""}:
        return True
    if right is None and left in {None, ""}:
        return True
    return str(left) == str(right)


def numeric_values_equal(left: Any, right: Any, *, tolerance: float = 1e-6) -> bool:
    if left is None and right in {None, ""}:
        return True
    if right is None and left in {None, ""}:
        return True
    try:
        return abs(float(left) - float(right)) <= tolerance
    except (TypeError, ValueError):
        return values_equal(left, right)


def normalized_json_value(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return str(value)
    return json.dumps(parsed, ensure_ascii=False, sort_keys=True)


def is_blank(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def is_portable_storage_path(value: Any) -> bool:
    if is_blank(value):
        return False
    path = Path(str(value))
    if path.is_absolute():
        return False
    return ".." not in path.parts


def path_parts(value: Any) -> list[str]:
    return [
        part
        for part in str(value).replace("\\", "/").split("/")
        if part and part not in {".", "/"}
    ]


def local_project_path_from_known_suffix(value: Any) -> Path | None:
    if is_blank(value):
        return None
    parts = path_parts(value)
    lowered = [part.lower() for part in parts]
    marker_roots: list[tuple[list[str], Path, int]] = [
        (["scripts", "researchtool"], TOOL_ROOT, 2),
        (["data"], TOOL_ROOT, 0),
        (["cache"], TOOL_ROOT / "Data", 0),
    ]
    for marker, root, skip in marker_roots:
        marker_len = len(marker)
        for index in range(0, len(parts) - marker_len + 1):
            if lowered[index:index + marker_len] != marker:
                continue
            suffix_parts = parts[index + skip:]
            if not suffix_parts:
                continue
            return root.joinpath(*suffix_parts)
    return None


def storage_path_candidates(value: str | None, *, incoming_db_path: Path | None = None) -> list[Path]:
    if not value:
        return []
    raw = Path(str(value))
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(PROJECT_ROOT / raw)
        candidates.append(TOOL_ROOT / raw)
        if incoming_db_path is not None:
            candidates.append(incoming_db_path.parent / raw)

    local_equivalent = local_project_path_from_known_suffix(value)
    if local_equivalent is not None:
        candidates.append(local_equivalent)

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).replace("\\", "/").lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def normalize_storage_path(value: Any, *, fallback: str | None = None) -> str:
    local_equivalent = local_project_path_from_known_suffix(value)
    if local_equivalent is not None:
        return to_storage_path(local_equivalent) or str(local_equivalent)
    if is_portable_storage_path(value):
        return str(value).replace("\\", "/")
    if value and not Path(str(value)).is_absolute():
        return str(value).replace("\\", "/")
    if fallback:
        return fallback
    return to_storage_path(value) or str(value)


def path_exists(value: str | None, *, incoming_db_path: Path | None = None) -> bool:
    if not value:
        return False
    try:
        return any(path.exists() for path in storage_path_candidates(value, incoming_db_path=incoming_db_path))
    except OSError:
        return False


def hansen_mask_path_exists(row: dict[str, Any], *, incoming_db_path: Path | None = None) -> bool:
    path = row.get("local_path")
    if path_exists(path, incoming_db_path=incoming_db_path):
        return True
    sample_id = text_or_none(row.get("sample_id"))
    analysis_id = text_or_none(row.get("analysis_id"))
    mask_type = text_or_none(row.get("mask_type"))
    if not sample_id or not analysis_id or not mask_type:
        return False
    canonical_path = FOREST_HANSEN_CACHE_DIR / "masks" / sample_id / analysis_id / f"{mask_type}.tif"
    return canonical_path.exists()


def allowed_bases_dir() -> Path:
    DB_SYNC_BASES_DIR.mkdir(parents=True, exist_ok=True)
    return DB_SYNC_BASES_DIR.resolve(strict=False)


def resolve_incoming_db_path(raw_path: str) -> Path:
    if not raw_path or not raw_path.strip():
        raise DbSyncError("incoming_path is required.")

    raw = Path(raw_path.strip())
    base_dir = allowed_bases_dir()
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(PROJECT_ROOT / raw)
        candidates.append(base_dir / raw)

    existing = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    resolved = existing.resolve(strict=False)
    try:
        resolved.relative_to(base_dir)
    except ValueError as exc:
        raise DbSyncError(f"Incoming DB must be inside {base_dir}") from exc
    if resolved.suffix.lower() not in {".sqlite", ".db"}:
        raise DbSyncError("Incoming DB must be a .sqlite or .db file.")
    if not resolved.exists():
        raise DbSyncError(f"Incoming DB was not found: {resolved}")
    return resolved


def candidate_databases() -> dict[str, Any]:
    base_dir = allowed_bases_dir()
    candidates = []
    for path in sorted(base_dir.glob("*.sqlite")) + sorted(base_dir.glob("*.db")):
        candidates.append(
            {
                "name": path.name,
                "path": to_storage_path(path) or str(path),
                "size_bytes": path.stat().st_size,
                "modified_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            }
        )
    return {
        "base_dir": to_storage_path(base_dir) or str(base_dir),
        "items": candidates,
    }


def connect_incoming(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
    return {str(row["name"]) for row in rows}


def table_exists(connection: sqlite3.Connection, table: str) -> bool:
    row = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def read_rows(connection: sqlite3.Connection, table: str, columns: list[str]) -> list[dict[str, Any]]:
    if not table_exists(connection, table):
        return []
    existing_columns = table_columns(connection, table)
    selected = [column for column in columns if column in existing_columns]
    if not selected:
        return []
    sql = f"SELECT {', '.join(selected)} FROM {table}"
    rows = []
    for row in connection.execute(sql).fetchall():
        payload = {column: row[column] if column in selected else None for column in columns}
        rows.append(payload)
    return rows


def count_rows(connection: sqlite3.Connection, table: str) -> int | None:
    if not table_exists(connection, table):
        return None
    row = connection.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
    return int(row["count"])


def schema_warnings(connection: sqlite3.Connection) -> list[str]:
    warnings = []
    for table in sorted(REQUIRED_TABLES):
        if not table_exists(connection, table):
            warnings.append(f"Incoming DB is missing table {table}; this entity will be skipped.")
    return warnings


def append_note(local_note: str | None, incoming_note: str | None, *, db_name: str, stamp: str) -> str | None:
    incoming = text_or_none(incoming_note)
    local = text_or_none(local_note)
    if not incoming:
        return local
    if not local:
        return incoming
    if incoming in local:
        return local
    return f"{local.rstrip()}\n\n--- incoming note from {db_name} at {stamp} ---\n{incoming}"


def op(
    *,
    entity: str,
    key: str,
    action: str,
    status: str,
    message: str,
    safe: bool = False,
    field: str | None = None,
    local_value: Any = None,
    incoming_value: Any = None,
    missing_cache: bool = False,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    operation_id = f"{entity}:{key}:{action}:{field or ''}"
    return {
        "operation_id": operation_id,
        "entity": entity,
        "key": key,
        "action": action,
        "status": status,
        "safe": safe,
        "field": field,
        "local_value": json_value(local_value),
        "incoming_value": json_value(incoming_value),
        "message": message,
        "missing_cache": missing_cache,
        "payload": payload or {},
    }


def add_summary(summary: dict[str, Counter], entity: str, status: str) -> None:
    summary[entity][status] += 1


def manual_rows(db: Session) -> dict[str, dict[str, Any]]:
    return {
        row.sample_id: {
            "sample_id": row.sample_id,
            "validation": row.validation,
            "notes": row.notes,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }
        for row in db.scalars(select(ManualValidation))
    }


def status_rows(db: Session) -> dict[str, dict[str, Any]]:
    rows = {}
    for row in db.scalars(select(SampleStatus)):
        key = f"{row.sample_id}|{row.source}"
        rows[key] = {
            "sample_id": row.sample_id,
            "source": row.source,
            "status": row.status,
            "reason": row.reason,
            "metrics_json": row.metrics_json,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }
    return rows


def editable_sample_rows(db: Session) -> dict[str, dict[str, Any]]:
    return {
        row.sample_id: {
            "sample_id": row.sample_id,
            "display_name": row.display_name,
            "notes": row.notes,
            "has_multiple_events": bool(row.has_multiple_events),
        }
        for row in db.scalars(select(Sample))
    }


def incoming_manual_rows(connection: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    return {
        str(row["sample_id"]): row
        for row in read_rows(
            connection,
            "forest_manual_validations",
            ["sample_id", "validation", "notes", "updated_at"],
        )
        if row.get("sample_id")
    }


def incoming_status_rows(connection: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    rows = {}
    for row in read_rows(
        connection,
        "forest_sample_statuses",
        ["sample_id", "source", "status", "reason", "metrics_json", "updated_at"],
    ):
        if row.get("sample_id") and row.get("source"):
            rows[f"{row['sample_id']}|{row['source']}"] = row
    return rows


def incoming_sample_rows(connection: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    return {
        str(row["sample_id"]): {
            **row,
            "has_multiple_events": bool_value(row.get("has_multiple_events")),
        }
        for row in read_rows(
            connection,
            "forest_samples",
            ["sample_id", "display_name", "notes", "has_multiple_events"],
        )
        if row.get("sample_id")
    }


def incoming_download_rows(connection: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    return {
        str(row["download_id"]): row
        for row in read_rows(connection, "forest_sentinel_downloads", DOWNLOAD_COLUMNS)
        if row.get("download_id")
    }


def incoming_preview_rows(connection: sqlite3.Connection) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_rows(connection, "forest_derived_previews", PREVIEW_COLUMNS):
        if row.get("download_id") and row.get("preview_id"):
            rows[str(row["download_id"])].append(row)
    return rows


def review_rows(db: Session) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    reasons: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for reason in db.scalars(select(SentinelSceneReviewReason)):
        reasons[reason.download_id].append(
            {
                "reason_code": reason.reason_code,
                "reason_text": reason.reason_text,
            }
        )
    for review in db.scalars(select(SentinelSceneReview)):
        rows[review.download_id] = {
            "download_id": review.download_id,
            "is_excluded": bool(review.is_excluded),
            "reason_code": review.reason_code,
            "reason_text": review.reason_text,
            "notes": review.notes,
            "created_at": review.created_at.isoformat() if review.created_at else None,
            "updated_at": review.updated_at.isoformat() if review.updated_at else None,
            "reasons": reasons.get(review.download_id, []),
        }
    return rows


def incoming_review_rows(connection: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    reasons: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for reason in read_rows(
        connection,
        "forest_sentinel_scene_review_reasons",
        ["download_id", "reason_code", "reason_text"],
    ):
        if reason.get("download_id") and reason.get("reason_code"):
            reasons[str(reason["download_id"])].append(
                {
                    "reason_code": reason["reason_code"],
                    "reason_text": reason.get("reason_text"),
                }
            )

    rows: dict[str, dict[str, Any]] = {}
    for review in read_rows(
        connection,
        "forest_sentinel_scene_reviews",
        ["download_id", "is_excluded", "reason_code", "reason_text", "notes", "created_at", "updated_at"],
    ):
        download_id = review.get("download_id")
        if not download_id:
            continue
        reason_payloads = reasons.get(str(download_id), [])
        if not reason_payloads and review.get("reason_code"):
            reason_payloads = [
                {
                    "reason_code": review.get("reason_code"),
                    "reason_text": review.get("reason_text"),
                }
            ]
        rows[str(download_id)] = {
            **review,
            "is_excluded": bool_value(review.get("is_excluded")),
            "reasons": reason_payloads,
        }
    return rows


def compare_manual_validations(
    db: Session,
    connection: sqlite3.Connection,
    *,
    source_name: str,
    stamp: str,
    summary: dict[str, Counter],
    operations: list[dict[str, Any]],
) -> None:
    local = manual_rows(db)
    incoming = incoming_manual_rows(connection)
    local_samples = set(db.scalars(select(Sample.sample_id)))
    entity = "manual_validations"
    for key in sorted(set(local) | set(incoming)):
        left = local.get(key)
        right = incoming.get(key)
        if left is None and right is not None:
            if key not in local_samples:
                add_summary(summary, entity, "only_incoming")
                operations.append(
                    op(
                        entity=entity,
                        key=key,
                        action="manual_missing_sample",
                        status="report-only",
                        message="Incoming manual validation references a sample that does not exist locally.",
                        incoming_value=right,
                    )
                )
                continue
            add_summary(summary, entity, "only_incoming")
            operations.append(
                op(
                    entity=entity,
                    key=key,
                    action="manual_insert",
                    status="safe",
                    safe=True,
                    message="Incoming manual validation can be inserted.",
                    incoming_value=right,
                    payload={"row": right},
                )
            )
        elif left is not None and right is None:
            add_summary(summary, entity, "only_local")
            operations.append(
                op(
                    entity=entity,
                    key=key,
                    action="sentinel_review_only_local",
                    status="report-only",
                    message="Local Sentinel review is absent in incoming DB.",
                    local_value=left,
                )
            )
        elif left and right:
            if left["validation"] != right["validation"]:
                add_summary(summary, entity, "conflict")
                operations.append(
                    op(
                        entity=entity,
                        key=key,
                        action="manual_conflict",
                        status="conflict",
                        field="validation",
                        message="Manual validation differs.",
                        local_value=left["validation"],
                        incoming_value=right["validation"],
                    )
                )
            elif not values_equal(left.get("notes"), right.get("notes")):
                merged = append_note(left.get("notes"), right.get("notes"), db_name=source_name, stamp=stamp)
                if merged != left.get("notes"):
                    add_summary(summary, entity, "safe_operations")
                    operations.append(
                        op(
                            entity=entity,
                            key=key,
                            action="manual_note_append",
                            status="safe",
                            safe=True,
                            field="notes",
                            message="Manual notes will be appended without losing local text.",
                            local_value=left.get("notes"),
                            incoming_value=right.get("notes"),
                            payload={"notes": merged},
                        )
                    )
                else:
                    add_summary(summary, entity, "same")
            else:
                add_summary(summary, entity, "same")


def compare_sample_statuses(
    db: Session,
    connection: sqlite3.Connection,
    *,
    summary: dict[str, Counter],
    operations: list[dict[str, Any]],
) -> None:
    local = status_rows(db)
    incoming = incoming_status_rows(connection)
    local_samples = set(db.scalars(select(Sample.sample_id)))
    entity = "sample_statuses"
    for key in sorted(set(local) | set(incoming)):
        left = local.get(key)
        right = incoming.get(key)
        if left is None and right is not None:
            sample_id = str(right.get("sample_id") or "")
            if sample_id not in local_samples:
                add_summary(summary, entity, "only_incoming")
                operations.append(
                    op(
                        entity=entity,
                        key=key,
                        action="sample_status_missing_sample",
                        status="report-only",
                        message="Incoming sample status references a sample that does not exist locally.",
                        incoming_value=right,
                    )
                )
                continue
            add_summary(summary, entity, "only_incoming")
            operations.append(
                op(
                    entity=entity,
                    key=key,
                    action="sample_status_insert",
                    status="safe",
                    safe=True,
                    message="Incoming sample status can be inserted.",
                    incoming_value=right,
                    payload={"row": right},
                )
            )
        elif left is not None and right is None:
            add_summary(summary, entity, "only_local")
        elif left and right:
            full_equal = all(
                values_equal(left.get(field), right.get(field))
                for field in ("status", "reason", "metrics_json")
            )
            if full_equal:
                add_summary(summary, entity, "same")
            else:
                add_summary(summary, entity, "conflict")
                operations.append(
                    op(
                        entity=entity,
                        key=key,
                        action="sample_status_conflict",
                        status="conflict",
                        message="Sample status row differs.",
                        local_value={field: left.get(field) for field in ("status", "reason", "metrics_json")},
                        incoming_value={field: right.get(field) for field in ("status", "reason", "metrics_json")},
                    )
                )


def compare_sample_metadata(
    db: Session,
    connection: sqlite3.Connection,
    *,
    source_name: str,
    stamp: str,
    summary: dict[str, Counter],
    operations: list[dict[str, Any]],
) -> None:
    local = editable_sample_rows(db)
    incoming = incoming_sample_rows(connection)
    entity = "sample_metadata"
    for key in sorted(set(local) | set(incoming)):
        left = local.get(key)
        right = incoming.get(key)
        if left is None:
            add_summary(summary, entity, "only_incoming")
            operations.append(
                op(
                    entity=entity,
                    key=key,
                    action="sample_missing_local",
                    status="report-only",
                    message="Incoming sample does not exist locally; sample import is outside this MVP.",
                    incoming_value=right,
                )
            )
            continue
        if right is None:
            add_summary(summary, entity, "only_local")
            continue

        changed = False
        for field in ("display_name",):
            if is_blank(left.get(field)) and not is_blank(right.get(field)):
                changed = True
                add_summary(summary, entity, "safe_operations")
                operations.append(
                    op(
                        entity=entity,
                        key=key,
                        action="sample_metadata_set",
                        status="safe",
                        safe=True,
                        field=field,
                        message=f"{field} is empty locally and can be filled from incoming.",
                        local_value=left.get(field),
                        incoming_value=right.get(field),
                        payload={"field": field, "value": right.get(field)},
                    )
                )
            elif not values_equal(left.get(field), right.get(field)):
                changed = True
                add_summary(summary, entity, "conflict")
                operations.append(
                    op(
                        entity=entity,
                        key=key,
                        action="sample_metadata_conflict",
                        status="conflict",
                        field=field,
                        message=f"{field} differs.",
                        local_value=left.get(field),
                        incoming_value=right.get(field),
                    )
                )

        if not values_equal(left.get("notes"), right.get("notes")):
            changed = True
            merged = append_note(left.get("notes"), right.get("notes"), db_name=source_name, stamp=stamp)
            if merged != left.get("notes"):
                add_summary(summary, entity, "safe_operations")
                operations.append(
                    op(
                        entity=entity,
                        key=key,
                        action="sample_notes_append",
                        status="safe",
                        safe=True,
                        field="notes",
                        message="Sample notes will be appended without losing local text.",
                        local_value=left.get("notes"),
                        incoming_value=right.get("notes"),
                        payload={"notes": merged},
                    )
                )

        if bool(left.get("has_multiple_events")) != bool(right.get("has_multiple_events")):
            changed = True
            add_summary(summary, entity, "conflict")
            operations.append(
                op(
                    entity=entity,
                    key=key,
                    action="sample_metadata_conflict",
                    status="conflict",
                    field="has_multiple_events",
                    message="has_multiple_events differs.",
                    local_value=bool(left.get("has_multiple_events")),
                    incoming_value=bool(right.get("has_multiple_events")),
                )
            )
        if not changed:
            add_summary(summary, entity, "same")


def download_import_operation(
    *,
    download_id: str,
    download: dict[str, Any] | None,
    previews: list[dict[str, Any]],
    local_samples: set[str],
) -> list[dict[str, Any]]:
    if not download:
        return [
            op(
                entity="sentinel_downloads",
                key=download_id,
                action="sentinel_download_missing_incoming",
                status="report-only",
                message="Review references a download_id that is missing from incoming forest_sentinel_downloads.",
            )
        ]
    sample_id = str(download.get("sample_id") or "")
    if sample_id not in local_samples:
        return [
            op(
                entity="sentinel_downloads",
                key=download_id,
                action="sentinel_download_missing_sample",
                status="report-only",
                message="Incoming download references a sample that does not exist locally.",
                incoming_value={"sample_id": sample_id},
            )
        ]

    fallback_path = f"Scripts/ResearchTool/Data/cache/forest/sentinel/rasters/_missing/{download_id}.tif"
    normalized_download = {
        **download,
        "local_path": normalize_storage_path(download.get("local_path"), fallback=fallback_path),
    }
    operations = [
        op(
            entity="sentinel_downloads",
            key=download_id,
            action="sentinel_download_import",
            status="safe",
            safe=True,
            message="Import Sentinel download metadata without copying cache files.",
            missing_cache=not path_exists(normalized_download["local_path"]),
            incoming_value=normalized_download,
            payload={"row": normalized_download},
        )
    ]
    for preview in previews:
        normalized_preview = {
            **preview,
            "png_path": normalize_storage_path(preview.get("png_path")),
            "metadata_path": normalize_storage_path(preview.get("metadata_path")),
        }
        if (
            not is_portable_storage_path(normalized_preview.get("png_path"))
            or not is_portable_storage_path(normalized_preview.get("metadata_path"))
        ):
            operations.append(
                op(
                    entity="derived_previews",
                    key=str(preview.get("preview_id") or f"{download_id}:{preview.get('view')}"),
                    action="sentinel_preview_skip_nonportable",
                    status="report-only",
                    message="Preview metadata has non-portable paths and will not be imported.",
                    incoming_value=preview,
                )
            )
            continue
        operations.append(
            op(
                entity="derived_previews",
                key=str(preview["preview_id"]),
                action="sentinel_preview_import",
                status="safe",
                safe=True,
                message="Import derived preview metadata without copying cache files.",
                missing_cache=not path_exists(normalized_preview["png_path"]),
                incoming_value=normalized_preview,
                payload={"row": normalized_preview},
            )
        )
    return operations


SENTINEL_DOWNLOAD_DIFF_FIELDS = [
    "local_path",
    "cloud_fraction",
    "shadow_fraction",
    "nodata_fraction",
    "dark_fraction",
    "is_bad_cloud",
    "is_bad_quality",
    "quality_flags_json",
]


def local_download_row(download: SentinelDownload) -> dict[str, Any]:
    return {
        "download_id": download.download_id,
        "sample_id": download.sample_id,
        "scene_id": download.scene_id,
        "period": download.period,
        "local_path": normalize_storage_path(download.local_path),
        "bbox_json": download.bbox_json,
        "bands_json": download.bands_json,
        "cloud_fraction": download.cloud_fraction,
        "shadow_fraction": download.shadow_fraction,
        "nodata_fraction": download.nodata_fraction,
        "dark_fraction": download.dark_fraction,
        "is_bad_cloud": download.is_bad_cloud,
        "is_bad_quality": download.is_bad_quality,
        "quality_flags_json": download.quality_flags_json,
        "metadata_json": download.metadata_json,
    }


def download_values_equal(field: str, left: Any, right: Any) -> bool:
    if field in {"cloud_fraction", "shadow_fraction", "nodata_fraction", "dark_fraction"}:
        return numeric_values_equal(left, right, tolerance=1e-4)
    if field in {"is_bad_cloud", "is_bad_quality"}:
        return bool_value(left) == bool_value(right)
    if field == "quality_flags_json":
        return normalized_json_value(left) == normalized_json_value(right)
    if field == "local_path":
        return normalize_storage_path(left) == normalize_storage_path(right)
    return values_equal(left, right)


def compare_sentinel_download_metadata(
    db: Session,
    connection: sqlite3.Connection,
    *,
    summary: dict[str, Counter],
    operations: list[dict[str, Any]],
) -> None:
    entity = "sentinel_downloads"
    incoming = incoming_download_rows(connection)
    local = {
        download.download_id: local_download_row(download)
        for download in db.scalars(select(SentinelDownload))
    }
    for key in sorted(set(local) & set(incoming)):
        left = local[key]
        right = incoming[key]
        for field in SENTINEL_DOWNLOAD_DIFF_FIELDS:
            if download_values_equal(field, left.get(field), right.get(field)):
                continue
            add_summary(summary, entity, "metadata_diff")
            operations.append(
                op(
                    entity=entity,
                    key=key,
                    action="sentinel_download_metadata_diff",
                    status="report-only",
                    field=field,
                    message="Sentinel download metadata differs between local and incoming DB.",
                    local_value=left.get(field),
                    incoming_value=right.get(field),
                )
            )


def compare_sentinel_reviews(
    db: Session,
    connection: sqlite3.Connection,
    *,
    source_name: str,
    stamp: str,
    summary: dict[str, Counter],
    operations: list[dict[str, Any]],
) -> None:
    local = review_rows(db)
    incoming = incoming_review_rows(connection)
    local_download_ids = set(db.scalars(select(SentinelDownload.download_id)))
    local_samples = set(db.scalars(select(Sample.sample_id)))
    incoming_downloads = incoming_download_rows(connection)
    incoming_previews = incoming_preview_rows(connection)
    entity = "sentinel_reviews"

    for key in sorted(set(local) | set(incoming)):
        left = local.get(key)
        right = incoming.get(key)
        if left is None and right is not None:
            add_summary(summary, entity, "only_incoming")
            if key not in local_download_ids:
                download_ops = download_import_operation(
                    download_id=key,
                    download=incoming_downloads.get(key),
                    previews=incoming_previews.get(key, []),
                    local_samples=local_samples,
                )
                operations.extend(download_ops)
                if any(item["status"] == "report-only" and not item["safe"] for item in download_ops):
                    operations.append(
                        op(
                            entity=entity,
                            key=key,
                            action="sentinel_review_blocked",
                            status="report-only",
                            message="Review cannot be imported until download metadata is available.",
                            incoming_value=right,
                        )
                    )
                    continue
                add_summary(summary, entity, "missing_cache")
            operations.append(
                op(
                    entity=entity,
                    key=key,
                    action="sentinel_review_insert",
                    status="safe",
                    safe=True,
                    message="Incoming Sentinel review can be inserted.",
                    missing_cache=key not in local_download_ids,
                    incoming_value=right,
                    payload={"row": right},
                )
            )
        elif left is not None and right is None:
            add_summary(summary, entity, "only_local")
        elif left and right:
            left_excluded = bool(left.get("is_excluded"))
            right_excluded = bool(right.get("is_excluded"))
            if left_excluded != right_excluded:
                add_summary(summary, entity, "conflict")
                operations.append(
                    op(
                        entity=entity,
                        key=key,
                        action="sentinel_review_conflict",
                        status="conflict",
                        field="is_excluded",
                        message="Sentinel review exclusion state differs.",
                        local_value=left_excluded,
                        incoming_value=right_excluded,
                    )
                )
                continue

            left_reasons = {reason["reason_code"]: reason for reason in left.get("reasons", [])}
            right_reasons = {reason["reason_code"]: reason for reason in right.get("reasons", [])}
            inserted_reason = False
            reason_diff_added = False
            if left_excluded and right_excluded:
                for reason_code, reason in sorted(right_reasons.items()):
                    if reason_code not in left_reasons:
                        inserted_reason = True
                        add_summary(summary, entity, "safe_operations")
                        operations.append(
                            op(
                                entity=entity,
                                key=key,
                                action="sentinel_review_reason_insert",
                                status="safe",
                                safe=True,
                                field="reason_code",
                                message="Incoming reason can be added to local reason set.",
                                incoming_value=reason,
                                payload={"download_id": key, "reason": reason},
                            )
                        )
                for reason_code, reason in sorted(left_reasons.items()):
                    if reason_code not in right_reasons:
                        reason_diff_added = True
                        add_summary(summary, entity, "local_only_reason")
                        operations.append(
                            op(
                                entity=entity,
                                key=key,
                                action="sentinel_review_reason_only_local",
                                status="report-only",
                                field="reason_code",
                                message="Local Sentinel review reason is absent in incoming DB.",
                                local_value=reason,
                            )
                        )
                    elif not values_equal(
                        reason.get("reason_text"),
                        right_reasons[reason_code].get("reason_text"),
                    ):
                        reason_diff_added = True
                        add_summary(summary, entity, "reason_text_diff")
                        operations.append(
                            op(
                                entity=entity,
                                key=key,
                                action="sentinel_review_reason_text_diff",
                                status="report-only",
                                field="reason_text",
                                message="Sentinel review reason text differs.",
                                local_value=reason,
                                incoming_value=right_reasons[reason_code],
                            )
                        )

            note_op_added = False
            if not values_equal(left.get("notes"), right.get("notes")):
                merged = append_note(left.get("notes"), right.get("notes"), db_name=source_name, stamp=stamp)
                if merged != left.get("notes"):
                    note_op_added = True
                    add_summary(summary, entity, "safe_operations")
                    operations.append(
                        op(
                            entity=entity,
                            key=key,
                            action="sentinel_review_notes_append",
                            status="safe",
                            safe=True,
                            field="notes",
                            message="Sentinel review notes will be appended without losing local text.",
                            local_value=left.get("notes"),
                            incoming_value=right.get("notes"),
                            payload={"notes": merged},
                        )
                    )
            if not inserted_reason and not reason_diff_added and not note_op_added:
                add_summary(summary, entity, "same")


def compare_cache_diagnostics(
    db: Session,
    connection: sqlite3.Connection,
    *,
    incoming_db_path: Path,
    summary: dict[str, Counter],
    operations: list[dict[str, Any]],
) -> None:
    entity = "cache_diagnostics"
    for download in db.scalars(select(SentinelDownload)):
        if not path_exists(download.local_path):
            add_summary(summary, entity, "missing_cache")
            operations.append(
                op(
                    entity=entity,
                    key=download.download_id,
                    action="missing_local_geotiff",
                    status="report-only",
                    message="Local Sentinel GeoTIFF is missing on disk.",
                    local_value=download.local_path,
                    missing_cache=True,
                )
            )
    for preview in db.scalars(select(DerivedPreview)):
        if not path_exists(preview.png_path) or not path_exists(preview.metadata_path):
            add_summary(summary, entity, "missing_cache")
            operations.append(
                op(
                    entity=entity,
                    key=preview.preview_id,
                    action="missing_local_preview",
                    status="report-only",
                    message="Local derived preview PNG/metadata is missing on disk.",
                    local_value={"png_path": preview.png_path, "metadata_path": preview.metadata_path},
                    missing_cache=True,
                )
            )

    for row in read_rows(connection, "forest_hansen_tiles", ["id", "local_path"]):
        path = row.get("local_path")
        key = str(row.get("id") or path)
        if path and not path_exists(path, incoming_db_path=incoming_db_path):
            add_summary(summary, entity, "incoming_hansen_missing_cache")
            operations.append(
                op(
                    entity=entity,
                    key=key,
                    action="incoming_hansen_missing_cache",
                    status="report-only",
                    message="Incoming forest_hansen_tiles path is not present locally.",
                    incoming_value=path,
                    missing_cache=True,
                )
            )

    for row in read_rows(
        connection,
        "forest_hansen_masks",
        ["mask_id", "sample_id", "analysis_id", "mask_type", "local_path"],
    ):
        path = row.get("local_path")
        key = str(row.get("mask_id") or path)
        if path and not hansen_mask_path_exists(row, incoming_db_path=incoming_db_path):
            add_summary(summary, entity, "incoming_hansen_missing_cache")
            operations.append(
                op(
                    entity=entity,
                    key=key,
                    action="incoming_hansen_missing_cache",
                    status="report-only",
                    message="Incoming forest_hansen_masks path is not present locally.",
                    incoming_value=path,
                    missing_cache=True,
                )
            )


def compare_hansen_summary(
    connection: sqlite3.Connection,
    *,
    summary: dict[str, Counter],
) -> None:
    entity = "hansen_analyses"
    count = count_rows(connection, "forest_hansen_analyses")
    if count is None:
        add_summary(summary, entity, "missing_table")
    else:
        summary[entity]["incoming_rows"] = count


def sentinel_cache_repair_payload(db: Session) -> dict[str, Any]:
    preview_issues: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for preview in db.scalars(select(DerivedPreview).order_by(DerivedPreview.download_id, DerivedPreview.view)):
        png_exists = path_exists(preview.png_path)
        metadata_exists = path_exists(preview.metadata_path)
        if png_exists and metadata_exists:
            continue
        preview_issues[preview.download_id].append(
            {
                "preview_id": preview.preview_id,
                "view": preview.view,
                "png_exists": png_exists,
                "metadata_exists": metadata_exists,
                "png_path": preview.png_path,
                "metadata_path": preview.metadata_path,
            }
        )

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
    sample_ids = sorted({download.sample_id for download in downloads})
    samples = {
        sample.sample_id: sample
        for sample in db.scalars(select(Sample).where(Sample.sample_id.in_(sample_ids)))
    } if sample_ids else {}

    missing_items: list[dict[str, Any]] = []
    for download in downloads:
        geotiff_exists = path_exists(download.local_path)
        missing_previews = preview_issues.get(download.download_id, [])
        if geotiff_exists and not missing_previews:
            continue
        sample = samples.get(download.sample_id)
        metadata = json.loads(download.metadata_json or "{}")
        item = metadata.get("item") if isinstance(metadata, dict) else {}
        missing_items.append(
            {
                "download_id": download.download_id,
                "sample_id": download.sample_id,
                "sample_name": sample.display_name if sample is not None else None,
                "sample_short_id": download.sample_id[:8],
                "period": download.period,
                "scene_id": download.scene_id,
                "datetime": item.get("datetime") if isinstance(item, dict) else None,
                "cloud_fraction": download.cloud_fraction,
                "geotiff_exists": geotiff_exists,
                "missing_geotiff": not geotiff_exists,
                "missing_preview_count": len(missing_previews),
                "missing_preview_views": [preview["view"] for preview in missing_previews],
                "local_path": download.local_path,
            }
        )

    return {
        "sentinel": {
            "missing_download_count": sum(1 for item in missing_items if item["missing_geotiff"]),
            "missing_preview_download_count": sum(1 for item in missing_items if item["missing_preview_count"]),
            "missing_preview_row_count": sum(int(item["missing_preview_count"]) for item in missing_items),
            "items": missing_items,
        }
    }


def build_db_sync_plan(db: Session, incoming_path: str) -> dict[str, Any]:
    resolved = resolve_incoming_db_path(incoming_path)
    stamp = utc_stamp()
    source_name = resolved.name
    summary: dict[str, Counter] = defaultdict(Counter)
    operations: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    with connect_incoming(resolved) as connection:
        warnings = schema_warnings(connection)
        compare_manual_validations(db, connection, source_name=source_name, stamp=stamp, summary=summary, operations=operations)
        compare_sample_statuses(db, connection, summary=summary, operations=operations)
        compare_sample_metadata(db, connection, source_name=source_name, stamp=stamp, summary=summary, operations=operations)
        compare_sentinel_reviews(db, connection, source_name=source_name, stamp=stamp, summary=summary, operations=operations)
        compare_sentinel_download_metadata(db, connection, summary=summary, operations=operations)
        compare_cache_diagnostics(db, connection, incoming_db_path=resolved, summary=summary, operations=diagnostics)
        compare_hansen_summary(connection, summary=summary)
    cache_repair = sentinel_cache_repair_payload(db)

    safe_count = sum(1 for item in operations if item["safe"])
    conflict_count = sum(1 for item in operations if item["status"] == "conflict")
    report_only_count = sum(1 for item in operations if item["status"] == "report-only")
    missing_cache_count = sum(1 for item in operations if item["missing_cache"])
    diagnostic_missing_cache_count = sum(1 for item in diagnostics if item["missing_cache"])
    return {
        "incoming_path": to_storage_path(resolved) or str(resolved),
        "incoming_name": source_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": {entity: dict(counter) for entity, counter in sorted(summary.items())},
        "totals": {
            "operations": len(operations),
            "safe": safe_count,
            "conflicts": conflict_count,
            "report_only": report_only_count,
            "missing_cache": missing_cache_count,
        },
        "diagnostic_totals": {
            "operations": len(diagnostics),
            "missing_cache": diagnostic_missing_cache_count,
        },
        "cache_repair": cache_repair,
        "warnings": warnings,
        "operations": operations,
        "diagnostics": diagnostics,
    }


def create_backup() -> Path:
    DB_SYNC_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup_path = DB_SYNC_BACKUP_DIR / f"state_before_db_sync_{utc_stamp()}.sqlite"
    shutil.copy2(FOREST_STATE_DB_PATH, backup_path)
    return backup_path


def report_dir() -> Path:
    path = DB_SYNC_OUTPUTS_DIR / f"db_sync_{utc_stamp()}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_report(plan: dict[str, Any], applied: list[dict[str, Any]], skipped: list[dict[str, Any]], backup_path: Path) -> dict[str, str]:
    output_dir = report_dir()
    summary_path = output_dir / "summary.json"
    operations_path = output_dir / "operations.csv"
    conflicts_path = output_dir / "conflicts.csv"
    diagnostics_path = output_dir / "diagnostics.csv"

    summary_path.write_text(
        json.dumps(
            {
                "incoming_path": plan["incoming_path"],
                "incoming_name": plan["incoming_name"],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "backup_path": to_storage_path(backup_path) or str(backup_path),
                "plan_totals": plan["totals"],
                "diagnostic_totals": plan.get("diagnostic_totals", {}),
                "applied_count": len(applied),
                "skipped_count": len(skipped),
                "warnings": plan["warnings"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    fieldnames = [
        "operation_id",
        "entity",
        "key",
        "action",
        "field",
        "status",
        "safe",
        "missing_cache",
        "local_value",
        "incoming_value",
        "message",
        "apply_status",
    ]
    with operations_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item, apply_status in [(item, "applied") for item in applied] + [(item, "skipped") for item in skipped]:
            writer.writerow({key: item.get(key) for key in fieldnames if key != "apply_status"} | {"apply_status": apply_status})

    with conflicts_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames[:-1])
        writer.writeheader()
        for item in plan["operations"]:
            if item["status"] in {"conflict", "report-only"}:
                writer.writerow({key: item.get(key) for key in fieldnames[:-1]})

    with diagnostics_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames[:-1])
        writer.writeheader()
        for item in plan.get("diagnostics", []):
            writer.writerow({key: item.get(key) for key in fieldnames[:-1]})

    return {
        "summary": to_storage_path(summary_path) or str(summary_path),
        "operations": to_storage_path(operations_path) or str(operations_path),
        "conflicts": to_storage_path(conflicts_path) or str(conflicts_path),
        "diagnostics": to_storage_path(diagnostics_path) or str(diagnostics_path),
    }


def selected_safe_operations(plan: dict[str, Any], groups: list[str] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_groups = set(groups or [])
    applied = []
    skipped = []
    for item in plan["operations"]:
        if not item["safe"]:
            skipped.append(item)
            continue
        if selected_groups and item["entity"] not in selected_groups:
            skipped.append(item)
            continue
        applied.append(item)
    return applied, skipped


def upsert_manual_status(db: Session, sample_id: str, validation: str, notes: str | None) -> None:
    status = db.scalar(
        select(SampleStatus).where(
            SampleStatus.sample_id == sample_id,
            SampleStatus.source == "manual",
        )
    )
    if status is None:
        status = SampleStatus(sample_id=sample_id, source="manual", status=validation)
        db.add(status)
    status.status = validation
    status.reason = notes
    status.metrics_json = None
    status.updated_at = utc_now()


def apply_operation(db: Session, item: dict[str, Any]) -> None:
    action = item["action"]
    key = item["key"]
    payload = item.get("payload") or {}

    if action == "manual_insert":
        row = payload["row"]
        manual = db.get(ManualValidation, key)
        if manual is None:
            manual = ManualValidation(sample_id=key, validation=str(row["validation"]))
            db.add(manual)
        manual.validation = str(row["validation"])
        manual.notes = text_or_none(row.get("notes"))
        manual.updated_at = utc_now()
        upsert_manual_status(db, key, manual.validation, manual.notes)
        return

    if action == "manual_note_append":
        manual = db.get(ManualValidation, key)
        if manual is not None:
            manual.notes = payload.get("notes")
            manual.updated_at = utc_now()
            upsert_manual_status(db, key, manual.validation, manual.notes)
        return

    if action == "sample_status_insert":
        row = payload["row"]
        status = db.scalar(
            select(SampleStatus).where(
                SampleStatus.sample_id == row["sample_id"],
                SampleStatus.source == row["source"],
            )
        )
        if status is None:
            status = SampleStatus(
                sample_id=row["sample_id"],
                source=row["source"],
                status=row["status"],
            )
            db.add(status)
        status.status = row["status"]
        status.reason = row.get("reason")
        status.metrics_json = row.get("metrics_json")
        status.updated_at = utc_now()
        return

    if action == "sample_metadata_set":
        sample = db.get(Sample, key)
        if sample is not None:
            setattr(sample, payload["field"], payload.get("value"))
            sample.updated_at = utc_now()
        return

    if action == "sample_notes_append":
        sample = db.get(Sample, key)
        if sample is not None:
            sample.notes = payload.get("notes")
            sample.updated_at = utc_now()
        return

    if action == "sentinel_download_import":
        row = payload["row"]
        existing = db.get(SentinelDownload, key)
        if existing is None:
            existing = SentinelDownload(
                download_id=key,
                sample_id=row["sample_id"],
                scene_id=row.get("scene_id") or key,
                period=row.get("period"),
                local_path=row.get("local_path") or f"Scripts/ResearchTool/Data/cache/forest/sentinel/rasters/_missing/{key}.tif",
                bbox_json=row.get("bbox_json") or "[]",
                bands_json=row.get("bands_json") or "[]",
                metadata_json=row.get("metadata_json") or "{}",
            )
            db.add(existing)
        existing.cloud_fraction = row.get("cloud_fraction")
        existing.shadow_fraction = row.get("shadow_fraction")
        existing.nodata_fraction = row.get("nodata_fraction")
        existing.dark_fraction = row.get("dark_fraction")
        existing.is_bad_cloud = bool_value(row.get("is_bad_cloud"))
        existing.is_bad_quality = bool_value(row.get("is_bad_quality"))
        existing.quality_flags_json = row.get("quality_flags_json")
        return

    if action == "sentinel_preview_import":
        row = payload["row"]
        preview = db.get(DerivedPreview, row["preview_id"])
        if preview is None:
            db.add(
                DerivedPreview(
                    preview_id=row["preview_id"],
                    download_id=row["download_id"],
                    view=row["view"],
                    png_path=row["png_path"],
                    metadata_path=row["metadata_path"],
                )
            )
        return

    if action == "sentinel_review_insert":
        row = payload["row"]
        review = db.scalar(select(SentinelSceneReview).where(SentinelSceneReview.download_id == key))
        if review is None:
            review = SentinelSceneReview(download_id=key)
            db.add(review)
        review.is_excluded = bool_value(row.get("is_excluded"))
        review.reason_code = row.get("reason_code")
        review.reason_text = row.get("reason_text")
        review.notes = row.get("notes")
        review.updated_at = utc_now()
        for reason in row.get("reasons") or []:
            reason_code = reason.get("reason_code")
            if not reason_code:
                continue
            existing_reason = db.scalar(
                select(SentinelSceneReviewReason).where(
                    SentinelSceneReviewReason.download_id == key,
                    SentinelSceneReviewReason.reason_code == reason_code,
                )
            )
            if existing_reason is None:
                db.add(
                    SentinelSceneReviewReason(
                        download_id=key,
                        reason_code=reason_code,
                        reason_text=reason.get("reason_text"),
                    )
                )
        return

    if action == "sentinel_review_notes_append":
        review = db.scalar(select(SentinelSceneReview).where(SentinelSceneReview.download_id == key))
        if review is not None:
            review.notes = payload.get("notes")
            review.updated_at = utc_now()
        return

    if action == "sentinel_review_reason_insert":
        reason = payload["reason"]
        reason_code = reason.get("reason_code")
        existing_reason = db.scalar(
            select(SentinelSceneReviewReason).where(
                SentinelSceneReviewReason.download_id == key,
                SentinelSceneReviewReason.reason_code == reason_code,
            )
        )
        if existing_reason is None:
            db.add(
                SentinelSceneReviewReason(
                    download_id=key,
                    reason_code=reason_code,
                    reason_text=reason.get("reason_text"),
                )
            )
        return


def apply_db_sync_plan(db: Session, incoming_path: str, groups: list[str] | None = None) -> dict[str, Any]:
    plan = build_db_sync_plan(db, incoming_path)
    to_apply, skipped = selected_safe_operations(plan, groups)
    backup_path = create_backup()
    try:
        for item in to_apply:
            apply_operation(db, item)
        db.commit()
    except Exception:
        db.rollback()
        raise

    reports = write_report(plan, to_apply, skipped, backup_path)
    return {
        "incoming_path": plan["incoming_path"],
        "backup_path": to_storage_path(backup_path) or str(backup_path),
        "applied_count": len(to_apply),
        "skipped_count": len(skipped),
        "reports": reports,
        "totals": plan["totals"],
    }
