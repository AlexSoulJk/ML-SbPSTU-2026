from __future__ import annotations

import csv
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath


DRY_RUN = True

# First real-run scope: cache/artifact paths that affect Hansen/Sentinel UI.
UPDATE_IMPORT_EXPORTS = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "Scripts" / "ResearchTool" / "Data" / "forest" / "state.sqlite"
OUTPUT_ROOT = PROJECT_ROOT / "Scripts" / "ResearchScripts" / "Outputs"


@dataclass(frozen=True)
class PathField:
    table: str
    primary_key: str
    column: str
    priority: int


PATH_FIELDS = [
    PathField("forest_hansen_tiles", "tile_id", "local_path", 1),
    PathField("forest_hansen_masks", "mask_id", "local_path", 1),
    PathField("forest_sentinel_searches", "search_id", "cache_path", 1),
    PathField("forest_sentinel_downloads", "download_id", "local_path", 1),
    PathField("forest_derived_previews", "preview_id", "png_path", 1),
    PathField("forest_derived_previews", "preview_id", "metadata_path", 1),
    PathField("forest_exports", "export_id", "csv_path", 2),
    PathField("forest_exports", "export_id", "geojson_path", 2),
    PathField("forest_import_batches", "import_id", "stored_path", 2),
]


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def output_dir() -> Path:
    path = OUTPUT_ROOT / f"forest_path_repair_{now_tag()}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_relative(path: Path | PureWindowsPath | PurePosixPath) -> str:
    return PurePosixPath(*path.parts).as_posix()


def is_absolute_path(value: str) -> bool:
    return (
        Path(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or PurePosixPath(value).is_absolute()
    )


def relative_to_project(value: str) -> str | None:
    if not is_absolute_path(value):
        return normalize_relative(Path(value))

    path = Path(value)
    try:
        resolved = path.resolve(strict=False)
        return normalize_relative(resolved.relative_to(PROJECT_ROOT.resolve(strict=False)))
    except (OSError, ValueError):
        pass

    # Fallback for Windows-style paths parsed on non-Windows systems.
    project_win = PureWindowsPath(str(PROJECT_ROOT.resolve(strict=False)))
    value_win = PureWindowsPath(value)
    try:
        return normalize_relative(value_win.relative_to(project_win))
    except ValueError:
        return None


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def table_exists(connection: sqlite3.Connection, table: str) -> bool:
    row = connection.execute(
        "select 1 from sqlite_master where type = 'table' and name = ?",
        (table,),
    ).fetchone()
    return row is not None


def table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in connection.execute(f"pragma table_info({quote_identifier(table)})")}


def action_for(field: PathField, value: str | None) -> tuple[str, str | None, str]:
    if value is None or not str(value).strip():
        return "skip", None, "empty"

    raw = str(value)
    if not is_absolute_path(raw):
        return "skip", raw, "already_relative"

    new_path = relative_to_project(raw)
    if new_path is None:
        return "skip", None, "absolute_outside_project"

    if new_path == raw:
        return "skip", new_path, "no_change"

    if field.priority > 1 and not UPDATE_IMPORT_EXPORTS:
        return "report_only", new_path, "priority_2_report_only"

    return "would_update" if DRY_RUN else "updated", new_path, "absolute_inside_project"


def repair_field(connection: sqlite3.Connection, field: PathField) -> list[dict[str, object]]:
    if not table_exists(connection, field.table):
        return [
            {
                "table": field.table,
                "primary_key": "",
                "column": field.column,
                "old_path": "",
                "new_path": "",
                "action": "skip",
                "reason": "missing_table",
            }
        ]

    columns = table_columns(connection, field.table)
    if field.primary_key not in columns or field.column not in columns:
        return [
            {
                "table": field.table,
                "primary_key": "",
                "column": field.column,
                "old_path": "",
                "new_path": "",
                "action": "skip",
                "reason": "missing_column",
            }
        ]

    table = quote_identifier(field.table)
    primary_key = quote_identifier(field.primary_key)
    column = quote_identifier(field.column)
    rows = connection.execute(
        f"select {primary_key}, {column} from {table}"
    ).fetchall()

    report_rows: list[dict[str, object]] = []
    for pk_value, old_path in rows:
        action, new_path, reason = action_for(field, old_path)
        report_rows.append(
            {
                "table": field.table,
                "primary_key": pk_value,
                "column": field.column,
                "old_path": old_path or "",
                "new_path": new_path or "",
                "action": action,
                "reason": reason,
            }
        )
        if action == "updated" and new_path is not None:
            connection.execute(
                f"update {table} set {column} = ? where {primary_key} = ?",
                (new_path, pk_value),
            )

    return report_rows


def write_report(rows: list[dict[str, object]], report_path: Path) -> None:
    fieldnames = [
        "table",
        "primary_key",
        "column",
        "old_path",
        "new_path",
        "action",
        "reason",
    ]
    with report_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    run_dir = output_dir()
    report_path = run_dir / "path_repair_report.csv"
    backup_path = run_dir / "state.sqlite.bak"

    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database was not found: {DB_PATH}")

    if not DRY_RUN:
        shutil.copy2(DB_PATH, backup_path)

    rows: list[dict[str, object]] = []
    with sqlite3.connect(DB_PATH) as connection:
        for field in PATH_FIELDS:
            rows.extend(repair_field(connection, field))
        if DRY_RUN:
            connection.rollback()
        else:
            connection.commit()

    write_report(rows, report_path)
    action_counts: dict[str, int] = {}
    for row in rows:
        action = str(row["action"])
        action_counts[action] = action_counts.get(action, 0) + 1

    print(f"DB: {DB_PATH}")
    print(f"DRY_RUN: {DRY_RUN}")
    print(f"UPDATE_IMPORT_EXPORTS: {UPDATE_IMPORT_EXPORTS}")
    if not DRY_RUN:
        print(f"Backup: {backup_path}")
    print(f"Report: {report_path}")
    print(f"Actions: {action_counts}")


if __name__ == "__main__":
    main()
