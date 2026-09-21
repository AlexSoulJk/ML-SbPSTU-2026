from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


# Edit only this path before running the script.
COLLEAGUE_DB_PATH = Path(r"Scripts\ResearchTool\Data\forest\bases\state_Оля.sqlite")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_DB_PATH = PROJECT_ROOT / "Scripts" / "ResearchTool" / "Data" / "forest" / "bases" /"state.sqlite"
OUTPUT_ROOT = Path(__file__).resolve().parent / "Outputs"

# Notes can be personal/noisy. Keep True while investigating, set False if you
# only care about validation labels.
COMPARE_MANUAL_NOTES = True


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def connect(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(f"Database does not exist: {db_path}")

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


def table_exists(con: sqlite3.Connection, table: str) -> bool:
    row = con.execute(
        "select 1 from sqlite_master where type = 'table' and name = ?",
        (table,),
    ).fetchone()
    return row is not None


def column_names(con: sqlite3.Connection, table: str) -> list[str]:
    if not table_exists(con, table):
        return []
    return [row["name"] for row in con.execute(f"pragma table_info({table})")]


def row_hash(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:12]


def is_abs_path(value: str | None) -> bool:
    if not value:
        return False
    return (
        Path(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or PurePosixPath(value).is_absolute()
    )


def path_kind(value: str | None) -> str:
    if not value:
        return ""
    return "absolute" if is_abs_path(value) else "relative"


def resolve_project_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if is_abs_path(value):
        return path
    return PROJECT_ROOT / path


def path_exists_here(value: str | None) -> bool | str:
    path = resolve_project_path(value)
    return path.exists() if path is not None else ""


def csv_safe(value: Any) -> Any:
    if value is None:
        return ""
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fetch_samples(con: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    if not table_exists(con, "forest_samples"):
        return {}

    cols = column_names(con, "forest_samples")
    wanted = [
        "sample_id",
        "display_name",
        "source_file",
        "source_id",
        "lat",
        "lon",
        "source_event_year",
        "driver_primary",
        "confidence_primary",
        "region",
    ]
    selected = [col for col in wanted if col in cols]
    rows = con.execute(f"select {', '.join(selected)} from forest_samples").fetchall()
    return {row["sample_id"]: dict(row) for row in rows}


def sample_fields(samples: dict[str, dict[str, Any]], sample_id: str) -> dict[str, Any]:
    sample = samples.get(sample_id, {})
    return {
        "sample_id": sample_id,
        "display_name": csv_safe(sample.get("display_name")),
        "source_file": csv_safe(sample.get("source_file")),
        "source_id": csv_safe(sample.get("source_id")),
        "lat": csv_safe(sample.get("lat")),
        "lon": csv_safe(sample.get("lon")),
        "source_event_year": csv_safe(sample.get("source_event_year")),
        "driver_primary": csv_safe(sample.get("driver_primary")),
        "confidence_primary": csv_safe(sample.get("confidence_primary")),
        "region": csv_safe(sample.get("region")),
    }


def fetch_manual_validations(con: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    if not table_exists(con, "forest_manual_validations"):
        return {}
    rows = con.execute(
        """
        select sample_id, validation, notes, updated_at
        from forest_manual_validations
        """
    ).fetchall()
    return {row["sample_id"]: dict(row) for row in rows}


def fetch_statuses(con: sqlite3.Connection) -> dict[tuple[str, str], dict[str, Any]]:
    if not table_exists(con, "forest_sample_statuses"):
        return {}
    rows = con.execute(
        """
        select sample_id, source, status, reason, metrics_json, updated_at
        from forest_sample_statuses
        """
    ).fetchall()
    return {(row["sample_id"], row["source"]): dict(row) for row in rows}


def fetch_hansen_analyses(con: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    if not table_exists(con, "forest_hansen_analyses"):
        return {}

    rows = con.execute(
        """
        select sample_id,
               analysis_id,
               analysis_area_type,
               plot_size_m,
               treecover_threshold,
               event_year,
               total_loss_area_ha,
               dominant_loss_area_ha,
               dominant_year_share,
               status,
               histogram_json,
               created_at
        from forest_hansen_analyses
        order by sample_id, created_at desc
        """
    ).fetchall()

    by_sample: dict[str, dict[str, Any]] = {}
    duplicates: dict[str, int] = {}
    for row in rows:
        sample_id = row["sample_id"]
        duplicates[sample_id] = duplicates.get(sample_id, 0) + 1
        if sample_id not in by_sample:
            by_sample[sample_id] = dict(row)

    for sample_id, count in duplicates.items():
        by_sample[sample_id]["analysis_count"] = count
    return by_sample


def fetch_hansen_tiles(con: sqlite3.Connection) -> dict[tuple[str, str, str], dict[str, Any]]:
    if not table_exists(con, "forest_hansen_tiles"):
        return {}
    rows = con.execute(
        """
        select layer, tile_id, version, local_path, downloaded_at
        from forest_hansen_tiles
        """
    ).fetchall()
    return {(row["layer"], row["tile_id"], row["version"]): dict(row) for row in rows}


def count_by(rows: dict[Any, dict[str, Any]], field: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows.values():
        key = str(row.get(field) or "")
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items()))


def compare_manual(
    local: dict[str, dict[str, Any]],
    colleague: dict[str, dict[str, Any]],
    samples: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    summary = {"same": 0, "only_local": 0, "only_colleague": 0, "different": 0}

    for sample_id in sorted(set(local) | set(colleague)):
        left = local.get(sample_id)
        right = colleague.get(sample_id)
        if left is None:
            diff_type = "only_colleague"
        elif right is None:
            diff_type = "only_local"
        else:
            same_validation = left["validation"] == right["validation"]
            same_notes = (left.get("notes") or "") == (right.get("notes") or "")
            diff_type = "same" if same_validation and (same_notes or not COMPARE_MANUAL_NOTES) else "different"

        summary[diff_type] += 1
        if diff_type == "same":
            continue

        rows.append(
            {
                **sample_fields(samples, sample_id),
                "diff_type": diff_type,
                "local_validation": csv_safe(left.get("validation") if left else None),
                "colleague_validation": csv_safe(right.get("validation") if right else None),
                "local_notes": csv_safe(left.get("notes") if left else None),
                "colleague_notes": csv_safe(right.get("notes") if right else None),
                "local_updated_at": csv_safe(left.get("updated_at") if left else None),
                "colleague_updated_at": csv_safe(right.get("updated_at") if right else None),
            }
        )

    return rows, summary


def compare_statuses(
    local: dict[tuple[str, str], dict[str, Any]],
    colleague: dict[tuple[str, str], dict[str, Any]],
    samples: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    summary = {"same": 0, "only_local": 0, "only_colleague": 0, "different": 0}

    for key in sorted(set(local) | set(colleague)):
        sample_id, source = key
        left = local.get(key)
        right = colleague.get(key)
        if left is None:
            diff_type = "only_colleague"
        elif right is None:
            diff_type = "only_local"
        else:
            same = (
                left.get("status") == right.get("status")
                and (left.get("reason") or "") == (right.get("reason") or "")
                and row_hash(left.get("metrics_json")) == row_hash(right.get("metrics_json"))
            )
            diff_type = "same" if same else "different"

        summary[diff_type] += 1
        if diff_type == "same":
            continue

        rows.append(
            {
                **sample_fields(samples, sample_id),
                "source": source,
                "diff_type": diff_type,
                "local_status": csv_safe(left.get("status") if left else None),
                "colleague_status": csv_safe(right.get("status") if right else None),
                "local_reason": csv_safe(left.get("reason") if left else None),
                "colleague_reason": csv_safe(right.get("reason") if right else None),
                "local_metrics_hash": row_hash(left.get("metrics_json") if left else None),
                "colleague_metrics_hash": row_hash(right.get("metrics_json") if right else None),
                "local_updated_at": csv_safe(left.get("updated_at") if left else None),
                "colleague_updated_at": csv_safe(right.get("updated_at") if right else None),
            }
        )

    return rows, summary


def compare_hansen_analyses(
    local: dict[str, dict[str, Any]],
    colleague: dict[str, dict[str, Any]],
    samples: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    summary = {"same": 0, "only_local": 0, "only_colleague": 0, "different": 0}

    fields = [
        "analysis_area_type",
        "plot_size_m",
        "treecover_threshold",
        "event_year",
        "total_loss_area_ha",
        "dominant_loss_area_ha",
        "dominant_year_share",
        "status",
        "analysis_count",
    ]

    for sample_id in sorted(set(local) | set(colleague)):
        left = local.get(sample_id)
        right = colleague.get(sample_id)
        if left is None:
            diff_type = "only_colleague"
        elif right is None:
            diff_type = "only_local"
        else:
            same = all(left.get(field) == right.get(field) for field in fields)
            same = same and row_hash(left.get("histogram_json")) == row_hash(right.get("histogram_json"))
            diff_type = "same" if same else "different"

        summary[diff_type] += 1
        if diff_type == "same":
            continue

        row = {
            **sample_fields(samples, sample_id),
            "diff_type": diff_type,
            "local_analysis_id": csv_safe(left.get("analysis_id") if left else None),
            "colleague_analysis_id": csv_safe(right.get("analysis_id") if right else None),
            "local_created_at": csv_safe(left.get("created_at") if left else None),
            "colleague_created_at": csv_safe(right.get("created_at") if right else None),
            "local_histogram_hash": row_hash(left.get("histogram_json") if left else None),
            "colleague_histogram_hash": row_hash(right.get("histogram_json") if right else None),
        }
        for field in fields:
            row[f"local_{field}"] = csv_safe(left.get(field) if left else None)
            row[f"colleague_{field}"] = csv_safe(right.get(field) if right else None)
        rows.append(row)

    return rows, summary


def compare_hansen_tiles(
    local: dict[tuple[str, str, str], dict[str, Any]],
    colleague: dict[tuple[str, str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    summary = {"same": 0, "only_local": 0, "only_colleague": 0, "different": 0}

    for layer, tile_id, version in sorted(set(local) | set(colleague)):
        left = local.get((layer, tile_id, version))
        right = colleague.get((layer, tile_id, version))
        if left is None:
            diff_type = "only_colleague"
        elif right is None:
            diff_type = "only_local"
        else:
            same = (left.get("local_path") or "") == (right.get("local_path") or "")
            diff_type = "same" if same else "different"

        summary[diff_type] += 1
        if diff_type == "same":
            continue

        local_path = left.get("local_path") if left else None
        colleague_path = right.get("local_path") if right else None
        rows.append(
            {
                "diff_type": diff_type,
                "layer": layer,
                "tile_id": tile_id,
                "version": version,
                "local_path": csv_safe(local_path),
                "colleague_path": csv_safe(colleague_path),
                "local_path_kind": path_kind(local_path),
                "colleague_path_kind": path_kind(colleague_path),
                "local_file_exists_here": path_exists_here(local_path),
                "colleague_file_exists_here": path_exists_here(colleague_path),
                "local_downloaded_at": csv_safe(left.get("downloaded_at") if left else None),
                "colleague_downloaded_at": csv_safe(right.get("downloaded_at") if right else None),
            }
        )

    return rows, summary


def table_count(con: sqlite3.Connection, table: str) -> int | None:
    if not table_exists(con, table):
        return None
    return int(con.execute(f"select count(*) from {table}").fetchone()[0])


def main() -> None:
    output_dir = OUTPUT_ROOT / f"forest_db_compare_{now_tag()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with connect(LOCAL_DB_PATH) as local_con, connect(COLLEAGUE_DB_PATH) as colleague_con:
        local_samples = fetch_samples(local_con)
        colleague_samples = fetch_samples(colleague_con)
        samples = {**colleague_samples, **local_samples}

        local_manual = fetch_manual_validations(local_con)
        colleague_manual = fetch_manual_validations(colleague_con)
        manual_rows, manual_summary = compare_manual(local_manual, colleague_manual, samples)

        local_statuses = fetch_statuses(local_con)
        colleague_statuses = fetch_statuses(colleague_con)
        status_rows, status_summary = compare_statuses(local_statuses, colleague_statuses, samples)

        local_analyses = fetch_hansen_analyses(local_con)
        colleague_analyses = fetch_hansen_analyses(colleague_con)
        analysis_rows, analysis_summary = compare_hansen_analyses(local_analyses, colleague_analyses, samples)

        local_tiles = fetch_hansen_tiles(local_con)
        colleague_tiles = fetch_hansen_tiles(colleague_con)
        tile_rows, tile_summary = compare_hansen_tiles(local_tiles, colleague_tiles)

        write_csv(
            output_dir / "manual_validations_diff.csv",
            manual_rows,
            [
                "diff_type",
                "sample_id",
                "display_name",
                "source_file",
                "source_id",
                "lat",
                "lon",
                "source_event_year",
                "driver_primary",
                "confidence_primary",
                "region",
                "local_validation",
                "colleague_validation",
                "local_notes",
                "colleague_notes",
                "local_updated_at",
                "colleague_updated_at",
            ],
        )
        write_csv(
            output_dir / "sample_statuses_diff.csv",
            status_rows,
            [
                "diff_type",
                "source",
                "sample_id",
                "display_name",
                "source_file",
                "source_id",
                "lat",
                "lon",
                "source_event_year",
                "driver_primary",
                "confidence_primary",
                "region",
                "local_status",
                "colleague_status",
                "local_reason",
                "colleague_reason",
                "local_metrics_hash",
                "colleague_metrics_hash",
                "local_updated_at",
                "colleague_updated_at",
            ],
        )
        analysis_fields = [
            "analysis_area_type",
            "plot_size_m",
            "treecover_threshold",
            "event_year",
            "total_loss_area_ha",
            "dominant_loss_area_ha",
            "dominant_year_share",
            "status",
            "analysis_count",
        ]
        write_csv(
            output_dir / "hansen_analyses_diff.csv",
            analysis_rows,
            [
                "diff_type",
                "sample_id",
                "display_name",
                "source_file",
                "source_id",
                "lat",
                "lon",
                "source_event_year",
                "driver_primary",
                "confidence_primary",
                "region",
                "local_analysis_id",
                "colleague_analysis_id",
                "local_created_at",
                "colleague_created_at",
                "local_histogram_hash",
                "colleague_histogram_hash",
                *[f"{side}_{field}" for field in analysis_fields for side in ("local", "colleague")],
            ],
        )
        write_csv(
            output_dir / "hansen_tiles_diff.csv",
            tile_rows,
            [
                "diff_type",
                "layer",
                "tile_id",
                "version",
                "local_path",
                "colleague_path",
                "local_path_kind",
                "colleague_path_kind",
                "local_file_exists_here",
                "colleague_file_exists_here",
                "local_downloaded_at",
                "colleague_downloaded_at",
            ],
        )

        summary = {
            "local_db": str(LOCAL_DB_PATH),
            "colleague_db": str(COLLEAGUE_DB_PATH),
            "output_dir": str(output_dir),
            "tables": {
                table: {
                    "local_count": table_count(local_con, table),
                    "colleague_count": table_count(colleague_con, table),
                }
                for table in [
                    "forest_samples",
                    "forest_manual_validations",
                    "forest_sample_statuses",
                    "forest_hansen_analyses",
                    "forest_hansen_tiles",
                    "forest_sentinel_downloads",
                    "forest_derived_previews",
                ]
            },
            "manual_validation_counts": {
                "local": count_by(local_manual, "validation"),
                "colleague": count_by(colleague_manual, "validation"),
            },
            "hansen_status_counts": {
                "local": count_by(local_analyses, "status"),
                "colleague": count_by(colleague_analyses, "status"),
            },
            "hansen_tile_layer_counts": {
                "local": count_by(local_tiles, "layer"),
                "colleague": count_by(colleague_tiles, "layer"),
            },
            "diffs": {
                "manual_validations": manual_summary,
                "sample_statuses": status_summary,
                "hansen_analyses": analysis_summary,
                "hansen_tiles": tile_summary,
            },
        }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Local DB: {LOCAL_DB_PATH}")
    print(f"Colleague DB: {COLLEAGUE_DB_PATH}")
    print(f"Output: {output_dir}")
    print("")
    for name, values in summary["diffs"].items():
        print(
            f"{name}: "
            f"same={values['same']} "
            f"only_local={values['only_local']} "
            f"only_colleague={values['only_colleague']} "
            f"different={values['different']}"
        )
    print("")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
