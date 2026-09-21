from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

from sqlalchemy import select


DRY_RUN = False
ENSURE_SCHEMA = False

# By default we warm only the layer needed for event-year analysis.
# Add "treecover2000" here only if you want the optional treecover layer cached too.
LAYERS = ("lossyear",)

# Set to an integer for a small test batch, or None for all missing tiles.
LIMIT_TILES = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOL_ROOT = PROJECT_ROOT / "Scripts" / "ResearchTool"
sys.path.insert(0, str(TOOL_ROOT))

from backend.forest.database import SessionLocal, init_forest_database
from backend.forest.hansen_provider import (
    HANSEN_VERSION,
    download_hansen_tile,
    hansen_tile_path,
    tile_id_for_lonlat,
)
from backend.forest.models import HansenTile, Sample
from backend.forest.storage_paths import to_storage_path


SCRIPT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_ROOT / "Outputs"


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def existing_tile_record(db, layer: str, tile_id: str) -> HansenTile | None:
    return db.scalar(
        select(HansenTile).where(
            HansenTile.layer == layer,
            HansenTile.tile_id == tile_id,
            HansenTile.version == HANSEN_VERSION,
        )
    )


def upsert_tile_record(db, layer: str, tile_id: str, local_path: str) -> None:
    tile = existing_tile_record(db, layer, tile_id)
    if tile is None:
        db.add(
            HansenTile(
                layer=layer,
                tile_id=tile_id,
                version=HANSEN_VERSION,
                local_path=local_path,
            )
        )
        return
    tile.local_path = local_path


def unique_sample_tiles(db) -> dict[str, int]:
    counts: dict[str, int] = {}
    for lon, lat in db.execute(select(Sample.lon, Sample.lat)):
        tile_id = tile_id_for_lonlat(float(lon), float(lat))
        counts[tile_id] = counts.get(tile_id, 0) + 1
    return dict(sorted(counts.items()))


def planned_tile_rows(db) -> list[dict[str, object]]:
    sample_counts = unique_sample_tiles(db)
    rows: list[dict[str, object]] = []

    for tile_id, sample_count in sample_counts.items():
        for layer in LAYERS:
            expected_path = Path(hansen_tile_path(layer, tile_id))
            record = existing_tile_record(db, layer, tile_id)
            file_exists = expected_path.exists()
            record_exists = record is not None

            if file_exists and not record_exists and not DRY_RUN:
                upsert_tile_record(db, layer, tile_id, to_storage_path(expected_path) or str(expected_path))

            action = "skip_cached" if file_exists else "download"
            if file_exists and not record_exists:
                action = "sync_record" if not DRY_RUN else "would_sync_record"

            rows.append(
                {
                    "tile_id": tile_id,
                    "layer": layer,
                    "sample_count": sample_count,
                    "file_exists": file_exists,
                    "record_exists": record_exists,
                    "local_path": to_storage_path(expected_path) or str(expected_path),
                    "action": action,
                }
            )

    return rows


def write_report(rows: list[dict[str, object]]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / f"hansen_tile_prefetch_{now_tag()}.csv"
    fieldnames = [
        "tile_id",
        "layer",
        "sample_count",
        "file_exists",
        "record_exists",
        "local_path",
        "action",
    ]
    with report_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return report_path


def progress(stage: str, message: str) -> None:
    print(f"[{stage}] {message}")


def main() -> None:
    if ENSURE_SCHEMA:
        init_forest_database()

    with SessionLocal() as db:
        rows = planned_tile_rows(db)
        missing_rows = [row for row in rows if row["action"] == "download"]
        if LIMIT_TILES is not None:
            missing_rows = missing_rows[:LIMIT_TILES]

        print(f"Hansen version: {HANSEN_VERSION}")
        print(f"Layers: {', '.join(LAYERS)}")
        print(f"Unique tile/layer rows: {len(rows)}")
        print(f"Missing tile/layer rows to download: {len(missing_rows)}")
        print(f"DRY_RUN: {DRY_RUN}")

        if DRY_RUN:
            report_path = write_report(rows)
            print(f"Report: {report_path}")
            print("Dry run only. Set DRY_RUN = False to download missing tiles.")
            return

        downloaded = 0
        failed = 0
        for row in missing_rows:
            layer = str(row["layer"])
            tile_id = str(row["tile_id"])
            try:
                local_path = download_hansen_tile(layer, tile_id, progress=progress)
                upsert_tile_record(db, layer, tile_id, local_path)
                db.commit()
                row["file_exists"] = True
                row["record_exists"] = True
                row["local_path"] = local_path
                row["action"] = "downloaded"
                downloaded += 1
            except Exception as exc:
                db.rollback()
                row["action"] = f"failed: {str(exc)[:300]}"
                failed += 1
                print(f"[failed] {layer} {tile_id}: {exc}")

        # Also persist records for files that already existed but were missing DB records.
        db.commit()
        report_path = write_report(rows)
        print(f"Downloaded: {downloaded}")
        print(f"Failed: {failed}")
        print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
