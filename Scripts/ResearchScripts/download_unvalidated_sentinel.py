from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from sqlalchemy import or_, select


# Edit these values in the file and run it again. No CLI arguments are required.
DRY_RUN = False
ENSURE_SCHEMA = True

LIMIT_SAMPLES = 100
DRIVERS = ("Logging", "Wildfire")
SAMPLE_ID_PREFIXES: tuple[str, ...] = ()
MODE = "empty_samples"
SEASONS = ("spring", "early_summer", "late_summer", "autumn")

TARGET_VALID_SCENES = 10
MAX_QC_SCENES = 50
MIN_DAYS_BETWEEN_ACCEPTED = 7
STAC_MAX_CLOUD = 30.0
LOCAL_CLOUD_THRESHOLD = 0.10
SEARCH_TOP_N_PER_WINDOW = 40


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOL_ROOT = PROJECT_ROOT / "Scripts" / "ResearchTool"
sys.path.insert(0, str(TOOL_ROOT))

from backend.config import ensure_directories, load_dotenv  # noqa: E402
from backend.forest.database import SessionLocal, init_forest_database  # noqa: E402
from backend.forest.event_service import backfill_hansen_events  # noqa: E402
from backend.forest.models import Sample  # noqa: E402
from backend.forest.sentinel_batch_service import (  # noqa: E402
    BatchOptions,
    execute_sentinel_batch,
    plan_sentinel_batch,
)


def selected_samples(db) -> list[Sample]:
    stmt = (
        select(Sample)
        .where(
            or_(
                Sample.driver_primary.in_(DRIVERS),
                Sample.driver_secondary.in_(DRIVERS),
            )
        )
        .order_by(Sample.source_file, Sample.source_id, Sample.sample_id)
    )
    samples = list(db.scalars(stmt))
    if SAMPLE_ID_PREFIXES:
        samples = [
            sample
            for sample in samples
            if any(sample.sample_id.startswith(prefix) for prefix in SAMPLE_ID_PREFIXES)
        ]
    return samples[:LIMIT_SAMPLES]


def main() -> None:
    load_dotenv()
    ensure_directories()
    if ENSURE_SCHEMA:
        init_forest_database()

    options = BatchOptions(
        mode=MODE,
        seasons=SEASONS,
        target_valid=TARGET_VALID_SCENES,
        max_qc_scenes=MAX_QC_SCENES,
        min_days=MIN_DAYS_BETWEEN_ACCEPTED,
        stac_max_cloud=STAC_MAX_CLOUD,
        local_cloud_threshold=LOCAL_CLOUD_THRESHOLD,
        search_top_n=SEARCH_TOP_N_PER_WINDOW,
    )

    with SessionLocal() as db:
        samples = selected_samples(db)
        event_sync = backfill_hansen_events(db, dry_run=DRY_RUN)
        plan = plan_sentinel_batch(db, samples, options)

        print(f"DRY_RUN: {DRY_RUN}")
        print(f"Selected samples: {len(samples)}")
        print(f"Mode: {MODE}")
        print(f"Drivers: {', '.join(DRIVERS)}")
        print(f"Seasons: {', '.join(SEASONS)}")
        print(f"Event rows: {event_sync['event_rows']}")
        print(f"Plan summary: {plan['summary']}")
        print(f"Skipped: {dict(Counter(row['reason'] for row in plan['skipped']))}")

        if DRY_RUN:
            print(
                "Dry run only. Existing event rows are used for the Sentinel plan; "
                "set DRY_RUN = False to sync missing events and download imagery."
            )
            return

        actionable_ids = {
            row["sample_id"]
            for row in plan["samples"]
            if row["stop_reason"] is None
        }
        result = execute_sentinel_batch(
            db,
            [sample for sample in samples if sample.sample_id in actionable_ids],
            options,
            progress=lambda stage, message, done, total: print(
                f"[{done}/{total}] {stage}: {message}"
            ),
        )
        print(f"Actions: {result['actions']}")
        print(f"Report: {result['operations_csv']}")
        print(f"Bundle selection: {result['bundle_selection']}")


if __name__ == "__main__":
    main()
