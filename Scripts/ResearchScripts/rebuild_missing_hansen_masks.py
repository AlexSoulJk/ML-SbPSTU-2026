from __future__ import annotations

import csv
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select


DRY_RUN = True
INCLUDE_TREECOVER = False
ENSURE_SCHEMA = False

# Set to an integer for a small test batch, or None for all affected samples.
LIMIT_SAMPLES = None

# Leave empty for all samples, or set prefixes for a targeted repair.
SAMPLE_ID_PREFIXES: tuple[str, ...] = ()


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOL_ROOT = PROJECT_ROOT / "Scripts" / "ResearchTool"
sys.path.insert(0, str(TOOL_ROOT))

from backend.config import ensure_directories, load_dotenv  # noqa: E402
from backend.forest.database import SessionLocal, init_forest_database  # noqa: E402
from backend.forest.hansen_service import analyze_sample_hansen  # noqa: E402
from backend.forest.models import HansenAnalysis, HansenMask, Sample  # noqa: E402
from backend.forest.storage_paths import resolve_storage_path  # noqa: E402


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


def latest_analysis(db, sample_id: str) -> HansenAnalysis | None:
    return db.scalar(
        select(HansenAnalysis)
        .where(HansenAnalysis.sample_id == sample_id)
        .order_by(HansenAnalysis.created_at.desc())
        .limit(1)
    )


def masks_for_analysis(db, analysis_id: str) -> list[HansenMask]:
    return list(
        db.scalars(
            select(HansenMask)
            .where(HansenMask.analysis_id == analysis_id)
            .order_by(HansenMask.mask_type)
        )
    )


def sample_rebuild_reason(db, sample: Sample) -> tuple[str, dict[str, Any]]:
    analysis = latest_analysis(db, sample.sample_id)
    if analysis is None:
        return "no_analysis", {
            "analysis_id": "",
            "missing_tif_count": "",
            "missing_tif_paths": "",
        }

    masks = masks_for_analysis(db, analysis.analysis_id)
    if not masks:
        return "no_masks", {
            "analysis_id": analysis.analysis_id,
            "missing_tif_count": "",
            "missing_tif_paths": "",
        }

    missing_paths = [
        display_path(resolve_storage_path(mask.local_path))
        for mask in masks
        if not resolve_storage_path(mask.local_path).exists()
    ]
    if missing_paths:
        return "missing_tif", {
            "analysis_id": analysis.analysis_id,
            "missing_tif_count": len(missing_paths),
            "missing_tif_paths": " | ".join(missing_paths[:8]),
        }

    return "ok", {
        "analysis_id": analysis.analysis_id,
        "missing_tif_count": 0,
        "missing_tif_paths": "",
    }


def selected_samples(db) -> list[Sample]:
    samples = list(db.scalars(select(Sample).order_by(Sample.source_file, Sample.source_id, Sample.sample_id)))
    if SAMPLE_ID_PREFIXES:
        samples = [
            sample for sample in samples
            if any(sample.sample_id.startswith(prefix) for prefix in SAMPLE_ID_PREFIXES)
        ]
    return samples


def write_report(rows: list[dict[str, Any]]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / f"hansen_mask_rebuild_{now_tag()}.csv"
    fieldnames = [
        "sample_id",
        "source_id",
        "display_name",
        "reason",
        "analysis_id",
        "missing_tif_count",
        "missing_tif_paths",
        "action",
        "new_analysis_id",
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

    rows: list[dict[str, Any]] = []
    with SessionLocal() as db:
        samples = selected_samples(db)
        for sample in samples:
            reason, details = sample_rebuild_reason(db, sample)
            action = "skip_ok" if reason == "ok" else ("would_rebuild" if DRY_RUN else "pending")
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "source_id": sample.source_id,
                    "display_name": sample.display_name or "",
                    "reason": reason,
                    "analysis_id": details["analysis_id"],
                    "missing_tif_count": details["missing_tif_count"],
                    "missing_tif_paths": details["missing_tif_paths"],
                    "action": action,
                    "new_analysis_id": "",
                    "error": "",
                }
            )

        work_rows = [row for row in rows if row["reason"] != "ok"]
        if LIMIT_SAMPLES is not None:
            work_rows = work_rows[:LIMIT_SAMPLES]

        reason_counts = Counter(row["reason"] for row in rows)
        print(f"Samples in DB: {len(rows)}")
        print(f"Samples to rebuild: {len(work_rows)}")
        print(f"Reasons: {dict(sorted(reason_counts.items()))}")
        print(f"INCLUDE_TREECOVER: {INCLUDE_TREECOVER}")
        print(f"LIMIT_SAMPLES: {LIMIT_SAMPLES}")
        print(f"DRY_RUN: {DRY_RUN}")

        if not DRY_RUN:
            row_by_sample = {row["sample_id"]: row for row in work_rows}
            samples_by_id = {sample.sample_id: sample for sample in samples}
            for index, row in enumerate(work_rows, start=1):
                sample = samples_by_id[row["sample_id"]]
                try:
                    payload = analyze_sample_hansen(
                        db,
                        sample,
                        include_treecover=INCLUDE_TREECOVER,
                        skip_existing=False,
                    )
                    row_by_sample[sample.sample_id]["new_analysis_id"] = payload.get("analysis_id", "")
                    row_by_sample[sample.sample_id]["action"] = "rebuilt"
                    print(f"[{index}/{len(work_rows)}] rebuilt {sample.sample_id[:8]}")
                except Exception as exc:
                    db.rollback()
                    row_by_sample[sample.sample_id]["action"] = "failed"
                    row_by_sample[sample.sample_id]["error"] = str(exc)[:500]
                    print(f"[{index}/{len(work_rows)}] failed {sample.sample_id[:8]}: {exc}")

    report_path = write_report(rows)
    print(f"Report: {report_path}")
    if DRY_RUN:
        print("Dry run only. Set DRY_RUN = False to rebuild missing Hansen sample masks.")


if __name__ == "__main__":
    main()
