from __future__ import annotations

import csv
import sys
from collections import Counter
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import rasterio
from sqlalchemy import select


DRY_RUN = True
FORCE_REWRITE = False
REPAIR_TIF_PATHS = True

# Set to None for all mask types, or restrict to a subset.
MASK_TYPES: set[str] | None = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "Scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from ResearchTool.backend.forest.database import SessionLocal  # noqa: E402
from ResearchTool.backend.forest.hansen_service import (  # noqa: E402
    rgba_binary,
    rgba_loss_year,
    rgba_treecover,
    write_png,
)
from ResearchTool.backend.config import FOREST_HANSEN_CACHE_DIR  # noqa: E402
from ResearchTool.backend.forest.models import HansenMask  # noqa: E402
from ResearchTool.backend.forest.storage_paths import resolve_storage_path, to_storage_path  # noqa: E402


SCRIPT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_ROOT / "Outputs"


LOSS_YEAR_TYPES = {"all_loss", "aoi_all_loss"}
DOMINANT_TYPES = {"dominant_year", "aoi_dominant_year"}
TREECOVER_TYPES = {"treecover2000", "aoi_treecover2000"}
MaskIndex = tuple[
    dict[tuple[str, str, str], Path],
    dict[tuple[str, str], list[Path]],
]


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def display_path(path: Path) -> str:
    try:
        return str(path.resolve(strict=False).relative_to(PROJECT_ROOT.resolve(strict=False)))
    except ValueError:
        return str(path)


def png_for_mask(mask_type: str, data):
    if mask_type in LOSS_YEAR_TYPES:
        return rgba_loss_year(data)
    if mask_type in DOMINANT_TYPES:
        return rgba_binary(data > 0, (220, 70, 45))
    if mask_type in TREECOVER_TYPES:
        return rgba_treecover(data)
    raise ValueError(f"Unsupported Hansen mask type: {mask_type}")


def build_existing_mask_index() -> MaskIndex:
    root = FOREST_HANSEN_CACHE_DIR / "masks"
    exact: dict[tuple[str, str, str], Path] = {}
    loose: dict[tuple[str, str], list[Path]] = defaultdict(list)
    if not root.exists():
        return exact, loose

    for tif_path in root.rglob("*.tif"):
        mask_type = tif_path.stem
        analysis_id = tif_path.parent.name
        sample_id = tif_path.parent.parent.name if tif_path.parent.parent != root else ""
        if not sample_id or not analysis_id:
            continue
        exact[(sample_id, analysis_id, mask_type)] = tif_path
        loose[(sample_id, mask_type)].append(tif_path)
    return exact, loose


def find_existing_tif(mask: HansenMask, index: MaskIndex) -> tuple[Path | None, str]:
    exact, loose = index
    candidate = exact.get((mask.sample_id, mask.analysis_id, mask.mask_type))
    if candidate and candidate.exists():
        return candidate, "exact_sample_analysis_type"

    candidates = [path for path in loose.get((mask.sample_id, mask.mask_type), []) if path.exists()]
    if len(candidates) == 1:
        return candidates[0], "unique_sample_type"
    if len(candidates) > 1:
        return None, "ambiguous_sample_type"
    return None, "not_found"


def repair_mask(mask: HansenMask, index: MaskIndex) -> dict[str, object]:
    tif_path = resolve_storage_path(mask.local_path)
    png_path = tif_path.with_suffix(".png")
    row = {
        "mask_id": mask.mask_id,
        "sample_id": mask.sample_id,
        "analysis_id": mask.analysis_id,
        "mask_type": mask.mask_type,
        "tif_path": display_path(tif_path),
        "candidate_tif_path": "",
        "candidate_match": "",
        "png_path": display_path(png_path),
        "tif_exists": tif_path.exists(),
        "png_exists": png_path.exists(),
        "action": "",
        "error": "",
    }

    if MASK_TYPES is not None and mask.mask_type not in MASK_TYPES:
        row["action"] = "skip_mask_type"
        return row
    if not tif_path.exists():
        candidate, match_reason = find_existing_tif(mask, index)
        row["candidate_match"] = match_reason
        if candidate is None or not REPAIR_TIF_PATHS:
            row["action"] = "missing_tif"
            return row

        row["candidate_tif_path"] = display_path(candidate)
        if DRY_RUN:
            row["action"] = "would_update_tif_path"
            return row

        mask.local_path = to_storage_path(candidate) or str(candidate)
        tif_path = candidate
        png_path = tif_path.with_suffix(".png")
        row["tif_path"] = display_path(tif_path)
        row["png_path"] = display_path(png_path)
        row["tif_exists"] = True
        row["png_exists"] = png_path.exists()
        row["action"] = "updated_tif_path"

    if png_path.exists() and not FORCE_REWRITE:
        if row["action"] != "updated_tif_path":
            row["action"] = "skip_png_exists"
        return row
    if DRY_RUN:
        row["action"] = "would_write_png"
        return row

    try:
        with rasterio.open(tif_path) as dataset:
            data = dataset.read(1)
        write_png(png_path, png_for_mask(mask.mask_type, data))
        row["png_exists"] = True
        row["action"] = "updated_tif_path_and_wrote_png" if row["action"] == "updated_tif_path" else "wrote_png"
    except Exception as exc:
        row["action"] = "failed"
        row["error"] = str(exc)[:500]
    return row


def write_report(rows: list[dict[str, object]]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / f"hansen_mask_png_repair_{now_tag()}.csv"
    fieldnames = [
        "mask_id",
        "sample_id",
        "analysis_id",
        "mask_type",
        "tif_path",
        "candidate_tif_path",
        "candidate_match",
        "png_path",
        "tif_exists",
        "png_exists",
        "action",
        "error",
    ]
    with report_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return report_path


def main() -> None:
    mask_index = build_existing_mask_index()
    with SessionLocal() as db:
        stmt = select(HansenMask).order_by(
            HansenMask.sample_id,
            HansenMask.analysis_id,
            HansenMask.mask_type,
        )
        masks = list(db.scalars(stmt))
        rows = [repair_mask(mask, mask_index) for mask in masks]
        if not DRY_RUN:
            db.commit()

    action_counts = Counter(str(row["action"]) for row in rows)
    type_counts = Counter(str(row["mask_type"]) for row in rows)
    report_path = write_report(rows)

    print(f"DRY_RUN: {DRY_RUN}")
    print(f"FORCE_REWRITE: {FORCE_REWRITE}")
    print(f"REPAIR_TIF_PATHS: {REPAIR_TIF_PATHS}")
    print(f"Indexed TIFF masks: {len(mask_index[0])}")
    print(f"Masks in DB: {len(rows)}")
    print(f"Mask types: {dict(sorted(type_counts.items()))}")
    print(f"Actions: {dict(sorted(action_counts.items()))}")
    print(f"Report: {report_path}")
    if DRY_RUN:
        print("Dry run only. Set DRY_RUN = False to write missing PNG files.")


if __name__ == "__main__":
    main()
