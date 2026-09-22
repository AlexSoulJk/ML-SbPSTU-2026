from __future__ import annotations

import sys
from pathlib import Path

import rasterio
from sqlalchemy import select


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "Scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from ResearchTool.backend.forest.database import SessionLocal  # noqa: E402
from ResearchTool.backend.forest.hansen_service import rgba_loss_year, write_png  # noqa: E402
from ResearchTool.backend.forest.models import HansenMask  # noqa: E402
from ResearchTool.backend.forest.storage_paths import resolve_storage_path  # noqa: E402


MASK_TYPES = {"all_loss", "aoi_all_loss"}
DRY_RUN = False


def refresh_mask_png(mask: HansenMask) -> str:
    tif_path = resolve_storage_path(mask.local_path)
    png_path = tif_path.with_suffix(".png")
    if not tif_path.exists():
        return "missing_tif"
    if DRY_RUN:
        return "would_update"

    with rasterio.open(tif_path) as dataset:
        loss = dataset.read(1)
    write_png(png_path, rgba_loss_year(loss))
    return "updated"


def main() -> None:
    counts: dict[str, int] = {}
    with SessionLocal() as db:
        masks = list(
            db.scalars(
                select(HansenMask).where(HansenMask.mask_type.in_(sorted(MASK_TYPES)))
            )
        )
        for mask in masks:
            result = refresh_mask_png(mask)
            counts[result] = counts.get(result, 0) + 1

    print(f"DRY_RUN: {DRY_RUN}")
    print(f"Masks: {sum(counts.values())}")
    print(f"Actions: {counts}")


if __name__ == "__main__":
    main()
