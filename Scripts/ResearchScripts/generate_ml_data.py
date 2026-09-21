from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import func, or_, select


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "Scripts"
RESEARCH_TOOL_ROOT = SCRIPTS_ROOT / "ResearchTool"
DERIVED_ROOT = RESEARCH_TOOL_ROOT / "Data" / "cache" / "forest" / "derived"
PATH_TO_IMAGES_DATA = PROJECT_ROOT / "Scripts" / "ResearchScripts" / "Data" / "Images"

sys.path.insert(0, str(SCRIPTS_ROOT))

from ResearchTool.backend.forest.database import SessionLocal  # noqa: E402
from ResearchTool.backend.forest.models import (  # noqa: E402
    DerivedPreview,
    ManualValidation,
    Sample,
    SentinelDownload,
    SentinelSceneReview,
)


def resolve_project_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def display_path(path: str | Path) -> str:
    resolved = resolve_project_path(path)
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def get_images_from_db(db) -> list[dict[str, Any]]:
    scene_datetime = func.json_extract(
        SentinelDownload.metadata_json,
        "$.item.datetime",
    )

    stmt = (
        select(
            Sample.sample_id,
            Sample.display_name,
            Sample.source_id,
            Sample.lat,
            Sample.lon,
            Sample.driver_primary,
            SentinelDownload.download_id,
            SentinelDownload.period,
            SentinelDownload.scene_id,
            SentinelDownload.local_path.label("geotiff_path"),
            scene_datetime.label("scene_datetime"),
            func.substr(scene_datetime, 1, 10).label("scene_date"),
            DerivedPreview.view,
            DerivedPreview.png_path,
            DerivedPreview.metadata_path,
        )
        .join(ManualValidation, ManualValidation.sample_id == Sample.sample_id)
        .join(SentinelDownload, SentinelDownload.sample_id == Sample.sample_id)
        .join(DerivedPreview, DerivedPreview.download_id == SentinelDownload.download_id)
        .outerjoin(SentinelSceneReview, SentinelSceneReview.download_id == SentinelDownload.download_id)
        .where(
            ManualValidation.validation == "Valid",
           or_(
            Sample.driver_primary.in_(["Logging", "Wildfire"]),
            Sample.driver_secondary.in_(["Logging", "Wildfire"]),
            ),
            DerivedPreview.view == "rgb",
            SentinelDownload.period.in_(["PRE", "POST"]),
            or_(
            SentinelSceneReview.id.is_(None),
            SentinelSceneReview.is_excluded == False,
        ),
        )
        .order_by(
            Sample.source_file,
            Sample.source_id,
            Sample.sample_id,
            SentinelDownload.period,
            SentinelDownload.scene_id,
            DerivedPreview.view,
        )
    )
    return [dict(row) for row in db.execute(stmt).mappings().all()]


def derived_relative_path(png_path: str | Path) -> Path:
    path = resolve_project_path(png_path)
    try:
        return path.resolve(strict=False).relative_to(DERIVED_ROOT.resolve(strict=False))
    except ValueError:
        parts = Path(png_path).parts
        if "derived" in parts:
            derived_index = parts.index("derived")
            return Path(*parts[derived_index + 1 :])
        raise


def get_original_image_path(image: dict[str, Any]) -> Path:
    return resolve_project_path(image["png_path"])


def get_path_to_image(image: dict[str, Any], folder_path: str | Path) -> Path:
    relative_path = derived_relative_path(image["png_path"])
    path = resolve_project_path(folder_path) / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_metadata_to_folder(folder_path: str | Path, images_info: dict[int, dict[str, Any]]) -> Path:
    metadata_folder = resolve_project_path(folder_path) / "metadata"
    metadata_folder.mkdir(parents=True, exist_ok=True)
    metadata_file = metadata_folder / "images_metadata.json"
    metadata_file.write_text(
        json.dumps(images_info, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )
    print(f"Saved metadata to {display_path(metadata_file)}")
    return metadata_file


def create_metadata_for_images(
    images: list[dict[str, Any]],
    folder_path: str | Path,
) -> dict[int, dict[str, Any]]:
    result = {}
    for index, image in enumerate(images):
        result[index] = {
            "sample_id": image["sample_id"],
            "period": image["period"],
            "scene_datetime": image["scene_datetime"],
            "path": display_path(get_path_to_image(image, folder_path)),
            "label": image["driver_primary"],
            "image_id": image["download_id"],
        }
    return result


def save_images_to_folder(folder_path: str | Path, images: list[dict[str, Any]]) -> dict[str, int]:
    stats = {
        "copied": 0,
        "already_exists": 0,
        "missing_source": 0,
    }

    for image in images:
        target_path = get_path_to_image(image, folder_path)
        source_path = get_original_image_path(image)
        if target_path.exists():
            stats["already_exists"] += 1
            print(f"Image already exists: {display_path(target_path)}")
            continue

        if not source_path.exists():
            stats["missing_source"] += 1
            print(f"Original image not found: {display_path(source_path)}")
            continue

        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        stats["copied"] += 1
        print(f"Copied image to {display_path(target_path)}")

    save_metadata_to_folder(
        folder_path,
        create_metadata_for_images(images, folder_path),
    )
    return stats


def restore_images_to_cache_from_folder(
    folder_path: str | Path,
    images: list[dict[str, Any]],
    *,
    overwrite_existing: bool = False,
) -> dict[str, int]:
    stats = {
        "restored": 0,
        "already_exists": 0,
        "missing_backup": 0,
    }

    for image in images:
        source_path = get_path_to_image(image, folder_path)
        target_path = get_original_image_path(image)
        if target_path.exists() and not overwrite_existing:
            stats["already_exists"] += 1
            continue

        if not source_path.exists():
            stats["missing_backup"] += 1
            print(f"Backup image not found: {display_path(source_path)}")
            continue

        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        stats["restored"] += 1
        print(f"Restored image to {display_path(target_path)}")

    return stats


def main() -> None:
    with SessionLocal() as db:
        images = get_images_from_db(db)

    print(f"Found RGB PRE/POST images for valid samples: {len(images)}")
    stats = save_images_to_folder(PATH_TO_IMAGES_DATA, images)
    print(f"Done: {stats}")


if __name__ == "__main__":
    main()
