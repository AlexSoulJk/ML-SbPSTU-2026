from __future__ import annotations

from generate_ml_data import (
    PATH_TO_IMAGES_DATA,
    SessionLocal,
    get_images_from_db,
    restore_images_to_cache_from_folder,
)


OVERWRITE_EXISTING = False


def main() -> None:
    with SessionLocal() as db:
        images = get_images_from_db(db)

    print(f"Found RGB PRE/POST images for valid samples: {len(images)}")
    stats = restore_images_to_cache_from_folder(
        PATH_TO_IMAGES_DATA,
        images,
        overwrite_existing=OVERWRITE_EXISTING,
    )
    print(f"Done: {stats}")


if __name__ == "__main__":
    main()
