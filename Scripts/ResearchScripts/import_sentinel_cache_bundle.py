from __future__ import annotations

import sys
from pathlib import Path


# Edit these values in the file and run it again. No CLI arguments are required.
DRY_RUN = True

# None selects the newest bundle from Data/forest/imports or data_operations outputs.
BUNDLE_PATH: Path | None = None
OVERWRITE_EXISTING = False
REBUILD_PREVIEWS = True


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOL_ROOT = PROJECT_ROOT / "Scripts" / "ResearchTool"
sys.path.insert(0, str(TOOL_ROOT))

from backend.forest.bundle_service import (  # noqa: E402
    bundle_candidates,
    import_bundle,
    import_bundle_plan,
)
from backend.forest.database import SessionLocal, init_forest_database  # noqa: E402


def selected_bundle_path() -> Path:
    if BUNDLE_PATH is not None:
        return BUNDLE_PATH
    candidates = bundle_candidates()
    if not candidates:
        raise FileNotFoundError(
            "No bundle was found. Put a ZIP or bundle directory into "
            "Scripts/ResearchTool/Data/forest/imports."
        )
    return Path(candidates[0]["path"])


def main() -> None:
    init_forest_database()
    bundle_path = selected_bundle_path()
    with SessionLocal() as db:
        plan = import_bundle_plan(db, str(bundle_path))
        print(f"DRY_RUN: {DRY_RUN}")
        print(f"Bundle: {bundle_path}")
        print(f"Files: {plan['summary']['files']}")
        print(f"Copy: {plan['summary']['copy']}")
        print(f"Existing: {plan['summary']['skip_exists']}")
        print(f"Missing bundle files: {plan['summary']['missing_bundle_file']}")
        print(f"Checksum mismatches: {plan['summary']['checksum_mismatches']}")
        print(f"Database not current: {plan['summary']['missing_db_downloads']}")
        print(f"Identity conflicts: {plan['summary']['identity_conflicts']}")
        print(f"Copy bytes: {plan['summary']['bytes_to_copy'] / 1024 / 1024:.1f} MiB")
        if not plan["can_apply"]:
            raise RuntimeError(
                "Bundle validation failed. Pull/merge the current database and check the bundle."
            )
        if DRY_RUN:
            print("Dry run only. Set DRY_RUN = False to copy files and rebuild previews.")
            return

        result = import_bundle(
            db,
            str(bundle_path),
            overwrite_existing=OVERWRITE_EXISTING,
            rebuild_previews=REBUILD_PREVIEWS,
            progress=lambda stage, message, done, total: print(
                f"[{done}/{total}] {stage}: {message}"
            ),
        )
        print(f"Copied: {result['copied']}")
        print(f"Failed: {result['failed']}")
        print(f"Previews rebuilt: {result['previews_rebuilt']}")
        print(f"DB backup: {result['backup_path'] or 'not required'}")
        print(f"Report: {result['operations_csv']}")


if __name__ == "__main__":
    main()
