from __future__ import annotations

from ml_tests.stage_b import run_stage_b


# Keep False for an inspection-only rehearsal. Stage B was applied to the working
# database only after the same migration passed on an isolated SQLite backup.
APPLY_TO_WORKING_DATABASE = False


def main() -> None:
    result = run_stage_b(apply_to_working_database=APPLY_TO_WORKING_DATABASE)
    print(f"Stage B status: {result['status']}")
    print(f"Backup: {result['backup_path']}")
    if result["status"] == "PASS":
        print("Report: Scripts/ResearchScripts/Outputs/pipeline_v2/stages/B/STAGE_REPORT.md")


if __name__ == "__main__":
    main()
