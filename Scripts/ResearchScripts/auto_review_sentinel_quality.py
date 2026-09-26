from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from sqlalchemy import select


DRY_RUN = True
LIMIT_DOWNLOADS: int | None = None
OVERWRITE_EXISTING_REVIEWS = False
INCLUDE_OK_ROWS = False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "Scripts"
RESEARCH_TOOL_ROOT = SCRIPTS_ROOT / "ResearchTool"
sys.path.insert(0, str(RESEARCH_TOOL_ROOT))

from backend.forest.database import SessionLocal, init_forest_database  # noqa: E402
from backend.forest.models import (  # noqa: E402
    ManualValidation,
    SentinelDownload,
    SentinelSceneReview,
    SentinelSceneReviewReason,
    utc_now,
)
from backend.forest.sentinel_service import (  # noqa: E402
    AUTO_REVIEW_NOTES,
    SENTINEL_CLOUD_THRESHOLD,
    SENTINEL_DARK_BRIGHTNESS_THRESHOLD,
    SENTINEL_DARK_LOW_CONTRAST_P95_THRESHOLD,
    SENTINEL_LOW_CONTRAST_RANGE_THRESHOLD,
    SENTINEL_LOW_CONTRAST_RELATIVE_THRESHOLD,
    SENTINEL_NODATA_THRESHOLD,
    SENTINEL_SHADOW_THRESHOLD,
    auto_review_reasons,
    cloud_metrics,
)
from backend.forest.storage_paths import resolve_storage_path  # noqa: E402


SCRIPT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_ROOT / "Outputs"


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_report(rows: list[dict[str, object]]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / f"sentinel_quality_auto_review_{now_tag()}.csv"
    fieldnames = [
        "action",
        "download_id",
        "sample_id",
        "sample_manual_validation",
        "sample_is_valid",
        "period",
        "scene_id",
        "local_path",
        "reason_code",
        "reason_codes",
        "comparison",
        "existing_review_state",
        "existing_is_excluded",
        "existing_reason_code",
        "existing_reason_codes",
        "existing_missing_auto_reason_codes",
        "existing_extra_reason_codes",
        "existing_reason_text",
        "existing_notes",
        "cloud_fraction",
        "shadow_fraction",
        "nodata_fraction",
        "dark_fraction",
        "brightness_p05",
        "brightness_p50",
        "brightness_p95",
        "brightness_p95_minus_p05",
        "brightness_relative_contrast",
        "quality_flags",
        "cloud_threshold",
        "shadow_threshold",
        "nodata_threshold",
        "dark_brightness_threshold",
        "dark_low_contrast_p95_threshold",
        "low_contrast_range_threshold",
        "low_contrast_relative_threshold",
        "error",
    ]
    with report_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return report_path


def row_base(download: SentinelDownload) -> dict[str, object]:
    return {
        "download_id": download.download_id,
        "sample_id": download.sample_id,
        "period": download.period or "",
        "scene_id": download.scene_id,
        "local_path": download.local_path,
    }


def review_payload(review: SentinelSceneReview | None) -> dict[str, object]:
    if review is None:
        return {
            "existing_review_state": "missing",
            "existing_is_excluded": "",
            "existing_reason_code": "",
            "existing_reason_codes": "",
            "existing_reason_text": "",
            "existing_notes": "",
        }
    return {
        "existing_review_state": "excluded" if review.is_excluded else "ok",
        "existing_is_excluded": bool(review.is_excluded),
        "existing_reason_code": review.reason_code or "",
        "existing_reason_codes": "",
        "existing_reason_text": review.reason_text or "",
        "existing_notes": review.notes or "",
    }


def review_reason_codes(db, download_id: str, review: SentinelSceneReview | None = None) -> list[str]:
    codes = list(
        db.scalars(
            select(SentinelSceneReviewReason.reason_code)
            .where(SentinelSceneReviewReason.download_id == download_id)
            .order_by(SentinelSceneReviewReason.id)
        )
    )
    if not codes and review is not None and review.reason_code:
        codes.append(review.reason_code)
    return codes


def replace_review_reasons(db, download_id: str, reason_codes: list[str]) -> None:
    existing = list(
        db.scalars(
            select(SentinelSceneReviewReason).where(SentinelSceneReviewReason.download_id == download_id)
        )
    )
    for reason in existing:
        db.delete(reason)
    if existing:
        db.flush()
    for reason_code in reason_codes:
        db.add(
            SentinelSceneReviewReason(
                download_id=download_id,
                reason_code=reason_code,
                reason_text=None,
            )
        )


def clean_reason_codes(reason_codes: list[str]) -> list[str]:
    cleaned: list[str] = []
    for reason_code in reason_codes:
        if reason_code and reason_code not in cleaned:
            cleaned.append(reason_code)
    return cleaned


def manual_validation_payload(manual: ManualValidation | None) -> dict[str, object]:
    validation = manual.validation if manual is not None else ""
    return {
        "sample_manual_validation": validation,
        "sample_is_valid": validation == "Valid",
    }


def classify_review_action(
    *,
    reason_codes: list[str],
    review: SentinelSceneReview | None,
    existing_reason_codes: list[str],
) -> tuple[str, str]:
    auto_excludes = bool(reason_codes)
    auto_set = set(reason_codes)
    existing_set = set(existing_reason_codes)
    if auto_excludes:
        if review is None:
            return (
                "would_auto_exclude" if DRY_RUN else "auto_excluded",
                "auto_exclude_no_review",
            )
        if review.is_excluded:
            if auto_set == existing_set:
                return "matches_existing_exclusion", "auto_exclude_matches_existing_exclusion"
            if auto_set.issubset(existing_set):
                return "matches_existing_exclusion_extra_reasons", "auto_exclude_existing_exclusion_extra_reasons"
            if OVERWRITE_EXISTING_REVIEWS:
                return (
                    "would_update_review" if DRY_RUN else "updated_review",
                    "auto_exclude_existing_exclusion_different_reasons",
                )
            return "existing_exclusion_different_reasons", "auto_exclude_existing_exclusion_different_reasons"
        if OVERWRITE_EXISTING_REVIEWS:
            return (
                "would_update_review" if DRY_RUN else "updated_review",
                "auto_exclude_existing_ok",
            )
        return "existing_ok_auto_exclude", "auto_exclude_existing_ok"

    if review is None:
        return "ok", "auto_ok_no_review"
    if review.is_excluded:
        return "auto_ok_existing_excluded", "auto_ok_existing_excluded"
    return "ok", "auto_ok_matches_existing_ok"


def should_write_row(row: dict[str, object]) -> bool:
    return INCLUDE_OK_ROWS or row.get("action") != "ok"


def row_reason_codes(row: dict[str, object]) -> list[str]:
    raw_codes = str(row.get("reason_codes") or "")
    if raw_codes:
        return [code for code in raw_codes.split(",") if code]
    raw_code = str(row.get("reason_code") or "")
    return [raw_code] if raw_code else []


def print_reason_stats(rows: list[dict[str, object]]) -> None:
    reason_rows = [row for row in rows if row_reason_codes(row)]
    if not reason_rows:
        print("Auto exclusion by reason: {}")
        return

    print("Auto exclusion by reason:")
    all_reasons = sorted({reason for row in reason_rows for reason in row_reason_codes(row)})
    for reason in all_reasons:
        current = [row for row in reason_rows if reason in row_reason_codes(row)]
        actions = Counter(str(row["action"]) for row in current)
        print(f"  {reason}: total={len(current)} actions={dict(sorted(actions.items()))}")

    combinations = Counter(",".join(row_reason_codes(row)) for row in reason_rows)
    print("Auto exclusion by reason set:")
    for reason_set, count in sorted(combinations.items()):
        print(f"  {reason_set}: {count}")


def print_valid_sample_conflicts(rows: list[dict[str, object]]) -> None:
    conflicts = [
        row
        for row in rows
        if row.get("sample_is_valid") is True
        and row_reason_codes(row)
        and row.get("existing_review_state") in {"missing", "ok"}
    ]
    unique_samples = {row.get("sample_id") for row in conflicts}
    print(
        "Valid samples, currently non-excluded images auto would exclude: "
        f"images={len(conflicts)}, samples={len(unique_samples)}"
    )
    if not conflicts:
        return

    print("Valid-sample conflicts by reason:")
    all_reasons = sorted({reason for row in conflicts for reason in row_reason_codes(row)})
    for reason in all_reasons:
        current = [row for row in conflicts if reason in row_reason_codes(row)]
        review_states = Counter(str(row["existing_review_state"]) for row in current)
        actions = Counter(str(row["action"]) for row in current)
        print(
            f"  {reason}: images={len(current)}, "
            f"samples={len({row.get('sample_id') for row in current})}, "
            f"states={dict(sorted(review_states.items()))}, "
            f"actions={dict(sorted(actions.items()))}"
        )


def main() -> None:
    init_forest_database()
    rows: list[dict[str, object]] = []

    with SessionLocal() as db:
        stmt = select(SentinelDownload).order_by(
            SentinelDownload.sample_id,
            SentinelDownload.period,
            SentinelDownload.scene_id,
            SentinelDownload.download_id,
        )
        if LIMIT_DOWNLOADS is not None:
            stmt = stmt.limit(LIMIT_DOWNLOADS)
        downloads = list(db.scalars(stmt))

        for download in downloads:
            base = row_base(download)
            path = resolve_storage_path(download.local_path)
            if not path.exists():
                rows.append({**base, "action": "missing_tif", "error": str(path)})
                continue

            try:
                metrics = cloud_metrics(path)
            except Exception as exc:
                rows.append({**base, "action": "failed", "error": str(exc)[:500]})
                continue

            reason_codes = clean_reason_codes(auto_review_reasons(metrics))
            reason_code = reason_codes[0] if reason_codes else None
            flags_json = json.dumps(metrics["quality_flags"], ensure_ascii=False, sort_keys=True)
            review = db.scalar(
                select(SentinelSceneReview).where(
                    SentinelSceneReview.download_id == download.download_id
                )
            )
            manual = db.get(ManualValidation, download.sample_id)
            existing_reason_codes = review_reason_codes(db, download.download_id, review)
            existing_set = set(existing_reason_codes)
            auto_set = set(reason_codes)
            missing_auto_reason_codes = sorted(auto_set - existing_set)
            extra_existing_reason_codes = sorted(existing_set - auto_set)

            action, comparison = classify_review_action(
                reason_codes=reason_codes,
                review=review,
                existing_reason_codes=existing_reason_codes,
            )
            if reason_codes and not DRY_RUN:
                if review is None:
                    db.add(
                        SentinelSceneReview(
                            download_id=download.download_id,
                            is_excluded=True,
                            reason_code=reason_code,
                            reason_text=None,
                            notes=AUTO_REVIEW_NOTES,
                        )
                    )
                    replace_review_reasons(db, download.download_id, reason_codes)
                elif OVERWRITE_EXISTING_REVIEWS and (
                    not review.is_excluded or auto_set != existing_set
                ):
                    review.is_excluded = True
                    review.reason_code = reason_code
                    review.reason_text = None
                    review.notes = AUTO_REVIEW_NOTES
                    review.updated_at = utc_now()
                    replace_review_reasons(db, download.download_id, reason_codes)

            if not DRY_RUN:
                download.cloud_fraction = float(metrics["cloud_fraction"])
                download.shadow_fraction = float(metrics["shadow_fraction"])
                download.nodata_fraction = float(metrics["nodata_fraction"])
                download.dark_fraction = float(metrics["dark_fraction"])
                download.is_bad_cloud = bool(metrics["is_bad_cloud"])
                download.is_bad_quality = bool(metrics["is_bad_quality"])
                download.quality_flags_json = flags_json

            rows.append(
                {
                    **base,
                    **manual_validation_payload(manual),
                    "action": action,
                    "reason_code": reason_code or "",
                    "reason_codes": ",".join(reason_codes),
                    "comparison": comparison,
                    **{
                        **review_payload(review),
                        "existing_reason_codes": ",".join(existing_reason_codes),
                        "existing_missing_auto_reason_codes": ",".join(missing_auto_reason_codes),
                        "existing_extra_reason_codes": ",".join(extra_existing_reason_codes),
                    },
                    "cloud_fraction": metrics["cloud_fraction"],
                    "shadow_fraction": metrics["shadow_fraction"],
                    "nodata_fraction": metrics["nodata_fraction"],
                    "dark_fraction": metrics["dark_fraction"],
                    "brightness_p05": metrics["brightness_p05"],
                    "brightness_p50": metrics["brightness_p50"],
                    "brightness_p95": metrics["brightness_p95"],
                    "brightness_p95_minus_p05": metrics["brightness_p95_minus_p05"],
                    "brightness_relative_contrast": metrics["brightness_relative_contrast"],
                    "quality_flags": flags_json,
                    "cloud_threshold": SENTINEL_CLOUD_THRESHOLD,
                    "shadow_threshold": SENTINEL_SHADOW_THRESHOLD,
                    "nodata_threshold": SENTINEL_NODATA_THRESHOLD,
                    "dark_brightness_threshold": SENTINEL_DARK_BRIGHTNESS_THRESHOLD,
                    "dark_low_contrast_p95_threshold": SENTINEL_DARK_LOW_CONTRAST_P95_THRESHOLD,
                    "low_contrast_range_threshold": SENTINEL_LOW_CONTRAST_RANGE_THRESHOLD,
                    "low_contrast_relative_threshold": SENTINEL_LOW_CONTRAST_RELATIVE_THRESHOLD,
                    "error": "",
                }
            )

        if DRY_RUN:
            db.rollback()
        else:
            db.commit()

    report_rows = [row for row in rows if should_write_row(row)]
    report_path = write_report(report_rows)
    print(f"DRY_RUN: {DRY_RUN}")
    print(f"LIMIT_DOWNLOADS: {LIMIT_DOWNLOADS}")
    print(f"OVERWRITE_EXISTING_REVIEWS: {OVERWRITE_EXISTING_REVIEWS}")
    print(f"INCLUDE_OK_ROWS: {INCLUDE_OK_ROWS}")
    print(
        "Thresholds: "
        f"cloud>{SENTINEL_CLOUD_THRESHOLD:.0%}, "
        f"shadow>{SENTINEL_SHADOW_THRESHOLD:.0%}, "
        f"nodata>{SENTINEL_NODATA_THRESHOLD:.0%}, "
        "DARK_AND_LOW_CONTRAST: "
        f"p95<{SENTINEL_DARK_LOW_CONTRAST_P95_THRESHOLD:g} and "
        f"(range<{SENTINEL_LOW_CONTRAST_RANGE_THRESHOLD:g} or "
        f"relative<{SENTINEL_LOW_CONTRAST_RELATIVE_THRESHOLD:.2f})"
    )
    print(f"Downloads checked: {len(rows)}")
    print(f"Rows written to report: {len(report_rows)}")
    print(f"Actions: {dict(sorted(Counter(row['action'] for row in rows).items()))}")
    print(f"Comparisons: {dict(sorted(Counter(row.get('comparison', '') for row in rows).items()))}")
    print_reason_stats(rows)
    print_valid_sample_conflicts(rows)
    print(f"Report: {report_path}")
    if DRY_RUN:
        print("Dry run only. Set DRY_RUN = False to write metrics and auto-exclusions.")


if __name__ == "__main__":
    main()
