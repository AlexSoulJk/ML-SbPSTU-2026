from __future__ import annotations

import csv
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from sqlalchemy import select


LIMIT_DOWNLOADS: int | None = None
PERIODS: tuple[str, ...] | None = None
ONLY_VALID_SAMPLES = False
INCLUDE_EXCLUDED_REVIEWS = True
SAMPLE_ID_PREFIXES: tuple[str, ...] = ()
DOWNLOAD_ID_PREFIXES: tuple[str, ...] = ()

# Legacy OR-rule thresholds kept for calibration sweeps.
CURRENT_DARK_BRIGHTNESS_THRESHOLD = 350.0
CURRENT_DARK_FRACTION_THRESHOLD = 0.65
CURRENT_DARK_P95_THRESHOLD = 600.0
PRODUCTION_DARK_LOW_CONTRAST_P95_THRESHOLD = 350.0
PRODUCTION_LOW_CONTRAST_RANGE_THRESHOLD = 100.0
PRODUCTION_LOW_CONTRAST_RELATIVE_THRESHOLD = 0.30

# Extra thresholds to inspect without changing application behavior.
DARK_PIXEL_THRESHOLDS = (100, 150, 200, 250, 300, 350, 450, 600, 800)
P95_CANDIDATE_THRESHOLDS = (200, 300, 400, 500, 600, 800, 1000, 1500, 2000)
DARK_FRACTION_CANDIDATE_THRESHOLDS = (0.30, 0.50, 0.65, 0.80, 0.90, 0.95)
COMBINED_OR_RULE_CANDIDATES = (
    (300.0, 0.95),
    (350.0, 0.95),
    (400.0, 0.95),
    (350.0, 0.90),
    (400.0, 0.90),
    (600.0, 0.65),
)
QC_TOO_DARK_P95_THRESHOLDS = (300.0, 350.0, 400.0)
QC_LOW_CONTRAST_RANGE_THRESHOLDS = (75.0, 100.0, 125.0, 150.0)
QC_LOW_CONTRAST_RELATIVE_THRESHOLDS = (0.20, 0.25, 0.30, 0.35)
STRICT_QC_DARK_P95_THRESHOLD = 350.0
STRICT_QC_RANGE_THRESHOLD = 100.0
STRICT_QC_RELATIVE_CONTRAST_THRESHOLD = 0.30

WRITE_PLOT_PNGS = True


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "Scripts"
RESEARCH_TOOL_ROOT = SCRIPTS_ROOT / "ResearchTool"
sys.path.insert(0, str(RESEARCH_TOOL_ROOT))

from backend.forest.database import SessionLocal, init_forest_database  # noqa: E402
from backend.forest.models import ManualValidation, SentinelDownload, SentinelSceneReview  # noqa: E402
from backend.forest.sentinel_service import SCL_CLOUD_CLASSES, SCL_SHADOW_CLASSES  # noqa: E402
from backend.forest.storage_paths import resolve_storage_path  # noqa: E402


SCRIPT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_ROOT / "Outputs"

PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 98, 99)
BRIGHTNESS_BINS = [0, 50, 100, 150, 200, 250, 300, 350, 450, 600, 800, 1000, 1500, 2000, 3000, 5000, 10000]
BRIGHTNESS_RANGE_BINS = [0, 25, 50, 75, 100, 150, 200, 250, 300, 350, 450, 600, 800, 1000, 1500, 2000, 3000, 5000, 10000]
FRACTION_BINS = [round(value / 20, 2) for value in range(21)]
RELATIVE_CONTRAST_BINS = [0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 5.00, 10.00]


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def selected_downloads(db) -> list[SentinelDownload]:
    stmt = select(SentinelDownload).order_by(
        SentinelDownload.sample_id,
        SentinelDownload.period,
        SentinelDownload.scene_id,
        SentinelDownload.download_id,
    )
    if PERIODS:
        stmt = stmt.where(SentinelDownload.period.in_(PERIODS))
    downloads = list(db.scalars(stmt))
    if SAMPLE_ID_PREFIXES:
        downloads = [
            download for download in downloads
            if any(download.sample_id.startswith(prefix) for prefix in SAMPLE_ID_PREFIXES)
        ]
    if DOWNLOAD_ID_PREFIXES:
        downloads = [
            download for download in downloads
            if any(download.download_id.startswith(prefix) for prefix in DOWNLOAD_ID_PREFIXES)
        ]
    if LIMIT_DOWNLOADS is not None:
        downloads = downloads[:LIMIT_DOWNLOADS]
    return downloads


def manual_payload(db, sample_id: str) -> dict[str, Any]:
    manual = db.get(ManualValidation, sample_id)
    validation = manual.validation if manual is not None else ""
    return {
        "manual_validation": validation,
        "sample_is_valid": validation == "Valid",
    }


def review_payload(db, download_id: str) -> dict[str, Any]:
    review = db.scalar(
        select(SentinelSceneReview).where(SentinelSceneReview.download_id == download_id)
    )
    if review is None:
        return {
            "review_state": "missing",
            "is_excluded": False,
            "review_reason_code": "",
            "review_notes": "",
        }
    return {
        "review_state": "excluded" if review.is_excluded else "ok",
        "is_excluded": bool(review.is_excluded),
        "review_reason_code": review.reason_code or "",
        "review_notes": review.notes or "",
    }


def finite_values(values: np.ndarray) -> np.ndarray:
    data = values[np.isfinite(values)]
    return data.astype(np.float32, copy=False)


def percentile_payload(prefix: str, values: np.ndarray) -> dict[str, Any]:
    values = finite_values(values)
    if values.size == 0:
        return {f"{prefix}_p{percentile:02d}": "" for percentile in PERCENTILES}
    return {
        f"{prefix}_p{percentile:02d}": round(float(np.percentile(values, percentile)), 3)
        for percentile in PERCENTILES
    }


def contrast_payload(prefix: str, percentiles: dict[str, Any]) -> dict[str, Any]:
    p05 = percentiles.get(f"{prefix}_p05")
    p50 = percentiles.get(f"{prefix}_p50")
    p95 = percentiles.get(f"{prefix}_p95")
    if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in (p05, p50, p95)):
        return {
            f"{prefix}_p95_minus_p05": "",
            f"{prefix}_p95_minus_p05_div_p50": "",
        }

    brightness_range = float(p95) - float(p05)
    relative_range = brightness_range / float(p50) if float(p50) > 0 else ""
    return {
        f"{prefix}_p95_minus_p05": round(brightness_range, 3),
        f"{prefix}_p95_minus_p05_div_p50": "" if relative_range == "" else round(float(relative_range), 4),
    }


def is_production_dark_low_contrast(prefix: str, percentiles: dict[str, Any], pixel_count: int) -> bool:
    p05 = percentiles.get(f"{prefix}_p05")
    p50 = percentiles.get(f"{prefix}_p50")
    p95 = percentiles.get(f"{prefix}_p95")
    if pixel_count <= 0:
        return False
    if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in (p05, p50, p95)):
        return False

    brightness_range = float(p95) - float(p05)
    relative_contrast = brightness_range / float(p50) if float(p50) > 0 else 0.0
    return (
        float(p95) < PRODUCTION_DARK_LOW_CONTRAST_P95_THRESHOLD
        and (
            brightness_range < PRODUCTION_LOW_CONTRAST_RANGE_THRESHOLD
            or relative_contrast < PRODUCTION_LOW_CONTRAST_RELATIVE_THRESHOLD
        )
    )


def fraction_below(values: np.ndarray, threshold: float) -> float | None:
    values = finite_values(values)
    if values.size == 0:
        return None
    return float((values < threshold).sum() / values.size)


def read_darkness_metrics(path: Path) -> dict[str, Any]:
    with rasterio.open(path) as dataset:
        if dataset.count < 8:
            raise ValueError(f"Expected at least 8 bands, got {dataset.count}")
        blue = dataset.read(1).astype(np.float32)
        green = dataset.read(2).astype(np.float32)
        red = dataset.read(3).astype(np.float32)
        nir = dataset.read(4).astype(np.float32)
        swir1 = dataset.read(5).astype(np.float32)
        swir2 = dataset.read(6).astype(np.float32)
        scl = dataset.read(7)
        data_mask = dataset.read(8) > 0

    total_pixels = int(data_mask.size)
    emptyish = data_mask & ((blue + green + red + nir + swir1 + swir2) <= 0)
    valid = data_mask & ~emptyish
    cloud = np.isin(scl, list(SCL_CLOUD_CLASSES)) & valid
    shadow = np.isin(scl, list(SCL_SHADOW_CLASSES)) & valid
    clear = valid & ~cloud & ~shadow

    valid_count = int(valid.sum())
    clear_count = int(clear.sum())
    nodata_fraction = float((~valid).sum() / total_pixels) if total_pixels else 1.0
    data_mask_fraction = float(data_mask.sum() / total_pixels) if total_pixels else 0.0
    cloud_fraction = float(cloud.sum() / valid_count) if valid_count else 0.0
    shadow_fraction = float(shadow.sum() / valid_count) if valid_count else 0.0

    brightness = (red + green + blue) / 3.0
    valid_brightness = brightness[valid]
    clear_brightness = brightness[clear]
    valid_dark_fraction = fraction_below(valid_brightness, CURRENT_DARK_BRIGHTNESS_THRESHOLD)
    clear_dark_fraction = fraction_below(clear_brightness, CURRENT_DARK_BRIGHTNESS_THRESHOLD)
    valid_brightness_percentiles = percentile_payload("valid_brightness", valid_brightness)
    clear_brightness_percentiles = percentile_payload("clear_brightness", clear_brightness)
    current_dark_valid = is_production_dark_low_contrast(
        "valid_brightness",
        valid_brightness_percentiles,
        int(valid_brightness.size),
    )
    current_dark_clear = is_production_dark_low_contrast(
        "clear_brightness",
        clear_brightness_percentiles,
        int(clear_brightness.size),
    )

    row: dict[str, Any] = {
        "total_pixels": total_pixels,
        "valid_pixels": valid_count,
        "clear_pixels": clear_count,
        "data_mask_fraction": round(data_mask_fraction, 4),
        "nodata_fraction": round(nodata_fraction, 4),
        "cloud_fraction": round(cloud_fraction, 4),
        "shadow_fraction": round(shadow_fraction, 4),
        "current_dark_valid": bool(current_dark_valid),
        "current_dark_clear": bool(current_dark_clear),
    }
    row.update(valid_brightness_percentiles)
    row.update(clear_brightness_percentiles)
    row.update(contrast_payload("clear_brightness", clear_brightness_percentiles))
    row["clear_relative_contrast"] = row.get("clear_brightness_p95_minus_p05_div_p50", "")
    for threshold in DARK_PIXEL_THRESHOLDS:
        valid_fraction = fraction_below(valid_brightness, threshold)
        clear_fraction = fraction_below(clear_brightness, threshold)
        row[f"valid_dark_fraction_lt_{threshold}"] = "" if valid_fraction is None else round(valid_fraction, 4)
        row[f"clear_dark_fraction_lt_{threshold}"] = "" if clear_fraction is None else round(clear_fraction, 4)
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def histogram(values: list[float], bins: list[float]) -> list[dict[str, Any]]:
    clean = [value for value in values if isinstance(value, (int, float)) and math.isfinite(value)]
    counts = [0 for _ in range(len(bins) - 1)]
    overflow = 0
    for value in clean:
        placed = False
        for index in range(len(bins) - 1):
            if bins[index] <= value < bins[index + 1]:
                counts[index] += 1
                placed = True
                break
        if not placed and value >= bins[-1]:
            overflow += 1
    rows = [
        {
            "bucket_start": bins[index],
            "bucket_end": bins[index + 1],
            "count": counts[index],
            "pct": round(counts[index] / len(clean), 4) if clean else 0.0,
        }
        for index in range(len(counts))
    ]
    if overflow:
        rows.append(
            {
                "bucket_start": bins[-1],
                "bucket_end": "inf",
                "count": overflow,
                "pct": round(overflow / len(clean), 4) if clean else 0.0,
            }
        )
    return rows


def print_histogram(title: str, values: list[float], bins: list[float]) -> None:
    rows = histogram(values, bins)
    max_count = max((int(row["count"]) for row in rows), default=0)
    print(title)
    for row in rows:
        count = int(row["count"])
        if count == 0:
            continue
        bar = "#" * max(1, round(42 * count / max_count)) if max_count else ""
        print(f"  {row['bucket_start']:>7} - {row['bucket_end']:<7} {count:>5} {bar}")


def metric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            values.append(float(value))
    return values


def finite_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def print_metric_distribution(title: str, values: list[float]) -> None:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        print(f"{title}: no values")
        return
    data = np.array(clean, dtype=np.float32)
    print(
        f"{title}: "
        f"n={data.size}, "
        f"p05={np.percentile(data, 5):.3g}, "
        f"p50={np.percentile(data, 50):.3g}, "
        f"p95={np.percentile(data, 95):.3g}, "
        f"min={float(np.min(data)):.3g}, "
        f"max={float(np.max(data)):.3g}"
    )


def candidate_stats(rows: list[dict[str, Any]], label: str, *, p95_key: str, fraction_key: str) -> None:
    print(label)
    for threshold in P95_CANDIDATE_THRESHOLDS:
        count = sum(1 for row in rows if isinstance(row.get(p95_key), (int, float)) and float(row[p95_key]) < threshold)
        print(f"  {p95_key} < {threshold:>4}: {count}")
    for threshold in DARK_FRACTION_CANDIDATE_THRESHOLDS:
        count = sum(
            1
            for row in rows
            if isinstance(row.get(fraction_key), (int, float)) and float(row[fraction_key]) > threshold
        )
        print(f"  {fraction_key} > {threshold:.2f}: {count}")


def threshold_count(
    rows: list[dict[str, Any]],
    *,
    key: str,
    threshold: float,
    mode: str,
) -> dict[str, Any]:
    eligible = 0
    count = 0
    for row in rows:
        value = finite_number(row.get(key))
        if value is None:
            continue
        eligible += 1
        if (mode == "lt" and value < threshold) or (mode == "gt" and value > threshold):
            count += 1
    return {
        "eligible_rows": eligible,
        "selected_rows": count,
        "selected_pct": round(count / eligible, 4) if eligible else 0.0,
    }


def strict_qc_overlap_stats(
    rows: list[dict[str, Any]],
    *,
    dark_p95_threshold: float,
    range_threshold: float,
    relative_threshold: float,
) -> dict[str, Any]:
    eligible = 0
    too_dark_only = 0
    low_contrast_only = 0
    both = 0
    low_contrast_absolute = 0
    low_contrast_relative = 0
    low_contrast_absolute_only = 0
    low_contrast_relative_only = 0
    low_contrast_absolute_and_relative = 0

    for row in rows:
        p95 = finite_number(row.get("clear_brightness_p95"))
        brightness_range = finite_number(row.get("clear_brightness_p95_minus_p05"))
        relative_contrast = finite_number(row.get("clear_relative_contrast"))
        if p95 is None or brightness_range is None or relative_contrast is None:
            continue

        eligible += 1
        too_dark = p95 < dark_p95_threshold
        low_abs = brightness_range < range_threshold
        low_rel = relative_contrast < relative_threshold
        low_contrast = low_abs or low_rel

        if low_abs:
            low_contrast_absolute += 1
        if low_rel:
            low_contrast_relative += 1
        if low_abs and low_rel:
            low_contrast_absolute_and_relative += 1
        elif low_abs:
            low_contrast_absolute_only += 1
        elif low_rel:
            low_contrast_relative_only += 1

        if too_dark and low_contrast:
            both += 1
        elif too_dark:
            too_dark_only += 1
        elif low_contrast:
            low_contrast_only += 1

    selected_or = too_dark_only + low_contrast_only + both
    return {
        "eligible_rows": eligible,
        "too_dark_only": too_dark_only,
        "low_contrast_only": low_contrast_only,
        "both": both,
        "dark_and_low_contrast": both,
        "dark_or_low_contrast": selected_or,
        "dark_or_low_contrast_pct": round(selected_or / eligible, 4) if eligible else 0.0,
        "low_contrast_absolute": low_contrast_absolute,
        "low_contrast_relative": low_contrast_relative,
        "low_contrast_absolute_only": low_contrast_absolute_only,
        "low_contrast_relative_only": low_contrast_relative_only,
        "low_contrast_absolute_and_relative": low_contrast_absolute_and_relative,
    }


def strict_qc_sweep_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sweep_rows = []
    for group in ("all", "valid_samples", "valid_non_excluded"):
        group_rows = rows_for_group(rows, group)
        for threshold in QC_TOO_DARK_P95_THRESHOLDS:
            sweep_rows.append(
                {
                    "group": group,
                    "rule_family": "too_dark",
                    "metric": "clear_brightness_p95",
                    "operator": "<",
                    "threshold": threshold,
                    **threshold_count(
                        group_rows,
                        key="clear_brightness_p95",
                        threshold=threshold,
                        mode="lt",
                    ),
                }
            )
        for threshold in QC_LOW_CONTRAST_RANGE_THRESHOLDS:
            sweep_rows.append(
                {
                    "group": group,
                    "rule_family": "low_contrast_absolute",
                    "metric": "clear_brightness_p95_minus_p05",
                    "operator": "<",
                    "threshold": threshold,
                    **threshold_count(
                        group_rows,
                        key="clear_brightness_p95_minus_p05",
                        threshold=threshold,
                        mode="lt",
                    ),
                }
            )
        for threshold in QC_LOW_CONTRAST_RELATIVE_THRESHOLDS:
            sweep_rows.append(
                {
                    "group": group,
                    "rule_family": "low_contrast_relative",
                    "metric": "clear_relative_contrast",
                    "operator": "<",
                    "threshold": threshold,
                    **threshold_count(
                        group_rows,
                        key="clear_relative_contrast",
                        threshold=threshold,
                        mode="lt",
                    ),
                }
            )
        for dark_threshold in QC_TOO_DARK_P95_THRESHOLDS:
            for range_threshold in QC_LOW_CONTRAST_RANGE_THRESHOLDS:
                for relative_threshold in QC_LOW_CONTRAST_RELATIVE_THRESHOLDS:
                    sweep_rows.append(
                        {
                            "group": group,
                            "rule_family": "combined_dark_low_contrast",
                            "dark_p95_threshold": dark_threshold,
                            "range_threshold": range_threshold,
                            "relative_threshold": relative_threshold,
                            **strict_qc_overlap_stats(
                                group_rows,
                                dark_p95_threshold=dark_threshold,
                                range_threshold=range_threshold,
                                relative_threshold=relative_threshold,
                            ),
                        }
                    )
    return sweep_rows


def print_strict_qc_summary(rows: list[dict[str, Any]]) -> None:
    print("  STRICT QC sweep on clear pixels:")
    print("    TOO_DARK:")
    for threshold in QC_TOO_DARK_P95_THRESHOLDS:
        stats = threshold_count(rows, key="clear_brightness_p95", threshold=threshold, mode="lt")
        print(
            f"      p95 < {threshold:g}: "
            f"{stats['selected_rows']} ({stats['selected_pct']:.1%}); eligible={stats['eligible_rows']}"
        )

    print("    LOW_CONTRAST absolute:")
    for threshold in QC_LOW_CONTRAST_RANGE_THRESHOLDS:
        stats = threshold_count(rows, key="clear_brightness_p95_minus_p05", threshold=threshold, mode="lt")
        print(
            f"      range < {threshold:g}: "
            f"{stats['selected_rows']} ({stats['selected_pct']:.1%}); eligible={stats['eligible_rows']}"
        )

    print("    LOW_CONTRAST relative:")
    for threshold in QC_LOW_CONTRAST_RELATIVE_THRESHOLDS:
        stats = threshold_count(rows, key="clear_relative_contrast", threshold=threshold, mode="lt")
        print(
            f"      relative < {threshold:.2f}: "
            f"{stats['selected_rows']} ({stats['selected_pct']:.1%}); eligible={stats['eligible_rows']}"
        )

    stats = strict_qc_overlap_stats(
        rows,
        dark_p95_threshold=STRICT_QC_DARK_P95_THRESHOLD,
        range_threshold=STRICT_QC_RANGE_THRESHOLD,
        relative_threshold=STRICT_QC_RELATIVE_CONTRAST_THRESHOLD,
    )
    print(
        "    COMBINED strict "
        f"(p95 < {STRICT_QC_DARK_P95_THRESHOLD:g}; "
        f"range < {STRICT_QC_RANGE_THRESHOLD:g} OR relative < {STRICT_QC_RELATIVE_CONTRAST_THRESHOLD:.2f}):"
    )
    print(
        "      "
        f"too_dark_only={stats['too_dark_only']}, "
        f"low_contrast_only={stats['low_contrast_only']}, "
        f"both={stats['both']}, "
        f"dark_AND_low_contrast={stats['dark_and_low_contrast']}, "
        f"dark_OR_low_contrast={stats['dark_or_low_contrast']} "
        f"({stats['dark_or_low_contrast_pct']:.1%}); "
        f"eligible={stats['eligible_rows']}"
    )
    print(
        "      "
        f"low_abs_only={stats['low_contrast_absolute_only']}, "
        f"low_relative_only={stats['low_contrast_relative_only']}, "
        f"low_abs_and_relative={stats['low_contrast_absolute_and_relative']}"
    )


def combined_or_rule_stats(
    rows: list[dict[str, Any]],
    *,
    p95_key: str,
    fraction_key: str,
    p95_threshold: float,
    fraction_threshold: float,
) -> dict[str, Any]:
    eligible = 0
    only_p95 = 0
    only_fraction = 0
    both = 0
    for row in rows:
        p95 = row.get(p95_key)
        fraction = row.get(fraction_key)
        if not isinstance(p95, (int, float)) or not isinstance(fraction, (int, float)):
            continue
        if not math.isfinite(float(p95)) or not math.isfinite(float(fraction)):
            continue
        eligible += 1
        p95_hit = float(p95) < p95_threshold
        fraction_hit = float(fraction) > fraction_threshold
        if p95_hit and fraction_hit:
            both += 1
        elif p95_hit:
            only_p95 += 1
        elif fraction_hit:
            only_fraction += 1

    selected = only_p95 + only_fraction + both
    return {
        "eligible_rows": eligible,
        "selected_rows": selected,
        "selected_pct": round(selected / eligible, 4) if eligible else 0.0,
        "only_p95": only_p95,
        "only_dark_fraction": only_fraction,
        "both": both,
    }


def combined_or_rule_sweep_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sweep_rows = []
    metric_specs = (
        ("valid_pixels", "valid_brightness_p95", f"valid_dark_fraction_lt_{int(CURRENT_DARK_BRIGHTNESS_THRESHOLD)}"),
        ("clear_pixels", "clear_brightness_p95", f"clear_dark_fraction_lt_{int(CURRENT_DARK_BRIGHTNESS_THRESHOLD)}"),
    )
    for group in ("all", "valid_samples", "valid_non_excluded"):
        group_rows = rows_for_group(rows, group)
        for metric_space, p95_key, fraction_key in metric_specs:
            for p95_threshold, fraction_threshold in COMBINED_OR_RULE_CANDIDATES:
                sweep_rows.append(
                    {
                        "group": group,
                        "metric_space": metric_space,
                        "p95_key": p95_key,
                        "dark_fraction_key": fraction_key,
                        "p95_threshold": p95_threshold,
                        "dark_fraction_threshold": fraction_threshold,
                        **combined_or_rule_stats(
                            group_rows,
                            p95_key=p95_key,
                            fraction_key=fraction_key,
                            p95_threshold=p95_threshold,
                            fraction_threshold=fraction_threshold,
                        ),
                    }
                )
    return sweep_rows


def print_combined_or_rule_sweep(
    rows: list[dict[str, Any]],
    label: str,
    *,
    p95_key: str,
    fraction_key: str,
) -> None:
    print(label)
    for p95_threshold, fraction_threshold in COMBINED_OR_RULE_CANDIDATES:
        stats = combined_or_rule_stats(
            rows,
            p95_key=p95_key,
            fraction_key=fraction_key,
            p95_threshold=p95_threshold,
            fraction_threshold=fraction_threshold,
        )
        print(
            f"  p95 < {p95_threshold:g} OR dark > {fraction_threshold:.2f}: "
            f"{stats['selected_rows']} ({stats['selected_pct']:.1%}); "
            f"only_p95={stats['only_p95']}, "
            f"only_dark_fraction={stats['only_dark_fraction']}, "
            f"both={stats['both']}, "
            f"eligible={stats['eligible_rows']}"
        )


def maybe_write_plots(plot_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not WRITE_PLOT_PNGS:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping PNG plots.")
        return

    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_specs = [
        ("valid_brightness_p95", BRIGHTNESS_BINS),
        ("clear_brightness_p95", BRIGHTNESS_BINS),
        ("valid_brightness_p50", BRIGHTNESS_BINS),
        ("clear_brightness_p50", BRIGHTNESS_BINS),
        ("clear_brightness_p95_minus_p05", BRIGHTNESS_RANGE_BINS),
        ("clear_brightness_p95_minus_p05_div_p50", RELATIVE_CONTRAST_BINS),
        ("clear_relative_contrast", RELATIVE_CONTRAST_BINS),
        ("valid_dark_fraction_lt_350", FRACTION_BINS),
        ("clear_dark_fraction_lt_350", FRACTION_BINS),
    ]
    for key, bins in plot_specs:
        values = metric_values(rows, key)
        if not values:
            continue
        plt.figure(figsize=(9, 4))
        plt.hist(values, bins=bins, edgecolor="black")
        plt.title(key)
        plt.xlabel(key)
        plt.ylabel("images")
        plt.tight_layout()
        plt.savefig(plot_dir / f"{key}.png", dpi=140)
        plt.close()


def rows_for_group(rows: list[dict[str, Any]], group: str) -> list[dict[str, Any]]:
    if group == "valid_samples":
        return [row for row in rows if row.get("sample_is_valid") is True]
    if group == "valid_non_excluded":
        return [
            row for row in rows
            if row.get("sample_is_valid") is True and not bool(row.get("is_excluded"))
        ]
    return rows


def print_summary(rows: list[dict[str, Any]]) -> None:
    print(f"Rows analyzed: {len(rows)}")
    print(f"Manual validation: {dict(sorted(Counter(row.get('manual_validation', '') for row in rows).items()))}")
    print(f"Review states: {dict(sorted(Counter(row.get('review_state', '') for row in rows).items()))}")
    for group in ("all", "valid_samples", "valid_non_excluded"):
        group_rows = rows_for_group(rows, group)
        print(f"\n[{group}] rows={len(group_rows)}")
        print(
            "  current_dark_valid="
            f"{sum(1 for row in group_rows if row.get('current_dark_valid') is True)}; "
            "current_dark_clear="
            f"{sum(1 for row in group_rows if row.get('current_dark_clear') is True)}"
        )
        candidate_stats(
            group_rows,
            "  Candidate p95/fraction counts on valid pixels:",
            p95_key="valid_brightness_p95",
            fraction_key=f"valid_dark_fraction_lt_{int(CURRENT_DARK_BRIGHTNESS_THRESHOLD)}",
        )
        candidate_stats(
            group_rows,
            "  Candidate p95/fraction counts on clear pixels:",
            p95_key="clear_brightness_p95",
            fraction_key=f"clear_dark_fraction_lt_{int(CURRENT_DARK_BRIGHTNESS_THRESHOLD)}",
        )
        print_combined_or_rule_sweep(
            group_rows,
            "  Combined OR-rule sweep on valid pixels:",
            p95_key="valid_brightness_p95",
            fraction_key=f"valid_dark_fraction_lt_{int(CURRENT_DARK_BRIGHTNESS_THRESHOLD)}",
        )
        print_combined_or_rule_sweep(
            group_rows,
            "  Combined OR-rule sweep on clear pixels:",
            p95_key="clear_brightness_p95",
            fraction_key=f"clear_dark_fraction_lt_{int(CURRENT_DARK_BRIGHTNESS_THRESHOLD)}",
        )
        print_metric_distribution("  Clear brightness p05", metric_values(group_rows, "clear_brightness_p05"))
        print_metric_distribution("  Clear brightness p50", metric_values(group_rows, "clear_brightness_p50"))
        print_metric_distribution("  Clear brightness p95", metric_values(group_rows, "clear_brightness_p95"))
        print_metric_distribution(
            "  Clear brightness p95-p05",
            metric_values(group_rows, "clear_brightness_p95_minus_p05"),
        )
        print_metric_distribution(
            "  Clear brightness (p95-p05)/p50",
            metric_values(group_rows, "clear_relative_contrast"),
        )
        print_strict_qc_summary(group_rows)
        print_histogram("  Histogram valid_brightness_p95:", metric_values(group_rows, "valid_brightness_p95"), BRIGHTNESS_BINS)
        print_histogram("  Histogram valid_dark_fraction_lt_350:", metric_values(group_rows, "valid_dark_fraction_lt_350"), FRACTION_BINS)
        print_histogram(
            "  Histogram clear_brightness_p95_minus_p05:",
            metric_values(group_rows, "clear_brightness_p95_minus_p05"),
            BRIGHTNESS_RANGE_BINS,
        )
        print_histogram(
            "  Histogram clear_relative_contrast:",
            metric_values(group_rows, "clear_relative_contrast"),
            RELATIVE_CONTRAST_BINS,
        )


def main() -> None:
    init_forest_database()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = now_tag()
    rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    with SessionLocal() as db:
        downloads = selected_downloads(db)
        for download in downloads:
            manual = manual_payload(db, download.sample_id)
            review = review_payload(db, download.download_id)
            if ONLY_VALID_SAMPLES and not manual["sample_is_valid"]:
                continue
            if not INCLUDE_EXCLUDED_REVIEWS and review["is_excluded"]:
                continue

            path = resolve_storage_path(download.local_path)
            base = {
                "download_id": download.download_id,
                "sample_id": download.sample_id,
                "period": download.period or "",
                "scene_id": download.scene_id,
                "local_path": download.local_path,
                **manual,
                **review,
            }
            if not path.exists():
                failed.append({**base, "error": f"missing_tif: {path}"})
                continue
            try:
                rows.append({**base, **read_darkness_metrics(path)})
            except Exception as exc:
                failed.append({**base, "error": str(exc)[:500]})

    csv_path = OUTPUT_DIR / f"sentinel_darkness_metrics_{tag}.csv"
    write_csv(csv_path, rows)
    if failed:
        write_csv(OUTPUT_DIR / f"sentinel_darkness_metrics_failed_{tag}.csv", failed)

    hist_rows = []
    for key, bins in (
        ("valid_brightness_p95", BRIGHTNESS_BINS),
        ("clear_brightness_p95", BRIGHTNESS_BINS),
        ("valid_dark_fraction_lt_350", FRACTION_BINS),
        ("clear_dark_fraction_lt_350", FRACTION_BINS),
        ("clear_brightness_p95_minus_p05", BRIGHTNESS_RANGE_BINS),
        ("clear_brightness_p95_minus_p05_div_p50", RELATIVE_CONTRAST_BINS),
    ):
        for row in histogram(metric_values(rows, key), bins):
            hist_rows.append({"metric": key, **row})
    hist_path = OUTPUT_DIR / f"sentinel_darkness_histograms_{tag}.csv"
    write_csv(hist_path, hist_rows)

    sweep_rows = combined_or_rule_sweep_rows(rows)
    sweep_path = OUTPUT_DIR / f"sentinel_darkness_or_rule_sweep_{tag}.csv"
    write_csv(sweep_path, sweep_rows)

    qc_sweep_rows = strict_qc_sweep_rows(rows)
    qc_sweep_path = OUTPUT_DIR / f"sentinel_darkness_strict_qc_sweep_{tag}.csv"
    write_csv(qc_sweep_path, qc_sweep_rows)

    plot_dir = OUTPUT_DIR / f"sentinel_darkness_plots_{tag}"
    maybe_write_plots(plot_dir, rows)

    print(f"LIMIT_DOWNLOADS: {LIMIT_DOWNLOADS}")
    print(f"ONLY_VALID_SAMPLES: {ONLY_VALID_SAMPLES}")
    print(f"INCLUDE_EXCLUDED_REVIEWS: {INCLUDE_EXCLUDED_REVIEWS}")
    print(
        "Current production brightness rule: "
        f"p95 < {PRODUCTION_DARK_LOW_CONTRAST_P95_THRESHOLD:g} and "
        f"(p95-p05 < {PRODUCTION_LOW_CONTRAST_RANGE_THRESHOLD:g} or "
        f"(p95-p05)/p50 < {PRODUCTION_LOW_CONTRAST_RELATIVE_THRESHOLD:.2f})"
    )
    print_summary(rows)
    print(f"\nRows CSV: {csv_path}")
    print(f"Histogram CSV: {hist_path}")
    print(f"Combined OR-rule sweep CSV: {sweep_path}")
    print(f"Strict QC sweep CSV: {qc_sweep_path}")
    if WRITE_PLOT_PNGS:
        print(f"Plot dir: {plot_dir}")
    if failed:
        print(f"Failed/missing: {len(failed)}")


if __name__ == "__main__":
    main()
