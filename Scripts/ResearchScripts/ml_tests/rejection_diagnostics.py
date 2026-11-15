from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import select
from sqlalchemy.orm import Session

from ResearchTool.backend.forest.models import (
    EventReview,
    HansenCandidate,
    SampleEvent,
    SampleEventDownload,
    SentinelDownload,
    SentinelSceneReview,
    SentinelSceneReviewReason,
)

from .dataset import parse_scene_datetime
from .reporting import create_run_directory, write_csv, write_json
from .review_validation import (
    DEFAULT_MODEL_SOURCES,
    TARGET_LABELS,
    load_reviewer_validation,
)
from .year_mismatch_research import (
    build_year_mismatch_research,
    year_mismatch_ranking_figure,
    year_mismatch_timeline_figure,
)


REVIEW_HIERARCHY = (
    "VALID_LOGGING",
    "VALID_WILDFIRE",
    "NO_PRE",
    "QC_FAILED",
    "YEAR_MISMATCH",
    "EVENT_TOO_SMALL",
    "NO_VISIBLE_CHANGE",
    "TYPE_UNCLEAR",
    "OTHER_INVALID",
)

LATENT_FLOAT_FIELDS = (
    "knownness_k3",
    "knownness_k5",
    "knownness_k10",
    "nearest_similarity",
    "sim_logging",
    "sim_wildfire",
    "similarity_margin",
    "logreg_probability_wildfire",
    "event_loss_area_ha",
    "cluster_membership_stability",
)

LATENT_INT_FIELDS = (
    "kmeans_cluster",
    "kmeans_cluster_raw",
    "hdbscan_cluster",
    "pre_image_count",
    "post_image_count",
    "hansen_event_count",
)

LATENT_BOOL_FIELDS = (
    "hdbscan_is_noise",
    "ambiguous_flag",
    "unknown_flag",
    "is_multi_event_area",
)

VALIDITY_FEATURES = (
    "knownness_k5",
    "remoteclip_probability_wildfire",
    "remoteclip_probability_margin",
    "xgboost_probability_wildfire",
    "xgboost_probability_margin",
    "hdbscan_is_noise_number",
    "event_loss_area_log1p",
    "pre_image_count",
    "post_image_count",
    "hansen_event_count",
    "is_multi_event_area_number",
    "similarity_margin_abs",
    "event_hansen_pixels_log1p",
    "hansen_loss_aoi_percent",
    "usable_pre_download_count",
    "usable_post_download_count",
    "bad_quality_download_count",
    "excluded_download_count",
    "qc_cloud_fraction_mean",
    "qc_nodata_fraction_mean",
)

REASON_NUMERIC_FEATURES = (
    ("event_loss_area_ha", "event_area_ha"),
    ("event_hansen_pixels", "hansen_pixels"),
    ("hansen_loss_aoi_percent", "hansen_loss_aoi_percent"),
    ("pre_download_count", "pre_count"),
    ("event_download_count", "event_count"),
    ("post_download_count", "post_count"),
    ("usable_pre_download_count", "usable_pre_count"),
    ("usable_event_download_count", "usable_event_count"),
    ("usable_post_download_count", "usable_post_count"),
    ("bad_cloud_download_count", "bad_cloud_count"),
    ("bad_quality_download_count", "bad_quality_count"),
    ("excluded_download_count", "excluded_count"),
    ("qc_cloud_fraction_mean", "cloud_fraction_mean"),
    ("qc_shadow_fraction_mean", "shadow_fraction_mean"),
    ("qc_snow_fraction_mean", "snow_fraction_mean"),
    ("qc_nodata_fraction_mean", "nodata_fraction_mean"),
    ("qc_dark_fraction_mean", "dark_fraction_mean"),
    ("knownness_k5", "knownness"),
    ("remoteclip_probability_wildfire", "remoteclip_p_wildfire"),
    ("xgboost_probability_wildfire", "xgboost_p_wildfire"),
    ("sim_logging", "sim_logging"),
    ("sim_wildfire", "sim_wildfire"),
)

PIPELINE_REJECTION_GROUPS = frozenset({"NO_PRE", "QC_FAILED"})
ONE_VS_VALID_REASONS = ("EVENT_TOO_SMALL", "YEAR_MISMATCH", "NO_VISIBLE_CHANGE")


def find_latest_latent_run(output_root: str | Path) -> Path:
    root = Path(output_root)
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "latent_event_level_table.csv").exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No latent_event_level_table.csv was found under {root}. "
            "Run valid_post_sample_evaluation.ipynb first."
        )
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def _optional_float(value: Any) -> float | None:
    if value in (None, "", "None", "null", "nan"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _optional_int(value: Any) -> int | None:
    parsed = _optional_float(value)
    return int(parsed) if parsed is not None else None


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (None, "", "None", "null"):
        return None
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def _json_value(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _calendar_season(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.month in {12, 1, 2}:
        return "winter"
    if value.month in {3, 4, 5}:
        return "spring"
    if value.month in {6, 7, 8}:
        return "summer"
    return "autumn"


def _quality_flag_codes(value: str | None) -> set[str]:
    payload = _json_value(value, [])
    if isinstance(payload, list):
        return {str(item) for item in payload if str(item).strip()}
    if isinstance(payload, dict):
        result: set[str] = set()
        for key, item in payload.items():
            if isinstance(item, bool) and item:
                result.add(str(key))
            elif isinstance(item, (list, tuple, set)):
                result.update(str(entry) for entry in item if str(entry).strip())
            elif item not in (None, "", False, 0):
                result.add(str(key))
        return result
    return {str(payload)} if str(payload).strip() else set()


def _mean_or_none(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def load_event_database_features(
    db: Session,
    event_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Load Hansen geometry, Sentinel acquisition, QC, and neighboring-event features."""

    requested_ids = sorted({str(value) for value in event_ids if value})
    if not requested_ids:
        return {}

    events = list(
        db.scalars(select(SampleEvent).where(SampleEvent.event_id.in_(requested_ids))).all()
    )
    sample_ids = sorted({event.sample_id for event in events})
    candidates = list(
        db.scalars(
            select(HansenCandidate).where(HansenCandidate.created_sample_id.in_(sample_ids))
        ).all()
    )
    candidate_by_sample = {
        str(candidate.created_sample_id): candidate
        for candidate in candidates
        if candidate.created_sample_id
    }

    sibling_events = list(
        db.scalars(select(SampleEvent).where(SampleEvent.sample_id.in_(sample_ids))).all()
    )
    siblings_by_sample: dict[str, list[SampleEvent]] = defaultdict(list)
    for event in sibling_events:
        siblings_by_sample[str(event.sample_id)].append(event)
    sibling_reviews = list(
        db.scalars(
            select(EventReview).where(
                EventReview.event_id.in_([event.event_id for event in sibling_events])
            )
        ).all()
    )
    review_by_event = {str(review.event_id): review for review in sibling_reviews}

    linked_downloads = db.execute(
        select(SampleEventDownload, SentinelDownload)
        .join(
            SentinelDownload,
            SentinelDownload.download_id == SampleEventDownload.download_id,
        )
        .where(SampleEventDownload.event_id.in_(requested_ids))
        .order_by(
            SampleEventDownload.event_id,
            SampleEventDownload.period,
            SampleEventDownload.rank,
            SentinelDownload.download_id,
        )
    ).all()
    download_ids = sorted({download.download_id for _, download in linked_downloads})
    scene_reviews = (
        list(
            db.scalars(
                select(SentinelSceneReview).where(
                    SentinelSceneReview.download_id.in_(download_ids)
                )
            ).all()
        )
        if download_ids
        else []
    )
    scene_review_by_download = {
        str(review.download_id): review for review in scene_reviews
    }
    scene_reason_rows = (
        list(
            db.scalars(
                select(SentinelSceneReviewReason).where(
                    SentinelSceneReviewReason.download_id.in_(download_ids)
                )
            ).all()
        )
        if download_ids
        else []
    )
    scene_reasons_by_download: dict[str, set[str]] = defaultdict(set)
    for reason in scene_reason_rows:
        scene_reasons_by_download[str(reason.download_id)].add(str(reason.reason_code))

    downloads_by_event: dict[str, list[tuple[SampleEventDownload, SentinelDownload]]] = (
        defaultdict(list)
    )
    for link, download in linked_downloads:
        downloads_by_event[str(link.event_id)].append((link, download))

    features: dict[str, dict[str, Any]] = {}
    for event in events:
        candidate = candidate_by_sample.get(str(event.sample_id))
        event_metadata = _json_value(event.metadata_json, {})
        candidate_metadata = _json_value(candidate.metadata_json, {}) if candidate else {}
        event_pixels = _optional_int(event_metadata.get("pixels"))
        if event_pixels is None:
            histogram = candidate_metadata.get("histogram", [])
            for item in histogram if isinstance(histogram, list) else []:
                if _optional_int(item.get("year")) == int(event.event_year):
                    event_pixels = _optional_int(item.get("pixels"))
                    break

        aoi_area_ha = (
            float(candidate.aoi_size_m) ** 2 / 10_000.0 if candidate is not None else None
        )
        event_loss_area = _optional_float(event.loss_area_ha)
        loss_aoi_percent = (
            100.0 * event_loss_area / aoi_area_ha
            if event_loss_area is not None and aoi_area_ha
            else None
        )

        siblings = sorted(
            siblings_by_sample.get(str(event.sample_id), []),
            key=lambda item: (item.event_year, item.event_id),
        )
        sibling_years = [int(item.event_year) for item in siblings]
        sibling_status_by_year = {
            str(item.event_year): (
                review_by_event[str(item.event_id)].review_status
                if str(item.event_id) in review_by_event
                else None
            )
            for item in siblings
        }
        record: dict[str, Any] = {
            "candidate_id": candidate.candidate_id if candidate is not None else None,
            "candidate_aoi_size_m": float(candidate.aoi_size_m) if candidate else None,
            "candidate_aoi_area_ha": aoi_area_ha,
            "candidate_component_area_ha": (
                float(candidate.component_area_ha) if candidate is not None else None
            ),
            "candidate_dominant_loss_area_ha": (
                float(candidate.dominant_loss_area_ha) if candidate is not None else None
            ),
            "candidate_total_loss_area_ha": (
                float(candidate.total_loss_area_ha) if candidate is not None else None
            ),
            "candidate_post_2016_loss_aoi_percent": (
                float(candidate.post_2016_loss_aoi_percent) if candidate else None
            ),
            "candidate_dominant_event_aoi_percent": (
                float(candidate.dominant_event_aoi_percent) if candidate else None
            ),
            "candidate_pixel_count": int(candidate.pixel_count) if candidate else None,
            "event_hansen_pixels": event_pixels,
            "hansen_loss_aoi_percent": loss_aoi_percent,
            "neighbor_event_years": sibling_years,
            "neighbor_event_status_by_year": sibling_status_by_year,
            "has_event_y_minus_1": int(event.event_year) - 1 in sibling_years,
            "has_event_y_plus_1": int(event.event_year) + 1 in sibling_years,
        }

        period_counts: Counter[str] = Counter()
        usable_counts: Counter[str] = Counter()
        period_season_counts: Counter[str] = Counter()
        period_scene_years: dict[str, set[int]] = defaultdict(set)
        metric_values: dict[str, list[float]] = defaultdict(list)
        flags: set[str] = set()
        bad_cloud_count = 0
        bad_quality_count = 0
        excluded_count = 0
        for link, download in downloads_by_event.get(str(event.event_id), []):
            period = str(link.period or download.period or "UNKNOWN").upper()
            period_counts[period] += 1
            scene_review = scene_review_by_download.get(str(download.download_id))
            is_excluded = bool(scene_review and scene_review.is_excluded)
            is_bad_cloud = bool(download.is_bad_cloud)
            is_bad_quality = bool(download.is_bad_quality)
            bad_cloud_count += int(is_bad_cloud)
            bad_quality_count += int(is_bad_quality)
            excluded_count += int(is_excluded)
            if not (is_bad_cloud or is_bad_quality or is_excluded):
                usable_counts[period] += 1

            flags.update(_quality_flag_codes(download.quality_flags_json))
            flags.update(scene_reasons_by_download.get(str(download.download_id), set()))
            if scene_review is not None and scene_review.reason_code:
                flags.add(str(scene_review.reason_code))

            for field in ("cloud", "shadow", "snow", "nodata", "dark"):
                value = _optional_float(getattr(download, f"{field}_fraction"))
                if value is not None:
                    metric_values[field].append(value)
            scene_datetime = parse_scene_datetime(download.metadata_json, download.scene_id)
            season = _calendar_season(scene_datetime)
            if season is not None:
                period_season_counts[f"{period}:{season}"] += 1
            if scene_datetime is not None:
                period_scene_years[period].add(int(scene_datetime.year))

        record.update(
            {
                "download_count": sum(period_counts.values()),
                "pre_download_count": period_counts["PRE"],
                "event_download_count": period_counts["EVENT"],
                "post_download_count": period_counts["POST"],
                "usable_pre_download_count": usable_counts["PRE"],
                "usable_event_download_count": usable_counts["EVENT"],
                "usable_post_download_count": usable_counts["POST"],
                "bad_cloud_download_count": bad_cloud_count,
                "bad_quality_download_count": bad_quality_count,
                "excluded_download_count": excluded_count,
                "qc_flag_codes": sorted(flags),
                "qc_flag_count": len(flags),
                "period_season_counts": dict(sorted(period_season_counts.items())),
                "pre_scene_years": sorted(period_scene_years["PRE"]),
                "event_scene_years": sorted(period_scene_years["EVENT"]),
                "post_scene_years": sorted(period_scene_years["POST"]),
            }
        )
        for field in ("cloud", "shadow", "snow", "nodata", "dark"):
            values = metric_values[field]
            record[f"qc_{field}_fraction_mean"] = _mean_or_none(values)
            record[f"qc_{field}_fraction_max"] = max(values) if values else None
        features[str(event.event_id)] = record
    return features


def load_latent_event_table(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: list[dict[str, Any]] = []
    for raw in rows:
        row: dict[str, Any] = dict(raw)
        for field in LATENT_FLOAT_FIELDS:
            row[field] = _optional_float(row.get(field))
        for field in LATENT_INT_FIELDS:
            row[field] = _optional_int(row.get(field))
        for field in LATENT_BOOL_FIELDS:
            row[field] = _optional_bool(row.get(field))
        result.append(row)
    return result


def rejection_reason_group(reason: str | None) -> str:
    text = " ".join(str(reason or "").strip().casefold().replace("ё", "е").split())
    if not text:
        return "OTHER_INVALID"

    has_pre = "pre" in text
    missing_tokens = ("нет", "no ", "missing", "отсутств")
    if has_pre and any(token in text for token in missing_tokens):
        return "NO_PRE"

    year_tokens = (
        "прошлого года",
        "другого года",
        "следующего года",
        "год события",
        "для следующего года",
        "year mismatch",
        "wrong year",
        "last year",
        "previous year",
        "next year",
        "next yea",
        "another year",
        "other year",
        "different year",
    )
    explicit_year_reference = re.search(r"\b(?:19|20)\d{2}\b", text) is not None
    if any(token in text for token in year_tokens) or explicit_year_reference:
        return "YEAR_MISMATCH"

    no_change_tokens = (
        "изменений не видно",
        "не видно разницы",
        "не видно рвзницы",
        "no visible change",
        "no change",
    )
    if any(token in text for token in no_change_tokens):
        return "NO_VISIBLE_CHANGE"

    small_tokens = (
        "маленьк",
        "небольш",
        "слишком мал",
        "small event",
        "too small",
    )
    if any(token in text for token in small_tokens):
        return "EVENT_TOO_SMALL"

    unclear_tokens = (
        "невозможно определить",
        "невозможно понять",
        "сложно определить",
        "сложно понять",
        "не понятно",
        "непонятно",
        "type unclear",
        "unclear",
    )
    if any(token in text for token in unclear_tokens):
        return "TYPE_UNCLEAR"

    qc_tokens = (
        "нет валидных изображений",
        "нет ни одного валидного снимка",
        "no valid",
        "qc",
        "cloud",
        "облак",
        "no data",
    )
    if any(token in text for token in qc_tokens):
        return "QC_FAILED"
    return "OTHER_INVALID"


def review_hierarchy_class(row: dict[str, Any]) -> str:
    if row.get("review_status") == "VALID":
        if row.get("human_label") == "Logging":
            return "VALID_LOGGING"
        if row.get("human_label") == "Wildfire":
            return "VALID_WILDFIRE"
        return "OTHER_INVALID"
    return rejection_reason_group(row.get("rejection_reason"))


def human_state(row: dict[str, Any]) -> str:
    if row.get("review_status") != "VALID":
        return "Rejected"
    value = str(row.get("human_label") or "")
    return value if value in TARGET_LABELS else "Rejected"


def _probability_wildfire(row: dict[str, Any], prefix: str) -> float | None:
    probabilities = row.get(f"{prefix}_probabilities")
    if isinstance(probabilities, dict):
        for key in ("Wildfire", "wildfire", "WILDFIRE"):
            if key in probabilities:
                return _optional_float(probabilities[key])
    label = row.get(f"{prefix}_label")
    confidence = _optional_float(row.get(f"{prefix}_confidence"))
    if confidence is None:
        return None
    if label == "Wildfire":
        return confidence
    if label == "Logging":
        return 1.0 - confidence
    return None


def _normalized_entropy(probability: float | None) -> float | None:
    if probability is None:
        return None
    value = min(max(float(probability), 1e-12), 1.0 - 1e-12)
    return float(-(value * math.log2(value) + (1.0 - value) * math.log2(1.0 - value)))


def merge_review_and_latent_rows(
    review_rows: Sequence[dict[str, Any]],
    latent_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    latent_by_event = {
        str(row["event_id"]): row
        for row in latent_rows
        if row.get("source") == "HANSEN_MINED" and row.get("event_id")
    }
    merged: list[dict[str, Any]] = []
    for source_row in review_rows:
        row = dict(source_row)
        latent = latent_by_event.get(str(row["event_id"]))
        row["human_state"] = human_state(row)
        row["review_hierarchy_class"] = review_hierarchy_class(row)
        row["rejection_reason_group"] = (
            rejection_reason_group(row.get("rejection_reason"))
            if row["human_state"] == "Rejected"
            else None
        )
        row["latent_available"] = latent is not None
        if latent is not None:
            for key, value in latent.items():
                if key in {"sample_id", "event_id", "event_year", "true_label"}:
                    continue
                row[key] = value
        remote_probability = _probability_wildfire(row, "remoteclip")
        xgboost_probability = _probability_wildfire(row, "xgboost")
        if remote_probability is None and latent is not None:
            remote_probability = _optional_float(latent.get("logreg_probability_wildfire"))
        row["remoteclip_probability_wildfire"] = remote_probability
        row["xgboost_probability_wildfire"] = xgboost_probability
        row["remoteclip_probability_margin"] = (
            abs(remote_probability - 0.5) * 2.0 if remote_probability is not None else None
        )
        row["xgboost_probability_margin"] = (
            abs(xgboost_probability - 0.5) * 2.0 if xgboost_probability is not None else None
        )
        row["remoteclip_probability_entropy"] = _normalized_entropy(remote_probability)
        row["xgboost_probability_entropy"] = _normalized_entropy(xgboost_probability)
        row["hdbscan_is_noise_number"] = (
            float(bool(row.get("hdbscan_is_noise")))
            if row.get("hdbscan_is_noise") is not None
            else None
        )
        row["is_multi_event_area_number"] = (
            float(bool(row.get("is_multi_event_area")))
            if row.get("is_multi_event_area") is not None
            else None
        )
        area = _optional_float(row.get("event_loss_area_ha"))
        row["event_loss_area_log1p"] = math.log1p(max(area, 0.0)) if area is not None else None
        event_pixels = _optional_float(row.get("event_hansen_pixels"))
        row["event_hansen_pixels_log1p"] = (
            math.log1p(max(event_pixels, 0.0)) if event_pixels is not None else None
        )
        similarity_margin = _optional_float(row.get("similarity_margin"))
        row["similarity_margin_abs"] = (
            abs(similarity_margin) if similarity_margin is not None else None
        )
        if row.get("remoteclip_label") in TARGET_LABELS and row.get("xgboost_label") in TARGET_LABELS:
            row["model_agreement_type"] = (
                f"RC={row['remoteclip_label']} / XGB={row['xgboost_label']}"
            )
            row["models_disagree"] = row["remoteclip_label"] != row["xgboost_label"]
        else:
            row["model_agreement_type"] = "MISSING_MODEL_LABEL"
            row["models_disagree"] = None
        merged.append(row)
    return merged


def _distribution_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "q05": None,
            "q25": None,
            "median": None,
            "q75": None,
            "q95": None,
        }
    array = np.asarray(values, dtype=float)
    quantiles = np.quantile(array, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "q05": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "q95": float(quantiles[4]),
    }


def _group_distribution(
    rows: Sequence[dict[str, Any]],
    *,
    group_key: str,
    value_key: str,
    value_name: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _optional_float(row.get(value_key))
        if value is not None:
            grouped[str(row.get(group_key) or "(missing)")].append(value)
    return [
        {group_key: group, "metric": value_name, **_distribution_summary(values)}
        for group, values in sorted(grouped.items())
    ]


def three_state_tables(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for model, prefix in (("REMOTECLIP_LOGREG", "remoteclip"), ("XGBOOST", "xgboost")):
        available = [row for row in rows if row.get(f"{prefix}_label") in TARGET_LABELS]
        human_totals = Counter(str(row["human_state"]) for row in available)
        counts = Counter(
            (str(row[f"{prefix}_label"]), str(row["human_state"])) for row in available
        )
        for prediction in TARGET_LABELS:
            for state in (*TARGET_LABELS, "Rejected"):
                count = counts[(prediction, state)]
                result.append(
                    {
                        "model": model,
                        "predicted_label": prediction,
                        "human_state": state,
                        "count": count,
                        "human_state_percent": (
                            100.0 * count / human_totals[state] if human_totals[state] else 0.0
                        ),
                    }
                )
    return result


def probability_summaries(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for model, prefix in (("REMOTECLIP_LOGREG", "remoteclip"), ("XGBOOST", "xgboost")):
        for state in (*TARGET_LABELS, "Rejected"):
            selected = [row for row in rows if row["human_state"] == state]
            probabilities = [
                value
                for row in selected
                if (value := _optional_float(row.get(f"{prefix}_probability_wildfire")))
                is not None
            ]
            entropies = [
                value
                for row in selected
                if (value := _optional_float(row.get(f"{prefix}_probability_entropy")))
                is not None
            ]
            result.append(
                {
                    "model": model,
                    "human_state": state,
                    **_distribution_summary(probabilities),
                    "entropy_mean": float(np.mean(entropies)) if entropies else None,
                    "extreme_probability_share": (
                        sum(value <= 0.1 or value >= 0.9 for value in probabilities)
                        / len(probabilities)
                        if probabilities
                        else None
                    ),
                }
            )
    return result


def knownness_summaries(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    status_rows = _group_distribution(
        rows,
        group_key="review_status",
        value_key="knownness_k5",
        value_name="knownness_k5",
    )
    rejected = [row for row in rows if row["human_state"] == "Rejected"]
    reason_rows: list[dict[str, Any]] = []
    for group in REVIEW_HIERARCHY[2:]:
        selected = [row for row in rejected if row["rejection_reason_group"] == group]
        if not selected:
            continue
        knownness = [
            value
            for row in selected
            if (value := _optional_float(row.get("knownness_k5"))) is not None
        ]
        remote = [
            value
            for row in selected
            if (value := _optional_float(row.get("remoteclip_probability_wildfire"))) is not None
        ]
        xgboost = [
            value
            for row in selected
            if (value := _optional_float(row.get("xgboost_probability_wildfire"))) is not None
        ]
        hdbscan = [
            bool(row["hdbscan_is_noise"])
            for row in selected
            if row.get("hdbscan_is_noise") is not None
        ]
        reason_rows.append(
            {
                "rejection_reason_group": group,
                "events": len(selected),
                "latent_events": len(knownness),
                "knownness_mean": float(np.mean(knownness)) if knownness else None,
                "knownness_median": float(np.median(knownness)) if knownness else None,
                "remoteclip_p_wildfire_mean": float(np.mean(remote)) if remote else None,
                "xgboost_p_wildfire_mean": float(np.mean(xgboost)) if xgboost else None,
                "hdbscan_noise_share": float(np.mean(hdbscan)) if hdbscan else None,
                "raw_reason_variants": len(
                    {str(row.get("rejection_reason") or "") for row in selected}
                ),
            }
        )
    return status_rows, reason_rows


def hdbscan_rejection_test(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    available = [row for row in rows if row.get("hdbscan_is_noise") is not None]
    rejected_noise = sum(
        row["human_state"] == "Rejected" and bool(row["hdbscan_is_noise"])
        for row in available
    )
    rejected_clustered = sum(
        row["human_state"] == "Rejected" and not bool(row["hdbscan_is_noise"])
        for row in available
    )
    valid_noise = sum(
        row["human_state"] != "Rejected" and bool(row["hdbscan_is_noise"])
        for row in available
    )
    valid_clustered = sum(
        row["human_state"] != "Rejected" and not bool(row["hdbscan_is_noise"])
        for row in available
    )
    corrected_odds = (
        (rejected_noise + 0.5)
        * (valid_clustered + 0.5)
        / ((rejected_clustered + 0.5) * (valid_noise + 0.5))
    )
    try:
        from scipy.stats import fisher_exact

        fisher_odds, p_value = fisher_exact(
            [[rejected_noise, rejected_clustered], [valid_noise, valid_clustered]],
            alternative="two-sided",
        )
        fisher_odds = float(fisher_odds)
        p_value = float(p_value)
    except (ImportError, ValueError):
        fisher_odds = None
        p_value = None
    return {
        "eligible_events": len(available),
        "rejected_noise": rejected_noise,
        "rejected_clustered": rejected_clustered,
        "valid_noise": valid_noise,
        "valid_clustered": valid_clustered,
        "noise_share_rejected": (
            rejected_noise / (rejected_noise + rejected_clustered)
            if rejected_noise + rejected_clustered
            else None
        ),
        "noise_share_valid": (
            valid_noise / (valid_noise + valid_clustered)
            if valid_noise + valid_clustered
            else None
        ),
        "odds_ratio_haldane_corrected": float(corrected_odds),
        "fisher_odds_ratio": fisher_odds,
        "fisher_p_value": p_value,
        "interpretation_note": (
            "For HANSEN_MINED rows the current latent pipeline assigns HDBSCAN status "
            "from the nearest manual event; it is diagnostic, not an inductive HDBSCAN prediction."
        ),
    }


def model_disagreement_table(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model_agreement_type"])].append(row)
    result: list[dict[str, Any]] = []
    for group, selected in sorted(grouped.items()):
        total = len(selected)
        counts = Counter(str(row["human_state"]) for row in selected)
        result.append(
            {
                "model_agreement_type": group,
                "events": total,
                "valid_events": counts["Logging"] + counts["Wildfire"],
                "rejected_events": counts["Rejected"],
                "valid_share": (counts["Logging"] + counts["Wildfire"]) / total,
                "rejected_share": counts["Rejected"] / total,
                "human_logging_share": counts["Logging"] / total,
                "human_wildfire_share": counts["Wildfire"] / total,
            }
        )
    return result


def remoteclip_error_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "sample_id",
        "display_name",
        "event_id",
        "event_year",
        "human_label",
        "remoteclip_label",
        "xgboost_label",
        "remoteclip_probability_wildfire",
        "knownness_k5",
        "kmeans_cluster",
        "kmeans_cluster_role",
        "hdbscan_is_noise",
        "sim_logging",
        "sim_wildfire",
        "similarity_margin",
        "event_loss_area_ha",
        "pre_image_count",
        "post_image_count",
        "review_notes",
        "rejection_reason",
        "rejection_reason_group",
    )
    selected = [
        row
        for row in rows
        if row["human_state"] in TARGET_LABELS
        and row.get("remoteclip_label") in TARGET_LABELS
        and row["remoteclip_label"] != row["human_state"]
    ]
    return [{key: row.get(key) for key in fields} for row in selected]


def point_level_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["sample_id"])].append(row)
    result: list[dict[str, Any]] = []
    for sample_id, events in sorted(grouped.items()):
        first = events[0]
        valid = [row for row in events if row["human_state"] in TARGET_LABELS]
        labels = {str(row["human_state"]) for row in valid}
        if not valid:
            point_state = "ONLY_REJECTED"
        elif len(labels) > 1:
            point_state = "MIXED_VALID_HISTORY"
        elif "Logging" in labels:
            point_state = "VALID_LOGGING_ONLY"
        else:
            point_state = "VALID_WILDFIRE_ONLY"
        record: dict[str, Any] = {
            "sample_id": sample_id,
            "display_name": first.get("display_name"),
            "reviewed_events": len(events),
            "valid_events": len(valid),
            "rejected_events": len(events) - len(valid),
            "has_any_valid_logging": "Logging" in labels,
            "has_any_valid_wildfire": "Wildfire" in labels,
            "has_only_rejected_events": not valid,
            "has_mixed_valid_history": len(labels) > 1,
            "point_state": point_state,
            "reviewed_years": sorted(int(row["event_year"]) for row in events),
            "human_valid_history": sorted(labels),
        }
        for prefix in ("remoteclip", "xgboost"):
            evaluable = [row for row in valid if row.get(f"{prefix}_label") in TARGET_LABELS]
            predicted_history = {str(row[f"{prefix}_label"]) for row in evaluable}
            record[f"{prefix}_evaluable_valid_events"] = len(evaluable)
            record[f"{prefix}_all_valid_events_correct"] = (
                all(row[f"{prefix}_label"] == row["human_state"] for row in evaluable)
                if len(evaluable) == len(valid) and valid
                else None
            )
            record[f"{prefix}_predicted_valid_history"] = sorted(predicted_history)
            record[f"{prefix}_history_set_exact"] = (
                predicted_history == labels if len(evaluable) == len(valid) and valid else None
            )
        result.append(record)
    return result


def point_level_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for state, count in Counter(str(row["point_state"]) for row in rows).most_common():
        result.append({"metric": "point_state", "value": state, "count": count})
    for prefix in ("remoteclip", "xgboost"):
        eligible = [
            row for row in rows if row.get(f"{prefix}_all_valid_events_correct") is not None
        ]
        result.extend(
            [
                {
                    "metric": f"{prefix}_all_valid_events_correct",
                    "value": "True",
                    "count": sum(bool(row[f"{prefix}_all_valid_events_correct"]) for row in eligible),
                    "eligible_points": len(eligible),
                },
                {
                    "metric": f"{prefix}_history_set_exact",
                    "value": "True",
                    "count": sum(bool(row[f"{prefix}_history_set_exact"]) for row in eligible),
                    "eligible_points": len(eligible),
                },
            ]
        )
    return result


def _analysis_reason(row: dict[str, Any]) -> str:
    return (
        "VALID"
        if row.get("human_state") in TARGET_LABELS
        else str(row.get("rejection_reason_group") or "OTHER_INVALID")
    )


def _numeric_values(rows: Sequence[dict[str, Any]], key: str) -> list[float]:
    return [
        value
        for row in rows
        if (value := _optional_float(row.get(key))) is not None
    ]


def _preferred_count(row: dict[str, Any], database_key: str, latent_key: str) -> float | None:
    database_value = _optional_float(row.get(database_key))
    return database_value if database_value is not None else _optional_float(row.get(latent_key))


def reason_feature_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_analysis_reason(row)].append(row)
    total = len(rows)
    order = {"VALID": 0, **{name: index + 1 for index, name in enumerate(REVIEW_HIERARCHY[2:])}}
    result: list[dict[str, Any]] = []
    for reason, selected in sorted(grouped.items(), key=lambda item: order.get(item[0], 999)):
        model_rows = [
            row for row in selected if row.get("models_disagree") is not None
        ]
        pre_counts = [
            value
            for row in selected
            if (value := _preferred_count(row, "pre_download_count", "pre_image_count"))
            is not None
        ]
        post_counts = [
            value
            for row in selected
            if (value := _preferred_count(row, "post_download_count", "post_image_count"))
            is not None
        ]
        hdbscan = [
            float(bool(row["hdbscan_is_noise"]))
            for row in selected
            if row.get("hdbscan_is_noise") is not None
        ]
        result.append(
            {
                "reason": reason,
                "count": len(selected),
                "share": len(selected) / total if total else None,
                "median_event_area": (
                    float(np.median(values))
                    if (values := _numeric_values(selected, "event_loss_area_ha"))
                    else None
                ),
                "median_hansen_loss_aoi_percent": (
                    float(np.median(values))
                    if (values := _numeric_values(selected, "hansen_loss_aoi_percent"))
                    else None
                ),
                "median_hansen_pixels": (
                    float(np.median(values))
                    if (values := _numeric_values(selected, "event_hansen_pixels"))
                    else None
                ),
                "median_knownness": (
                    float(np.median(values))
                    if (values := _numeric_values(selected, "knownness_k5"))
                    else None
                ),
                "mean_RC_P(W)": _mean_or_none(
                    _numeric_values(selected, "remoteclip_probability_wildfire")
                ),
                "mean_XGB_P(W)": _mean_or_none(
                    _numeric_values(selected, "xgboost_probability_wildfire")
                ),
                "model_disagreement_share": (
                    float(np.mean([bool(row["models_disagree"]) for row in model_rows]))
                    if model_rows
                    else None
                ),
                "median_PRE_count": float(np.median(pre_counts)) if pre_counts else None,
                "median_POST_count": float(np.median(post_counts)) if post_counts else None,
                "HDBSCAN_noise_share": _mean_or_none(hdbscan),
                "mean_bad_quality_count": _mean_or_none(
                    _numeric_values(selected, "bad_quality_download_count")
                ),
                "mean_excluded_count": _mean_or_none(
                    _numeric_values(selected, "excluded_download_count")
                ),
            }
        )
    return result


def reason_feature_distributions(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_analysis_reason(row)].append(row)
    result: list[dict[str, Any]] = []
    for reason, selected in sorted(grouped.items()):
        for key, label in REASON_NUMERIC_FEATURES:
            result.append(
                {
                    "reason": reason,
                    "feature": label,
                    **_distribution_summary(_numeric_values(selected, key)),
                }
            )
    return result


def reason_year_counts(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter((_analysis_reason(row), int(row["event_year"])) for row in rows)
    return [
        {"reason": reason, "event_year": year, "count": count}
        for (reason, year), count in sorted(counts.items())
    ]


def reason_season_counts(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        reason = _analysis_reason(row)
        for key, count in dict(row.get("period_season_counts") or {}).items():
            period, _, season = str(key).partition(":")
            counts[(reason, period or "UNKNOWN", season or "unknown")] += int(count)
    return [
        {"reason": reason, "period": period, "season": season, "image_count": count}
        for (reason, period, season), count in sorted(counts.items())
    ]


def reason_qc_flag_counts(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        reason = _analysis_reason(row)
        for flag in row.get("qc_flag_codes") or []:
            counts[(reason, str(flag))] += 1
    return [
        {"reason": reason, "qc_flag": flag, "event_count": count}
        for (reason, flag), count in sorted(counts.items())
    ]


def small_event_threshold_analysis(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    cohort = [
        row
        for row in rows
        if _analysis_reason(row) in {"VALID", "EVENT_TOO_SMALL"}
    ]
    threshold_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    feature_summaries: list[dict[str, Any]] = []
    features = (
        ("event_loss_area_ha", "event_area_ha", "ha"),
        ("event_hansen_pixels", "hansen_pixels", "pixels"),
        ("hansen_loss_aoi_percent", "hansen_loss_aoi_percent", "% AOI"),
    )
    for key, label, unit in features:
        available = [
            row for row in cohort if _optional_float(row.get(key)) is not None
        ]
        targets = np.asarray(
            [_analysis_reason(row) == "EVENT_TOO_SMALL" for row in available], dtype=int
        )
        values = np.asarray([float(row[key]) for row in available], dtype=float)
        if not len(available) or len(set(targets.tolist())) < 2:
            feature_summaries.append(
                {"feature": label, "unit": unit, "status": "insufficient_data"}
            )
            continue
        for threshold in sorted(set(values.tolist())):
            predicted = values <= threshold
            true_positive = int(np.sum(predicted & (targets == 1)))
            false_positive = int(np.sum(predicted & (targets == 0)))
            false_negative = int(np.sum((~predicted) & (targets == 1)))
            precision = true_positive / (true_positive + false_positive) if np.sum(predicted) else 0.0
            recall = true_positive / int(np.sum(targets == 1))
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            threshold_rows.append(
                {
                    "feature": label,
                    "unit": unit,
                    "threshold": float(threshold),
                    "eligible_events": len(available),
                    "below_or_equal_count": int(np.sum(predicted)),
                    "event_too_small_below": true_positive,
                    "valid_below": false_positive,
                    "event_too_small_above": false_negative,
                    "reject_rate_below": precision,
                    "event_too_small_recall": recall,
                    "f1": f1,
                }
            )

        feature_thresholds = [
            row for row in threshold_rows if row["feature"] == label
        ]
        qualifying = [
            row
            for row in feature_thresholds
            if row["below_or_equal_count"] >= 5 and row["reject_rate_below"] >= 0.70
        ]
        if qualifying:
            recommended = max(
                qualifying,
                key=lambda row: (
                    row["event_too_small_recall"],
                    row["reject_rate_below"],
                    row["threshold"],
                ),
            )
            recommendation_status = "candidate_rule_at_least_70_percent_precision"
        else:
            recommended = max(
                feature_thresholds,
                key=lambda row: (row["f1"], row["reject_rate_below"]),
            )
            recommendation_status = "best_exploratory_f1_below_70_percent_precision"
        feature_summaries.append(
            {
                "feature": label,
                "unit": unit,
                "status": "ok",
                "eligible_events": len(available),
                "event_too_small_events": int(np.sum(targets == 1)),
                "valid_events": int(np.sum(targets == 0)),
                "roc_auc_for_smaller_value": float(roc_auc_score(targets, -values)),
                "recommendation_status": recommendation_status,
                "recommended_threshold": recommended["threshold"],
                "reject_rate_below": recommended["reject_rate_below"],
                "event_too_small_recall": recommended["event_too_small_recall"],
                "events_below": recommended["below_or_equal_count"],
                "valid_events_below": recommended["valid_below"],
            }
        )

        edges = sorted(set(float(value) for value in np.quantile(values, np.linspace(0, 1, 6))))
        for index in range(max(0, len(edges) - 1)):
            lower, upper = edges[index], edges[index + 1]
            in_bin = (values >= lower) & (values <= upper) if index == 0 else (values > lower) & (values <= upper)
            if not np.any(in_bin):
                continue
            bin_targets = targets[in_bin]
            bin_rows.append(
                {
                    "feature": label,
                    "bin_index": index + 1,
                    "lower": lower,
                    "upper": upper,
                    "events": int(np.sum(in_bin)),
                    "event_too_small_events": int(np.sum(bin_targets)),
                    "event_too_small_share": float(np.mean(bin_targets)),
                }
            )
    return {
        "summary": feature_summaries,
        "thresholds": threshold_rows,
        "bins": bin_rows,
        "interpretation_note": (
            "Thresholds are descriptive and use the same reviewed cohort that motivated the rule. "
            "A candidate deterministic rule requires confirmation on a later review batch."
        ),
    }


def infer_expert_event_year(
    rejection_reason: str | None,
    hansen_year: int,
) -> tuple[int | None, int | None, str]:
    text = " ".join(str(rejection_reason or "").casefold().split())
    explicit_years = [int(value) for value in re.findall(r"\b20\d{2}\b", text)]
    if explicit_years:
        year = explicit_years[0]
        return year, year - hansen_year, "explicit_year_in_free_text"
    if any(token in text for token in ("прошл", "previous", "prior year")):
        return hansen_year - 1, -1, "relative_year_in_free_text"
    if any(token in text for token in ("следующ", "next year")):
        return hansen_year + 1, 1, "relative_year_in_free_text"
    if any(token in text for token in ("другого года", "wrong year", "year mismatch")):
        return None, None, "different_year_without_direction"
    return None, None, "expert_year_not_structured"


def year_mismatch_temporal_analysis(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    for row in rows:
        if _analysis_reason(row) != "YEAR_MISMATCH":
            continue
        hansen_year = int(row["event_year"])
        expert_year, offset, source = infer_expert_event_year(
            row.get("rejection_reason"), hansen_year
        )
        neighbor_years = [int(value) for value in row.get("neighbor_event_years") or []]
        status_by_year = dict(row.get("neighbor_event_status_by_year") or {})
        details.append(
            {
                "sample_id": row.get("sample_id"),
                "event_id": row.get("event_id"),
                "hansen_event_year": hansen_year,
                "inferred_expert_year": expert_year,
                "expert_year_offset": offset,
                "expert_year_source": source,
                "has_event_y_minus_1": bool(row.get("has_event_y_minus_1")),
                "has_event_y": hansen_year in neighbor_years,
                "has_event_y_plus_1": bool(row.get("has_event_y_plus_1")),
                "inferred_expert_year_exists_as_hansen_event": (
                    expert_year in neighbor_years if expert_year is not None else None
                ),
                "inferred_expert_year_review_status": (
                    status_by_year.get(str(expert_year)) if expert_year is not None else None
                ),
                "pre_download_count": row.get("pre_download_count"),
                "event_download_count": row.get("event_download_count"),
                "post_download_count": row.get("post_download_count"),
                "usable_pre_download_count": row.get("usable_pre_download_count"),
                "usable_event_download_count": row.get("usable_event_download_count"),
                "usable_post_download_count": row.get("usable_post_download_count"),
                "pre_scene_years": row.get("pre_scene_years"),
                "event_scene_years": row.get("event_scene_years"),
                "post_scene_years": row.get("post_scene_years"),
                "neighbor_event_years": neighbor_years,
                "rejection_reason": row.get("rejection_reason"),
                "review_notes": row.get("review_notes"),
                "firms_available": False,
            }
        )
    offsets = Counter(
        "unknown" if row["expert_year_offset"] is None else str(row["expert_year_offset"])
        for row in details
    )
    inferred = [row for row in details if row["inferred_expert_year"] is not None]
    summary = {
        "events": len(details),
        "expert_year_inferred": len(inferred),
        "expert_year_not_inferred": len(details) - len(inferred),
        "offset_counts": dict(sorted(offsets.items())),
        "inferred_year_exists_as_hansen_event": sum(
            bool(row["inferred_expert_year_exists_as_hansen_event"]) for row in inferred
        ),
        "has_y_minus_1": sum(bool(row["has_event_y_minus_1"]) for row in details),
        "has_y_plus_1": sum(bool(row["has_event_y_plus_1"]) for row in details),
        "structured_expert_year_available": False,
        "firms_available": False,
        "data_gap_note": (
            "The schema stores the expert-visible year only in free text, so Y-1/Y+1 is inferred "
            "only when the reviewer explicitly wrote a relative year. FIRMS is not integrated in "
            "the current database and is therefore reported as unavailable."
        ),
    }
    return {"summary": summary, "events": details}


def acquisition_pipeline_statistics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if _analysis_reason(row) in PIPELINE_REJECTION_GROUPS]
    summary: list[dict[str, Any]] = []
    for reason in sorted(PIPELINE_REJECTION_GROUPS):
        group = [row for row in selected if _analysis_reason(row) == reason]
        if not group:
            continue
        summary.append(
            {
                "reason": reason,
                "events": len(group),
                "points": len({str(row["sample_id"]) for row in group}),
                "median_pre_count": (
                    float(np.median(values))
                    if (values := _numeric_values(group, "pre_download_count"))
                    else None
                ),
                "median_event_count": (
                    float(np.median(values))
                    if (values := _numeric_values(group, "event_download_count"))
                    else None
                ),
                "median_post_count": (
                    float(np.median(values))
                    if (values := _numeric_values(group, "post_download_count"))
                    else None
                ),
                "median_usable_pre_count": (
                    float(np.median(values))
                    if (values := _numeric_values(group, "usable_pre_download_count"))
                    else None
                ),
                "median_usable_post_count": (
                    float(np.median(values))
                    if (values := _numeric_values(group, "usable_post_download_count"))
                    else None
                ),
                "bad_cloud_downloads": int(
                    sum(_numeric_values(group, "bad_cloud_download_count"))
                ),
                "bad_quality_downloads": int(
                    sum(_numeric_values(group, "bad_quality_download_count"))
                ),
                "manually_excluded_downloads": int(
                    sum(_numeric_values(group, "excluded_download_count"))
                ),
                "mean_cloud_fraction": _mean_or_none(
                    _numeric_values(group, "qc_cloud_fraction_mean")
                ),
                "mean_nodata_fraction": _mean_or_none(
                    _numeric_values(group, "qc_nodata_fraction_mean")
                ),
            }
        )
    return {
        "summary": summary,
        "by_year": [
            row for row in reason_year_counts(selected) if row["reason"] in PIPELINE_REJECTION_GROUPS
        ],
        "by_season": [
            row for row in reason_season_counts(selected) if row["reason"] in PIPELINE_REJECTION_GROUPS
        ],
        "qc_flags": [
            row for row in reason_qc_flag_counts(selected) if row["reason"] in PIPELINE_REJECTION_GROUPS
        ],
        "event_rows": selected,
        "interpretation_note": (
            "NO_PRE and QC_FAILED are acquisition-pipeline outcomes. They are excluded from the "
            "exploratory validity classifiers and summarized only as coverage/QC losses."
        ),
    }


def reason_model_agreement_statistics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    available = [
        row
        for row in rows
        if row.get("remoteclip_label") in TARGET_LABELS
        and row.get("xgboost_label") in TARGET_LABELS
    ]
    cross_counts = Counter(
        (_analysis_reason(row), str(row["model_agreement_type"])) for row in available
    )
    reason_totals = Counter(_analysis_reason(row) for row in available)
    pattern_totals = Counter(str(row["model_agreement_type"]) for row in available)
    cross_rows = [
        {
            "reason": reason,
            "model_agreement_type": pattern,
            "count": count,
            "share_within_reason": count / reason_totals[reason],
            "share_within_model_pattern": count / pattern_totals[pattern],
        }
        for (reason, pattern), count in sorted(cross_counts.items())
    ]
    conditionals = [
        {
            "model_condition": pattern,
            "reason": reason,
            "count": count,
            "condition_total": pattern_totals[pattern],
            "probability_reason_given_condition": count / pattern_totals[pattern],
        }
        for (reason, pattern), count in sorted(cross_counts.items(), key=lambda item: (item[0][1], item[0][0]))
    ]
    disagreement = [row for row in available if bool(row["models_disagree"])]
    both_wildfire = [
        row
        for row in available
        if row["remoteclip_label"] == "Wildfire" and row["xgboost_label"] == "Wildfire"
    ]
    headline = {
        "P(EVENT_TOO_SMALL | RC!=XGB)": (
            sum(_analysis_reason(row) == "EVENT_TOO_SMALL" for row in disagreement)
            / len(disagreement)
            if disagreement
            else None
        ),
        "P(YEAR_MISMATCH | RC=XGB=Wildfire)": (
            sum(_analysis_reason(row) == "YEAR_MISMATCH" for row in both_wildfire)
            / len(both_wildfire)
            if both_wildfire
            else None
        ),
        "model_complete_events": len(available),
        "model_disagreement_events": len(disagreement),
        "both_wildfire_events": len(both_wildfire),
    }
    return {"cross_table": cross_rows, "conditionals": conditionals, "headline": headline}


def _one_vs_valid_feature_rows(
    rows: Sequence[dict[str, Any]],
    reason: str,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if _analysis_reason(row) in {"VALID", reason}
        and _analysis_reason(row) not in PIPELINE_REJECTION_GROUPS
    ]


def one_vs_valid_diagnostics(
    rows: Sequence[dict[str, Any]],
    *,
    reasons: Sequence[str] = ONE_VS_VALID_REASONS,
    seed: int = 42,
    requested_splits: int = 5,
) -> dict[str, Any]:
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    for reason in reasons:
        selected = _one_vs_valid_feature_rows(rows, reason)
        positive = [row for row in selected if _analysis_reason(row) == reason]
        valid = [row for row in selected if _analysis_reason(row) == "VALID"]
        base = {
            "reason": reason,
            "events": len(selected),
            "valid_events": len(valid),
            "reason_events": len(positive),
            "samples": len({str(row["sample_id"]) for row in selected}),
        }
        if len(positive) < 5 or len(valid) < 5:
            metric_rows.append({**base, "status": "insufficient_data_minimum_5_per_class"})
            continue
        feature_names = [
            feature
            for feature in VALIDITY_FEATURES
            if any(_optional_float(row.get(feature)) is not None for row in selected)
        ]
        if not feature_names:
            metric_rows.append({**base, "status": "no_available_features"})
            continue
        matrix = np.asarray(
            [
                [
                    np.nan if _optional_float(row.get(feature)) is None else float(row[feature])
                    for feature in feature_names
                ]
                for row in selected
            ],
            dtype=float,
        )
        target = np.asarray([int(_analysis_reason(row) == reason) for row in selected], dtype=int)
        groups = np.asarray([str(row["sample_id"]) for row in selected])
        group_counts = [len(set(groups[target == value])) for value in (0, 1)]
        n_splits = min(requested_splits, *group_counts)
        if n_splits < 2:
            metric_rows.append({**base, "status": "insufficient_groups"})
            continue
        estimator = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced", max_iter=3000, random_state=seed
                    ),
                ),
            ]
        )
        probabilities = np.full(len(selected), np.nan, dtype=float)
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        for fold, (train_index, test_index) in enumerate(
            splitter.split(matrix, target, groups=groups), start=1
        ):
            fitted = clone(estimator).fit(matrix[train_index], target[train_index])
            probabilities[test_index] = fitted.predict_proba(matrix[test_index])[:, 1]
            for index in test_index:
                prediction_rows.append(
                    {
                        "reason": reason,
                        "fold": fold,
                        "sample_id": selected[index]["sample_id"],
                        "event_id": selected[index]["event_id"],
                        "is_reason": bool(target[index]),
                        "probability_reason": float(probabilities[index]),
                    }
                )
        predicted = (probabilities >= 0.5).astype(int)
        metric_rows.append(
            {
                **base,
                "status": "ok",
                "folds": n_splits,
                "features": feature_names,
                "balanced_accuracy": float(balanced_accuracy_score(target, predicted)),
                "precision_reason": float(precision_score(target, predicted, zero_division=0)),
                "recall_reason": float(recall_score(target, predicted, zero_division=0)),
                "f1_reason": float(f1_score(target, predicted, zero_division=0)),
                "roc_auc": float(roc_auc_score(target, probabilities)),
                "average_precision": float(average_precision_score(target, probabilities)),
            }
        )
        fitted_full = clone(estimator).fit(matrix, target)
        for feature, coefficient in zip(
            feature_names,
            fitted_full.named_steps["classifier"].coef_[0],
            strict=True,
        ):
            coefficient_rows.append(
                {
                    "reason": reason,
                    "feature": feature,
                    "standardized_coefficient": float(coefficient),
                    "absolute_value": abs(float(coefficient)),
                }
            )
    return {
        "metrics": metric_rows,
        "predictions": prediction_rows,
        "coefficients": sorted(
            coefficient_rows,
            key=lambda row: (str(row["reason"]), -float(row["absolute_value"])),
        ),
        "interpretation_note": (
            "One-vs-VALID logistic models are exploratory grouped-CV diagnostics. NO_PRE and "
            "QC_FAILED are intentionally excluded because they describe acquisition failures."
        ),
    }


def point_reason_relationships(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["sample_id"])].append(row)
    cooccurrence: Counter[tuple[str, str]] = Counter()
    transitions: Counter[tuple[str, str, int]] = Counter()
    histories: list[dict[str, Any]] = []
    for sample_id, events in sorted(grouped.items()):
        ordered = sorted(events, key=lambda row: (int(row["event_year"]), str(row["event_id"])))
        labels = [str(row["review_hierarchy_class"]) for row in ordered]
        label_counts = Counter(labels)
        unique_labels = sorted(label_counts)
        for left, right in combinations(unique_labels, 2):
            cooccurrence[(left, right)] += 1
        for label, count in label_counts.items():
            if count > 1:
                cooccurrence[(label, label)] += 1
        for left, right in zip(ordered, ordered[1:]):
            gap = int(right["event_year"]) - int(left["event_year"])
            transitions[
                (
                    str(left["review_hierarchy_class"]),
                    str(right["review_hierarchy_class"]),
                    gap,
                )
            ] += 1
        histories.append(
            {
                "sample_id": sample_id,
                "display_name": ordered[0].get("display_name"),
                "reviewed_events": len(ordered),
                "year_reason_history": [
                    {
                        "event_year": int(row["event_year"]),
                        "review_class": row["review_hierarchy_class"],
                    }
                    for row in ordered
                ],
                "review_classes": unique_labels,
            }
        )
    return {
        "cooccurrence": [
            {"class_a": left, "class_b": right, "point_count": count}
            for (left, right), count in sorted(
                cooccurrence.items(), key=lambda item: (-item[1], item[0])
            )
        ],
        "adjacent_transitions": [
            {
                "from_class": left,
                "to_class": right,
                "year_gap": gap,
                "transition_count": count,
            }
            for (left, right, gap), count in sorted(
                transitions.items(), key=lambda item: (-item[1], item[0])
            )
        ],
        "histories": histories,
    }


def coverage_analysis(
    rows: Sequence[dict[str, Any]],
    latent_reference_rows: Sequence[dict[str, Any]],
    *,
    quantiles: Sequence[float] = (0.01, 0.05, 0.10, 0.15, 0.20, 0.25),
) -> list[dict[str, Any]]:
    reference_values = [
        float(row["knownness_k5"])
        for row in latent_reference_rows
        if row.get("source") == "MANUAL_TRAIN" and row.get("knownness_k5") is not None
    ]
    eligible = [row for row in rows if row.get("knownness_k5") is not None]
    if not reference_values or not eligible:
        return []
    result: list[dict[str, Any]] = []
    for quantile in quantiles:
        threshold = float(np.quantile(reference_values, quantile))
        predicted_rejected = [row for row in eligible if float(row["knownness_k5"]) < threshold]
        retained = [row for row in eligible if float(row["knownness_k5"]) >= threshold]
        true_positive = sum(row["human_state"] == "Rejected" for row in predicted_rejected)
        false_positive = len(predicted_rejected) - true_positive
        human_rejected = sum(row["human_state"] == "Rejected" for row in eligible)
        human_valid = len(eligible) - human_rejected
        retained_valid = [
            row
            for row in retained
            if row["human_state"] in TARGET_LABELS
            and row.get("remoteclip_label") in TARGET_LABELS
        ]
        y_true = [str(row["human_state"]) for row in retained_valid]
        y_pred = [str(row["remoteclip_label"]) for row in retained_valid]
        result.append(
            {
                "reference_quantile": quantile,
                "knownness_threshold": threshold,
                "eligible_events": len(eligible),
                "model_rejected_events": len(predicted_rejected),
                "human_rejected_events": human_rejected,
                "human_valid_events": human_valid,
                "human_rejected_caught": true_positive,
                "human_valid_lost": false_positive,
                "reject_precision": (
                    true_positive / len(predicted_rejected) if predicted_rejected else None
                ),
                "reject_recall": true_positive / human_rejected if human_rejected else None,
                "coverage": len(retained) / len(eligible),
                "retained_valid_events": len(retained_valid),
                "remoteclip_accuracy_on_retained_valid": (
                    float(accuracy_score(y_true, y_pred)) if y_true else None
                ),
                "remoteclip_balanced_accuracy_on_retained_valid": (
                    float(balanced_accuracy_score(y_true, y_pred))
                    if len(set(y_true)) == len(TARGET_LABELS)
                    else None
                ),
            }
        )
    return result


def _validity_feature_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("knownness_k5") is not None
        and row.get("remoteclip_probability_wildfire") is not None
        and row.get("xgboost_probability_wildfire") is not None
        and _analysis_reason(row) not in PIPELINE_REJECTION_GROUPS
    ]


def validity_model_diagnostics(
    rows: Sequence[dict[str, Any]],
    *,
    seed: int = 42,
    requested_splits: int = 5,
) -> dict[str, Any]:
    selected = _validity_feature_rows(rows)
    if len(selected) < 20 or len({row["human_state"] for row in selected}) < 2:
        return {
            "status": "insufficient_data",
            "events": len(selected),
            "feature_names": list(VALIDITY_FEATURES),
            "metrics": [],
            "predictions": [],
            "feature_importance": [],
        }

    matrix = np.asarray(
        [
            [
                np.nan if row.get(feature) is None else float(row[feature])
                for feature in VALIDITY_FEATURES
            ]
            for row in selected
        ],
        dtype=float,
    )
    target = np.asarray([int(row["human_state"] == "Rejected") for row in selected], dtype=int)
    groups = np.asarray([str(row["sample_id"]) for row in selected])
    group_counts = [len(set(groups[target == value])) for value in (0, 1)]
    n_splits = min(requested_splits, *group_counts)
    if n_splits < 2:
        return {
            "status": "insufficient_groups",
            "events": len(selected),
            "feature_names": list(VALIDITY_FEATURES),
            "metrics": [],
            "predictions": [],
            "feature_importance": [],
        }

    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                list(range(len(VALIDITY_FEATURES))),
            )
        ],
        remainder="drop",
    )
    models: dict[str, Any] = {
        "LOGISTIC_REGRESSION": Pipeline(
            [
                ("preprocess", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=seed,
                    ),
                ),
            ]
        )
    }
    try:
        from xgboost import XGBClassifier

        tree_preprocessor = ColumnTransformer(
            [
                (
                    "numeric",
                    SimpleImputer(strategy="median"),
                    list(range(len(VALIDITY_FEATURES))),
                )
            ],
            remainder="drop",
        )
        models["XGBOOST"] = Pipeline(
            [
                ("preprocess", tree_preprocessor),
                (
                    "classifier",
                    XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        n_estimators=120,
                        max_depth=2,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=2.0,
                        n_jobs=1,
                        random_state=seed,
                    ),
                ),
            ]
        )
    except ImportError:
        pass

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    metrics: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    importances: list[dict[str, Any]] = []
    for model_name, estimator in models.items():
        probabilities = np.full(len(selected), np.nan, dtype=float)
        for fold, (train_index, test_index) in enumerate(
            splitter.split(matrix, target, groups=groups),
            start=1,
        ):
            fitted = clone(estimator)
            fitted.fit(matrix[train_index], target[train_index])
            probabilities[test_index] = fitted.predict_proba(matrix[test_index])[:, 1]
            for index in test_index:
                predictions.append(
                    {
                        "model": model_name,
                        "fold": fold,
                        "sample_id": selected[index]["sample_id"],
                        "event_id": selected[index]["event_id"],
                        "human_rejected": bool(target[index]),
                        "probability_rejected": float(probabilities[index]),
                    }
                )
        predicted = (probabilities >= 0.5).astype(int)
        metrics.append(
            {
                "model": model_name,
                "events": len(selected),
                "samples": len(set(groups)),
                "folds": n_splits,
                "accuracy": float(accuracy_score(target, predicted)),
                "balanced_accuracy": float(balanced_accuracy_score(target, predicted)),
                "precision_rejected": float(precision_score(target, predicted, zero_division=0)),
                "recall_rejected": float(recall_score(target, predicted, zero_division=0)),
                "f1_rejected": float(f1_score(target, predicted, zero_division=0)),
                "roc_auc": float(roc_auc_score(target, probabilities)),
                "average_precision": float(average_precision_score(target, probabilities)),
            }
        )
        fitted_full = clone(estimator).fit(matrix, target)
        classifier = fitted_full.named_steps["classifier"]
        if model_name == "LOGISTIC_REGRESSION":
            values = classifier.coef_[0]
            importance_type = "standardized_coefficient"
        else:
            values = classifier.feature_importances_
            importance_type = "gain_importance"
        for feature, value in zip(VALIDITY_FEATURES, values, strict=True):
            importances.append(
                {
                    "model": model_name,
                    "feature": feature,
                    "importance_type": importance_type,
                    "value": float(value),
                    "absolute_value": abs(float(value)),
                }
            )
    return {
        "status": "ok",
        "events": len(selected),
        "samples": len(set(groups)),
        "folds": n_splits,
        "feature_names": list(VALIDITY_FEATURES),
        "metrics": metrics,
        "predictions": predictions,
        "feature_importance": sorted(
            importances,
            key=lambda row: (str(row["model"]), -float(row["absolute_value"])),
        ),
        "interpretation_note": (
            "Exploratory grouped-CV diagnostic only. The cohort is small and was not held out "
            "from feature/rule exploration; do not treat these values as production performance. "
            "NO_PRE and QC_FAILED are excluded because they are acquisition-pipeline failures."
        ),
    }


def _count_rows(rows: Iterable[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    counts = Counter(str(row.get(key) or "(missing)") for row in rows)
    total = sum(counts.values())
    return [
        {key: value, "count": count, "percent": 100.0 * count / total if total else 0.0}
        for value, count in counts.most_common()
    ]


def build_rejection_diagnostics(
    review_rows: Sequence[dict[str, Any]],
    latent_rows: Sequence[dict[str, Any]],
    *,
    seed: int = 42,
) -> dict[str, Any]:
    rows = merge_review_and_latent_rows(review_rows, latent_rows)
    point_rows = point_level_rows(rows)
    knownness_status, reason_summary = knownness_summaries(rows)
    latent_available = [row for row in rows if row["latent_available"]]
    valid = [row for row in rows if row["human_state"] in TARGET_LABELS]
    rejected = [row for row in rows if row["human_state"] == "Rejected"]
    small_event = small_event_threshold_analysis(rows)
    temporal = year_mismatch_temporal_analysis(rows)
    year_mismatch_research = build_year_mismatch_research(rows, seed=seed)
    pipeline = acquisition_pipeline_statistics(rows)
    agreement = reason_model_agreement_statistics(rows)
    one_vs_valid = one_vs_valid_diagnostics(rows, seed=seed)
    point_relationships = point_reason_relationships(rows)
    return {
        "summary": {
            "reviewed_events": len(rows),
            "reviewed_points": len({row["sample_id"] for row in rows}),
            "valid_events": len(valid),
            "rejected_events": len(rejected),
            "latent_available_events": len(latent_available),
            "latent_missing_events": len(rows) - len(latent_available),
            "latent_coverage": len(latent_available) / len(rows) if rows else 0.0,
        },
        "event_rows": rows,
        "point_rows": point_rows,
        "hierarchy_counts": _count_rows(rows, "review_hierarchy_class"),
        "raw_reason_mapping": [
            {
                "raw_rejection_reason": reason,
                "rejection_reason_group": rejection_reason_group(reason),
                "count": count,
            }
            for reason, count in Counter(
                str(row.get("rejection_reason") or "") for row in rejected
            ).most_common()
        ],
        "three_state_tables": three_state_tables(rows),
        "probability_summaries": probability_summaries(rows),
        "knownness_status_summary": knownness_status,
        "rejection_reason_summary": reason_summary,
        "reason_feature_summary": reason_feature_summary(rows),
        "reason_feature_distributions": reason_feature_distributions(rows),
        "reason_year_counts": reason_year_counts(rows),
        "reason_season_counts": reason_season_counts(rows),
        "reason_qc_flag_counts": reason_qc_flag_counts(rows),
        "small_event_analysis": small_event,
        "year_mismatch_analysis": temporal,
        "year_mismatch_research": year_mismatch_research,
        "acquisition_pipeline": pipeline,
        "reason_model_agreement": agreement,
        "one_vs_valid": one_vs_valid,
        "point_reason_relationships": point_relationships,
        "hdbscan_rejection": hdbscan_rejection_test(rows),
        "remoteclip_errors": remoteclip_error_rows(rows),
        "model_disagreement": model_disagreement_table(rows),
        "point_summary": point_level_summary(point_rows),
        "coverage_analysis": coverage_analysis(rows, latent_rows),
        "validity_models": validity_model_diagnostics(rows, seed=seed),
        "settings": {
            "seed": seed,
            "knownness_metric": "knownness_k5",
            "hierarchy": list(REVIEW_HIERARCHY),
            "validity_features": list(VALIDITY_FEATURES),
        },
    }


def _format_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def rejection_diagnostics_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    reviewer = result["reviewer_name"]
    hdbscan = result["hdbscan_rejection"]
    lines = [
        f"# Hierarchical review diagnostics for {reviewer}",
        "",
        "## Cohort and hierarchy",
        "",
        f"- Reviewed events: **{summary['reviewed_events']}** from "
        f"**{summary['reviewed_points']}** points.",
        f"- Human VALID: **{summary['valid_events']}**; human REJECTED: "
        f"**{summary['rejected_events']}**.",
        f"- Events with saved latent diagnostics: **{summary['latent_available_events']}** "
        f"({summary['latent_coverage']:.1%}); missing: **{summary['latent_missing_events']}**.",
        "",
        "The hierarchy first separates VALID from REJECTED and only then separates "
        "Logging from Wildfire. Raw reviewer comments remain unchanged in the event export.",
        "",
        "| Hierarchy class | Events | Percent |",
        "|---|---:|---:|",
    ]
    for row in result["hierarchy_counts"]:
        lines.append(
            f"| {row['review_hierarchy_class']} | {row['count']} | {row['percent']:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Closed-set three-state audit",
            "",
            "Models can predict only Logging or Wildfire. Human Rejected is therefore shown "
            "as a third target state rather than treated as a classification error.",
            "",
            "| Model | Prediction | Human Logging | Human Wildfire | Human Rejected |",
            "|---|---|---:|---:|---:|",
        ]
    )
    three_state = result["three_state_tables"]
    for model in DEFAULT_MODEL_SOURCES:
        for prediction in TARGET_LABELS:
            lookup = {
                row["human_state"]: row["count"]
                for row in three_state
                if row["model"] == model and row["predicted_label"] == prediction
            }
            lines.append(
                f"| {model} | {prediction} | {lookup.get('Logging', 0)} | "
                f"{lookup.get('Wildfire', 0)} | {lookup.get('Rejected', 0)} |"
            )

    lines.extend(
        [
            "",
            "## Model disagreement as a review signal",
            "",
            "| Prediction pair | Events | VALID share | REJECTED share | Human Logging | Human Wildfire |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["model_disagreement"]:
        lines.append(
            f"| {row['model_agreement_type']} | {row['events']} | "
            f"{row['valid_share']:.1%} | {row['rejected_share']:.1%} | "
            f"{row['human_logging_share']:.1%} | {row['human_wildfire_share']:.1%} |"
        )

    lines.extend(
        [
            "",
            "## Probability distributions",
            "",
            "| Model | Human state | N | Mean P(Wildfire) | Median | Mean entropy | Extreme P share |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["probability_summaries"]:
        lines.append(
            f"| {row['model']} | {row['human_state']} | {row['count']} | "
            f"{_format_number(row['mean'])} | {_format_number(row['median'])} | "
            f"{_format_number(row['entropy_mean'])} | "
            f"{_format_number(row['extreme_probability_share'])} |"
        )

    lines.extend(
        [
            "",
            "## Knownness by review state",
            "",
            "| Review status | N | Mean | Median | p05 | p95 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["knownness_status_summary"]:
        lines.append(
            f"| {row['review_status']} | {row['count']} | {_format_number(row['mean'])} | "
            f"{_format_number(row['median'])} | {_format_number(row['q05'])} | "
            f"{_format_number(row['q95'])} |"
        )

    lines.extend(
        [
            "",
            "### Rejection groups",
            "",
            "| Group | Events | With latent data | Mean knownness | RC P(W) | XGB P(W) | HDBSCAN noise |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["rejection_reason_summary"]:
        lines.append(
            f"| {row['rejection_reason_group']} | {row['events']} | {row['latent_events']} | "
            f"{_format_number(row['knownness_mean'])} | "
            f"{_format_number(row['remoteclip_p_wildfire_mean'])} | "
            f"{_format_number(row['xgboost_p_wildfire_mean'])} | "
            f"{_format_number(row['hdbscan_noise_share'])} |"
        )

    lines.extend(
        [
            "",
            "## Per-reason feature profile",
            "",
            "This table keeps VALID as the reference group. PRE/POST counts come from the "
            "event-download links in SQLite; Hansen area and pixels come from the materialized event.",
            "",
            "| Reason | Count | Share | Median area, ha | Median Hansen % AOI | Median pixels | Median knownness | RC P(W) | XGB P(W) | Disagree | PRE | POST | HDB noise |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["reason_feature_summary"]:
        lines.append(
            f"| {row['reason']} | {row['count']} | {_format_number(row['share'])} | "
            f"{_format_number(row['median_event_area'])} | "
            f"{_format_number(row['median_hansen_loss_aoi_percent'])} | "
            f"{_format_number(row['median_hansen_pixels'])} | "
            f"{_format_number(row['median_knownness'])} | "
            f"{_format_number(row['mean_RC_P(W)'])} | {_format_number(row['mean_XGB_P(W)'])} | "
            f"{_format_number(row['model_disagreement_share'])} | "
            f"{_format_number(row['median_PRE_count'])} | {_format_number(row['median_POST_count'])} | "
            f"{_format_number(row['HDBSCAN_noise_share'])} |"
        )

    small_event = result["small_event_analysis"]
    lines.extend(
        [
            "",
            "## EVENT_TOO_SMALL vs VALID: threshold search",
            "",
            "A threshold means `feature <= threshold -> reject as EVENT_TOO_SMALL`. The recommended "
            "candidate first requires at least five events below the threshold and at least 70% "
            "precision; otherwise the best exploratory F1 is shown and explicitly marked.",
            "",
            "| Feature | N | ROC-AUC (smaller = reject) | Candidate threshold | Reject rate below | Recall | VALID lost | Status |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in small_event["summary"]:
        lines.append(
            f"| {row['feature']} | {row.get('eligible_events', 0)} | "
            f"{_format_number(row.get('roc_auc_for_smaller_value'))} | "
            f"{_format_number(row.get('recommended_threshold'))} | "
            f"{_format_number(row.get('reject_rate_below'))} | "
            f"{_format_number(row.get('event_too_small_recall'))} | "
            f"{row.get('valid_events_below', '-')} | {row['status'] if row['status'] != 'ok' else row.get('recommendation_status')} |"
        )
    lines.extend(["", small_event["interpretation_note"]])

    temporal = result["year_mismatch_analysis"]["summary"]
    lines.extend(
        [
            "",
            "## YEAR_MISMATCH temporal consistency",
            "",
            f"- YEAR_MISMATCH events: **{temporal['events']}**.",
            f"- Expert year inferred from explicit relative/absolute wording: "
            f"**{temporal['expert_year_inferred']}**; unavailable: "
            f"**{temporal['expert_year_not_inferred']}**.",
            f"- Inferred expert year also exists as a Hansen event at the same point: "
            f"**{temporal['inferred_year_exists_as_hansen_event']}**.",
            f"- Neighboring Hansen event Y-1: **{temporal['has_y_minus_1']}**; "
            f"Y+1: **{temporal['has_y_plus_1']}**.",
            f"- Offset counts: `{json.dumps(temporal['offset_counts'], ensure_ascii=False)}`.",
            "",
            temporal["data_gap_note"],
            "",
            "Detailed PRE/EVENT/POST scene years and neighboring-event statuses are exported to "
            "`year_mismatch_events.csv`. No FIRMS values are fabricated.",
        ]
    )

    year_research = result["year_mismatch_research"]
    year_summary = year_research["summary"]
    lines.extend(
        [
            "",
            "### Selecting the main physical event year",
            "",
            f"The focused cohort contains **{year_summary['year_mismatch_points']}** "
            f"YEAR_MISMATCH points. **{year_summary['evaluable_points']}** have a selectable "
            f"target year; **{year_summary['non_evaluable_points']}** remain ambiguous or have "
            f"no expert-indicated year among the Hansen candidates.",
            "",
            "Ground truth first uses an explicit expert-visible year from the rejection text. "
            "Only when the text gives no direction does a unique VALID_WILDFIRE year act as "
            "fallback. Multiple VALID years are not collapsed into a made-up single target.",
            "",
            "| Ranking method | Points | Hit@1 | Mean reciprocal rank | Median target rank |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in year_research["ranking_metrics"]:
        lines.append(
            f"| {row['method']} | {row['evaluable_points']} | "
            f"{_format_number(row['hit_at_1'])} | "
            f"{_format_number(row['mean_reciprocal_rank'])} | "
            f"{_format_number(row['median_target_rank'])} |"
        )
    lines.extend(
        [
            "",
            f"Hypothesis status: **{year_summary['hypothesis_status']}**. Best current method: "
            f"**{year_summary['best_method']}**, Hit@1="
            f"**{_format_number(year_summary['best_hit_at_1'])}**, MRR="
            f"**{_format_number(year_summary['best_mean_reciprocal_rank'])}**.",
            "",
            year_summary["interpretation_note"],
            "",
            "`year_mismatch_timeline.csv` and `year_mismatch_timeline_wide.csv` contain the "
            "Y-2/Y-1/Y/Y+1 point histories with area, model probabilities, expert status, "
            "PRE/EVENT/POST counts, seasons, and reserved FIRMS fields.",
        ]
    )

    pipeline = result["acquisition_pipeline"]
    lines.extend(
        [
            "",
            "## NO_PRE and QC_FAILED: acquisition pipeline",
            "",
            pipeline["interpretation_note"],
            "",
            "| Reason | Events | Points | PRE | Usable PRE | POST | Usable POST | Bad cloud | Bad quality | Manual excluded |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in pipeline["summary"]:
        lines.append(
            f"| {row['reason']} | {row['events']} | {row['points']} | "
            f"{_format_number(row['median_pre_count'])} | "
            f"{_format_number(row['median_usable_pre_count'])} | "
            f"{_format_number(row['median_post_count'])} | "
            f"{_format_number(row['median_usable_post_count'])} | "
            f"{row['bad_cloud_downloads']} | {row['bad_quality_downloads']} | "
            f"{row['manually_excluded_downloads']} |"
        )

    agreement = result["reason_model_agreement"]["headline"]
    lines.extend(
        [
            "",
            "## Rejection reasons and model disagreement",
            "",
            f"- P(EVENT_TOO_SMALL | RC != XGB): "
            f"**{_format_number(agreement['P(EVENT_TOO_SMALL | RC!=XGB)'])}**.",
            f"- P(YEAR_MISMATCH | RC = XGB = Wildfire): "
            f"**{_format_number(agreement['P(YEAR_MISMATCH | RC=XGB=Wildfire)'])}**.",
            "",
            "The complete reason x prediction-pair table is in "
            "`reason_model_agreement.csv` and `reason_given_model_pattern.csv`.",
        ]
    )

    lines.extend(
        [
            "",
            "## One-vs-VALID exploratory diagnostics",
            "",
            "| Reason | Status | N reason | N valid | Folds | Balanced accuracy | Precision | Recall | ROC-AUC |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["one_vs_valid"]["metrics"]:
        lines.append(
            f"| {row['reason']} | {row['status']} | {row['reason_events']} | "
            f"{row['valid_events']} | {row.get('folds', '-')} | "
            f"{_format_number(row.get('balanced_accuracy'))} | "
            f"{_format_number(row.get('precision_reason'))} | "
            f"{_format_number(row.get('recall_reason'))} | "
            f"{_format_number(row.get('roc_auc'))} |"
        )

    lines.extend(
        [
            "",
            "## Point-level reason relationships",
            "",
            "Most frequent pairs of review outcomes occurring at the same point:",
            "",
            "| Class A | Class B | Points |",
            "|---|---|---:|",
        ]
    )
    for row in result["point_reason_relationships"]["cooccurrence"][:15]:
        lines.append(
            f"| {row['class_a']} | {row['class_b']} | {row['point_count']} |"
        )

    lines.extend(
        [
            "",
            "## HDBSCAN noise and human rejection",
            "",
            f"- P(noise | VALID): **{_format_number(hdbscan['noise_share_valid'])}**.",
            f"- P(noise | REJECTED): **{_format_number(hdbscan['noise_share_rejected'])}**.",
            f"- Haldane-corrected odds ratio: "
            f"**{_format_number(hdbscan['odds_ratio_haldane_corrected'])}**.",
            f"- Fisher exact p-value: **{_format_number(hdbscan['fisher_p_value'])}**.",
            "",
            hdbscan["interpretation_note"],
            "",
            "## Knownness reject coverage",
            "",
            "Thresholds are quantiles of the MANUAL_TRAIN knownness reference, not thresholds "
            "optimized on Olga's reviews.",
            "",
            "| Reference tail | Threshold | Reject precision | Reject recall | Coverage | VALID lost | RC balanced accuracy retained |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["coverage_analysis"]:
        lines.append(
            f"| {row['reference_quantile']:.0%} | {_format_number(row['knownness_threshold'], 5)} | "
            f"{_format_number(row['reject_precision'])} | {_format_number(row['reject_recall'])} | "
            f"{_format_number(row['coverage'])} | {row['human_valid_lost']} | "
            f"{_format_number(row['remoteclip_balanced_accuracy_on_retained_valid'])} |"
        )

    validity = result["validity_models"]
    lines.extend(
        [
            "",
            "## Exploratory VALID vs REJECTED models",
            "",
        ]
    )
    if validity["status"] == "ok":
        lines.extend(
            [
                f"Grouped CV uses **{validity['events']}** events from "
                f"**{validity['samples']}** points and **{validity['folds']}** folds.",
                "",
                "| Model | Balanced accuracy | Precision REJECTED | Recall REJECTED | F1 REJECTED | ROC-AUC | AP |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in validity["metrics"]:
            lines.append(
                f"| {row['model']} | {_format_number(row['balanced_accuracy'])} | "
                f"{_format_number(row['precision_rejected'])} | "
                f"{_format_number(row['recall_rejected'])} | "
                f"{_format_number(row['f1_rejected'])} | {_format_number(row['roc_auc'])} | "
                f"{_format_number(row['average_precision'])} |"
            )
        lines.extend(["", validity["interpretation_note"]])
    else:
        lines.append(f"Validity model was not run: `{validity['status']}`.")

    lines.extend(
        [
            "",
            "## RemoteCLIP errors on accepted events",
            "",
            f"Accepted classification errors: **{len(result['remoteclip_errors'])}**. "
            "See `remoteclip_errors.csv` for knownness, cluster, similarities, event size, "
            "probability, and reviewer notes for every error.",
            "",
            "## Interpretation limits",
            "",
            "- Missing latent rows are never silently imputed into knownness/HDBSCAN analyses.",
            "- Rejection groups are deterministic text normalization for analysis; raw comments remain authoritative.",
            "- HDBSCAN status for Hansen events is inherited from the nearest manual reference event.",
            "- The validity model is exploratory because the same small cohort informed feature and rule selection.",
            "- Events from one point are dependent; grouped CV keeps a point in one fold.",
            "- Expert-visible years are not structured in the database; relative years are parsed only from explicit reviewer wording.",
            "- FIRMS is not present in the current schema, so temporal tables mark it unavailable instead of inferring fire evidence.",
            "",
            "## Generated artifacts",
            "",
            "- `event_level_hierarchy.csv` — joined human, model, latent, and hierarchy fields.",
            "- `raw_reason_mapping.csv` and `rejection_reason_summary.csv` — normalized rejection audit.",
            "- `three_state_model_table.csv` — Logging/Wildfire/Rejected closed-set table.",
            "- `model_disagreement.csv` — human states by model prediction pair.",
            "- `knownness_*.csv`, `probability_summary.csv`, and `hdbscan_rejection.json`.",
            "- `coverage_analysis.csv` — reject precision/recall/coverage sweep.",
            "- `validity_model_*.csv` — grouped-CV pre-filter diagnostics.",
            "- `reason_feature_*.csv` — per-reason distributions, years, seasons, and QC flags.",
            "- `small_event_*.csv` — EVENT_TOO_SMALL threshold and binned reject-rate sweeps.",
            "- `year_mismatch_*.csv` — temporal evidence, point timelines, target-year ranking, and explicit data gaps.",
            "- `acquisition_pipeline_*.csv` — NO_PRE/QC_FAILED engineering losses.",
            "- `one_vs_valid_*.csv` — exploratory reason-specific classifiers.",
            "- `point_reason_*.csv` — co-occurrence and adjacent year transitions.",
            "- `plots/*.png` — notebook figures.",
            "",
        ]
    )
    return "\n".join(lines)


def rejection_diagnostic_figures(result: dict[str, Any]) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    figures: dict[str, Any] = {}
    rows = result["event_rows"]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for axis, (model, prefix) in zip(
        axes,
        (("RemoteCLIP", "remoteclip"), ("XGBoost", "xgboost")),
        strict=True,
    ):
        matrix = np.zeros((2, 3), dtype=int)
        state_index = {"Logging": 0, "Wildfire": 1, "Rejected": 2}
        prediction_index = {"Logging": 0, "Wildfire": 1}
        for row in rows:
            prediction = row.get(f"{prefix}_label")
            if prediction in prediction_index:
                matrix[prediction_index[prediction], state_index[row["human_state"]]] += 1
        image = axis.imshow(matrix, cmap="Blues")
        for y_index in range(2):
            for x_index in range(3):
                axis.text(x_index, y_index, str(matrix[y_index, x_index]), ha="center", va="center")
        axis.set_xticks(range(3), ["Human L", "Human W", "Human Reject"])
        axis.set_yticks(range(2), ["Predict L", "Predict W"])
        axis.set_title(model)
        figure.colorbar(image, ax=axis, shrink=0.8)
    figures["three_state_matrices"] = figure

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    states = ["Logging", "Wildfire", "Rejected"]
    colors = ["#4b8f3a", "#d76536", "#747b86"]
    for axis, (model, key) in zip(
        axes,
        (
            ("RemoteCLIP", "remoteclip_probability_wildfire"),
            ("XGBoost", "xgboost_probability_wildfire"),
        ),
        strict=True,
    ):
        values = [
            [float(row[key]) for row in rows if row["human_state"] == state and row.get(key) is not None]
            for state in states
        ]
        violin = axis.violinplot(values, showmeans=True, showmedians=True)
        for body, color in zip(violin["bodies"], colors, strict=True):
            body.set_facecolor(color)
            body.set_alpha(0.65)
        axis.set_xticks(range(1, 4), states)
        axis.set_ylim(-0.03, 1.03)
        axis.set_ylabel("P(Wildfire)")
        axis.set_title(model)
        axis.grid(axis="y", alpha=0.25)
    figures["probability_by_human_state"] = figure

    figure, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    status_names = ["VALID", "REJECTED"]
    values = [
        [
            float(row["knownness_k5"])
            for row in rows
            if row["review_status"] == status and row.get("knownness_k5") is not None
        ]
        for status in status_names
    ]
    box = axis.boxplot(values, tick_labels=status_names, patch_artist=True, showfliers=True)
    for patch, color in zip(box["boxes"], ["#4b8f3a", "#747b86"], strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axis.set_ylabel("knownness_k5")
    axis.set_title("Knownness by human review status")
    axis.grid(axis="y", alpha=0.25)
    figures["knownness_valid_vs_rejected"] = figure

    reason_rows = [row for row in rows if row["human_state"] == "Rejected" and row.get("knownness_k5") is not None]
    reason_groups = [
        group
        for group, _ in Counter(str(row["rejection_reason_group"]) for row in reason_rows).most_common()
    ]
    if reason_groups:
        figure, axis = plt.subplots(figsize=(11, 5.2), constrained_layout=True)
        values = [
            [
                float(row["knownness_k5"])
                for row in reason_rows
                if row["rejection_reason_group"] == group
            ]
            for group in reason_groups
        ]
        axis.boxplot(values, tick_labels=reason_groups, patch_artist=True, showfliers=True)
        axis.tick_params(axis="x", rotation=30)
        axis.set_ylabel("knownness_k5")
        axis.set_title("Knownness by normalized rejection reason")
        axis.grid(axis="y", alpha=0.25)
        figures["knownness_by_rejection_reason"] = figure

    coverage = result["coverage_analysis"]
    if coverage:
        figure, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
        tails = [100.0 * float(row["reference_quantile"]) for row in coverage]
        axis.plot(tails, [row["reject_precision"] for row in coverage], marker="o", label="Reject precision")
        axis.plot(tails, [row["reject_recall"] for row in coverage], marker="o", label="Reject recall")
        axis.plot(tails, [row["coverage"] for row in coverage], marker="o", label="Coverage")
        axis.set_xlabel("Manual-reference lower tail, %")
        axis.set_ylabel("Score")
        axis.set_ylim(0.0, 1.05)
        axis.set_title("Knownness rejection trade-off")
        axis.grid(alpha=0.25)
        axis.legend()
        figures["knownness_coverage"] = figure

    small_thresholds = result["small_event_analysis"]["thresholds"]
    small_features = [
        value
        for value in ("event_area_ha", "hansen_pixels", "hansen_loss_aoi_percent")
        if any(row["feature"] == value for row in small_thresholds)
    ]
    if small_features:
        figure, axes = plt.subplots(
            1, len(small_features), figsize=(5.2 * len(small_features), 4.5), constrained_layout=True
        )
        axes_array = np.atleast_1d(axes)
        for axis, feature in zip(axes_array, small_features, strict=True):
            feature_rows = [
                row
                for row in small_thresholds
                if row["feature"] == feature and row["below_or_equal_count"] >= 3
            ]
            axis.plot(
                [row["threshold"] for row in feature_rows],
                [row["reject_rate_below"] for row in feature_rows],
                label="P(EVENT_TOO_SMALL | value <= threshold)",
            )
            axis.plot(
                [row["threshold"] for row in feature_rows],
                [row["event_too_small_recall"] for row in feature_rows],
                label="EVENT_TOO_SMALL recall",
            )
            axis.axhline(0.70, color="#c94f45", linestyle="--", linewidth=1, label="70%")
            axis.set_title(feature)
            axis.set_xlabel("Threshold")
            axis.set_ylim(0, 1.05)
            axis.grid(alpha=0.25)
        axes_array[0].set_ylabel("Share")
        axes_array[-1].legend(fontsize=8)
        figures["event_too_small_threshold_sweep"] = figure

    comparison_groups = ("VALID", "EVENT_TOO_SMALL", "YEAR_MISMATCH")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for axis, (key, title) in zip(
        axes,
        (("event_loss_area_ha", "Event area, ha"), ("hansen_loss_aoi_percent", "Hansen loss inside AOI, %")),
        strict=True,
    ):
        values = [
            [
                float(row[key])
                for row in rows
                if _analysis_reason(row) == group and row.get(key) is not None
            ]
            for group in comparison_groups
        ]
        axis.boxplot(values, tick_labels=comparison_groups, patch_artist=True, showfliers=True)
        axis.tick_params(axis="x", rotation=20)
        axis.set_ylabel(title)
        axis.set_title(f"{title}: research reasons vs VALID")
        axis.grid(axis="y", alpha=0.25)
    figures["research_reasons_vs_valid"] = figure

    temporal_events = result["year_mismatch_analysis"]["events"]
    if temporal_events:
        offset_counts = Counter(
            "unknown" if row["expert_year_offset"] is None else str(row["expert_year_offset"])
            for row in temporal_events
        )
        figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        labels = list(offset_counts)
        axis.bar(labels, [offset_counts[label] for label in labels], color="#7b8fa1")
        axis.set_xlabel("Expert-visible year offset from Hansen Y")
        axis.set_ylabel("Events")
        axis.set_title("YEAR_MISMATCH free-text temporal evidence")
        axis.grid(axis="y", alpha=0.25)
        figures["year_mismatch_offsets"] = figure

    pipeline_years = result["acquisition_pipeline"]["by_year"]
    if pipeline_years:
        years = sorted({int(row["event_year"]) for row in pipeline_years})
        figure, axis = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
        bottom = np.zeros(len(years), dtype=float)
        for reason, color in (("NO_PRE", "#6f7f90"), ("QC_FAILED", "#c26a4a")):
            lookup = {int(row["event_year"]): row["count"] for row in pipeline_years if row["reason"] == reason}
            values = np.asarray([lookup.get(year, 0) for year in years], dtype=float)
            axis.bar(years, values, bottom=bottom, label=reason, color=color)
            bottom += values
        axis.set_xlabel("Hansen event year")
        axis.set_ylabel("Rejected events")
        axis.set_title("Acquisition-pipeline losses by event year")
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
        figures["acquisition_losses_by_year"] = figure

    one_vs_metrics = [
        row for row in result["one_vs_valid"]["metrics"] if row.get("status") == "ok"
    ]
    if one_vs_metrics:
        figure, axis = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
        names = [row["reason"] for row in one_vs_metrics]
        positions = np.arange(len(names))
        width = 0.36
        axis.bar(
            positions - width / 2,
            [row["balanced_accuracy"] for row in one_vs_metrics],
            width,
            label="Balanced accuracy",
        )
        axis.bar(
            positions + width / 2,
            [row["roc_auc"] for row in one_vs_metrics],
            width,
            label="ROC-AUC",
        )
        axis.set_xticks(positions, names, rotation=20)
        axis.set_ylim(0, 1.05)
        axis.set_title("One-vs-VALID exploratory diagnostics")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
        figures["one_vs_valid_metrics"] = figure
    timeline_figure = year_mismatch_timeline_figure(result["year_mismatch_research"])
    if timeline_figure is not None:
        figures["year_mismatch_point_timelines"] = timeline_figure
    ranking_figure = year_mismatch_ranking_figure(result["year_mismatch_research"])
    if ranking_figure is not None:
        figures["year_mismatch_event_year_ranking"] = ranking_figure
    return figures


def export_rejection_diagnostics(
    result: dict[str, Any],
    *,
    output_root: str | Path,
    run_id: str | None = None,
) -> Path:
    run_dir = create_run_directory(output_root, run_id)
    write_csv(run_dir / "event_level_hierarchy.csv", result["event_rows"])
    write_csv(run_dir / "point_level_history.csv", result["point_rows"])
    write_csv(run_dir / "point_level_summary.csv", result["point_summary"])
    write_csv(run_dir / "hierarchy_counts.csv", result["hierarchy_counts"])
    write_csv(run_dir / "raw_reason_mapping.csv", result["raw_reason_mapping"])
    write_csv(run_dir / "three_state_model_table.csv", result["three_state_tables"])
    write_csv(run_dir / "probability_summary.csv", result["probability_summaries"])
    write_csv(run_dir / "knownness_status_summary.csv", result["knownness_status_summary"])
    write_csv(run_dir / "rejection_reason_summary.csv", result["rejection_reason_summary"])
    write_csv(run_dir / "reason_feature_summary.csv", result["reason_feature_summary"])
    write_csv(
        run_dir / "reason_feature_distributions.csv",
        result["reason_feature_distributions"],
    )
    write_csv(run_dir / "reason_year_counts.csv", result["reason_year_counts"])
    write_csv(run_dir / "reason_season_counts.csv", result["reason_season_counts"])
    write_csv(run_dir / "reason_qc_flag_counts.csv", result["reason_qc_flag_counts"])
    small_event = result["small_event_analysis"]
    write_csv(run_dir / "small_event_thresholds.csv", small_event["thresholds"])
    write_csv(run_dir / "small_event_bins.csv", small_event["bins"])
    write_json(run_dir / "small_event_summary.json", small_event["summary"])
    temporal = result["year_mismatch_analysis"]
    write_csv(run_dir / "year_mismatch_events.csv", temporal["events"])
    write_json(run_dir / "year_mismatch_summary.json", temporal["summary"])
    year_research = result["year_mismatch_research"]
    write_csv(run_dir / "year_mismatch_point_summary.csv", year_research["point_summaries"])
    write_csv(run_dir / "year_mismatch_timeline.csv", year_research["timeline_rows"])
    write_csv(run_dir / "year_mismatch_timeline_wide.csv", year_research["timeline_wide"])
    write_csv(run_dir / "year_mismatch_ranking_metrics.csv", year_research["ranking_metrics"])
    write_csv(
        run_dir / "year_mismatch_point_predictions.csv",
        year_research["point_predictions"],
    )
    write_csv(
        run_dir / "year_mismatch_ranking_details.csv", year_research["ranking_details"]
    )
    write_json(run_dir / "year_mismatch_research_summary.json", year_research["summary"])
    pipeline = result["acquisition_pipeline"]
    write_csv(run_dir / "acquisition_pipeline_summary.csv", pipeline["summary"])
    write_csv(run_dir / "acquisition_pipeline_by_year.csv", pipeline["by_year"])
    write_csv(run_dir / "acquisition_pipeline_by_season.csv", pipeline["by_season"])
    write_csv(run_dir / "acquisition_pipeline_qc_flags.csv", pipeline["qc_flags"])
    agreement = result["reason_model_agreement"]
    write_csv(run_dir / "reason_model_agreement.csv", agreement["cross_table"])
    write_csv(run_dir / "reason_given_model_pattern.csv", agreement["conditionals"])
    write_json(run_dir / "reason_model_agreement_headline.json", agreement["headline"])
    one_vs_valid = result["one_vs_valid"]
    write_csv(run_dir / "one_vs_valid_metrics.csv", one_vs_valid["metrics"])
    write_csv(run_dir / "one_vs_valid_predictions.csv", one_vs_valid["predictions"])
    write_csv(run_dir / "one_vs_valid_coefficients.csv", one_vs_valid["coefficients"])
    point_relationships = result["point_reason_relationships"]
    write_csv(run_dir / "point_reason_cooccurrence.csv", point_relationships["cooccurrence"])
    write_csv(
        run_dir / "point_reason_transitions.csv", point_relationships["adjacent_transitions"]
    )
    write_csv(run_dir / "point_reason_histories.csv", point_relationships["histories"])
    write_json(run_dir / "hdbscan_rejection.json", result["hdbscan_rejection"])
    write_csv(run_dir / "remoteclip_errors.csv", result["remoteclip_errors"])
    write_csv(run_dir / "model_disagreement.csv", result["model_disagreement"])
    write_csv(run_dir / "coverage_analysis.csv", result["coverage_analysis"])
    validity = result["validity_models"]
    write_csv(run_dir / "validity_model_metrics.csv", validity.get("metrics", []))
    write_csv(run_dir / "validity_model_predictions.csv", validity.get("predictions", []))
    write_csv(run_dir / "validity_model_feature_importance.csv", validity.get("feature_importance", []))
    summary = {
        key: value
        for key, value in result.items()
        if key not in {"event_rows", "point_rows"}
    }
    write_json(run_dir / "summary.json", summary)
    report_path = run_dir / "REPORT.md"
    report_path.write_text(rejection_diagnostics_report(result), encoding="utf-8")
    figures = rejection_diagnostic_figures(result)
    for name, figure in figures.items():
        figure.savefig(run_dir / "plots" / f"{name}.png", dpi=160, bbox_inches="tight")
    result["run_dir"] = str(run_dir)
    result["report_path"] = str(report_path)
    return run_dir


def run_rejection_diagnostics(
    db: Session,
    *,
    reviewer_name: str,
    latent_run_dir: str | Path,
    output_root: str | Path | None = None,
    run_id: str | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    latent_run = Path(latent_run_dir)
    latent_path = latent_run / "latent_event_level_table.csv"
    review_rows = load_reviewer_validation(
        db,
        reviewer_name=reviewer_name,
        model_sources=DEFAULT_MODEL_SOURCES,
    )
    database_features = load_event_database_features(
        db,
        [str(row["event_id"]) for row in review_rows],
    )
    for row in review_rows:
        row.update(database_features.get(str(row["event_id"]), {}))
    latent_rows = load_latent_event_table(latent_path)
    result = build_rejection_diagnostics(review_rows, latent_rows, seed=seed)
    result["reviewer_name"] = reviewer_name
    result["latent_run_dir"] = str(latent_run)
    result["latent_event_table"] = str(latent_path)
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    if output_root is not None:
        export_rejection_diagnostics(result, output_root=output_root, run_id=run_id)
    return result
