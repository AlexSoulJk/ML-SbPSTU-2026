from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import select
from sqlalchemy.orm import Session

from ResearchTool.backend.forest.models import HansenMask
from ResearchTool.backend.forest.storage_paths import resolve_storage_path

from .reporting import create_run_directory, write_csv, write_json
from .year_mismatch_research import RANKING_FEATURES


TARGET_LABELS = ("Logging", "Wildfire")
MODEL_PREFIXES = ("remoteclip", "xgboost")
PAIRED_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "recall_logging",
    "recall_wildfire",
)
FIXED_SMALL_EVENT_THRESHOLDS = {
    "event_loss_area_ha": 3.285,
    "event_hansen_pixels": 84.0,
    "hansen_loss_aoi_percent": 0.821,
}
MERGE_RELATIVE_AREA_THRESHOLD = 0.5
DOMINANT_METHODS = (
    "HISTORICAL_DOMINANT",
    "MAX_HANSEN_AREA",
    "HAND_BUILT_COMPOSITE",
    "LOPO_LOGISTIC_REGRESSION",
    "MAX_MODEL_MEAN_P_WILDFIRE",
)


def _number(value: Any) -> float | None:
    if value in (None, "", "None", "null", "nan"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"true", "1", "yes"}


def _review_class(row: dict[str, Any]) -> str:
    value = row.get("review_hierarchy_class")
    if value:
        return str(value)
    if row.get("review_status") == "VALID":
        label = str(row.get("human_label") or "")
        return f"VALID_{label.upper()}" if label in TARGET_LABELS else "OTHER_INVALID"
    return str(row.get("rejection_reason_group") or "OTHER_INVALID")


def _model_probability(row: dict[str, Any], prefix: str) -> float | None:
    return _number(row.get(f"{prefix}_probability_wildfire"))


def _model_mean_probability(row: dict[str, Any]) -> float | None:
    values = [
        value
        for prefix in MODEL_PREFIXES
        if (value := _model_probability(row, prefix)) is not None
    ]
    return float(np.mean(values)) if values else None


def _group_rows(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        row = dict(raw)
        row["review_hierarchy_class"] = _review_class(row)
        grouped[str(row["sample_id"])].append(row)
    for events in grouped.values():
        events.sort(key=lambda row: (int(row["event_year"]), str(row.get("event_id") or "")))
    return grouped


def _metric_values(rows: Sequence[dict[str, Any]], prefix: str) -> dict[str, float] | None:
    selected = [
        row
        for row in rows
        if row.get("review_status") == "VALID"
        and row.get("human_label") in TARGET_LABELS
        and row.get(f"{prefix}_label") in TARGET_LABELS
    ]
    if not selected:
        return None
    truth = [str(row["human_label"]) for row in selected]
    predicted = [str(row[f"{prefix}_label"]) for row in selected]
    recalls = recall_score(
        truth,
        predicted,
        labels=list(TARGET_LABELS),
        average=None,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(
            f1_score(
                truth,
                predicted,
                labels=list(TARGET_LABELS),
                average="macro",
                zero_division=0,
            )
        ),
        "recall_logging": float(recalls[0]),
        "recall_wildfire": float(recalls[1]),
    }


def paired_model_bootstrap(
    rows: Sequence[dict[str, Any]],
    *,
    bootstrap_runs: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    eligible = [
        dict(row)
        for row in rows
        if row.get("review_status") == "VALID"
        and row.get("human_label") in TARGET_LABELS
        and all(row.get(f"{prefix}_label") in TARGET_LABELS for prefix in MODEL_PREFIXES)
    ]
    grouped = _group_rows(eligible)
    sample_ids = sorted(grouped)
    point_metrics = {
        prefix: _metric_values(eligible, prefix) or {} for prefix in MODEL_PREFIXES
    }
    rng = np.random.default_rng(seed)
    bootstrap_rows: list[dict[str, Any]] = []
    if sample_ids:
        for run in range(bootstrap_runs):
            sampled_ids = rng.choice(sample_ids, size=len(sample_ids), replace=True)
            sampled_rows = [row for sample_id in sampled_ids for row in grouped[str(sample_id)]]
            rc = _metric_values(sampled_rows, "remoteclip")
            xgb = _metric_values(sampled_rows, "xgboost")
            if rc is None or xgb is None:
                continue
            for metric in PAIRED_METRICS:
                bootstrap_rows.append(
                    {
                        "bootstrap_run": run,
                        "metric": metric,
                        "remoteclip": rc[metric],
                        "xgboost": xgb[metric],
                        "delta_remoteclip_minus_xgboost": rc[metric] - xgb[metric],
                    }
                )
    summary_rows: list[dict[str, Any]] = []
    for metric in PAIRED_METRICS:
        values = np.asarray(
            [
                float(row["delta_remoteclip_minus_xgboost"])
                for row in bootstrap_rows
                if row["metric"] == metric
            ],
            dtype=float,
        )
        rc_value = point_metrics["remoteclip"].get(metric)
        xgb_value = point_metrics["xgboost"].get(metric)
        summary_rows.append(
            {
                "metric": metric,
                "remoteclip": rc_value,
                "xgboost": xgb_value,
                "point_delta": (
                    float(rc_value) - float(xgb_value)
                    if rc_value is not None and xgb_value is not None
                    else None
                ),
                "bootstrap_runs": len(values),
                "mean_delta": float(np.mean(values)) if len(values) else None,
                "median_delta": float(np.median(values)) if len(values) else None,
                "ci_low": float(np.quantile(values, 0.025)) if len(values) else None,
                "ci_high": float(np.quantile(values, 0.975)) if len(values) else None,
                "probability_delta_gt_zero": float(np.mean(values > 0.0)) if len(values) else None,
            }
        )
    joint = Counter(str(row.get("joint_outcome") or "UNAVAILABLE") for row in eligible)
    return {
        "summary": {
            "events": len(eligible),
            "points": len(sample_ids),
            "bootstrap_unit": "sample_id",
            "bootstrap_runs_requested": bootstrap_runs,
        },
        "metric_rows": summary_rows,
        "bootstrap_rows": bootstrap_rows,
        "joint_outcomes": [
            {
                "joint_outcome": key,
                "count": count,
                "percent": 100.0 * count / len(eligible) if eligible else None,
            }
            for key, count in joint.most_common()
        ],
    }


def _binary_rule_metrics(
    rows: Sequence[dict[str, Any]],
    predicted: Sequence[bool],
    *,
    rule: str,
    threshold: float | None,
) -> dict[str, Any]:
    target = np.asarray([_review_class(row) == "EVENT_TOO_SMALL" for row in rows], dtype=bool)
    pred = np.asarray(predicted, dtype=bool)
    true_positive = int(np.sum(pred & target))
    false_positive = int(np.sum(pred & ~target))
    false_negative = int(np.sum(~pred & target))
    true_negative = int(np.sum(~pred & ~target))
    valid_mask = np.asarray([row.get("review_status") == "VALID" for row in rows], dtype=bool)
    valid_lost = int(np.sum(pred & valid_mask))
    valid_total = int(np.sum(valid_mask))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else None
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else None
    specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive else None
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "rule": rule,
        "threshold": threshold,
        "events": len(rows),
        "event_too_small_events": int(np.sum(target)),
        "predicted_small_events": int(np.sum(pred)),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "valid_events": valid_total,
        "valid_events_lost": valid_lost,
        "valid_lost_rate": valid_lost / valid_total if valid_total else None,
    }


def validate_fixed_small_event_rules(
    rows: Sequence[dict[str, Any]],
    *,
    baseline_sample_ids: set[str],
) -> dict[str, Any]:
    validation = [dict(row) for row in rows if str(row["sample_id"]) not in baseline_sample_ids]
    metric_rows: list[dict[str, Any]] = []
    feature_predictions: dict[str, dict[str, bool]] = defaultdict(dict)
    for feature, threshold in FIXED_SMALL_EVENT_THRESHOLDS.items():
        selected = [row for row in validation if _number(row.get(feature)) is not None]
        predicted = [float(row[feature]) <= threshold for row in selected]
        metric_rows.append(
            {
                **_binary_rule_metrics(
                    selected,
                    predicted,
                    rule=f"{feature} <= {threshold}",
                    threshold=threshold,
                ),
                "feature": feature,
                "missing_events": len(validation) - len(selected),
            }
        )
        for row, value in zip(selected, predicted, strict=True):
            feature_predictions[str(row["event_id"])][feature] = value
    combined_rows = [
        row
        for row in validation
        if any(_number(row.get(feature)) is not None for feature in FIXED_SMALL_EVENT_THRESHOLDS)
    ]
    combined_predictions = [
        any(feature_predictions[str(row["event_id"])].values()) for row in combined_rows
    ]
    metric_rows.append(
        {
            **_binary_rule_metrics(
                combined_rows,
                combined_predictions,
                rule="ANY_FIXED_THRESHOLD",
                threshold=None,
            ),
            "feature": "combined_or",
            "missing_events": len(validation) - len(combined_rows),
        }
    )
    return {
        "summary": {
            "validation_points": len({str(row["sample_id"]) for row in validation}),
            "validation_events": len(validation),
            "baseline_points": len(baseline_sample_ids),
            "thresholds_frozen_before_validation": True,
        },
        "metrics": metric_rows,
    }


def point_taxonomy(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped = _group_rows(rows)
    point_rows: list[dict[str, Any]] = []
    for sample_id, events in sorted(grouped.items()):
        counts = Counter(_review_class(row) for row in events)
        valid_logging = counts["VALID_LOGGING"]
        valid_wildfire = counts["VALID_WILDFIRE"]
        valid_total = valid_logging + valid_wildfire
        multi = len(events) >= 2
        flags = {
            "has_temporal_tails": valid_total >= 1 and counts["YEAR_MISMATCH"] >= 1,
            "has_small_components": valid_total >= 1 and counts["EVENT_TOO_SMALL"] >= 1,
            "has_repeated_same_class": valid_logging >= 2 or valid_wildfire >= 2,
            "has_true_class_change": valid_logging >= 1 and valid_wildfire >= 1,
        }
        if not multi:
            point_type = "SINGLE_HANSEN_EVENT"
        elif flags["has_true_class_change"]:
            point_type = "TRUE_CLASS_CHANGE"
        elif flags["has_repeated_same_class"]:
            point_type = "REPEATED_SAME_CLASS"
        elif flags["has_temporal_tails"]:
            point_type = "SINGLE_PHYSICAL_EVENT_WITH_TEMPORAL_TAILS"
        elif flags["has_small_components"]:
            point_type = "DOMINANT_EVENT_WITH_SMALL_COMPONENTS"
        else:
            point_type = "UNRESOLVED"
        point_rows.append(
            {
                "sample_id": sample_id,
                "display_name": events[0].get("display_name"),
                "point_type": point_type,
                "n_events": len(events),
                "event_years": [int(row["event_year"]) for row in events],
                "n_valid_events": valid_total,
                "n_rejected_events": len(events) - valid_total,
                "n_valid_logging": valid_logging,
                "n_valid_wildfire": valid_wildfire,
                "n_year_mismatch": counts["YEAR_MISMATCH"],
                "n_event_too_small": counts["EVENT_TOO_SMALL"],
                **flags,
            }
        )
    event_count_distribution = Counter(int(row["n_events"]) for row in point_rows)
    multi_rows = [row for row in point_rows if int(row["n_events"]) >= 2]
    taxonomy_counts = Counter(str(row["point_type"]) for row in multi_rows)
    taxonomy_rows = []
    for point_type, count in taxonomy_counts.most_common():
        selected = [row for row in multi_rows if row["point_type"] == point_type]
        taxonomy_rows.append(
            {
                "point_type": point_type,
                "points": count,
                "share_multi_event_points": 100.0 * count / len(multi_rows) if multi_rows else None,
                "mean_event_count": float(np.mean([int(row["n_events"]) for row in selected])),
            }
        )
    return {
        "summary": {
            "reviewed_points": len(point_rows),
            "multi_event_points": len(multi_rows),
            "multi_event_share": len(multi_rows) / len(point_rows) if point_rows else None,
        },
        "point_rows": point_rows,
        "event_count_distribution": [
            {
                "events_per_point": event_count,
                "points": count,
                "percent": 100.0 * count / len(point_rows) if point_rows else None,
            }
            for event_count, count in sorted(event_count_distribution.items())
        ],
        "taxonomy_rows": taxonomy_rows,
    }


def _rank_fraction(values: Sequence[float | None]) -> list[float]:
    available = sorted({float(value) for value in values if value is not None})
    if not available:
        return [0.0 for _ in values]
    if len(available) == 1:
        return [1.0 if value is not None else 0.0 for value in values]
    lookup = {value: index / (len(available) - 1) for index, value in enumerate(available)}
    return [lookup.get(float(value), 0.0) if value is not None else 0.0 for value in values]


def _enrich_ranking_row(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    area = _number(result.get("event_loss_area_ha"))
    pixels = _number(result.get("event_hansen_pixels"))
    result["event_loss_area_log1p"] = math.log1p(max(area, 0.0)) if area is not None else None
    result["event_hansen_pixels_log1p"] = (
        math.log1p(max(pixels, 0.0)) if pixels is not None else None
    )
    result["model_mean_probability_wildfire"] = _model_mean_probability(result)
    downloads = _number(result.get("download_count")) or 0.0
    result["bad_quality_share"] = (
        (_number(result.get("bad_quality_download_count")) or 0.0) / downloads
        if downloads
        else None
    )
    result["excluded_download_share"] = (
        (_number(result.get("excluded_download_count")) or 0.0) / downloads
        if downloads
        else None
    )
    return result


def _base_method_scores(events: Sequence[dict[str, Any]]) -> dict[str, list[float]]:
    areas = [_number(row.get("event_loss_area_ha")) for row in events]
    model_probabilities = [_model_mean_probability(row) for row in events]
    usable = [
        sum(
            _number(row.get(key)) or 0.0
            for key in (
                "usable_pre_download_count",
                "usable_event_download_count",
                "usable_post_download_count",
            )
        )
        for row in events
    ]
    quality = [
        1.0
        - 0.5 * (_number(row.get("bad_quality_share")) or 0.0)
        - 0.5 * (_number(row.get("excluded_download_share")) or 0.0)
        for row in events
    ]
    area_rank = _rank_fraction(areas)
    model_rank = _rank_fraction(model_probabilities)
    usable_rank = _rank_fraction(usable)
    quality_rank = _rank_fraction(quality)
    return {
        "HISTORICAL_DOMINANT": [1.0 if _bool(row.get("event_is_dominant")) else 0.0 for row in events],
        "MAX_HANSEN_AREA": [value if value is not None else -1.0 for value in areas],
        "MAX_MODEL_MEAN_P_WILDFIRE": [
            value if value is not None else -1.0 for value in model_probabilities
        ],
        "HAND_BUILT_COMPOSITE": [
            0.45 * area_rank[index]
            + 0.30 * model_rank[index]
            + 0.15 * usable_rank[index]
            + 0.10 * quality_rank[index]
            for index in range(len(events))
        ],
    }


def _rank_events(
    sample_id: str,
    events: Sequence[dict[str, Any]],
    *,
    method: str,
    scores: Sequence[float],
    target_years: set[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ordered = sorted(
        range(len(events)),
        key=lambda index: (
            -float(scores[index]),
            -(_number(events[index].get("event_loss_area_ha")) or -1.0),
            int(events[index]["event_year"]),
        ),
    )
    details: list[dict[str, Any]] = []
    target_ranks: list[int] = []
    for rank, index in enumerate(ordered, start=1):
        event = events[index]
        year = int(event["event_year"])
        is_target = year in target_years
        if is_target:
            target_ranks.append(rank)
        details.append(
            {
                "method": method,
                "sample_id": sample_id,
                "event_id": event.get("event_id"),
                "event_year": year,
                "review_class": _review_class(event),
                "score": float(scores[index]),
                "rank": rank,
                "is_target_year": is_target,
            }
        )
    selected = events[ordered[0]]
    selected_year = int(selected["event_year"])
    best_rank = min(target_ranks) if target_ranks else None
    point = {
        "method": method,
        "sample_id": sample_id,
        "candidate_events": len(events),
        "target_years": sorted(target_years),
        "selected_year": selected_year,
        "selected_event_id": selected.get("event_id"),
        "hit_at_1": selected_year in target_years,
        "hit_at_2": bool(best_rank is not None and best_rank <= 2),
        "target_rank": best_rank,
        "reciprocal_rank": 1.0 / best_rank if best_rank else 0.0,
        "year_error": min(abs(selected_year - year) for year in target_years),
    }
    return details, point


def _lopo_scores(
    grouped: dict[str, list[dict[str, Any]]],
    target_years: dict[str, set[int]],
    *,
    seed: int,
) -> dict[str, list[float]]:
    scores_by_sample: dict[str, list[float]] = {}
    evaluable = sorted(target_years)
    for test_sample_id in evaluable:
        training_rows = [
            row
            for sample_id in evaluable
            if sample_id != test_sample_id
            for row in grouped[sample_id]
        ]
        target = np.asarray(
            [
                int(int(row["event_year"]) in target_years[str(row["sample_id"])])
                for row in training_rows
            ],
            dtype=int,
        )
        if len(training_rows) < 10 or len(set(target.tolist())) < 2:
            continue
        feature_names = [
            feature
            for feature in RANKING_FEATURES
            if any(_number(row.get(feature)) is not None for row in training_rows)
        ]
        if not feature_names:
            continue
        train_matrix = np.asarray(
            [
                [
                    np.nan if _number(row.get(feature)) is None else float(row[feature])
                    for feature in feature_names
                ]
                for row in training_rows
            ],
            dtype=float,
        )
        test_events = grouped[test_sample_id]
        test_matrix = np.asarray(
            [
                [
                    np.nan if _number(row.get(feature)) is None else float(row[feature])
                    for feature in feature_names
                ]
                for row in test_events
            ],
            dtype=float,
        )
        weights = np.asarray(
            [1.0 / len(grouped[str(row["sample_id"])]) for row in training_rows],
            dtype=float,
        )
        estimator = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
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
        estimator.fit(train_matrix, target, classifier__sample_weight=weights)
        scores_by_sample[test_sample_id] = estimator.predict_proba(test_matrix)[:, 1].tolist()
    return scores_by_sample


def dominant_year_ranking(
    rows: Sequence[dict[str, Any]],
    *,
    seed: int = 42,
) -> dict[str, Any]:
    raw_grouped = _group_rows(rows)
    grouped = {
        sample_id: [_enrich_ranking_row(row) for row in events]
        for sample_id, events in raw_grouped.items()
        if len(events) >= 2
    }
    target_years = {
        sample_id: {
            int(row["event_year"])
            for row in events
            if row.get("review_status") == "VALID" and row.get("human_label") in TARGET_LABELS
        }
        for sample_id, events in grouped.items()
    }
    target_years = {key: value for key, value in target_years.items() if value}
    lopo = _lopo_scores(grouped, target_years, seed=seed)
    detail_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    for sample_id, targets in sorted(target_years.items()):
        events = grouped[sample_id]
        method_scores = _base_method_scores(events)
        if sample_id in lopo:
            method_scores["LOPO_LOGISTIC_REGRESSION"] = lopo[sample_id]
        for method, scores in method_scores.items():
            details, point = _rank_events(
                sample_id,
                events,
                method=method,
                scores=scores,
                target_years=targets,
            )
            detail_rows.extend(details)
            point_rows.append(point)
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in point_rows:
        by_method[str(row["method"])].append(row)
    metric_rows: list[dict[str, Any]] = []
    for method, selected in sorted(by_method.items()):
        metric_rows.append(
            {
                "method": method,
                "eligible_points": len(selected),
                "hit_at_1": float(np.mean([bool(row["hit_at_1"]) for row in selected])),
                "hit_at_2": float(np.mean([bool(row["hit_at_2"]) for row in selected])),
                "mean_reciprocal_rank": float(
                    np.mean([float(row["reciprocal_rank"]) for row in selected])
                ),
                "median_target_rank": float(
                    np.median([int(row["target_rank"]) for row in selected])
                ),
                "mean_year_error": float(np.mean([float(row["year_error"]) for row in selected])),
            }
        )
    selected_by_point: dict[str, set[int]] = defaultdict(set)
    for row in point_rows:
        selected_by_point[str(row["sample_id"])].add(int(row["selected_year"]))
    differing_points = sum(len(years) > 1 for years in selected_by_point.values())
    return {
        "summary": {
            "multi_event_points": len(grouped),
            "evaluable_points": len(target_years),
            "points_where_methods_differ": differing_points,
            "target_definition": "all human VALID Logging/Wildfire years in the point",
        },
        "metrics": metric_rows,
        "point_predictions": point_rows,
        "ranking_details": detail_rows,
        "grouped_events": grouped,
    }


def _cluster_bootstrap_proportion(
    rows: Sequence[dict[str, Any]],
    predicate,
    *,
    bootstrap_runs: int,
    seed: int,
) -> dict[str, Any]:
    grouped = _group_rows(rows)
    sample_ids = sorted(grouped)
    estimate = float(np.mean([bool(predicate(row)) for row in rows])) if rows else None
    if not sample_ids:
        return {"estimate": estimate, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(bootstrap_runs):
        sampled = rng.choice(sample_ids, size=len(sample_ids), replace=True)
        batch = [row for sample_id in sampled for row in grouped[str(sample_id)]]
        if batch:
            values.append(float(np.mean([bool(predicate(row)) for row in batch])))
    return {
        "estimate": estimate,
        "ci_low": float(np.quantile(values, 0.025)) if values else None,
        "ci_high": float(np.quantile(values, 0.975)) if values else None,
    }


def dominant_relative_analysis(
    ranking: dict[str, Any],
    *,
    bootstrap_runs: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = ranking["grouped_events"]
    relative_rows: list[dict[str, Any]] = []
    for prediction in ranking["point_predictions"]:
        sample_id = str(prediction["sample_id"])
        method = str(prediction["method"])
        selected_year = int(prediction["selected_year"])
        events = grouped[sample_id]
        dominant = next(row for row in events if int(row["event_year"]) == selected_year)
        dominant_area = _number(dominant.get("event_loss_area_ha"))
        dominant_pixels = _number(dominant.get("event_hansen_pixels"))
        for event in events:
            area = _number(event.get("event_loss_area_ha"))
            pixels = _number(event.get("event_hansen_pixels"))
            relative_rows.append(
                {
                    **event,
                    "method": method,
                    "dominant_year": selected_year,
                    "delta_year": int(event["event_year"]) - selected_year,
                    "abs_delta_year": abs(int(event["event_year"]) - selected_year),
                    "relative_area": (
                        area / dominant_area
                        if area is not None and dominant_area not in (None, 0.0)
                        else None
                    ),
                    "relative_pixels": (
                        pixels / dominant_pixels
                        if pixels is not None and dominant_pixels not in (None, 0.0)
                        else None
                    ),
                    "review_hierarchy_class": _review_class(event),
                    "models_disagree": not _bool(event.get("models_agree")),
                }
            )
    summary_rows: list[dict[str, Any]] = []
    grouped_summary: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in relative_rows:
        grouped_summary[(str(row["method"]), int(row["delta_year"]))].append(row)
    for (method, delta), selected in sorted(grouped_summary.items()):
        classes = Counter(_review_class(row) for row in selected)
        areas = [value for row in selected if (value := _number(row.get("relative_area"))) is not None]
        rc = [value for row in selected if (value := _model_probability(row, "remoteclip")) is not None]
        xgb = [value for row in selected if (value := _model_probability(row, "xgboost")) is not None]
        summary_rows.append(
            {
                "method": method,
                "delta_year": delta,
                "events": len(selected),
                "points": len({str(row["sample_id"]) for row in selected}),
                "valid_share": sum(str(key).startswith("VALID_") for key in [_review_class(row) for row in selected]) / len(selected),
                "rejected_share": sum(not _review_class(row).startswith("VALID_") for row in selected) / len(selected),
                "year_mismatch_share": classes["YEAR_MISMATCH"] / len(selected),
                "event_too_small_share": classes["EVENT_TOO_SMALL"] / len(selected),
                "logging_share": classes["VALID_LOGGING"] / len(selected),
                "wildfire_share": classes["VALID_WILDFIRE"] / len(selected),
                "median_relative_area": float(np.median(areas)) if areas else None,
                "mean_remoteclip_p_wildfire": float(np.mean(rc)) if rc else None,
                "mean_xgboost_p_wildfire": float(np.mean(xgb)) if xgb else None,
                "model_disagreement_share": float(
                    np.mean([bool(row["models_disagree"]) for row in selected])
                ),
            }
        )
    headline_rows: list[dict[str, Any]] = []
    for method in DOMINANT_METHODS:
        method_rows = [row for row in relative_rows if row["method"] == method]
        strata = {
            "delta_0": [row for row in method_rows if int(row["delta_year"]) == 0],
            "abs_delta_1": [row for row in method_rows if int(row["abs_delta_year"]) == 1],
        }
        for stratum, selected in strata.items():
            for outcome, predicate in (
                ("VALID", lambda row: _review_class(row).startswith("VALID_")),
                ("YEAR_MISMATCH", lambda row: _review_class(row) == "YEAR_MISMATCH"),
                ("EVENT_TOO_SMALL", lambda row: _review_class(row) == "EVENT_TOO_SMALL"),
            ):
                stats = _cluster_bootstrap_proportion(
                    selected,
                    predicate,
                    bootstrap_runs=bootstrap_runs,
                    seed=seed,
                )
                headline_rows.append(
                    {
                        "method": method,
                        "stratum": stratum,
                        "outcome": outcome,
                        "events": len(selected),
                        "points": len({str(row["sample_id"]) for row in selected}),
                        **stats,
                    }
                )
    class_rows: list[dict[str, Any]] = []
    for review_class in ("VALID_LOGGING", "VALID_WILDFIRE", "YEAR_MISMATCH", "EVENT_TOO_SMALL"):
        selected = [
            row
            for row in relative_rows
            if row["method"] == "MAX_HANSEN_AREA" and _review_class(row) == review_class
        ]
        values = [value for row in selected if (value := _number(row.get("relative_area"))) is not None]
        class_rows.append(
            {
                "review_class": review_class,
                "events": len(selected),
                "median_relative_area": float(np.median(values)) if values else None,
                "mean_relative_area": float(np.mean(values)) if values else None,
                "p25_relative_area": float(np.quantile(values, 0.25)) if values else None,
                "p75_relative_area": float(np.quantile(values, 0.75)) if values else None,
            }
        )
    return {
        "event_rows": relative_rows,
        "summary_rows": summary_rows,
        "headline_rows": headline_rows,
        "relative_area_by_class": class_rows,
    }


def merge_simulation(
    rows: Sequence[dict[str, Any]],
    *,
    relative_area_threshold: float = MERGE_RELATIVE_AREA_THRESHOLD,
) -> dict[str, Any]:
    grouped = {key: value for key, value in _group_rows(rows).items() if len(value) >= 2}
    pair_rows: list[dict[str, Any]] = []
    identifiable_tails = 0
    before_events = sum(len(events) for events in grouped.values())
    components_after = 0
    components_after_good_only = 0
    for sample_id, events in sorted(grouped.items()):
        parents = list(range(len(events)))
        parents_good = list(range(len(events)))

        def find(parents_value: list[int], index: int) -> int:
            while parents_value[index] != index:
                parents_value[index] = parents_value[parents_value[index]]
                index = parents_value[index]
            return index

        def union(parents_value: list[int], left: int, right: int) -> None:
            root_left = find(parents_value, left)
            root_right = find(parents_value, right)
            if root_left != root_right:
                parents_value[root_right] = root_left

        for left_index in range(len(events)):
            for right_index in range(left_index + 1, len(events)):
                left = events[left_index]
                right = events[right_index]
                year_gap = abs(int(left["event_year"]) - int(right["event_year"]))
                if year_gap > 1:
                    continue
                areas = (
                    _number(left.get("event_loss_area_ha")),
                    _number(right.get("event_loss_area_ha")),
                )
                if areas[0] is None or areas[1] is None or max(areas) <= 0:
                    continue
                relative = min(areas) / max(areas)
                classes = (_review_class(left), _review_class(right))
                valid = tuple(value.startswith("VALID_") for value in classes)
                has_tail = any(value in {"YEAR_MISMATCH", "EVENT_TOO_SMALL"} for value in classes)
                identifiable = sum(valid) == 1 and has_tail
                identifiable_tails += int(identifiable)
                proposed = relative <= relative_area_threshold
                if not proposed:
                    outcome = "NOT_PROPOSED"
                elif all(valid):
                    outcome = "BAD_MERGE"
                elif identifiable:
                    outcome = "GOOD_MERGE"
                else:
                    outcome = "UNRESOLVED"
                if proposed:
                    union(parents, left_index, right_index)
                if outcome == "GOOD_MERGE":
                    union(parents_good, left_index, right_index)
                pair_rows.append(
                    {
                        "sample_id": sample_id,
                        "event_id_a": left.get("event_id"),
                        "event_id_b": right.get("event_id"),
                        "year_a": int(left["event_year"]),
                        "year_b": int(right["event_year"]),
                        "year_gap": year_gap,
                        "class_a": classes[0],
                        "class_b": classes[1],
                        "relative_area": relative,
                        "proposed": proposed,
                        "outcome": outcome,
                        "identifiable_tail_pair": identifiable,
                    }
                )
        components_after += len({find(parents, index) for index in range(len(events))})
        components_after_good_only += len(
            {find(parents_good, index) for index in range(len(events))}
        )
    proposed_rows = [row for row in pair_rows if row["proposed"]]
    outcomes = Counter(str(row["outcome"]) for row in proposed_rows)
    resolved = outcomes["GOOD_MERGE"] + outcomes["BAD_MERGE"]
    return {
        "summary": {
            "rule": f"adjacent year and relative area <= {relative_area_threshold}",
            "multi_event_points": len(grouped),
            "adjacent_pairs": len(pair_rows),
            "proposed_merges": len(proposed_rows),
            "good_merges": outcomes["GOOD_MERGE"],
            "bad_merges": outcomes["BAD_MERGE"],
            "unresolved_merges": outcomes["UNRESOLVED"],
            "merge_precision_resolved": outcomes["GOOD_MERGE"] / resolved if resolved else None,
            "merge_recall_identifiable_tails": (
                outcomes["GOOD_MERGE"] / identifiable_tails if identifiable_tails else None
            ),
            "physical_events_before": before_events,
            "event_groups_after_all_proposals": components_after,
            "event_groups_after_good_only": components_after_good_only,
        },
        "pair_rows": pair_rows,
    }


def load_hansen_mask_paths(
    db: Session,
    sample_ids: Iterable[str],
) -> dict[str, Path]:
    ids = sorted({str(value) for value in sample_ids})
    if not ids:
        return {}
    masks = db.scalars(
        select(HansenMask)
        .where(HansenMask.sample_id.in_(ids), HansenMask.mask_type == "aoi_all_loss")
        .order_by(HansenMask.sample_id, HansenMask.created_at.desc(), HansenMask.mask_id.desc())
    ).all()
    result: dict[str, Path] = {}
    for mask in masks:
        sample_id = str(mask.sample_id)
        if sample_id in result:
            continue
        path = resolve_storage_path(mask.local_path)
        if path.exists():
            result[sample_id] = path
    return result


def _spatial_pair_type(left: dict[str, Any], right: dict[str, Any]) -> str:
    classes = {_review_class(left), _review_class(right)}
    valid_classes = {value for value in classes if value.startswith("VALID_")}
    if "YEAR_MISMATCH" in classes and valid_classes:
        return "VALID_PLUS_YEAR_MISMATCH"
    if "EVENT_TOO_SMALL" in classes and valid_classes:
        return "VALID_PLUS_EVENT_TOO_SMALL"
    if classes == {"VALID_LOGGING", "VALID_WILDFIRE"}:
        return "VALID_LOGGING_PLUS_VALID_WILDFIRE"
    if len(valid_classes) == 1 and len(classes) == 1:
        return "VALID_PLUS_VALID_SAME_CLASS"
    return "OTHER"


def spatial_connectivity_analysis(
    rows: Sequence[dict[str, Any]],
    *,
    mask_paths: dict[str, Path],
) -> dict[str, Any]:
    try:
        import rasterio
        from scipy.ndimage import binary_dilation, distance_transform_edt
    except ImportError as exc:
        return {
            "summary": {"status": "dependency_missing", "detail": str(exc)},
            "pair_rows": [],
            "type_summary": [],
        }
    grouped = {key: value for key, value in _group_rows(rows).items() if len(value) >= 2}
    pair_rows: list[dict[str, Any]] = []
    missing_masks = 0
    missing_year_components = 0
    for sample_id, events in sorted(grouped.items()):
        path = mask_paths.get(sample_id)
        if path is None:
            missing_masks += 1
            continue
        with rasterio.open(path) as dataset:
            loss_year = dataset.read(1)
            transform = dataset.transform
            center_lat = (dataset.bounds.bottom + dataset.bounds.top) / 2.0
        pixel_width_m = abs(transform.a) * 111_320.0 * math.cos(math.radians(center_lat))
        pixel_height_m = abs(transform.e) * 110_574.0
        sampling = (pixel_height_m, pixel_width_m)
        structure = np.ones((3, 3), dtype=bool)
        for left_index in range(len(events)):
            for right_index in range(left_index + 1, len(events)):
                left = events[left_index]
                right = events[right_index]
                year_gap = abs(int(left["event_year"]) - int(right["event_year"]))
                if year_gap > 1:
                    continue
                left_mask = loss_year == int(left["event_year"]) - 2000
                right_mask = loss_year == int(right["event_year"]) - 2000
                if not np.any(left_mask) or not np.any(right_mask):
                    missing_year_components += 1
                    continue
                distance_map = distance_transform_edt(~left_mask, sampling=sampling)
                min_distance = float(np.min(distance_map[right_mask]))
                left_centroid = np.mean(np.argwhere(left_mask), axis=0)
                right_centroid = np.mean(np.argwhere(right_mask), axis=0)
                centroid_distance = float(
                    math.hypot(
                        (left_centroid[0] - right_centroid[0]) * pixel_height_m,
                        (left_centroid[1] - right_centroid[1]) * pixel_width_m,
                    )
                )
                area_left = _number(left.get("event_loss_area_ha"))
                area_right = _number(right.get("event_loss_area_ha"))
                relative_area = (
                    min(area_left, area_right) / max(area_left, area_right)
                    if area_left is not None and area_right is not None and max(area_left, area_right) > 0
                    else None
                )
                pair_rows.append(
                    {
                        "sample_id": sample_id,
                        "event_id_a": left.get("event_id"),
                        "event_id_b": right.get("event_id"),
                        "year_a": int(left["event_year"]),
                        "year_b": int(right["event_year"]),
                        "year_gap": year_gap,
                        "class_a": _review_class(left),
                        "class_b": _review_class(right),
                        "pair_type": _spatial_pair_type(left, right),
                        "minimum_pixel_center_distance_m": min_distance,
                        "centroid_distance_m": centroid_distance,
                        "touching_after_1px_dilation": bool(
                            np.any(binary_dilation(left_mask, structure=structure, iterations=1) & right_mask)
                        ),
                        "touching_after_2px_dilation": bool(
                            np.any(binary_dilation(left_mask, structure=structure, iterations=2) & right_mask)
                        ),
                        "relative_component_area": relative_area,
                        "mask_path": str(path),
                    }
                )
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        by_type[str(row["pair_type"])].append(row)
    type_summary: list[dict[str, Any]] = []
    for pair_type, selected in sorted(by_type.items()):
        min_distances = [float(row["minimum_pixel_center_distance_m"]) for row in selected]
        centroid_distances = [float(row["centroid_distance_m"]) for row in selected]
        relative_areas = [
            float(value)
            for row in selected
            if (value := _number(row.get("relative_component_area"))) is not None
        ]
        type_summary.append(
            {
                "pair_type": pair_type,
                "pairs": len(selected),
                "median_minimum_distance_m": float(np.median(min_distances)),
                "median_centroid_distance_m": float(np.median(centroid_distances)),
                "touching_1px_share": float(
                    np.mean([bool(row["touching_after_1px_dilation"]) for row in selected])
                ),
                "touching_2px_share": float(
                    np.mean([bool(row["touching_after_2px_dilation"]) for row in selected])
                ),
                "median_relative_area": float(np.median(relative_areas)) if relative_areas else None,
            }
        )
    return {
        "summary": {
            "status": "ok",
            "multi_event_points": len(grouped),
            "points_with_masks": len({str(row["sample_id"]) for row in pair_rows}),
            "adjacent_pairs_with_geometry": len(pair_rows),
            "points_missing_masks": missing_masks,
            "pairs_missing_year_components": missing_year_components,
            "distance_definition": "distance between raster pixel centers in local metre approximation",
        },
        "pair_rows": pair_rows,
        "type_summary": type_summary,
    }


def _count_rows(rows: Sequence[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    counts = Counter(str(row.get(key) or "(missing)") for row in rows)
    total = sum(counts.values())
    return [
        {key: value, "count": count, "percent": 100.0 * count / total if total else None}
        for value, count in counts.most_common()
    ]


def build_multi_event_research(
    rows: Sequence[dict[str, Any]],
    *,
    baseline_sample_ids: set[str],
    mask_paths: dict[str, Path] | None = None,
    bootstrap_runs: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    values = [dict(row) for row in rows]
    taxonomy = point_taxonomy(values)
    paired = paired_model_bootstrap(values, bootstrap_runs=bootstrap_runs, seed=seed)
    small_event = validate_fixed_small_event_rules(
        values,
        baseline_sample_ids=baseline_sample_ids,
    )
    ranking = dominant_year_ranking(values, seed=seed)
    dominant_relative = dominant_relative_analysis(
        ranking,
        bootstrap_runs=bootstrap_runs,
        seed=seed,
    )
    merge = merge_simulation(values)
    spatial = spatial_connectivity_analysis(values, mask_paths=mask_paths or {})
    hierarchy = _count_rows(values, "review_hierarchy_class")
    return {
        "summary": {
            "reviewed_events": len(values),
            "reviewed_points": len({str(row["sample_id"]) for row in values}),
            "valid_events": sum(row.get("review_status") == "VALID" for row in values),
            "rejected_events": sum(row.get("review_status") == "REJECTED" for row in values),
            "multi_event_points": taxonomy["summary"]["multi_event_points"],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "event_rows": values,
        "hierarchy_counts": hierarchy,
        "paired_models": paired,
        "fixed_small_event_validation": small_event,
        "taxonomy": taxonomy,
        "ranking": {key: value for key, value in ranking.items() if key != "grouped_events"},
        "dominant_relative": dominant_relative,
        "merge_simulation": merge,
        "spatial_connectivity": spatial,
        "settings": {
            "bootstrap_runs": bootstrap_runs,
            "bootstrap_unit": "sample_id",
            "seed": seed,
            "fixed_small_event_thresholds": FIXED_SMALL_EVENT_THRESHOLDS,
            "merge_relative_area_threshold": MERGE_RELATIVE_AREA_THRESHOLD,
            "firms_available": False,
        },
    }


def _fmt(value: Any, digits: int = 3) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def _pct(value: Any) -> str:
    return "-" if value is None else f"{100.0 * float(value):.1f}%"


def _headline_lookup(result: dict[str, Any], method: str) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row["stratum"]), str(row["outcome"])): row
        for row in result["dominant_relative"]["headline_rows"]
        if row["method"] == method
    }


def multi_event_report(
    result: dict[str, Any],
    *,
    base_diagnostics: dict[str, Any] | None = None,
    base_report_dir: str | Path | None = None,
) -> str:
    summary = result["summary"]
    taxonomy = result["taxonomy"]
    paired = result["paired_models"]
    small_event = result["fixed_small_event_validation"]
    ranking = result["ranking"]
    merge = result["merge_simulation"]["summary"]
    spatial = result["spatial_connectivity"]["summary"]
    headline = _headline_lookup(result, "MAX_HANSEN_AREA")
    valid_d0 = headline.get(("delta_0", "VALID"), {})
    valid_d1 = headline.get(("abs_delta_1", "VALID"), {})
    mismatch_d1 = headline.get(("abs_delta_1", "YEAR_MISMATCH"), {})
    small_d1 = headline.get(("abs_delta_1", "EVENT_TOO_SMALL"), {})
    combined_small_rule = next(
        (
            row
            for row in small_event["metrics"]
            if row.get("feature") == "combined_or"
        ),
        {},
    )
    best_ranking = max(
        ranking["metrics"],
        key=lambda row: (float(row["hit_at_1"]), float(row["mean_reciprocal_rank"])),
        default=None,
    )
    taxonomy_by_name = {
        str(row["point_type"]): row for row in taxonomy["taxonomy_rows"]
    }
    tail_points = sum(
        int(taxonomy_by_name.get(name, {}).get("points") or 0)
        for name in (
            "SINGLE_PHYSICAL_EVENT_WITH_TEMPORAL_TAILS",
            "DOMINANT_EVENT_WITH_SMALL_COMPONENTS",
        )
    )
    repeated_points = sum(
        int(taxonomy_by_name.get(name, {}).get("points") or 0)
        for name in ("REPEATED_SAME_CLASS", "TRUE_CLASS_CHANGE")
    )
    lines = [
        "# Multi-event Hansen structure and model validation",
        "",
        "## Research cohort",
        "",
        f"- Reviewed events: **{summary['reviewed_events']}**.",
        f"- Reviewed points: **{summary['reviewed_points']}**.",
        f"- VALID / REJECTED events: **{summary['valid_events']} / {summary['rejected_events']}**.",
        f"- Multi-event points: **{taxonomy['summary']['multi_event_points']}** "
        f"({_pct(taxonomy['summary']['multi_event_share'])}).",
        "",
        "All confidence intervals, paired tests, and cross-validation procedures use `sample_id` "
        "as the resampling/grouping unit. Events from the same point are never treated as "
        "independent observations.",
        "",
        "## Human review hierarchy",
        "",
        "| Class | Events | Share |",
        "|---|---:|---:|",
    ]
    for row in result["hierarchy_counts"]:
        lines.append(
            f"| {row['review_hierarchy_class']} | {row['count']} | {row['percent']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "Rejected events are not classification errors. `NO_PRE` and `QC_FAILED` remain "
            "acquisition-pipeline failures and are excluded from validity classifiers.",
            "",
            "## Paired RemoteCLIP vs XGBoost comparison",
            "",
            f"Paired grouped bootstrap uses **{paired['summary']['events']} valid events** from "
            f"**{paired['summary']['points']} points** and the same sampled points for both models.",
            "",
            "| Metric | RemoteCLIP | XGBoost | Point delta | Bootstrap mean delta | 95% CI | P(delta > 0) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in paired["metric_rows"]:
        lines.append(
            f"| {row['metric']} | {_fmt(row['remoteclip'])} | {_fmt(row['xgboost'])} | "
            f"{_fmt(row['point_delta'])} | {_fmt(row['mean_delta'])} | "
            f"[{_fmt(row['ci_low'])}, {_fmt(row['ci_high'])}] | "
            f"{_pct(row['probability_delta_gt_zero'])} |"
        )
    lines.extend(
        [
            "",
            "### Joint outcomes",
            "",
            "| Outcome | Events | Share |",
            "|---|---:|---:|",
        ]
    )
    for row in paired["joint_outcomes"]:
        lines.append(f"| {row['joint_outcome']} | {row['count']} | {row['percent']:.1f}% |")
    lines.extend(
        [
            "",
            "## Independent validation of frozen EVENT_TOO_SMALL rules",
            "",
            f"The thresholds are evaluated only on the **{small_event['summary']['validation_points']} "
            f"points / {small_event['summary']['validation_events']} events** added after the "
            "original 49-point analysis. They were not re-optimized before this test.",
            "",
            "| Rule | Precision | Recall | Specificity | F1 | VALID lost |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in small_event["metrics"]:
        lines.append(
            f"| `{row['rule']}` | {_fmt(row['precision'])} | {_fmt(row['recall'])} | "
            f"{_fmt(row['specificity'])} | {_fmt(row['f1'])} | "
            f"{row['valid_events_lost']} ({_pct(row['valid_lost_rate'])}) |"
        )
    lines.extend(
        [
            "",
            "The frozen signal generalizes, but it is not safe as a hard auto-reject rule: "
            f"the combined rule retains {_pct(combined_small_rule.get('recall'))} recall while "
            f"discarding {_pct(combined_small_rule.get('valid_lost_rate'))} of human-VALID "
            "holdout events. It should be used for review priority or as one feature in a "
            "conditional grouping model.",
        ]
    )
    lines.extend(
        [
            "",
            "## Multi-event point taxonomy",
            "",
            "Taxonomy priority is `TRUE_CLASS_CHANGE -> REPEATED_SAME_CLASS -> temporal tails -> "
            "small components -> UNRESOLVED`; overlapping boolean flags are preserved in the "
            "point-level artifact.",
            "",
            "| Point type | Points | Share of multi-event | Mean Hansen events |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in taxonomy["taxonomy_rows"]:
        lines.append(
            f"| {row['point_type']} | {row['points']} | "
            f"{row['share_multi_event_points']:.1f}% | {row['mean_event_count']:.2f} |"
        )
    lines.extend(
        [
            "",
            f"Tail/small-component types contain **{tail_points} points**; repeated or class-change "
            f"types contain **{repeated_points} points**. This is descriptive evidence, not an "
            "automatic physical-event ground truth.",
            "",
            "## Dominant-year ranking",
            "",
            "The target is a set of all human-VALID years per point; a point with two real VALID "
            "events is not forced to one target year.",
            "",
            "| Method | Points | Hit@1 | Hit@2 | MRR | Median target rank | Mean year error |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in ranking["metrics"]:
        lines.append(
            f"| {row['method']} | {row['eligible_points']} | {_fmt(row['hit_at_1'])} | "
            f"{_fmt(row['hit_at_2'])} | {_fmt(row['mean_reciprocal_rank'])} | "
            f"{_fmt(row['median_target_rank'])} | {_fmt(row['mean_year_error'])} |"
        )
    if best_ranking:
        lines.extend(
            [
                "",
                f"Best current Hit@1: **{best_ranking['method']} = "
                f"{_pct(best_ranking['hit_at_1'])}**. Methods select different years on "
                f"**{ranking['summary']['points_where_methods_differ']} points**.",
            ]
        )
    lines.extend(
        [
            "",
            "## Dominant-relative evidence (MAX_HANSEN_AREA)",
            "",
            "| Quantity | Estimate | 95% point-clustered CI | Events |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, row in (
        ("P(VALID | delta=0)", valid_d0),
        ("P(VALID | |delta|=1)", valid_d1),
        ("P(YEAR_MISMATCH | |delta|=1)", mismatch_d1),
        ("P(EVENT_TOO_SMALL | |delta|=1)", small_d1),
    ):
        lines.append(
            f"| {label} | {_fmt(row.get('estimate'))} | "
            f"[{_fmt(row.get('ci_low'))}, {_fmt(row.get('ci_high'))}] | "
            f"{row.get('events', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Candidate adjacent-year merge simulation",
            "",
            f"Rule: `{merge['rule']}`. Human review is used only to score proposed merges.",
            "",
            "| Proposed | GOOD | BAD | UNRESOLVED | Precision among resolved | Recall of identifiable tails | Groups before | Groups after |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
            f"| {merge['proposed_merges']} | {merge['good_merges']} | {merge['bad_merges']} | "
            f"{merge['unresolved_merges']} | {_fmt(merge['merge_precision_resolved'])} | "
            f"{_fmt(merge['merge_recall_identifiable_tails'])} | "
            f"{merge['physical_events_before']} | {merge['event_groups_after_all_proposals']} |",
            "",
            "## Spatial connectivity of adjacent Hansen components",
            "",
            f"Status: **{spatial.get('status')}**; geometry pairs: "
            f"**{spatial.get('adjacent_pairs_with_geometry', 0)}**; points with masks: "
            f"**{spatial.get('points_with_masks', 0)}**.",
            "",
            "| Pair type | Pairs | Median minimum distance, m | Median centroid distance, m | Touching after 1 px | Touching after 2 px | Median relative area |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["spatial_connectivity"]["type_summary"]:
        lines.append(
            f"| {row['pair_type']} | {row['pairs']} | "
            f"{_fmt(row['median_minimum_distance_m'], 1)} | "
            f"{_fmt(row['median_centroid_distance_m'], 1)} | "
            f"{_pct(row['touching_1px_share'])} | {_pct(row['touching_2px_share'])} | "
            f"{_fmt(row['median_relative_area'])} |"
        )
    if base_diagnostics:
        acquisition = base_diagnostics.get("acquisition_pipeline", {}).get("summary", [])
        validity = base_diagnostics.get("validity_models", {})
        lines.extend(
            [
                "",
                "## Acquisition and exploratory validity diagnostics",
                "",
                "`NO_PRE` and `QC_FAILED` are reported as pipeline losses, not semantic invalidity.",
                "",
                "| Pipeline reason | Events | Points |",
                "|---|---:|---:|",
            ]
        )
        for row in acquisition:
            lines.append(
                f"| {row.get('reason')} | {row.get('events')} | {row.get('points')} |"
            )
        lines.extend(
            [
                "",
                f"Exploratory VALID-vs-REJECTED models: status **{validity.get('status')}**. "
                "Grouped-CV metrics are exported separately and must not be interpreted as "
                "production performance.",
            ]
        )
    temporal_signal = (
        valid_d0.get("estimate") is not None
        and valid_d1.get("estimate") is not None
        and float(valid_d0["estimate"]) > float(valid_d1["estimate"])
    )
    bad_merge_share = (
        float(merge.get("bad_merges", 0)) / float(merge.get("proposed_merges", 0))
        if merge.get("proposed_merges", 0)
        else None
    )
    safe_merge = bad_merge_share is not None and bad_merge_share <= 0.10
    if temporal_signal and safe_merge:
        assessment = "supportive"
        assessment_detail = (
            "Both the dominant-relative pattern and the tested merge rule support conditional "
            "temporal grouping."
        )
    elif temporal_signal:
        assessment = "mixed"
        assessment_detail = (
            "The dominant-relative pattern strongly supports temporal tails, but the simple "
            f"merge rule would combine {merge.get('bad_merges', 0)} genuinely VALID pairs "
            f"({100.0 * bad_merge_share:.1f}% of proposals), so automatic merging is unsafe."
        )
    else:
        assessment = "not supported"
        assessment_detail = (
            "The current dominant-relative pattern does not provide stable evidence for temporal "
            "grouping."
        )
    lines.extend(
        [
            "",
            "## Main hypothesis assessment",
            "",
            f"Current evidence is **{assessment}** for conditional temporal grouping. "
            f"{assessment_detail} The result does not justify merging every adjacent Hansen year. "
            "Real repeated events and class changes remain explicit counterexamples and must stay "
            "separate.",
            "",
            "Recommended pipeline order remains:",
            "",
            "`Acquisition QC -> EVENT_TOO_SMALL screening -> temporal/EventGroup analysis -> "
            "dominant-year ranking -> RemoteCLIP/XGBoost -> knownness/disagreement -> manual review`.",
            "",
            "FIRMS data are not integrated; corresponding fields remain unavailable rather than "
            "being inferred.",
            "",
            "## Artifacts",
            "",
            "- `event_level_summary.csv` and `multi_event_points.csv`.",
            "- `paired_model_comparison.csv` and `paired_bootstrap_deltas.csv`.",
            "- `fixed_small_event_validation.csv`.",
            "- `dominant_year_ranking.csv`, `dominant_year_point_predictions.csv`.",
            "- `dominant_relative_events.csv`, `dominant_relative_summary.csv`, and bootstrap headlines.",
            "- `merge_simulation_pairs.csv`.",
            "- `spatial_connectivity_pairs.csv` and `spatial_connectivity_summary.csv`.",
            "- `plots/` contains the requested summary figures.",
        ]
    )
    if base_report_dir is not None:
        base_path = Path(base_report_dir)
        portable_base_reference = f"{base_path.parent.name}/{base_path.name}"
        lines.extend(
            [
                "",
                "Base rejection/knownness/acquisition report: "
                f"`{portable_base_reference}`.",
            ]
        )
    return "\n".join(lines)


def multi_event_figures(
    result: dict[str, Any],
    *,
    base_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    figures: dict[str, Any] = {}
    hierarchy = result["hierarchy_counts"]
    figure, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    axis.bar([row["review_hierarchy_class"] for row in hierarchy], [row["count"] for row in hierarchy])
    axis.set_title("Review hierarchy distribution")
    axis.set_ylabel("Events")
    axis.tick_params(axis="x", rotation=35)
    figures["review_hierarchy_distribution"] = figure

    relative = [
        row
        for row in result["dominant_relative"]["event_rows"]
        if row["method"] == "MAX_HANSEN_AREA"
    ]
    deltas = sorted({int(row["delta_year"]) for row in relative})
    classes = ("VALID_LOGGING", "VALID_WILDFIRE", "YEAR_MISMATCH", "EVENT_TOO_SMALL")
    figure, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    bottoms = np.zeros(len(deltas))
    for review_class in classes:
        values = np.asarray(
            [sum(_review_class(row) == review_class and int(row["delta_year"]) == delta for row in relative) for delta in deltas],
            dtype=float,
        )
        axis.bar(deltas, values, bottom=bottoms, label=review_class)
        bottoms += values
    axis.set_title("Review classes by year offset from max-area year")
    axis.set_xlabel("delta year")
    axis.set_ylabel("Events")
    axis.legend(fontsize=8)
    figures["review_class_by_delta_year"] = figure

    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    box_classes = ("VALID_LOGGING", "VALID_WILDFIRE", "YEAR_MISMATCH", "EVENT_TOO_SMALL")
    box_values = [
        [
            float(row["relative_area"])
            for row in relative
            if _review_class(row) == review_class and _number(row.get("relative_area")) is not None
        ]
        for review_class in box_classes
    ]
    present = [(name, values) for name, values in zip(box_classes, box_values, strict=True) if values]
    if present:
        axis.boxplot([values for _, values in present], tick_labels=[name for name, _ in present], showfliers=True)
    axis.set_yscale("log")
    axis.set_ylabel("Relative Hansen area (log scale)")
    axis.set_title("Relative Hansen area by review class")
    axis.tick_params(axis="x", rotation=25)
    figures["relative_area_by_review_class"] = figure

    max_area_summary = [
        row
        for row in result["dominant_relative"]["summary_rows"]
        if row["method"] == "MAX_HANSEN_AREA"
    ]
    figure, axis = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    axis.plot(
        [row["delta_year"] for row in max_area_summary],
        [row["valid_share"] for row in max_area_summary],
        marker="o",
    )
    axis.set_ylim(0.0, 1.05)
    axis.set_xlabel("delta year")
    axis.set_ylabel("P(VALID)")
    axis.set_title("VALID probability by offset from max-area year")
    axis.grid(alpha=0.25)
    figures["valid_probability_by_delta"] = figure

    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for review_class in classes:
        xs: list[int] = []
        ys: list[float] = []
        for delta in deltas:
            values = [
                value
                for row in relative
                if int(row["delta_year"]) == delta
                and _review_class(row) == review_class
                and (value := _model_probability(row, "remoteclip")) is not None
            ]
            if values:
                xs.append(delta)
                ys.append(float(np.mean(values)))
        if xs:
            axis.plot(xs, ys, marker="o", label=review_class)
    axis.set_ylim(0.0, 1.05)
    axis.set_xlabel("delta year")
    axis.set_ylabel("Mean RemoteCLIP P(Wildfire)")
    axis.set_title("RemoteCLIP confidence around max-area year")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    figures["remoteclip_probability_by_delta"] = figure

    ranking = result["ranking"]["metrics"]
    figure, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    x = np.arange(len(ranking))
    width = 0.25
    axis.bar(x - width, [row["hit_at_1"] for row in ranking], width, label="Hit@1")
    axis.bar(x, [row["hit_at_2"] for row in ranking], width, label="Hit@2")
    axis.bar(x + width, [row["mean_reciprocal_rank"] for row in ranking], width, label="MRR")
    axis.set_xticks(x, [str(row["method"]).replace("_", "\n") for row in ranking])
    axis.set_ylim(0.0, 1.05)
    axis.set_title("Dominant-year method comparison")
    axis.legend()
    figures["dominant_year_method_comparison"] = figure

    taxonomy = result["taxonomy"]["taxonomy_rows"]
    figure, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    axis.barh([row["point_type"] for row in taxonomy], [row["points"] for row in taxonomy])
    axis.set_title("Multi-event point taxonomy")
    axis.set_xlabel("Points")
    figures["multi_event_taxonomy"] = figure

    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    colors = {
        "VALID_LOGGING": "#2a9d8f",
        "VALID_WILDFIRE": "#e76f51",
        "YEAR_MISMATCH": "#7b2cbf",
        "EVENT_TOO_SMALL": "#457b9d",
    }
    for review_class, color in colors.items():
        selected = [row for row in result["event_rows"] if _review_class(row) == review_class]
        axis.scatter(
            [int(row["event_year"]) for row in selected if _number(row.get("event_loss_area_ha")) is not None],
            [float(row["event_loss_area_ha"]) for row in selected if _number(row.get("event_loss_area_ha")) is not None],
            label=review_class,
            alpha=0.65,
            s=22,
            color=color,
        )
    axis.set_yscale("log")
    axis.set_xlabel("Event year")
    axis.set_ylabel("Hansen event area, ha (log scale)")
    axis.set_title("Event area by human review class")
    axis.legend(fontsize=8)
    figures["event_area_by_review_status"] = figure

    paired = result["paired_models"]["metric_rows"]
    figure, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    x = np.arange(len(paired))
    width = 0.36
    axis.bar(x - width / 2, [row["remoteclip"] for row in paired], width, label="RemoteCLIP")
    axis.bar(x + width / 2, [row["xgboost"] for row in paired], width, label="XGBoost")
    axis.set_xticks(x, [str(row["metric"]).replace("_", "\n") for row in paired])
    axis.set_ylim(0.0, 1.05)
    axis.set_title("Paired model performance on VALID events")
    axis.legend()
    figures["paired_model_performance"] = figure

    if base_diagnostics:
        try:
            from .year_mismatch_research import year_mismatch_timeline_figure

            figures["year_mismatch_timelines"] = year_mismatch_timeline_figure(
                base_diagnostics["year_mismatch_research"]
            )
        except (KeyError, ValueError):
            pass
    return figures


def export_multi_event_research(
    result: dict[str, Any],
    *,
    output_root: str | Path,
    base_diagnostics: dict[str, Any] | None = None,
    base_report_dir: str | Path | None = None,
    run_id: str | None = None,
) -> Path:
    run_dir = create_run_directory(output_root, run_id)
    write_csv(run_dir / "event_level_summary.csv", result["event_rows"])
    write_csv(run_dir / "review_hierarchy_counts.csv", result["hierarchy_counts"])
    write_csv(run_dir / "paired_model_comparison.csv", result["paired_models"]["metric_rows"])
    write_csv(run_dir / "paired_bootstrap_deltas.csv", result["paired_models"]["bootstrap_rows"])
    write_csv(run_dir / "joint_model_outcomes.csv", result["paired_models"]["joint_outcomes"])
    write_csv(
        run_dir / "fixed_small_event_validation.csv",
        result["fixed_small_event_validation"]["metrics"],
    )
    write_csv(run_dir / "multi_event_points.csv", result["taxonomy"]["point_rows"])
    write_csv(
        run_dir / "events_per_point_distribution.csv",
        result["taxonomy"]["event_count_distribution"],
    )
    write_csv(run_dir / "multi_event_taxonomy.csv", result["taxonomy"]["taxonomy_rows"])
    write_csv(run_dir / "dominant_year_ranking.csv", result["ranking"]["metrics"])
    write_csv(
        run_dir / "dominant_year_point_predictions.csv",
        result["ranking"]["point_predictions"],
    )
    write_csv(run_dir / "dominant_year_ranking_details.csv", result["ranking"]["ranking_details"])
    write_csv(
        run_dir / "dominant_relative_events.csv",
        result["dominant_relative"]["event_rows"],
    )
    write_csv(
        run_dir / "dominant_relative_summary.csv",
        result["dominant_relative"]["summary_rows"],
    )
    write_csv(
        run_dir / "dominant_relative_bootstrap.csv",
        result["dominant_relative"]["headline_rows"],
    )
    write_csv(
        run_dir / "relative_area_by_review_class.csv",
        result["dominant_relative"]["relative_area_by_class"],
    )
    write_csv(run_dir / "merge_simulation_pairs.csv", result["merge_simulation"]["pair_rows"])
    write_csv(
        run_dir / "spatial_connectivity_pairs.csv",
        result["spatial_connectivity"]["pair_rows"],
    )
    write_csv(
        run_dir / "spatial_connectivity_summary.csv",
        result["spatial_connectivity"]["type_summary"],
    )
    if base_diagnostics:
        write_csv(
            run_dir / "knownness_status_summary.csv",
            base_diagnostics.get("knownness_status_summary", []),
        )
        write_csv(
            run_dir / "acquisition_pipeline_summary.csv",
            base_diagnostics.get("acquisition_pipeline", {}).get("summary", []),
        )
        write_csv(
            run_dir / "validity_model_metrics.csv",
            base_diagnostics.get("validity_models", {}).get("metrics", []),
        )
        write_csv(
            run_dir / "year_mismatch_ranking.csv",
            base_diagnostics.get("year_mismatch_research", {}).get("ranking_metrics", []),
        )
    summary_payload = {
        key: value
        for key, value in result.items()
        if key not in {"event_rows", "paired_models", "dominant_relative"}
    }
    summary_payload["paired_models"] = {
        key: value for key, value in result["paired_models"].items() if key != "bootstrap_rows"
    }
    summary_payload["dominant_relative"] = {
        key: value for key, value in result["dominant_relative"].items() if key != "event_rows"
    }
    write_json(run_dir / "summary.json", summary_payload)
    (run_dir / "REPORT.md").write_text(
        multi_event_report(
            result,
            base_diagnostics=base_diagnostics,
            base_report_dir=base_report_dir,
        ),
        encoding="utf-8",
    )
    for name, figure in multi_event_figures(
        result,
        base_diagnostics=base_diagnostics,
    ).items():
        figure.savefig(run_dir / "plots" / f"{name}.png", dpi=170, bbox_inches="tight")
        try:
            import matplotlib.pyplot as plt

            plt.close(figure)
        except ImportError:
            pass
    result["run_dir"] = str(run_dir)
    result["report_path"] = str(run_dir / "REPORT.md")
    return run_dir


def run_multi_event_research(
    rows: Sequence[dict[str, Any]],
    *,
    baseline_sample_ids: set[str],
    mask_paths: dict[str, Path] | None = None,
    bootstrap_runs: int = 2000,
    seed: int = 42,
    output_root: str | Path | None = None,
    base_diagnostics: dict[str, Any] | None = None,
    base_report_dir: str | Path | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    result = build_multi_event_research(
        rows,
        baseline_sample_ids=baseline_sample_ids,
        mask_paths=mask_paths,
        bootstrap_runs=bootstrap_runs,
        seed=seed,
    )
    if output_root is not None:
        export_multi_event_research(
            result,
            output_root=output_root,
            base_diagnostics=base_diagnostics,
            base_report_dir=base_report_dir,
            run_id=run_id,
        )
    return result
