from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Sequence

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET_LABELS = ("Logging", "Wildfire")
TIMELINE_OFFSETS = (-2, -1, 0, 1)

RANKING_FEATURES = (
    "event_loss_area_log1p",
    "event_hansen_pixels_log1p",
    "hansen_loss_aoi_percent",
    "remoteclip_probability_wildfire",
    "xgboost_probability_wildfire",
    "usable_pre_download_count",
    "usable_event_download_count",
    "usable_post_download_count",
    "bad_quality_share",
    "excluded_download_share",
)


def _number(value: Any) -> float | None:
    if value in (None, "", "None", "null", "nan"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _review_class(row: dict[str, Any]) -> str:
    value = row.get("review_hierarchy_class")
    if value:
        return str(value)
    if row.get("review_status") == "VALID":
        label = str(row.get("human_label") or "")
        return f"VALID_{label.upper()}" if label in TARGET_LABELS else "VALID_OTHER"
    return str(row.get("rejection_reason_group") or "OTHER_INVALID")


def _infer_expert_year(reason: str | None, hansen_year: int) -> tuple[int | None, str]:
    text = " ".join(str(reason or "").casefold().split())
    explicit = [int(value) for value in re.findall(r"\b20\d{2}\b", text)]
    if explicit:
        return explicit[0], "explicit_year_in_free_text"
    if any(token in text for token in ("прошл", "previous", "prior year")):
        return hansen_year - 1, "relative_year_in_free_text"
    if any(token in text for token in ("следующ", "next year")):
        return hansen_year + 1, "relative_year_in_free_text"
    if any(token in text for token in ("другого года", "wrong year", "year mismatch")):
        return None, "different_year_without_direction"
    return None, "expert_year_not_structured"


def _model_mean(row: dict[str, Any]) -> float | None:
    values = [
        value
        for key in ("remoteclip_probability_wildfire", "xgboost_probability_wildfire")
        if (value := _number(row.get(key))) is not None
    ]
    return float(np.mean(values)) if values else None


def _rank_fraction(values: Sequence[float | None]) -> list[float]:
    available = sorted({float(value) for value in values if value is not None})
    if not available:
        return [0.0 for _ in values]
    if len(available) == 1:
        return [1.0 if value is not None else 0.0 for value in values]
    lookup = {value: index / (len(available) - 1) for index, value in enumerate(available)}
    return [lookup.get(float(value), 0.0) if value is not None else 0.0 for value in values]


def _enrich_event(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    area = _number(result.get("event_loss_area_ha"))
    pixels = _number(result.get("event_hansen_pixels"))
    result["event_loss_area_log1p"] = math.log1p(max(area, 0.0)) if area is not None else None
    result["event_hansen_pixels_log1p"] = (
        math.log1p(max(pixels, 0.0)) if pixels is not None else None
    )
    result["model_mean_probability_wildfire"] = _model_mean(result)
    download_count = _number(result.get("download_count")) or 0.0
    bad_quality = _number(result.get("bad_quality_download_count")) or 0.0
    excluded = _number(result.get("excluded_download_count")) or 0.0
    result["bad_quality_share"] = bad_quality / download_count if download_count else None
    result["excluded_download_share"] = excluded / download_count if download_count else None
    result["expert_status"] = _review_class(result)
    return result


def _target_for_point(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    years = {int(row["event_year"]) for row in events}
    mismatch_rows = [row for row in events if _review_class(row) == "YEAR_MISMATCH"]
    inferred: list[int] = []
    inference_sources: list[str] = []
    for row in mismatch_rows:
        year, source = _infer_expert_year(row.get("rejection_reason"), int(row["event_year"]))
        if year is not None:
            inferred.append(year)
        inference_sources.append(source)
    inferred_unique = sorted(set(inferred))
    if len(inferred_unique) == 1:
        target = inferred_unique[0]
        if target in years:
            return {
                "target_year": target,
                "target_years": [target],
                "target_source": "expert_free_text",
                "target_status": "evaluable",
                "inferred_expert_years": inferred_unique,
                "inference_sources": sorted(set(inference_sources)),
            }
        return {
            "target_year": None,
            "target_years": inferred_unique,
            "target_source": "expert_free_text",
            "target_status": "expert_year_not_in_hansen_candidates",
            "inferred_expert_years": inferred_unique,
            "inference_sources": sorted(set(inference_sources)),
        }
    if len(inferred_unique) > 1:
        return {
            "target_year": None,
            "target_years": inferred_unique,
            "target_source": "expert_free_text",
            "target_status": "conflicting_expert_years",
            "inferred_expert_years": inferred_unique,
            "inference_sources": sorted(set(inference_sources)),
        }

    valid_wildfire_years = sorted(
        {
            int(row["event_year"])
            for row in events
            if _review_class(row) == "VALID_WILDFIRE"
        }
    )
    if len(valid_wildfire_years) == 1:
        return {
            "target_year": valid_wildfire_years[0],
            "target_years": valid_wildfire_years,
            "target_source": "unique_valid_wildfire",
            "target_status": "evaluable",
            "inferred_expert_years": [],
            "inference_sources": sorted(set(inference_sources)),
        }
    if len(valid_wildfire_years) > 1:
        status = "multiple_valid_wildfire_years"
    else:
        status = "no_selectable_expert_year"
    return {
        "target_year": None,
        "target_years": valid_wildfire_years,
        "target_source": "valid_review_fallback",
        "target_status": status,
        "inferred_expert_years": [],
        "inference_sources": sorted(set(inference_sources)),
    }


def _timeline_cell(year: int, event: dict[str, Any] | None) -> str:
    if event is None:
        return f"{year}: no Hansen event"
    area = _number(event.get("event_loss_area_ha"))
    rc = _number(event.get("remoteclip_probability_wildfire"))
    xgb = _number(event.get("xgboost_probability_wildfire"))
    counts = "/".join(
        str(int(_number(event.get(key)) or 0))
        for key in ("pre_download_count", "event_download_count", "post_download_count")
    )
    return (
        f"{year}: {_review_class(event)}; area={area:.3f} ha" if area is not None else f"{year}: {_review_class(event)}; area=n/a"
    ) + (
        f"; RC={rc:.3f}" if rc is not None else "; RC=n/a"
    ) + (
        f"; XGB={xgb:.3f}" if xgb is not None else "; XGB=n/a"
    ) + f"; PRE/EVENT/POST={counts}; FIRMS=n/a"


def _timeline_tables(
    grouped: dict[str, list[dict[str, Any]]],
    target_by_sample: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    long_rows: list[dict[str, Any]] = []
    wide_rows: list[dict[str, Any]] = []
    for sample_id, events in sorted(grouped.items()):
        by_year = {int(row["event_year"]): row for row in events}
        mismatch_rows = [row for row in events if _review_class(row) == "YEAR_MISMATCH"]
        for mismatch in mismatch_rows:
            anchor_year = int(mismatch["event_year"])
            target = target_by_sample[sample_id]
            wide: dict[str, Any] = {
                "sample_id": sample_id,
                "display_name": mismatch.get("display_name"),
                "anchor_mismatch_year": anchor_year,
                "target_year": target["target_year"],
                "target_source": target["target_source"],
                "target_status": target["target_status"],
            }
            for offset in TIMELINE_OFFSETS:
                year = anchor_year + offset
                event = by_year.get(year)
                period_seasons = dict(event.get("period_season_counts") or {}) if event else {}
                row = {
                    "sample_id": sample_id,
                    "display_name": mismatch.get("display_name"),
                    "anchor_event_id": mismatch.get("event_id"),
                    "anchor_mismatch_year": anchor_year,
                    "offset": offset,
                    "timeline_year": year,
                    "has_hansen_event": event is not None,
                    "event_id": event.get("event_id") if event else None,
                    "event_area_ha": event.get("event_loss_area_ha") if event else None,
                    "hansen_pixels": event.get("event_hansen_pixels") if event else None,
                    "hansen_loss_aoi_percent": event.get("hansen_loss_aoi_percent") if event else None,
                    "remoteclip_p_wildfire": event.get("remoteclip_probability_wildfire") if event else None,
                    "xgboost_p_wildfire": event.get("xgboost_probability_wildfire") if event else None,
                    "expert_status": _review_class(event) if event else "NO_HANSEN_EVENT",
                    "pre_count": event.get("pre_download_count") if event else None,
                    "event_count": event.get("event_download_count") if event else None,
                    "post_count": event.get("post_download_count") if event else None,
                    "usable_pre_count": event.get("usable_pre_download_count") if event else None,
                    "usable_event_count": event.get("usable_event_download_count") if event else None,
                    "usable_post_count": event.get("usable_post_download_count") if event else None,
                    "period_season_counts": period_seasons,
                    "is_selection_target": bool(event and target["target_year"] == year),
                    "target_year": target["target_year"],
                    "target_source": target["target_source"],
                    "firms_available": False,
                    "firms_detection_count": None,
                    "firms_nearest_distance_km": None,
                }
                long_rows.append(row)
                wide[f"Y{offset:+d}"] = _timeline_cell(year, event)
            wide_rows.append(wide)
    return long_rows, wide_rows


def _method_scores(events: Sequence[dict[str, Any]]) -> dict[str, list[float]]:
    areas = [_number(row.get("event_loss_area_ha")) for row in events]
    model_probabilities = [_number(row.get("model_mean_probability_wildfire")) for row in events]
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


def _rank_method(
    method: str,
    sample_id: str,
    events: Sequence[dict[str, Any]],
    scores: Sequence[float],
    target_year: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ordered = sorted(
        range(len(events)),
        key=lambda index: (
            -float(scores[index]),
            -(_number(events[index].get("event_loss_area_ha")) or -1.0),
            int(events[index]["event_year"]),
        ),
    )
    detail_rows: list[dict[str, Any]] = []
    target_rank: int | None = None
    for rank, index in enumerate(ordered, start=1):
        event = events[index]
        is_target = int(event["event_year"]) == target_year
        if is_target:
            target_rank = rank
        detail_rows.append(
            {
                "method": method,
                "sample_id": sample_id,
                "event_id": event.get("event_id"),
                "event_year": int(event["event_year"]),
                "expert_status": _review_class(event),
                "score": float(scores[index]),
                "rank": rank,
                "is_target_year": is_target,
            }
        )
    selected = events[ordered[0]]
    point_row = {
        "method": method,
        "sample_id": sample_id,
        "candidate_events": len(events),
        "target_year": target_year,
        "selected_year": int(selected["event_year"]),
        "selected_event_id": selected.get("event_id"),
        "hit_at_1": int(selected["event_year"]) == target_year,
        "target_rank": target_rank,
        "reciprocal_rank": 1.0 / target_rank if target_rank else 0.0,
    }
    return detail_rows, point_row


def _learned_lopo_rankings(
    grouped: dict[str, list[dict[str, Any]]],
    target_by_sample: dict[str, dict[str, Any]],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evaluable_ids = [
        sample_id
        for sample_id, target in target_by_sample.items()
        if target["target_status"] == "evaluable"
    ]
    detail_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    for test_sample_id in evaluable_ids:
        training_rows = [
            row
            for sample_id in evaluable_ids
            if sample_id != test_sample_id
            for row in grouped[sample_id]
        ]
        training_target = np.asarray(
            [
                int(int(row["event_year"]) == int(target_by_sample[str(row["sample_id"])]["target_year"]))
                for row in training_rows
            ],
            dtype=int,
        )
        if len(set(training_target.tolist())) < 2:
            continue
        feature_names = [
            feature
            for feature in RANKING_FEATURES
            if any(_number(row.get(feature)) is not None for row in training_rows)
        ]
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
        sample_weights = np.asarray(
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
                        class_weight="balanced", max_iter=3000, random_state=seed
                    ),
                ),
            ]
        )
        estimator.fit(
            train_matrix,
            training_target,
            classifier__sample_weight=sample_weights,
        )
        scores = estimator.predict_proba(test_matrix)[:, 1]
        details, point = _rank_method(
            "LOPO_LOGISTIC_REGRESSION",
            test_sample_id,
            test_events,
            scores,
            int(target_by_sample[test_sample_id]["target_year"]),
        )
        for row in details:
            row["feature_names"] = feature_names
        detail_rows.extend(details)
        point_rows.append(point)
    return detail_rows, point_rows


def _ranking_metrics(point_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in point_rows:
        grouped[str(row["method"])].append(row)
    result: list[dict[str, Any]] = []
    for method, rows in sorted(grouped.items()):
        result.append(
            {
                "method": method,
                "evaluable_points": len(rows),
                "hit_at_1_count": sum(bool(row["hit_at_1"]) for row in rows),
                "hit_at_1": float(np.mean([bool(row["hit_at_1"]) for row in rows])),
                "mean_reciprocal_rank": float(
                    np.mean([float(row["reciprocal_rank"]) for row in rows])
                ),
                "median_target_rank": float(
                    np.median([int(row["target_rank"]) for row in rows])
                ),
            }
        )
    return result


def build_year_mismatch_research(
    rows: Sequence[dict[str, Any]],
    *,
    seed: int = 42,
) -> dict[str, Any]:
    enriched = [_enrich_event(row) for row in rows]
    mismatch_sample_ids = {
        str(row["sample_id"])
        for row in enriched
        if _review_class(row) == "YEAR_MISMATCH"
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        sample_id = str(row["sample_id"])
        if sample_id in mismatch_sample_ids:
            grouped[sample_id].append(row)
    for events in grouped.values():
        events.sort(key=lambda row: (int(row["event_year"]), str(row.get("event_id") or "")))

    target_by_sample = {
        sample_id: _target_for_point(events) for sample_id, events in grouped.items()
    }
    timeline_rows, timeline_wide = _timeline_tables(grouped, target_by_sample)
    point_summaries: list[dict[str, Any]] = []
    for sample_id, events in sorted(grouped.items()):
        target = target_by_sample[sample_id]
        point_summaries.append(
            {
                "sample_id": sample_id,
                "display_name": events[0].get("display_name"),
                "candidate_event_years": [int(row["event_year"]) for row in events],
                "mismatch_years": [
                    int(row["event_year"])
                    for row in events
                    if _review_class(row) == "YEAR_MISMATCH"
                ],
                "valid_wildfire_years": [
                    int(row["event_year"])
                    for row in events
                    if _review_class(row) == "VALID_WILDFIRE"
                ],
                **target,
                "firms_available": False,
            }
        )

    ranking_details: list[dict[str, Any]] = []
    point_predictions: list[dict[str, Any]] = []
    for sample_id, events in sorted(grouped.items()):
        target = target_by_sample[sample_id]
        if target["target_status"] != "evaluable":
            continue
        for method, scores in _method_scores(events).items():
            details, point = _rank_method(
                method,
                sample_id,
                events,
                scores,
                int(target["target_year"]),
            )
            ranking_details.extend(details)
            point_predictions.append(point)
    learned_details, learned_points = _learned_lopo_rankings(
        grouped, target_by_sample, seed=seed
    )
    ranking_details.extend(learned_details)
    point_predictions.extend(learned_points)
    metrics = _ranking_metrics(point_predictions)
    best = max(
        metrics,
        key=lambda row: (row["hit_at_1"], row["mean_reciprocal_rank"]),
        default=None,
    )
    if best is None:
        hypothesis_status = "insufficient_ground_truth"
    elif best["hit_at_1"] >= 0.75 and best["mean_reciprocal_rank"] >= 0.85:
        hypothesis_status = "promising_for_candidate_ranking"
    elif best["hit_at_1"] >= 0.60:
        hypothesis_status = "weak_signal_requires_more_reviews"
    else:
        hypothesis_status = "not_supported_by_current_cohort"
    status_counts = Counter(target["target_status"] for target in target_by_sample.values())
    return {
        "summary": {
            "year_mismatch_points": len(grouped),
            "evaluable_points": status_counts["evaluable"],
            "non_evaluable_points": len(grouped) - status_counts["evaluable"],
            "target_status_counts": dict(sorted(status_counts.items())),
            "hypothesis_status": hypothesis_status,
            "best_method": best["method"] if best else None,
            "best_hit_at_1": best["hit_at_1"] if best else None,
            "best_mean_reciprocal_rank": (
                best["mean_reciprocal_rank"] if best else None
            ),
            "firms_available": False,
            "interpretation_note": (
                "Ground truth uses an explicit expert-visible year first and a unique "
                "VALID_WILDFIRE year only as fallback. Leave-one-point-out evaluation prevents "
                "events from the tested point entering its training fold. This is a ranking "
                "diagnostic on a very small cohort, not a production selector. Expert status, "
                "rejection reason, and target year are never used as ranking features."
            ),
        },
        "point_summaries": point_summaries,
        "timeline_rows": timeline_rows,
        "timeline_wide": timeline_wide,
        "ranking_metrics": metrics,
        "point_predictions": point_predictions,
        "ranking_details": ranking_details,
        "settings": {
            "timeline_offsets": list(TIMELINE_OFFSETS),
            "ranking_features": list(RANKING_FEATURES),
            "seed": seed,
        },
    }


def year_mismatch_timeline_figure(result: dict[str, Any]) -> Any:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = result["timeline_rows"]
    point_rows = result["point_summaries"]
    if not rows or not point_rows:
        return None
    sample_order = [str(row["sample_id"]) for row in point_rows]
    labels = [
        str(row.get("display_name") or row["sample_id"][:10])[:32]
        for row in point_rows
    ]
    colors = {
        "VALID_WILDFIRE": "#3b8f5a",
        "VALID_LOGGING": "#3f75a2",
        "YEAR_MISMATCH": "#d47a35",
        "EVENT_TOO_SMALL": "#9b6bb3",
        "NO_PRE": "#78838d",
        "QC_FAILED": "#c94f45",
        "NO_VISIBLE_CHANGE": "#ba9a3f",
        "NO_HANSEN_EVENT": "#d7dadd",
    }
    figure, axis = plt.subplots(
        figsize=(13, max(7.0, 0.72 * len(sample_order) + 2.5)), constrained_layout=True
    )
    y_lookup = {sample_id: index for index, sample_id in enumerate(sample_order)}
    for sample_id in sample_order:
        y = y_lookup[sample_id]
        axis.plot(TIMELINE_OFFSETS, [y] * len(TIMELINE_OFFSETS), color="#d9dde1", linewidth=1)
    for row in rows:
        sample_id = str(row["sample_id"])
        y = y_lookup[sample_id]
        status = str(row["expert_status"])
        area = _number(row.get("event_area_ha"))
        size = 35.0 if area is None else 35.0 + 45.0 * math.log1p(max(area, 0.0))
        marker = "*" if row.get("is_selection_target") else "o"
        axis.scatter(
            int(row["offset"]),
            y,
            s=size,
            marker=marker,
            color=colors.get(status, "#8b9299"),
            edgecolor="#202428",
            linewidth=0.6,
            zorder=3,
        )
        rc = _number(row.get("remoteclip_p_wildfire"))
        xgb = _number(row.get("xgboost_p_wildfire"))
        annotation = str(row["timeline_year"])
        if row.get("has_hansen_event"):
            annotation += f"\n{area:.2f}ha" if area is not None else "\narea n/a"
            if rc is not None or xgb is not None:
                annotation += (
                    f"\nR:{rc:.2f}" if rc is not None else "\nR:-"
                ) + (f" X:{xgb:.2f}" if xgb is not None else " X:-")
        axis.annotate(
            annotation,
            (int(row["offset"]), y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    axis.set_yticks(range(len(sample_order)), labels)
    axis.set_xticks(TIMELINE_OFFSETS, ["Y-2", "Y-1", "Y (mismatch)", "Y+1"])
    axis.set_xlim(min(TIMELINE_OFFSETS) - 0.35, max(TIMELINE_OFFSETS) + 0.35)
    axis.invert_yaxis()
    axis.set_title("YEAR_MISMATCH point timelines (size = event area; star = selection target)")
    axis.grid(axis="x", alpha=0.25)
    present_statuses = [
        status
        for status in colors
        if any(str(row["expert_status"]) == status for row in rows)
    ]
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=colors[status],
            markeredgecolor="#202428",
            markersize=8,
            label=status,
        )
        for status in present_statuses
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor="#ffffff",
            markeredgecolor="#202428",
            markersize=11,
            label="selection target",
        )
    )
    axis.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), title="Expert status")
    return figure


def year_mismatch_ranking_figure(result: dict[str, Any]) -> Any:
    import matplotlib.pyplot as plt

    metrics = result["ranking_metrics"]
    if not metrics:
        return None
    names = [str(row["method"]).replace("_", "\n") for row in metrics]
    positions = np.arange(len(metrics))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    axis.bar(
        positions - width / 2,
        [row["hit_at_1"] for row in metrics],
        width,
        label="Hit@1",
        color="#3f75a2",
    )
    axis.bar(
        positions + width / 2,
        [row["mean_reciprocal_rank"] for row in metrics],
        width,
        label="MRR",
        color="#d47a35",
    )
    axis.set_xticks(positions, names, fontsize=8)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Point-level score")
    axis.set_title("Can available features select the physical event year?")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    return figure
