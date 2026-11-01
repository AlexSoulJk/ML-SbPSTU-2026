from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any, Iterable

import joblib
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
)
from sqlalchemy import select
from sqlalchemy.orm import Session

from ResearchTool.backend.config import FOREST_MODELS_DIR
from ResearchTool.backend.forest.models import EventLabel, SampleEvent

from .config import stable_json_hash
from .experiments import LABEL_TO_INT, aggregate_event_embeddings, records_for_variant


KNN_VALUES = (3, 5, 10)
QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
AMBIGUOUS_BANDS = ((0.45, 0.55), (0.40, 0.60), (0.35, 0.65), (0.30, 0.70))
KNOWNNESS_REJECT_FRACTIONS = (0.01, 0.05, 0.10, 0.15, 0.20)
DEFAULT_UNKNOWN_QUANTILE = 0.05
DEFAULT_AMBIGUOUS_BAND = (0.40, 0.60)


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _majority(values: Iterable[str]) -> str:
    counts = Counter(str(value) for value in values if value)
    if not counts:
        return "unknown"
    return sorted(counts, key=lambda value: (-counts[value], value))[0]


def _load_remoteclip_artifact(version: str) -> tuple[dict[str, Any], dict[str, Any]]:
    model_dir = FOREST_MODELS_DIR / "remoteclip_logreg" / version
    manifest_path = model_dir / "manifest.json"
    artifact_path = model_dir / "classifier.joblib"
    if not manifest_path.exists() or not artifact_path.exists():
        raise FileNotFoundError(
            f"RemoteCLIP artifact {version} is incomplete under {model_dir}."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = joblib.load(artifact_path)
    class_order = list(payload.get("class_order") or manifest.get("class_order") or [])
    if class_order != ["Logging", "Wildfire"]:
        raise ValueError(f"Unsupported RemoteCLIP class order: {class_order}")
    return manifest, payload


def _latest_model_labels(
    db: Session,
    event_ids: Iterable[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    ids = sorted(set(event_ids))
    if not ids:
        return {}
    latest: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    rows = db.scalars(
        select(EventLabel)
        .where(
            EventLabel.event_id.in_(ids),
            EventLabel.source_type == "MODEL",
        )
        .order_by(EventLabel.created_at, EventLabel.label_id)
    )
    for row in rows:
        try:
            probabilities = json.loads(row.probabilities_json or "{}")
        except json.JSONDecodeError:
            probabilities = {}
        latest[row.event_id][row.source_name] = {
            "label": row.label,
            "confidence": row.confidence,
            "probabilities": probabilities,
            "source_version": row.source_version,
            "created_at": row.created_at.isoformat() if row.created_at else "",
        }
    return dict(latest)


def _sample_histories(
    db: Session,
    sample_ids: Iterable[str],
) -> dict[str, dict[str, Any]]:
    ids = sorted(set(sample_ids))
    histories: dict[str, list[SampleEvent]] = defaultdict(list)
    if ids:
        for event in db.scalars(
            select(SampleEvent)
            .where(SampleEvent.sample_id.in_(ids))
            .order_by(SampleEvent.sample_id, SampleEvent.event_year, SampleEvent.event_id)
        ):
            histories[event.sample_id].append(event)
    result: dict[str, dict[str, Any]] = {}
    for sample_id, events in histories.items():
        years = [int(event.event_year) for event in events]
        areas = [float(event.loss_area_ha or 0.0) for event in events]
        result[sample_id] = {
            "hansen_event_count": len(events),
            "hansen_event_years": years,
            "hansen_total_loss_area_ha": float(sum(areas)),
        }
    return result


def _predict_remoteclip(
    records: list[dict[str, Any]],
    payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if not records:
        return {}
    matrix = np.stack([np.asarray(record["features"], dtype=np.float32) for record in records])
    expected = int(payload.get("feature_dimension") or matrix.shape[1])
    if matrix.shape[1] != expected:
        raise ValueError(
            f"RemoteCLIP feature dimension {matrix.shape[1]} does not match artifact {expected}."
        )
    probabilities = np.asarray(
        payload["classifier"].predict_proba(payload["scaler"].transform(matrix)),
        dtype=float,
    )[:, 1]
    threshold = float(payload.get("decision_threshold") or 0.5)
    return {
        str(record["event_id"]): {
            "probability_wildfire": float(probability),
            "prediction": "Wildfire" if probability >= threshold else "Logging",
        }
        for record, probability in zip(records, probabilities, strict=True)
    }


def _knn_scores(
    query: np.ndarray,
    reference: np.ndarray,
    *,
    k_values: Iterable[int] = KNN_VALUES,
    leave_one_out: bool = False,
) -> list[dict[str, float]]:
    similarities = np.asarray(query, dtype=np.float32) @ np.asarray(reference, dtype=np.float32).T
    if leave_one_out:
        if similarities.shape[0] != similarities.shape[1]:
            raise ValueError("LOO kNN requires matching query and reference rows.")
        np.fill_diagonal(similarities, -np.inf)
    available = similarities.shape[1] - (1 if leave_one_out else 0)
    requested = sorted(set(int(k) for k in k_values))
    if not requested or available < max(requested):
        raise ValueError(f"kNN needs at least {max(requested)} neighbours; found {available}.")
    ordered = np.sort(similarities, axis=1)[:, ::-1]
    rows: list[dict[str, float]] = []
    for values in ordered:
        row: dict[str, float] = {"nearest_similarity": float(values[0])}
        for k in requested:
            neighbours = values[:k]
            row[f"knownness_k{k}"] = float(np.mean(neighbours))
            row[f"median_similarity_k{k}"] = float(np.median(neighbours))
        rows.append(row)
    return rows


def _class_conditional_scores(
    query: np.ndarray,
    reference: np.ndarray,
    reference_labels: list[str],
    *,
    leave_one_out: bool = False,
) -> list[dict[str, Any]]:
    similarities = np.asarray(query, dtype=np.float32) @ np.asarray(reference, dtype=np.float32).T
    if leave_one_out:
        if similarities.shape[0] != similarities.shape[1]:
            raise ValueError("LOO class-conditional kNN requires matching rows.")
        np.fill_diagonal(similarities, -np.inf)
    labels = np.asarray(reference_labels, dtype=object)
    output: list[dict[str, Any]] = []
    for row in similarities:
        sim_logging = float(np.max(row[labels == "Logging"]))
        sim_wildfire = float(np.max(row[labels == "Wildfire"]))
        prediction = "Wildfire" if sim_wildfire > sim_logging else "Logging"
        output.append(
            {
                "sim_logging": sim_logging,
                "sim_wildfire": sim_wildfire,
                "best_class_similarity": max(sim_logging, sim_wildfire),
                "similarity_margin": abs(sim_logging - sim_wildfire),
                "nearest_class_prediction": prediction,
            }
        )
    return output


def _cluster_mapping(
    raw_labels: np.ndarray,
    true_labels: list[str],
) -> tuple[dict[int, int], list[dict[str, Any]]]:
    clusters = sorted(set(int(value) for value in raw_labels.tolist()))
    if len(clusters) != 3:
        raise ValueError(f"KMeans k=3 produced {len(clusters)} clusters.")
    counts = Counter(zip((int(value) for value in raw_labels), true_labels))
    totals = Counter(int(value) for value in raw_labels)
    logging_cluster = max(
        clusters,
        key=lambda cluster: (
            counts[(cluster, "Logging")] / totals[cluster],
            counts[(cluster, "Logging")],
            -cluster,
        ),
    )
    remaining = [cluster for cluster in clusters if cluster != logging_cluster]
    wildfire_cluster = max(
        remaining,
        key=lambda cluster: (
            counts[(cluster, "Wildfire")] / totals[cluster],
            counts[(cluster, "Wildfire")],
            -cluster,
        ),
    )
    mixed_cluster = next(
        cluster for cluster in clusters if cluster not in {logging_cluster, wildfire_cluster}
    )
    mapping = {logging_cluster: 0, wildfire_cluster: 1, mixed_cluster: 2}
    roles = {0: "logging_mode", 1: "wildfire_mode", 2: "mixed_mode"}
    rows = []
    for raw in clusters:
        canonical = mapping[raw]
        rows.append(
            {
                "kmeans_cluster_raw": raw,
                "kmeans_cluster": canonical,
                "cluster_role": roles[canonical],
                "events": totals[raw],
                "logging_events": counts[(raw, "Logging")],
                "wildfire_events": counts[(raw, "Wildfire")],
                "logging_share": counts[(raw, "Logging")] / totals[raw],
                "wildfire_share": counts[(raw, "Wildfire")] / totals[raw],
            }
        )
    return mapping, sorted(rows, key=lambda row: row["kmeans_cluster"])


def _classification_metrics(
    true_labels: list[str],
    predictions: list[str],
) -> dict[str, float | int | None]:
    if not true_labels:
        return {
            "evaluated_events": 0,
            "accuracy": None,
            "balanced_accuracy": None,
            "logging_precision": None,
            "logging_recall": None,
            "wildfire_precision": None,
            "wildfire_recall": None,
        }
    y_true = np.asarray([LABEL_TO_INT[label] for label in true_labels], dtype=int)
    y_pred = np.asarray([LABEL_TO_INT[label] for label in predictions], dtype=int)
    return {
        "evaluated_events": len(true_labels),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "logging_precision": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "logging_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "wildfire_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "wildfire_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def _reject_metrics(
    rows: list[dict[str, Any]],
    retained: np.ndarray,
) -> dict[str, Any]:
    selected = [row for row, keep in zip(rows, retained, strict=True) if bool(keep)]
    metrics = _classification_metrics(
        [str(row["true_label"]) for row in selected],
        [str(row["remoteclip_prediction"]) for row in selected],
    )
    return {
        "coverage": float(np.mean(retained)),
        "rejected_events": int((~retained).sum()),
        **metrics,
    }


def _quantile_rows(
    values_by_source: dict[str, np.ndarray],
    *,
    metric: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, values in values_by_source.items():
        for quantile in QUANTILES:
            rows.append(
                {
                    "source": source,
                    "metric": metric,
                    "quantile": quantile,
                    "value": float(np.quantile(values, quantile)),
                    "events": int(len(values)),
                }
            )
    return rows


def _cluster_tables(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    season_counts = Counter(
        (row["kmeans_cluster"], row["true_label"], row["season"]) for row in rows
    )
    multi_counts = Counter(
        (row["kmeans_cluster"], row["true_label"], bool(row["is_multi_event_area"]))
        for row in rows
    )
    season_rows = [
        {
            "kmeans_cluster": cluster,
            "true_label": label,
            "season": season,
            "events": events,
        }
        for (cluster, label, season), events in sorted(season_counts.items())
    ]
    multi_rows = [
        {
            "kmeans_cluster": cluster,
            "true_label": label,
            "is_multi_event_area": multi,
            "events": events,
        }
        for (cluster, label, multi), events in sorted(multi_counts.items())
    ]
    return season_rows, multi_rows


def _mode_comparison(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    specs = [
        ("Logging", 0, "Logging / logging mode"),
        ("Logging", 2, "Logging / mixed mode"),
        ("Wildfire", 1, "Wildfire / wildfire mode"),
        ("Wildfire", 2, "Wildfire / mixed mode"),
    ]
    numeric = (
        "event_year",
        "event_loss_area_ha",
        "pre_image_count",
        "post_image_count",
        "hansen_event_count",
        "hansen_total_loss_area_ha",
        "logreg_probability_wildfire",
        "knownness_k5",
        "similarity_margin",
    )
    summary: list[dict[str, Any]] = []
    seasons: list[dict[str, Any]] = []
    for label, cluster, group in specs:
        selected = [
            row for row in rows
            if row["true_label"] == label and row["kmeans_cluster"] == cluster
        ]
        if not selected:
            continue
        summary.append(
            {
                "group": group,
                "metric": "multi_event_rate",
                "events": len(selected),
                "mean": float(np.mean([bool(row["is_multi_event_area"]) for row in selected])),
                "median": None,
                "p25": None,
                "p75": None,
            }
        )
        for metric in numeric:
            values = [
                float(row[metric]) for row in selected if _as_float(row.get(metric)) is not None
            ]
            if values:
                summary.append(
                    {
                        "group": group,
                        "metric": metric,
                        "events": len(values),
                        "mean": float(np.mean(values)),
                        "median": float(np.median(values)),
                        "p25": float(np.quantile(values, 0.25)),
                        "p75": float(np.quantile(values, 0.75)),
                    }
                )
        counts = Counter(str(row["season"]) for row in selected)
        seasons.extend(
            {
                "group": group,
                "season": season,
                "events": count,
                "share": count / len(selected),
            }
            for season, count in sorted(counts.items())
        )
    return summary, seasons


def _ambiguity_reject(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    probabilities = np.asarray([row["logreg_probability_wildfire"] for row in rows])
    output = []
    for lower, upper in AMBIGUOUS_BANDS:
        retained = ~((probabilities >= lower) & (probabilities <= upper))
        output.append(
            {
                "ambiguous_lower": lower,
                "ambiguous_upper": upper,
                "ambiguous_events": int((~retained).sum()),
                **_reject_metrics(rows, retained),
            }
        )
    return output


def _knownness_reject(rows: list[dict[str, Any]], *, metric: str = "knownness_k5") -> list[dict[str, Any]]:
    values = np.asarray([row[metric] for row in rows], dtype=float)
    output = []
    for fraction in KNOWNNESS_REJECT_FRACTIONS:
        threshold = float(np.quantile(values, fraction))
        retained = values >= threshold
        output.append(
            {
                "knownness_metric": metric,
                "reject_fraction_target": fraction,
                "knownness_threshold": threshold,
                **_reject_metrics(rows, retained),
            }
        )
    return output


def _combined_reject(rows: list[dict[str, Any]], *, metric: str = "knownness_k5") -> list[dict[str, Any]]:
    knownness = np.asarray([row[metric] for row in rows], dtype=float)
    probabilities = np.asarray([row["logreg_probability_wildfire"] for row in rows])
    output = []
    for fraction in (0.01, 0.05, 0.10):
        threshold = float(np.quantile(knownness, fraction))
        unknown = knownness < threshold
        for lower, upper in AMBIGUOUS_BANDS:
            ambiguous = (~unknown) & (probabilities >= lower) & (probabilities <= upper)
            retained = ~(unknown | ambiguous)
            output.append(
                {
                    "knownness_metric": metric,
                    "unknown_quantile": fraction,
                    "knownness_threshold": threshold,
                    "ambiguous_lower": lower,
                    "ambiguous_upper": upper,
                    "unknown_events": int(unknown.sum()),
                    "ambiguous_events": int(ambiguous.sum()),
                    **_reject_metrics(rows, retained),
                }
            )
    return output


def _align_labels(reference: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    ref_values = sorted(set(int(value) for value in reference.tolist()))
    pred_values = sorted(set(int(value) for value in predicted.tolist()))
    matrix = np.zeros((len(ref_values), len(pred_values)), dtype=int)
    for i, ref in enumerate(ref_values):
        for j, pred in enumerate(pred_values):
            matrix[i, j] = int(np.sum((reference == ref) & (predicted == pred)))
    ref_index, pred_index = linear_sum_assignment(-matrix)
    mapping = {pred_values[j]: ref_values[i] for i, j in zip(ref_index, pred_index, strict=True)}
    return np.asarray([mapping[int(value)] for value in predicted], dtype=int)


def _membership_stability(
    coordinates: np.ndarray,
    reference: np.ndarray,
    metadata: list[dict[str, Any]],
    mapping: dict[int, int],
    *,
    runs: int,
    sample_fraction: float,
    random_state: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    generator = np.random.default_rng(random_state)
    matches = np.zeros(len(reference), dtype=int)
    details: list[dict[str, Any]] = []
    subset_size = max(4, int(round(len(reference) * sample_fraction)))
    for run in range(runs):
        indices = generator.choice(len(reference), size=subset_size, replace=False)
        model = KMeans(
            n_clusters=3,
            n_init=20,
            random_state=random_state + run,
        ).fit(coordinates[indices])
        aligned = _align_labels(reference, model.predict(coordinates))
        same = aligned == reference
        matches += same.astype(int)
        details.append(
            {
                "run": run,
                "subset_size": subset_size,
                "adjusted_rand_index": float(adjusted_rand_score(reference, aligned)),
                "membership_match_rate": float(np.mean(same)),
            }
        )
    rows = [
        {
            "event_id": row["event_id"],
            "sample_id": row["sample_id"],
            "true_label": row["label"],
            "kmeans_cluster_raw": int(cluster),
            "kmeans_cluster": mapping[int(cluster)],
            "membership_stability": float(match / runs),
            "runs": runs,
        }
        for row, cluster, match in zip(metadata, reference, matches, strict=True)
    ]
    grouped: list[dict[str, Any]] = []
    for cluster in (0, 1, 2):
        values = [row["membership_stability"] for row in rows if row["kmeans_cluster"] == cluster]
        grouped.append(
            {
                "kmeans_cluster": cluster,
                "events": len(values),
                "membership_stability_mean": float(np.mean(values)),
                "membership_stability_median": float(np.median(values)),
                "membership_stability_p10": float(np.quantile(values, 0.10)),
            }
        )
    return rows, grouped, details


def _agreement_type(remoteclip: str | None, xgboost: str | None) -> str:
    if not remoteclip and not xgboost:
        return "NO_MODEL_LABELS"
    if not xgboost:
        return "REMOTECLIP_ONLY"
    if not remoteclip:
        return "XGBOOST_ONLY"
    if remoteclip == xgboost:
        return f"AGREE_{remoteclip.upper()}"
    return f"RC_{remoteclip.upper()}_XGB_{xgboost.upper()}"


def run_latent_diagnostics(
    db: Session,
    manifest: list[dict[str, Any]],
    embeddings: dict[str, np.ndarray],
    evaluation: dict[str, Any],
    *,
    remoteclip_version: str = "v1",
    reference_k: int = 3,
    membership_runs: int = 40,
    membership_sample_fraction: float = 0.80,
    random_state: int = 42,
) -> dict[str, Any]:
    if reference_k != 3:
        raise ValueError("The diagnostic cluster interpretation is defined for KMeans k=3.")
    if reference_k not in evaluation["kmeans"]["models_by_k"]:
        raise ValueError("KMeans k=3 is missing; run the base evaluation with K_MIN<=3<=K_MAX.")

    artifact_manifest, artifact = _load_remoteclip_artifact(remoteclip_version)
    feature_variant = str(artifact.get("feature_variant") or "")
    if feature_variant != "PRE_POST":
        raise ValueError(
            "These diagnostics expect the production PRE_POST RemoteCLIP artifact; "
            f"{remoteclip_version} uses {feature_variant}."
        )
    event_records = aggregate_event_embeddings(manifest, embeddings)
    feature_records = records_for_variant(event_records, feature_variant)
    manual_records = [
        row for row in feature_records if row["validation_cohort"] == "MANUAL_VALID"
    ]
    hansen_records = [
        row for row in feature_records if row.get("sample_source_format") == "HANSEN_MINING"
    ]
    if not manual_records or not hansen_records:
        raise ValueError(
            f"Expected manual and Hansen cohorts; found {len(manual_records)} and "
            f"{len(hansen_records)} events."
        )

    reconstructed_ids = sorted(str(row["record_id"]) for row in manual_records)
    expected_hash = str(
        artifact_manifest.get("training_cohort", {}).get("record_ids_sha256") or ""
    )
    reconstructed_hash = stable_json_hash(reconstructed_ids)
    exact_training_cohort = reconstructed_hash == expected_hash

    manual_post = np.stack([row["vectors"]["POST"] for row in manual_records]).astype(np.float32)
    hansen_post = np.stack([row["vectors"]["POST"] for row in hansen_records]).astype(np.float32)
    manual_labels = [str(row["label"]) for row in manual_records]
    manual_knn = _knn_scores(manual_post, manual_post, leave_one_out=True)
    hansen_knn = _knn_scores(hansen_post, manual_post)
    manual_class_knn = _class_conditional_scores(
        manual_post,
        manual_post,
        manual_labels,
        leave_one_out=True,
    )
    hansen_class_knn = _class_conditional_scores(hansen_post, manual_post, manual_labels)

    all_predictions = _predict_remoteclip(manual_records + hansen_records, artifact)
    all_event_ids = [str(row["event_id"]) for row in manual_records + hansen_records]
    stored_labels = _latest_model_labels(db, all_event_ids)
    histories = _sample_histories(
        db,
        [str(row["sample_id"]) for row in manual_records + hansen_records],
    )

    eval_metadata = evaluation["dataset"]["metadata"]
    eval_matrix = evaluation["dataset"]["matrix"]
    pca_model = evaluation["pca"]["model"]
    component_count = int(evaluation["pca"]["selected_components"])
    eval_coordinates = evaluation["pca"]["coordinates"]
    kmeans_model = evaluation["kmeans"]["models_by_k"][reference_k]
    eval_raw_clusters = np.asarray(evaluation["kmeans"]["labels_by_k"][reference_k], dtype=int)
    cluster_mapping, cluster_roles = _cluster_mapping(
        eval_raw_clusters,
        [str(row["label"]) for row in eval_metadata],
    )
    eval_cluster_by_event = {
        str(row["event_id"]): int(cluster)
        for row, cluster in zip(eval_metadata, eval_raw_clusters, strict=True)
    }
    eval_hdbscan_by_event = {
        str(row["event_id"]): int(cluster)
        for row, cluster in zip(
            eval_metadata,
            evaluation["hdbscan"]["labels"],
            strict=True,
        )
    }
    hansen_coordinates = pca_model.transform(hansen_post)[:, :component_count]
    hansen_raw_clusters = kmeans_model.predict(hansen_coordinates)
    nearest_eval_indices = np.argmax(hansen_post @ eval_matrix.T, axis=1)
    hansen_hdbscan = np.asarray(
        [evaluation["hdbscan"]["labels"][index] for index in nearest_eval_indices],
        dtype=int,
    )

    membership_rows, membership_summary, membership_details = _membership_stability(
        eval_coordinates,
        eval_raw_clusters,
        eval_metadata,
        cluster_mapping,
        runs=membership_runs,
        sample_fraction=membership_sample_fraction,
        random_state=random_state,
    )
    membership_by_event = {
        str(row["event_id"]): float(row["membership_stability"])
        for row in membership_rows
    }

    def base_row(
        record: dict[str, Any],
        *,
        source: str,
        knn: dict[str, Any],
        class_knn: dict[str, Any],
        raw_cluster: int,
        hdbscan_cluster: int,
        hdbscan_method: str,
    ) -> dict[str, Any]:
        event_id = str(record["event_id"])
        sample_id = str(record["sample_id"])
        prediction = all_predictions[event_id]
        history = histories.get(
            sample_id,
            {
                "hansen_event_count": 0,
                "hansen_event_years": [],
                "hansen_total_loss_area_ha": 0.0,
            },
        )
        years = list(history["hansen_event_years"])
        event_year = int(record["event_year"])
        stored = stored_labels.get(event_id, {})
        xgboost = stored.get("XGBOOST", {})
        stored_remoteclip = stored.get("REMOTECLIP_LOGREG", {})
        canonical = cluster_mapping[int(raw_cluster)]
        return {
            "source": source,
            "sample_id": sample_id,
            "event_id": event_id,
            "true_label": str(record["label"]) if source == "MANUAL_TRAIN" else "",
            "event_year": event_year,
            "season": _majority(record.get("period_seasons", {}).get("POST", [])),
            "post_seasons": record.get("period_seasons", {}).get("POST", []),
            "is_multi_event_area": bool(record.get("is_multi_event_area")),
            "event_loss_area_ha": record.get("event_loss_area_ha"),
            "event_loss_share": record.get("event_loss_share"),
            "pre_image_count": int(record.get("image_counts", {}).get("PRE", 0)),
            "post_image_count": int(record.get("image_counts", {}).get("POST", 0)),
            "hansen_event_count": int(history["hansen_event_count"]),
            "hansen_event_years": years,
            "hansen_prior_event_count": sum(year < event_year for year in years),
            "hansen_later_event_count": sum(year > event_year for year in years),
            "hansen_total_loss_area_ha": float(history["hansen_total_loss_area_ha"]),
            "logreg_probability_wildfire": prediction["probability_wildfire"],
            "remoteclip_prediction": prediction["prediction"],
            "stored_remoteclip_prediction": stored_remoteclip.get("label", ""),
            "xgboost_prediction": xgboost.get("label", ""),
            "kmeans_cluster_raw": int(raw_cluster),
            "kmeans_cluster": canonical,
            "kmeans_cluster_role": {0: "logging_mode", 1: "wildfire_mode", 2: "mixed_mode"}[
                canonical
            ],
            "hdbscan_cluster": int(hdbscan_cluster),
            "hdbscan_is_noise": int(hdbscan_cluster) == -1,
            "hdbscan_assignment_method": hdbscan_method,
            "cluster_membership_stability": membership_by_event.get(event_id),
            **knn,
            **class_knn,
        }

    manual_rows: list[dict[str, Any]] = []
    for record, knn, class_knn in zip(
        manual_records,
        manual_knn,
        manual_class_knn,
        strict=True,
    ):
        event_id = str(record["event_id"])
        manual_rows.append(
            base_row(
                record,
                source="MANUAL_TRAIN",
                knn=knn,
                class_knn=class_knn,
                raw_cluster=eval_cluster_by_event[event_id],
                hdbscan_cluster=eval_hdbscan_by_event[event_id],
                hdbscan_method="fitted_manual_post",
            )
        )
    hansen_rows = [
        base_row(
            record,
            source="HANSEN_MINED",
            knn=knn,
            class_knn=class_knn,
            raw_cluster=int(raw_cluster),
            hdbscan_cluster=int(hdbscan_cluster),
            hdbscan_method="nearest_manual_post_event",
        )
        for record, knn, class_knn, raw_cluster, hdbscan_cluster in zip(
            hansen_records,
            hansen_knn,
            hansen_class_knn,
            hansen_raw_clusters,
            hansen_hdbscan,
            strict=True,
        )
    ]

    knownness_threshold = float(
        np.quantile(
            [row["knownness_k5"] for row in manual_rows],
            DEFAULT_UNKNOWN_QUANTILE,
        )
    )
    ambiguous_lower, ambiguous_upper = DEFAULT_AMBIGUOUS_BAND
    for row in manual_rows + hansen_rows:
        row["unknown_flag"] = bool(row["knownness_k5"] < knownness_threshold)
        row["ambiguous_flag"] = bool(
            not row["unknown_flag"]
            and ambiguous_lower <= row["logreg_probability_wildfire"] <= ambiguous_upper
        )
        row["agreement_type"] = _agreement_type(
            str(row["remoteclip_prediction"] or "") or None,
            str(row["xgboost_prediction"] or "") or None,
        )

    cluster_season, cluster_multi = _cluster_tables(manual_rows)
    mode_summary, mode_seasons = _mode_comparison(manual_rows)

    loo_quantiles: list[dict[str, Any]] = []
    for k in KNN_VALUES:
        loo_quantiles.extend(
            _quantile_rows(
                {"MANUAL_TRAIN_LOO": np.asarray([row[f"knownness_k{k}"] for row in manual_rows])},
                metric=f"knownness_k{k}",
            )
        )

    nearest_metrics = _classification_metrics(
        [str(row["true_label"]) for row in manual_rows],
        [str(row["nearest_class_prediction"]) for row in manual_rows],
    )
    nearest_logreg_agreement = float(
        np.mean(
            [
                row["nearest_class_prediction"] == row["remoteclip_prediction"]
                for row in manual_rows
            ]
        )
    )

    probability_by_cluster = []
    for cluster in (0, 1, 2):
        selected = [row for row in manual_rows if row["kmeans_cluster"] == cluster]
        values = np.asarray([row["logreg_probability_wildfire"] for row in selected])
        probability_by_cluster.append(
            {
                "kmeans_cluster": cluster,
                "events": len(selected),
                "probability_mean": float(np.mean(values)),
                "probability_median": float(np.median(values)),
                "probability_p25": float(np.quantile(values, 0.25)),
                "probability_p75": float(np.quantile(values, 0.75)),
                "central_0_4_0_6_share": float(np.mean((values >= 0.4) & (values <= 0.6))),
            }
        )
    probability_correlations = []
    probabilities = np.asarray([row["logreg_probability_wildfire"] for row in manual_rows])
    for metric in ("sim_logging", "sim_wildfire", "similarity_margin"):
        correlation, p_value = spearmanr(
            probabilities,
            [row[metric] for row in manual_rows],
        )
        probability_correlations.append(
            {
                "metric": metric,
                "spearman_correlation": float(correlation),
                "p_value": float(p_value),
            }
        )

    ambiguity_reject = _ambiguity_reject(manual_rows)
    knownness_reject = _knownness_reject(manual_rows)
    combined_reject = _combined_reject(manual_rows)

    train_hansen_quantiles: list[dict[str, Any]] = []
    domain_shift: list[dict[str, Any]] = []
    for k in KNN_VALUES:
        metric = f"knownness_k{k}"
        manual_values = np.asarray([row[metric] for row in manual_rows])
        hansen_values = np.asarray([row[metric] for row in hansen_rows])
        train_hansen_quantiles.extend(
            _quantile_rows(
                {"MANUAL_TRAIN_LOO": manual_values, "HANSEN_MINED": hansen_values},
                metric=metric,
            )
        )
        for quantile in (0.01, 0.05):
            threshold = float(np.quantile(manual_values, quantile))
            domain_shift.append(
                {
                    "metric": metric,
                    "manual_quantile": quantile,
                    "manual_threshold": threshold,
                    "hansen_below_events": int(np.sum(hansen_values < threshold)),
                    "hansen_below_share": float(np.mean(hansen_values < threshold)),
                }
            )

    hansen_group_summary: list[dict[str, Any]] = []
    for agreement, selected in sorted(
        (
            key,
            [row for row in hansen_rows if row["agreement_type"] == key],
        )
        for key in {row["agreement_type"] for row in hansen_rows}
    ):
        if not selected:
            continue
        hansen_group_summary.append(
            {
                "agreement_type": agreement,
                "events": len(selected),
                "knownness_k5_mean": float(np.mean([row["knownness_k5"] for row in selected])),
                "knownness_k5_median": float(np.median([row["knownness_k5"] for row in selected])),
                "similarity_margin_mean": float(
                    np.mean([row["similarity_margin"] for row in selected])
                ),
                "probability_wildfire_mean": float(
                    np.mean([row["logreg_probability_wildfire"] for row in selected])
                ),
                "unknown_share": float(np.mean([row["unknown_flag"] for row in selected])),
                "ambiguous_share": float(np.mean([row["ambiguous_flag"] for row in selected])),
            }
        )

    rc_logging_xgb_wildfire = [
        row
        for row in hansen_rows
        if row["agreement_type"] == "RC_LOGGING_XGB_WILDFIRE"
    ]
    disagreement_diagnosis = Counter()
    for row in rc_logging_xgb_wildfire:
        if row["unknown_flag"]:
            diagnosis = "OOD_BY_MANUAL_P05"
        elif row["ambiguous_flag"]:
            diagnosis = "AMBIGUOUS_0_4_0_6"
        elif row["sim_logging"] >= row["sim_wildfire"]:
            diagnosis = "IN_DISTRIBUTION_LOGGING_LIKE"
        else:
            diagnosis = "WILDFIRE_LIKE_LATENT_CONFLICT"
        row["disagreement_diagnosis"] = diagnosis
        disagreement_diagnosis[diagnosis] += 1

    hdbscan_noise_summary: list[dict[str, Any]] = []
    for is_noise in (False, True):
        selected = [row for row in manual_rows if row["hdbscan_is_noise"] == is_noise]
        errors = [row["remoteclip_prediction"] != row["true_label"] for row in selected]
        values = [row["knownness_k5"] for row in selected]
        hdbscan_noise_summary.append(
            {
                "hdbscan_status": "noise" if is_noise else "clustered",
                "events": len(selected),
                "knownness_k5_mean": float(np.mean(values)),
                "knownness_k5_median": float(np.median(values)),
                "logreg_error_rate": float(np.mean(errors)),
            }
        )

    rank_stability: list[dict[str, Any]] = []
    unknown_overlap: list[dict[str, Any]] = []
    for left, right in ((3, 5), (3, 10), (5, 10)):
        left_values = np.asarray([row[f"knownness_k{left}"] for row in manual_rows])
        right_values = np.asarray([row[f"knownness_k{right}"] for row in manual_rows])
        correlation, p_value = spearmanr(left_values, right_values)
        rank_stability.append(
            {
                "left_k": left,
                "right_k": right,
                "spearman_correlation": float(correlation),
                "p_value": float(p_value),
            }
        )
        for fraction in KNOWNNESS_REJECT_FRACTIONS:
            count = max(1, int(np.ceil(len(manual_rows) * fraction)))
            left_ids = set(np.argsort(left_values)[:count].tolist())
            right_ids = set(np.argsort(right_values)[:count].tolist())
            unknown_overlap.append(
                {
                    "left_k": left,
                    "right_k": right,
                    "bottom_fraction": fraction,
                    "events_per_set": count,
                    "intersection": len(left_ids & right_ids),
                    "jaccard": len(left_ids & right_ids) / len(left_ids | right_ids),
                }
            )

    final_columns = (
        "source",
        "sample_id",
        "event_id",
        "event_year",
        "season",
        "true_label",
        "logreg_probability_wildfire",
        "kmeans_cluster",
        "kmeans_cluster_raw",
        "kmeans_cluster_role",
        "hdbscan_cluster",
        "hdbscan_is_noise",
        "hdbscan_assignment_method",
        "knownness_k3",
        "knownness_k5",
        "knownness_k10",
        "nearest_similarity",
        "sim_logging",
        "sim_wildfire",
        "similarity_margin",
        "ambiguous_flag",
        "unknown_flag",
        "remoteclip_prediction",
        "xgboost_prediction",
        "agreement_type",
        "is_multi_event_area",
        "event_loss_area_ha",
        "pre_image_count",
        "post_image_count",
        "hansen_event_count",
        "hansen_event_years",
        "cluster_membership_stability",
    )
    final_rows = [
        {column: row.get(column) for column in final_columns}
        for row in manual_rows + hansen_rows
    ]

    summary = {
        "manual_train_events": len(manual_rows),
        "manual_train_samples": len({row["sample_id"] for row in manual_rows}),
        "hansen_mined_events": len(hansen_rows),
        "hansen_mined_samples": len({row["sample_id"] for row in hansen_rows}),
        "remoteclip_version": remoteclip_version,
        "remoteclip_feature_variant": feature_variant,
        "exact_production_training_cohort": exact_training_cohort,
        "production_training_events": int(
            artifact_manifest.get("training_cohort", {}).get("records") or 0
        ),
        "current_reference_events": len(manual_rows),
        "production_training_record_ids_sha256": expected_hash,
        "reconstructed_training_record_ids_sha256": reconstructed_hash,
        "kmeans_reference_k": reference_k,
        "cluster_id_policy": (
            "Raw IDs are remapped after fitting: 0=most Logging-pure, "
            "1=most Wildfire-pure among the remaining clusters, 2=mixed."
        ),
        "hansen_hdbscan_policy": (
            "HDBSCAN has no inductive predict here; each Hansen event inherits the status "
            "of its nearest manual POST event. Noise is diagnostic and never implies UNKNOWN."
        ),
        "default_unknown_rule": {
            "metric": "knownness_k5",
            "manual_quantile": DEFAULT_UNKNOWN_QUANTILE,
            "threshold": knownness_threshold,
        },
        "default_ambiguous_rule": {
            "probability_lower": ambiguous_lower,
            "probability_upper": ambiguous_upper,
            "applied_only_when_not_unknown": True,
        },
        "manual_nearest_class_metrics": nearest_metrics,
        "manual_nearest_class_vs_logreg_agreement": nearest_logreg_agreement,
        "hansen_agreement_counts": dict(
            sorted(Counter(row["agreement_type"] for row in hansen_rows).items())
        ),
        "rc_logging_xgb_wildfire_diagnosis": dict(sorted(disagreement_diagnosis.items())),
        "threshold_note": (
            "Reject thresholds are train-side diagnostics only. They are not optimized on "
            "Hansen-mined events."
        ),
        "logreg_metric_note": (
            "Manual LogReg probabilities come from the frozen full-fit production artifact; "
            "manual reject quality is descriptive and optimistic, unlike grouped-CV metrics."
        ),
        "cohort_audit_note": (
            "The current MANUAL_VALID PRE_POST event-id set differs from the historical "
            "production training set; probabilities still use the frozen artifact."
            if not exact_training_cohort
            else "The current MANUAL_VALID PRE_POST event-id set matches the production artifact."
        ),
    }
    return {
        "summary": summary,
        "manual_rows": manual_rows,
        "hansen_rows": hansen_rows,
        "final_rows": final_rows,
        "cluster_roles": cluster_roles,
        "cluster_label_season": cluster_season,
        "cluster_label_multi_event": cluster_multi,
        "mode_comparison": mode_summary,
        "mode_seasons": mode_seasons,
        "loo_quantiles": loo_quantiles,
        "probability_by_cluster": probability_by_cluster,
        "probability_correlations": probability_correlations,
        "ambiguity_reject": ambiguity_reject,
        "knownness_reject": knownness_reject,
        "combined_reject": combined_reject,
        "train_hansen_quantiles": train_hansen_quantiles,
        "domain_shift": domain_shift,
        "hansen_group_summary": hansen_group_summary,
        "hdbscan_noise_summary": hdbscan_noise_summary,
        "knn_rank_stability": rank_stability,
        "knn_unknown_overlap": unknown_overlap,
        "membership_stability": membership_rows,
        "membership_stability_summary": membership_summary,
        "membership_stability_runs": membership_details,
    }


def _report_value(value: Any, kind: str = "plain") -> str:
    if value is None or value == "":
        return "–"
    if kind == "percent":
        return f"{float(value):.1%}"
    if kind == "float":
        return f"{float(value):.5f}"
    if kind == "metric":
        return f"{float(value):.3f}"
    if kind == "int":
        return str(int(value))
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.5f}"
    if isinstance(value, (list, tuple, dict)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(
    rows: Iterable[dict[str, Any]],
    columns: Iterable[tuple[str, str, str]],
) -> str:
    data = list(rows)
    specs = list(columns)
    if not data:
        return "_No rows._"
    header = "| " + " | ".join(label for _key, label, _kind in specs) + " |"
    separator = "| " + " | ".join("---" for _ in specs) + " |"
    body = [
        "| "
        + " | ".join(_report_value(row.get(key), kind) for key, _label, kind in specs)
        + " |"
        for row in data
    ]
    return "\n".join([header, separator, *body])


def diagnostics_report_markdown(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# RemoteCLIP latent-space and Hansen candidate diagnostics",
        "",
        "## Executive summary",
        "",
        (
            f"The current reference contains **{summary['current_reference_events']}** "
            f"MANUAL_VALID PRE/POST events from **{summary['manual_train_samples']}** samples. "
            f"The frozen production artifact was trained on "
            f"**{summary['production_training_events']}** events."
        ),
        "",
        (
            f"Production-cohort hash match: "
            f"**{'yes' if summary['exact_production_training_cohort'] else 'no'}**. "
            f"{summary['cohort_audit_note']}"
        ),
        "",
        (
            f"The frozen `{summary['remoteclip_version']}` `{summary['remoteclip_feature_variant']}` "
            f"classifier and the manual POST embedding reference were applied to "
            f"**{summary['hansen_mined_events']}** Hansen-mined events from "
            f"**{summary['hansen_mined_samples']}** samples."
        ),
        "",
        (
            f"The default diagnostic UNKNOWN rule is manual LOO `knownness_k5` below p05 "
            f"(**{summary['default_unknown_rule']['threshold']:.5f}**). The default AMBIGUOUS "
            "rule is `P(Wildfire)` in **0.40–0.60**, applied only after UNKNOWN. "
            "Neither threshold is optimized on Hansen events."
        ),
        "",
        "## Cohort and method audit",
        "",
        _markdown_table(
            [
                {
                    "source": "Current manual reference",
                    "events": summary["current_reference_events"],
                    "samples": summary["manual_train_samples"],
                },
                {
                    "source": "Historical production train",
                    "events": summary["production_training_events"],
                    "samples": "–",
                },
                {
                    "source": "Hansen-mined",
                    "events": summary["hansen_mined_events"],
                    "samples": summary["hansen_mined_samples"],
                },
            ],
            (
                ("source", "Source", "plain"),
                ("events", "Events", "plain"),
                ("samples", "Samples", "plain"),
            ),
        ),
        "",
        f"- `{summary['cluster_id_policy']}`",
        f"- `{summary['hansen_hdbscan_policy']}`",
        f"- {summary['logreg_metric_note']}",
        "",
        "## KMeans k=3 structure",
        "",
        _markdown_table(
            result["cluster_roles"],
            (
                ("kmeans_cluster", "Canonical cluster", "int"),
                ("kmeans_cluster_raw", "Raw cluster", "int"),
                ("cluster_role", "Role", "plain"),
                ("events", "Events", "int"),
                ("logging_events", "Logging", "int"),
                ("wildfire_events", "Wildfire", "int"),
                ("logging_share", "Logging share", "percent"),
                ("wildfire_share", "Wildfire share", "percent"),
            ),
        ),
        "",
        "The canonical IDs are assigned only after unsupervised fitting so that the two pure modes "
        "and the mixed mode remain readable. Raw KMeans IDs are preserved in the event-level export.",
        "",
        "## Leave-one-out knownness reference",
        "",
    ]

    quantile_lookup = {
        (row["metric"], float(row["quantile"])): row["value"]
        for row in result["loo_quantiles"]
    }
    quantile_rows = []
    for k in KNN_VALUES:
        metric = f"knownness_k{k}"
        quantile_rows.append(
            {
                "metric": metric,
                "p01": quantile_lookup[(metric, 0.01)],
                "p05": quantile_lookup[(metric, 0.05)],
                "p50": quantile_lookup[(metric, 0.50)],
                "p95": quantile_lookup[(metric, 0.95)],
                "p99": quantile_lookup[(metric, 0.99)],
            }
        )
    lines.extend(
        [
            _markdown_table(
                quantile_rows,
                (
                    ("metric", "Metric", "plain"),
                    ("p01", "p01", "float"),
                    ("p05", "p05", "float"),
                    ("p50", "p50", "float"),
                    ("p95", "p95", "float"),
                    ("p99", "p99", "float"),
                ),
            ),
            "",
            "### Class-conditional nearest-neighbour diagnostic",
            "",
            _markdown_table(
                [
                    {
                        **summary["manual_nearest_class_metrics"],
                        "logreg_agreement": summary["manual_nearest_class_vs_logreg_agreement"],
                    }
                ],
                (
                    ("evaluated_events", "Events", "int"),
                    ("accuracy", "Accuracy", "metric"),
                    ("balanced_accuracy", "Balanced accuracy", "metric"),
                    ("logging_recall", "Logging recall", "metric"),
                    ("wildfire_recall", "Wildfire recall", "metric"),
                    ("logreg_agreement", "Agreement with LogReg", "percent"),
                ),
            ),
            "",
            "## Reject diagnostics on the manual reference",
            "",
            "### AMBIGUOUS probability bands",
            "",
            _markdown_table(
                result["ambiguity_reject"],
                (
                    ("ambiguous_lower", "Lower", "float"),
                    ("ambiguous_upper", "Upper", "float"),
                    ("ambiguous_events", "Rejected", "int"),
                    ("coverage", "Coverage", "percent"),
                    ("accuracy", "Accuracy", "metric"),
                    ("balanced_accuracy", "Balanced accuracy", "metric"),
                ),
            ),
            "",
            "### Low-knownness reject",
            "",
            _markdown_table(
                result["knownness_reject"],
                (
                    ("reject_fraction_target", "Target tail", "percent"),
                    ("knownness_threshold", "Threshold", "float"),
                    ("rejected_events", "Rejected", "int"),
                    ("coverage", "Coverage", "percent"),
                    ("accuracy", "Accuracy", "metric"),
                    ("balanced_accuracy", "Balanced accuracy", "metric"),
                ),
            ),
            "",
            "### Combined UNKNOWN + AMBIGUOUS grid",
            "",
            _markdown_table(
                result["combined_reject"],
                (
                    ("unknown_quantile", "UNKNOWN tail", "percent"),
                    ("ambiguous_lower", "Ambiguous lower", "float"),
                    ("ambiguous_upper", "Ambiguous upper", "float"),
                    ("unknown_events", "UNKNOWN", "int"),
                    ("ambiguous_events", "AMBIGUOUS", "int"),
                    ("coverage", "Coverage", "percent"),
                    ("balanced_accuracy", "Balanced accuracy", "metric"),
                ),
            ),
            "",
            "## Manual-to-Hansen domain shift",
            "",
            _markdown_table(
                result["domain_shift"],
                (
                    ("metric", "Metric", "plain"),
                    ("manual_quantile", "Manual tail", "percent"),
                    ("manual_threshold", "Threshold", "float"),
                    ("hansen_below_events", "Hansen below", "int"),
                    ("hansen_below_share", "Hansen below share", "percent"),
                ),
            ),
            "",
            "## RemoteCLIP × XGBoost on Hansen events",
            "",
            _markdown_table(
                result["hansen_group_summary"],
                (
                    ("agreement_type", "Interaction", "plain"),
                    ("events", "Events", "int"),
                    ("knownness_k5_mean", "Knownness mean", "float"),
                    ("similarity_margin_mean", "Margin mean", "float"),
                    ("probability_wildfire_mean", "Mean P(Wildfire)", "float"),
                    ("unknown_share", "UNKNOWN share", "percent"),
                    ("ambiguous_share", "AMBIGUOUS share", "percent"),
                ),
            ),
            "",
            "### RemoteCLIP=Logging / XGBoost=Wildfire diagnosis",
            "",
            _markdown_table(
                [
                    {"diagnosis": key, "events": value}
                    for key, value in summary["rc_logging_xgb_wildfire_diagnosis"].items()
                ],
                (("diagnosis", "Diagnosis", "plain"), ("events", "Events", "int")),
            ),
            "",
            "## HDBSCAN noise",
            "",
            _markdown_table(
                result["hdbscan_noise_summary"],
                (
                    ("hdbscan_status", "Status", "plain"),
                    ("events", "Events", "int"),
                    ("knownness_k5_mean", "Knownness mean", "float"),
                    ("knownness_k5_median", "Knownness median", "float"),
                    ("logreg_error_rate", "LogReg error", "percent"),
                ),
            ),
            "",
            "HDBSCAN noise is not used as an UNKNOWN rule.",
            "",
            "## Stability",
            "",
            "### kNN rank stability",
            "",
            _markdown_table(
                result["knn_rank_stability"],
                (
                    ("left_k", "k left", "int"),
                    ("right_k", "k right", "int"),
                    ("spearman_correlation", "Spearman", "float"),
                    ("p_value", "p-value", "plain"),
                ),
            ),
            "",
            "### KMeans event-membership stability",
            "",
            _markdown_table(
                result["membership_stability_summary"],
                (
                    ("kmeans_cluster", "Cluster", "int"),
                    ("events", "Events", "int"),
                    ("membership_stability_mean", "Mean stability", "percent"),
                    ("membership_stability_median", "Median stability", "percent"),
                    ("membership_stability_p10", "p10 stability", "percent"),
                ),
            ),
            "",
            "## Generated artifacts",
            "",
            "- `REPORT.md` — this human-readable summary.",
            "- `latent_event_level_table.csv` — combined manual and Hansen event-level table.",
            "- `latent_manual_events.csv` and `latent_hansen_events.csv` — full source-specific diagnostics.",
            "- `latent_*_reject.csv`, `latent_domain_shift.csv`, and stability CSV files — detailed tables.",
            "- `plots/latent_*.png` — figures used by the notebook.",
            "",
            "## Interpretation limits",
            "",
            "- Manual LogReg probabilities are in-sample outputs of the frozen full-fit artifact; use grouped CV for unbiased classifier performance.",
            "- Canonical cluster IDs are descriptive labels assigned after unsupervised fitting.",
            "- The Hansen HDBSCAN field is inherited from the nearest manual event because this HDBSCAN implementation has no inductive prediction step.",
            "- UNKNOWN and AMBIGUOUS thresholds remain diagnostic and were not fitted on Hansen candidates.",
            "",
        ]
    )
    return "\n".join(lines)


def diagnostic_figures(result: dict[str, Any]) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    manual = result["manual_rows"]
    hansen = result["hansen_rows"]
    figures: dict[str, Any] = {}

    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    labels = ("Logging", "Wildfire")
    x = np.arange(3)
    bottom = np.zeros(3)
    for label in labels:
        values = np.asarray(
            [sum(row["kmeans_cluster"] == cluster and row["true_label"] == label for row in manual)
             for cluster in range(3)],
            dtype=float,
        )
        axes[0].bar(x, values, bottom=bottom, label=label)
        bottom += values
    axes[0].set(title="Canonical KMeans k=3 composition", xlabel="Cluster", ylabel="Events")
    axes[0].set_xticks(x)
    axes[0].legend()
    multi = [
        np.mean([row["is_multi_event_area"] for row in manual if row["kmeans_cluster"] == cluster])
        for cluster in range(3)
    ]
    axes[1].bar(x, multi, color="#6d58a5")
    axes[1].set(title="Multi-event share by cluster", xlabel="Cluster", ylabel="Share")
    axes[1].set_xticks(x)
    axes[1].set_ylim(0, 1)
    figures["cluster_metadata"] = figure

    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    groups = [
        ("L / mode 0", "Logging", 0),
        ("L / mixed 2", "Logging", 2),
        ("W / mode 1", "Wildfire", 1),
        ("W / mixed 2", "Wildfire", 2),
    ]
    probability_data = [
        [row["logreg_probability_wildfire"] for row in manual
         if row["true_label"] == label and row["kmeans_cluster"] == cluster]
        for _, label, cluster in groups
    ]
    knownness_data = [
        [row["knownness_k5"] for row in manual
         if row["true_label"] == label and row["kmeans_cluster"] == cluster]
        for _, label, cluster in groups
    ]
    axes[0].boxplot(probability_data, tick_labels=[group[0] for group in groups])
    axes[0].set(title="LogReg probability across latent modes", ylabel="P(Wildfire)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].boxplot(knownness_data, tick_labels=[group[0] for group in groups])
    axes[1].set(title="LOO knownness across latent modes", ylabel="Knownness k=5")
    axes[1].tick_params(axis="x", rotation=20)
    figures["latent_mode_comparison"] = figure

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    for axis, metric in zip(
        axes,
        ("sim_logging", "sim_wildfire", "similarity_margin"),
        strict=True,
    ):
        axis.scatter(
            [row[metric] for row in manual],
            [row["logreg_probability_wildfire"] for row in manual],
            c=[row["kmeans_cluster"] for row in manual],
            cmap="viridis",
            alpha=0.75,
        )
        axis.set(xlabel=metric, ylabel="P(Wildfire)", title=f"P(Wildfire) vs {metric}")
        axis.grid(alpha=0.2)
    figures["probability_relations"] = figure

    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    ambiguous = result["ambiguity_reject"]
    knownness = result["knownness_reject"]
    axes[0].plot(
        [row["coverage"] for row in ambiguous],
        [row["balanced_accuracy"] for row in ambiguous],
        marker="o",
        label="Ambiguous probability band",
    )
    axes[0].plot(
        [row["coverage"] for row in knownness],
        [row["balanced_accuracy"] for row in knownness],
        marker="s",
        label="Low-knownness reject",
    )
    axes[0].set(title="Coverage -> quality", xlabel="Coverage", ylabel="Balanced accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    combined = result["combined_reject"]
    scatter = axes[1].scatter(
        [row["coverage"] for row in combined],
        [row["balanced_accuracy"] for row in combined],
        c=[row["unknown_quantile"] for row in combined],
        s=[50 + 500 * (0.5 - row["ambiguous_lower"]) for row in combined],
        cmap="plasma",
    )
    axes[1].set(title="Combined UNKNOWN + AMBIGUOUS reject", xlabel="Coverage", ylabel="Balanced accuracy")
    axes[1].grid(alpha=0.2)
    figure.colorbar(scatter, ax=axes[1], label="UNKNOWN quantile")
    figures["reject_curves"] = figure

    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    manual_values = np.asarray([row["knownness_k5"] for row in manual])
    hansen_values = np.asarray([row["knownness_k5"] for row in hansen])
    axes[0].hist(manual_values, bins=25, density=True, alpha=0.6, label="Manual LOO")
    axes[0].hist(hansen_values, bins=25, density=True, alpha=0.6, label="Hansen")
    axes[0].axvline(np.quantile(manual_values, 0.05), linestyle="--", color="#a84343", label="Manual p05")
    axes[0].set(title="Knownness distribution", xlabel="Knownness k=5", ylabel="Density")
    axes[0].legend()
    for values, label in ((manual_values, "Manual LOO"), (hansen_values, "Hansen")):
        ordered = np.sort(values)
        axes[1].plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered), label=label)
    axes[1].set(title="Knownness ECDF", xlabel="Knownness k=5", ylabel="ECDF")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    figures["train_hansen_knownness"] = figure

    figure, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    groups = sorted({row["agreement_type"] for row in hansen})
    group_values = [[row["knownness_k5"] for row in hansen if row["agreement_type"] == group] for group in groups]
    axes[0].boxplot(group_values, tick_labels=groups)
    axes[0].set(title="Hansen knownness by model interaction", ylabel="Knownness k=5")
    axes[0].tick_params(axis="x", rotation=25)
    counts = Counter(row["agreement_type"] for row in hansen)
    axes[1].bar(groups, [counts[group] for group in groups], color="#4776a8")
    axes[1].set(title="RemoteCLIP / XGBoost interaction", ylabel="Events")
    axes[1].tick_params(axis="x", rotation=25)
    figures["hansen_model_interaction"] = figure

    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    statuses = ("clustered", "noise")
    data = [
        [row["knownness_k5"] for row in manual
         if ("noise" if row["hdbscan_is_noise"] else "clustered") == status]
        for status in statuses
    ]
    axes[0].boxplot(data, tick_labels=statuses)
    axes[0].set(title="Knownness by HDBSCAN status", ylabel="Knownness k=5")
    noise_summary = result["hdbscan_noise_summary"]
    axes[1].bar(
        [row["hdbscan_status"] for row in noise_summary],
        [row["logreg_error_rate"] for row in noise_summary],
        color=["#4b9b73", "#9aa0a6"],
    )
    axes[1].set(title="LogReg error by HDBSCAN status", ylabel="Error rate", ylim=(0, 1))
    figures["hdbscan_noise"] = figure

    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    rank_rows = result["knn_rank_stability"]
    axes[0].bar(
        [f"{row['left_k']} vs {row['right_k']}" for row in rank_rows],
        [row["spearman_correlation"] for row in rank_rows],
        color="#6d58a5",
    )
    axes[0].set(title="Knownness rank stability", ylabel="Spearman correlation", ylim=(0, 1))
    membership = result["membership_stability_summary"]
    axes[1].bar(
        [str(row["kmeans_cluster"]) for row in membership],
        [row["membership_stability_mean"] for row in membership],
        color="#4776a8",
    )
    axes[1].set(title="KMeans event-membership stability", xlabel="Canonical cluster", ylabel="Mean stability", ylim=(0, 1))
    figures["stability_diagnostics"] = figure

    return figures
