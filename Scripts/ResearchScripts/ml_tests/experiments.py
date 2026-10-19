from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean, pstdev
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from .config import ExperimentConfig
from .embeddings import l2_normalize
from .splits import assert_no_group_leakage, build_saved_splits


LABEL_TO_INT = {"Logging": 0, "Wildfire": 1}
INT_TO_LABEL = {value: key for key, value in LABEL_TO_INT.items()}


def aggregate_event_embeddings(
    manifest: list[dict[str, Any]],
    embeddings: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in manifest:
        if not row.get("eligible") or row.get("period") not in {"PRE", "POST"}:
            continue
        image_id = f"{row['download_id']}:{row['view']}"
        embedding = embeddings.get(image_id)
        if embedding is None:
            continue
        key = (str(row["sample_id"]), str(row["event_id"]))
        record = grouped.setdefault(
            key,
            {
                "record_id": str(row["event_id"]),
                "sample_id": str(row["sample_id"]),
                "event_id": str(row["event_id"]),
                "event_year": int(row["event_year"]),
                "event_loss_area_ha": row.get("event_loss_area_ha"),
                "event_loss_share": row.get("event_loss_share"),
                "event_is_dominant": bool(row.get("event_is_dominant")),
                "is_multi_event_area": bool(row.get("is_multi_event_area")),
                "label": str(row["label"]),
                "validation_cohort": str(row["validation_cohort"]),
                "period_embeddings": defaultdict(list),
                "period_image_ids": defaultdict(list),
                "period_seasons": defaultdict(list),
            },
        )
        period = str(row["period"])
        record["period_embeddings"][period].append(l2_normalize(embedding))
        record["period_image_ids"][period].append(image_id)
        record["period_seasons"][period].append(row.get("season") or "")

    records: list[dict[str, Any]] = []
    for record in grouped.values():
        vectors: dict[str, np.ndarray] = {}
        image_counts: dict[str, int] = {}
        for period, values in record.pop("period_embeddings").items():
            vectors[period] = l2_normalize(np.stack(values).mean(axis=0))
            image_counts[period] = len(values)
        record["vectors"] = vectors
        record["image_counts"] = image_counts
        record["period_image_ids"] = dict(record["period_image_ids"])
        record["period_seasons"] = dict(record["period_seasons"])
        records.append(record)
    return sorted(records, key=lambda row: (row["sample_id"], row["event_year"], row["event_id"]))


def feature_vector(record: dict[str, Any], variant: str) -> np.ndarray | None:
    vectors = record["vectors"]
    pre = vectors.get("PRE")
    post = vectors.get("POST")
    if variant == "PRE":
        return pre
    if variant == "POST":
        return post
    if pre is None or post is None:
        return None
    delta = post - pre
    if variant == "DELTA":
        return delta
    if variant == "PRE_POST":
        return np.concatenate([pre, post])
    if variant == "POST_DELTA":
        return np.concatenate([post, delta])
    if variant == "PRE_POST_DELTA":
        return np.concatenate([pre, post, delta])
    raise ValueError(f"Unknown feature variant: {variant}")


def records_for_variant(
    event_records: list[dict[str, Any]],
    variant: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source in event_records:
        vector = feature_vector(source, variant)
        if vector is None:
            continue
        records.append({**source, "features": np.asarray(vector, dtype=np.float32)})

    event_counts: Counter[str] = Counter(record["sample_id"] for record in records)
    for record in records:
        record["sample_weight"] = 1.0 / event_counts[record["sample_id"]]
    return records


def _fit_model(
    train_records: list[dict[str, Any]],
    classifier_config: dict[str, Any],
) -> tuple[StandardScaler, LogisticRegression]:
    if not train_records:
        raise ValueError("Training set is empty.")
    labels = [record["label"] for record in train_records]
    if len(set(labels)) < 2:
        raise ValueError(f"Training set contains one class: {Counter(labels)}")
    x_train = np.stack([record["features"] for record in train_records])
    y_train = np.asarray([LABEL_TO_INT[label] for label in labels])
    weights = np.asarray([float(record["sample_weight"]) for record in train_records])

    scaler = StandardScaler()
    try:
        scaler.fit(x_train, sample_weight=weights)
    except TypeError:
        scaler.fit(x_train)
    x_scaled = scaler.transform(x_train)
    model = LogisticRegression(**classifier_config)
    model.fit(x_scaled, y_train, sample_weight=weights)
    return scaler, model


def _evaluate_model(
    scaler: StandardScaler,
    model: LogisticRegression,
    test_records: list[dict[str, Any]],
) -> dict[str, Any]:
    if not test_records:
        raise ValueError("Test set is empty.")
    x_test = np.stack([record["features"] for record in test_records])
    y_true = np.asarray([LABEL_TO_INT[record["label"]] for record in test_records])
    probabilities = model.predict_proba(scaler.transform(x_test))[:, 1]
    y_pred = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    auc = float("nan") if len(set(y_true.tolist())) < 2 else float(roc_auc_score(y_true, probabilities))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": auc,
        "logging_precision": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "logging_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "wildfire_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "wildfire_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _run_fold(
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    classifier_config: dict[str, Any],
) -> dict[str, Any]:
    try:
        scaler, model = _fit_model(train_records, classifier_config)
        return {
            "status": "ok",
            **_evaluate_model(scaler, model, test_records),
            "train_events": len(train_records),
            "train_groups": len({record["sample_id"] for record in train_records}),
            "test_events": len(test_records),
            "test_groups": len({record["sample_id"] for record in test_records}),
            "train_logging": sum(record["label"] == "Logging" for record in train_records),
            "train_wildfire": sum(record["label"] == "Wildfire" for record in train_records),
            "test_logging": sum(record["label"] == "Logging" for record in test_records),
            "test_wildfire": sum(record["label"] == "Wildfire" for record in test_records),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "error": str(exc)[:500],
            "train_events": len(train_records),
            "test_events": len(test_records),
        }


def _summaries(fold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = ["accuracy", "balanced_accuracy", "f1", "roc_auc"]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in fold_rows:
        if row.get("status") == "ok":
            grouped[(row["feature_variant"], row["mode"], row["split_strategy"])].append(row)
    summaries: list[dict[str, Any]] = []
    for (variant, mode, strategy), rows in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "feature_variant": variant,
            "mode": mode,
            "split_strategy": strategy,
            "metric_semantics": (
                "weak_label_agreement"
                if mode == "WEAK_INTERNAL_DIAGNOSTIC"
                else "wri_label_on_event_confirmed_manual_cohort"
            ),
            "fold_runs": len(rows),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in rows if np.isfinite(float(row[metric]))]
            summary[f"{metric}_mean"] = mean(values) if values else ""
            summary[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0 if values else ""
        summaries.append(summary)
    return summaries


def run_baseline_experiments(
    event_records: list[dict[str, Any]],
    config: ExperimentConfig,
) -> dict[str, Any]:
    feature_variants = list(config.data.get("experiments", {}).get("feature_variants", []))
    classifier_config = dict(config.data.get("classifier", {}))
    profile = config.profile
    fold_rows: list[dict[str, Any]] = []
    saved_splits: dict[str, Any] = {}

    for variant in feature_variants:
        records = records_for_variant(event_records, variant)
        manual = [record for record in records if record["validation_cohort"] == "MANUAL_VALID"]
        weak = [record for record in records if record["validation_cohort"] == "NO_MANUAL_WEAK"]
        if len({record["label"] for record in manual}) < 2:
            saved_splits[variant] = {"status": "insufficient_manual_data"}
            continue
        splits = build_saved_splits(
            manual,
            requested_n_splits=int(profile["n_splits"]),
            seeds=[int(seed) for seed in profile["seeds"]],
            include_ordinary=True,
        )
        assert_no_group_leakage(splits)
        saved_splits[variant] = splits
        manual_by_id = {record["record_id"]: record for record in manual}

        for split in splits["rows"]:
            train_manual = [
                manual_by_id[record_id]
                for record_id in split["train_record_ids"]
                if record_id in manual_by_id
            ]
            test_manual = [
                manual_by_id[record_id]
                for record_id in split["test_record_ids"]
                if record_id in manual_by_id
            ]
            modes = [("MANUAL_ONLY", train_manual)]
            if split["strategy"] == "stratified_group":
                modes.extend(
                    [
                        ("MANUAL_PLUS_WEAK", train_manual + weak),
                        ("WEAK_ONLY_TO_MANUAL", weak),
                    ]
                )
            for mode, train_records in modes:
                result = _run_fold(train_records, test_manual, classifier_config)
                fold_rows.append(
                    {
                        "feature_variant": variant,
                        "mode": mode,
                        "metric_semantics": "wri_label_on_event_confirmed_manual_cohort",
                        "split_strategy": split["strategy"],
                        "seed": split["seed"],
                        "fold": split["fold"],
                        **result,
                    }
                )

        if len({record["label"] for record in weak}) >= 2:
            weak_splits = build_saved_splits(
                weak,
                requested_n_splits=int(profile["n_splits"]),
                seeds=[int(seed) for seed in profile["seeds"]],
                include_ordinary=False,
            )
            assert_no_group_leakage(weak_splits)
            saved_splits[f"{variant}:WEAK_INTERNAL_DIAGNOSTIC"] = weak_splits
            weak_by_id = {record["record_id"]: record for record in weak}
            for split in weak_splits["rows"]:
                train = [weak_by_id[item] for item in split["train_record_ids"]]
                test = [weak_by_id[item] for item in split["test_record_ids"]]
                fold_rows.append(
                    {
                        "feature_variant": variant,
                        "mode": "WEAK_INTERNAL_DIAGNOSTIC",
                        "metric_semantics": "weak_label_agreement",
                        "split_strategy": split["strategy"],
                        "seed": split["seed"],
                        "fold": split["fold"],
                        **_run_fold(train, test, classifier_config),
                    }
                )

    return {
        "fold_metrics": fold_rows,
        "baseline_metrics": _summaries(fold_rows),
        "saved_splits": saved_splits,
    }


def build_nomanual_review_queue(
    event_records: list[dict[str, Any]],
    config: ExperimentConfig,
) -> list[dict[str, Any]]:
    variant = str(
        config.data.get("experiments", {}).get("nomanual_review_feature") or "POST"
    )
    records = records_for_variant(event_records, variant)
    manual = [record for record in records if record["validation_cohort"] == "MANUAL_VALID"]
    weak = [record for record in records if record["validation_cohort"] == "NO_MANUAL_WEAK"]
    if not weak:
        return []
    scaler, model = _fit_model(manual, dict(config.data.get("classifier", {})))
    x_weak = np.stack([record["features"] for record in weak])
    probabilities = model.predict_proba(scaler.transform(x_weak))[:, 1]
    rows: list[dict[str, Any]] = []
    for record, probability in zip(weak, probabilities, strict=True):
        predicted = INT_TO_LABEL[int(probability >= 0.5)]
        weak_label = record["label"]
        conflict = float(probability if weak_label == "Logging" else 1.0 - probability)
        rows.append(
            {
                "sample_id": record["sample_id"],
                "event_id": record["event_id"],
                "event_year": record["event_year"],
                "weak_label": weak_label,
                "predicted_class": predicted,
                "probability_wildfire": float(probability),
                "prediction_margin": float(abs(probability - 0.5) * 2.0),
                "agreement_with_weak_label": predicted == weak_label,
                "label_conflict_score": conflict,
                "review_priority": conflict,
                "feature_variant": variant,
                "model_source": "MANUAL_VALID_FULL",
                "is_multi_event_area": record["is_multi_event_area"],
                "event_loss_area_ha": record["event_loss_area_ha"],
            }
        )
    return sorted(rows, key=lambda row: (-row["review_priority"], row["sample_id"], row["event_id"]))

