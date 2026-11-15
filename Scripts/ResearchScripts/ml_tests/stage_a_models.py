from __future__ import annotations

import json
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable

import joblib
import numpy as np
import sklearn
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from ResearchTool.backend.config import FOREST_MODELS_DIR
from ResearchTool.backend.forest.ml.spectral_features import (
    median_feature_rows,
    missing_required_index_medians,
    pre_post_features_from_stats,
    stats_for_tif,
)
from ResearchTool.backend.forest.storage_paths import resolve_storage_path, to_storage_path

from .config import PROJECT_ROOT, stable_json_hash
from .experiments import fit_classifier


CLASS_ORDER = ("Logging", "Wildfire")
LABEL_TO_INT = {label: index for index, label in enumerate(CLASS_ORDER)}


def _portable_path(path: str | Path) -> str:
    resolved = Path(path).resolve(strict=False)
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def file_sha256(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_xgboost_event_records(
    event_rows: list[dict[str, Any]],
    *,
    feature_names: list[str],
    progress: Callable[[int, int], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stats_cache: dict[str, dict[str, float]] = {}
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, event in enumerate(event_rows, start=1):
        period_stats: dict[str, list[dict[str, float]]] = {"PRE": [], "POST": []}
        rejected: list[dict[str, Any]] = []
        for item in event.get("inputs", []):
            if not item.get("xgboost_usable"):
                continue
            path = resolve_storage_path(item.get("geotiff_path"))
            cache_key = str(path.resolve(strict=False))
            if cache_key not in stats_cache:
                stats_cache[cache_key] = stats_for_tif(path)
            stats = stats_cache[cache_key]
            missing = missing_required_index_medians(stats)
            if missing:
                rejected.append(
                    {
                        "download_id": item.get("download_id"),
                        "period": item.get("period"),
                        "missing_features": missing,
                    }
                )
                continue
            period_stats[str(item["period"])].append(stats)
        if not period_stats["PRE"] or not period_stats["POST"]:
            failures.append(
                {
                    "event_id": event["event_id"],
                    "sample_id": event["sample_id"],
                    "status": "NO_COMPLETE_SPECTRAL_PRE_OR_POST",
                    "pre_usable": len(period_stats["PRE"]),
                    "post_usable": len(period_stats["POST"]),
                    "rejected_inputs": rejected,
                }
            )
            if progress:
                progress(index, len(event_rows))
            continue
        values = pre_post_features_from_stats(
            median_feature_rows(period_stats["PRE"]),
            median_feature_rows(period_stats["POST"]),
        )
        for name in (
            "hansen_loss_fraction",
            "hansen_recent_loss_5y",
            "hansen_dominant_share",
        ):
            values[name] = float(event.get(name) or 0.0)
        records.append(
            {
                "record_id": str(event["event_id"]),
                "event_id": str(event["event_id"]),
                "sample_id": str(event["sample_id"]),
                "event_year": int(event["event_year"]),
                "label": str(event["human_label"]),
                "features": {name: values.get(name) for name in feature_names},
                "period_counts": {period: len(rows) for period, rows in period_stats.items()},
                "spectral_rejections": rejected,
            }
        )
        if progress:
            progress(index, len(event_rows))
    records.sort(key=lambda row: (row["sample_id"], row["event_year"], row["event_id"]))
    return records, {
        "events_considered": len(event_rows),
        "ready_events": len(records),
        "failed_events": len(failures),
        "unique_geotiffs_read": len(stats_cache),
        "failure_rows": failures,
    }


def restrict_remoteclip_records(
    event_records: list[dict[str, Any]],
    event_ids: set[str],
    *,
    feature_variant: str,
) -> list[dict[str, Any]]:
    from .experiments import records_for_variant

    records = records_for_variant(event_records, feature_variant)
    return [row for row in records if str(row["event_id"]) in event_ids]


def restrict_xgboost_records(
    records: list[dict[str, Any]],
    event_ids: set[str],
) -> list[dict[str, Any]]:
    return [row for row in records if str(row["event_id"]) in event_ids]


def _metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    y_pred = (probabilities >= 0.5).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "logging_precision": float(precision[0]),
        "logging_recall": float(recall[0]),
        "logging_f1": float(f1[0]),
        "logging_support": int(support[0]),
        "wildfire_precision": float(precision[1]),
        "wildfire_recall": float(recall[1]),
        "wildfire_f1": float(f1[1]),
        "wildfire_support": int(support[1]),
        "tn": int(matrix[0, 0]),
        "fp": int(matrix[0, 1]),
        "fn": int(matrix[1, 0]),
        "tp": int(matrix[1, 1]),
    }


def _summary_rows(fold_rows: list[dict[str, Any]], *, model_name: str) -> list[dict[str, Any]]:
    metric_names = (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "logging_precision",
        "logging_recall",
        "logging_f1",
        "wildfire_precision",
        "wildfire_recall",
        "wildfire_f1",
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in fold_rows:
        if row.get("status") == "ok":
            grouped[str(row["cohort_mode"])].append(row)
    summaries: list[dict[str, Any]] = []
    for mode, rows in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "model": model_name,
            "cohort_mode": mode,
            "fold_runs": len(rows),
            "failed_fold_runs": sum(
                item.get("cohort_mode") == mode and item.get("status") != "ok"
                for item in fold_rows
            ),
        }
        for metric in metric_names:
            values = [float(row[metric]) for row in rows]
            summary[f"{metric}_mean"] = mean(values) if values else None
            summary[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
        for key in ("tn", "fp", "fn", "tp"):
            summary[f"{key}_total"] = sum(int(row[key]) for row in rows)
        summaries.append(summary)
    return summaries


def _records_by_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["record_id"]): row for row in records}


def _split_records(
    records: dict[str, dict[str, Any]],
    split: dict[str, Any],
    old_event_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train_ids = [str(value) for value in split["train_record_ids"]]
    test_ids = [str(value) for value in split["test_record_ids"]]
    full_train = [records[value] for value in train_ids if value in records]
    old_train = [records[value] for value in train_ids if value in records and value in old_event_ids]
    test = [records[value] for value in test_ids if value in records]
    return old_train, full_train, test


def _remoteclip_fold(
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    classifier_config: dict[str, Any],
) -> dict[str, Any]:
    scaler, classifier = fit_classifier(train, classifier_config)
    matrix = np.stack([np.asarray(row["features"], dtype=np.float32) for row in test])
    probabilities = classifier.predict_proba(scaler.transform(matrix))[:, 1]
    y_true = np.asarray([LABEL_TO_INT[str(row["label"])] for row in test], dtype=int)
    return _metrics(y_true, probabilities)


def _xgboost_matrix(
    rows: list[dict[str, Any]],
    feature_names: list[str],
    fill_values: dict[str, float],
) -> np.ndarray:
    matrix = []
    for row in rows:
        values = []
        for name in feature_names:
            raw = row["features"].get(name)
            value = float(raw) if raw is not None else fill_values[name]
            values.append(value if np.isfinite(value) else fill_values[name])
        matrix.append(values)
    return np.asarray(matrix, dtype=np.float32)


def _xgboost_fill_values(
    rows: list[dict[str, Any]],
    feature_names: list[str],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in feature_names:
        values = [
            float(row["features"][name])
            for row in rows
            if row["features"].get(name) is not None
            and np.isfinite(float(row["features"][name]))
        ]
        result[name] = float(np.median(values)) if values else 0.0
    return result


def _xgboost_fold(
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    feature_names: list[str],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    fill_values = _xgboost_fill_values(train, feature_names)
    x_train = _xgboost_matrix(train, feature_names, fill_values)
    y_train = np.asarray([LABEL_TO_INT[str(row["label"])] for row in train], dtype=int)
    x_test = _xgboost_matrix(test, feature_names, fill_values)
    y_true = np.asarray([LABEL_TO_INT[str(row["label"])] for row in test], dtype=int)
    model = xgb.XGBClassifier(**parameters)
    model.fit(x_train, y_train)
    probabilities = np.asarray(model.predict_proba(x_test)[:, 1], dtype=float)
    return _metrics(y_true, probabilities)


def run_cohort_size_experiment(
    records: list[dict[str, Any]],
    *,
    old_event_ids: set[str],
    saved_splits: dict[str, Any],
    model_name: str,
    remoteclip_classifier_config: dict[str, Any] | None = None,
    xgboost_feature_names: list[str] | None = None,
    xgboost_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    by_id = _records_by_id(records)
    fold_rows: list[dict[str, Any]] = []
    for split in saved_splits.get("rows", []):
        if split.get("strategy") != "stratified_group":
            continue
        old_train, full_train, test = _split_records(by_id, split, old_event_ids)
        test_ids = {str(row["event_id"]) for row in test}
        if test_ids & {str(row["event_id"]) for row in old_train}:
            raise AssertionError("OLD_COHORT_CV train/test event leakage detected")
        for mode, train in (("OLD_COHORT_CV", old_train), ("FULL_COHORT_CV", full_train)):
            base = {
                "model": model_name,
                "cohort_mode": mode,
                "seed": int(split["seed"]),
                "fold": int(split["fold"]),
                "train_events": len(train),
                "train_samples": len({row["sample_id"] for row in train}),
                "train_logging": sum(row["label"] == "Logging" for row in train),
                "train_wildfire": sum(row["label"] == "Wildfire" for row in train),
                "test_events": len(test),
                "test_samples": len({row["sample_id"] for row in test}),
                "test_logging": sum(row["label"] == "Logging" for row in test),
                "test_wildfire": sum(row["label"] == "Wildfire" for row in test),
            }
            try:
                if len({row["label"] for row in train}) < 2:
                    raise ValueError("training fold contains fewer than two classes")
                if len({row["label"] for row in test}) < 2:
                    raise ValueError("test fold contains fewer than two classes")
                if model_name == "REMOTECLIP_LOGREG":
                    metrics = _remoteclip_fold(
                        train,
                        test,
                        dict(remoteclip_classifier_config or {}),
                    )
                elif model_name == "XGBOOST":
                    metrics = _xgboost_fold(
                        train,
                        test,
                        list(xgboost_feature_names or []),
                        dict(xgboost_parameters or {}),
                    )
                else:
                    raise ValueError(f"Unsupported model: {model_name}")
                fold_rows.append({**base, "status": "ok", **metrics})
            except Exception as exc:
                fold_rows.append({**base, "status": "failed", "error": str(exc)[:500]})
    failed = [row for row in fold_rows if row["status"] != "ok"]
    if failed:
        raise RuntimeError(
            f"{model_name} cohort-size CV has {len(failed)} failed fold runs; "
            f"first error: {failed[0].get('error')}"
        )
    return {
        "fold_metrics": fold_rows,
        "summary_metrics": _summary_rows(fold_rows, model_name=model_name),
    }


def export_xgboost_v2(
    records: list[dict[str, Any]],
    *,
    version: str,
    feature_names: list[str],
    parameters: dict[str, Any],
    feature_schema_version: str,
    training_manifest_hash: str,
    split_manifest_hash: str,
    config_hash: str,
    database_fingerprint: dict[str, Any],
    metrics: dict[str, Any],
    overwrite: bool = False,
) -> dict[str, Any]:
    model_dir = FOREST_MODELS_DIR / "xgboost" / version
    model_path = model_dir / "model.json"
    metrics_path = model_dir / "metrics.json"
    manifest_path = model_dir / "manifest.json"
    existing = [path.name for path in (model_path, metrics_path, manifest_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"XGBoost {version} already contains {existing}")
    model_dir.mkdir(parents=True, exist_ok=True)
    fill_values = _xgboost_fill_values(records, feature_names)
    matrix = _xgboost_matrix(records, feature_names, fill_values)
    labels = np.asarray([LABEL_TO_INT[str(row["label"])] for row in records], dtype=int)
    model = xgb.XGBClassifier(**parameters)
    model.fit(matrix, labels)
    training_ids = sorted(str(row["event_id"]) for row in records)
    class_counts = dict(sorted(Counter(str(row["label"]) for row in records).items()))

    with tempfile.TemporaryDirectory(prefix=".xgboost-v2-", dir=model_dir) as temp_dir:
        staging = Path(temp_dir)
        staged_model = staging / "model.json"
        staged_metrics = staging / "metrics.json"
        staged_manifest = staging / "manifest.json"
        model.save_model(staged_model)
        reloaded = xgb.XGBClassifier()
        reloaded.load_model(staged_model)
        smoke = np.asarray(reloaded.predict_proba(matrix[:1])[0], dtype=float)
        if smoke.shape != (2,) or not np.isfinite(smoke).all():
            raise RuntimeError("Reloaded XGBoost v2 artifact failed smoke prediction")
        model_hash = file_sha256(staged_model)
        metrics_payload = {
            **metrics,
            "training_records": len(records),
            "class_counts": class_counts,
            "smoke_probability_wildfire": float(smoke[1]),
        }
        manifest = {
            "artifact_schema_version": "xgboost_event_classifier_v2",
            "producer_name": "XGBOOST",
            "producer_version": version,
            "recipe_version": "legacy_recipe_event_level_v2",
            "artifact_path": _portable_path(model_path),
            "artifact_sha256": model_hash,
            "metrics_path": _portable_path(metrics_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "feature_schema_version": feature_schema_version,
            "feature_names": feature_names,
            "class_order": list(CLASS_ORDER),
            "preprocessing": {"fill_values": fill_values},
            "hyperparameters": parameters,
            "training_cohort": {
                "cohort_id": "FULL_COHORT",
                "records": len(records),
                "samples": len({row["sample_id"] for row in records}),
                "class_counts": class_counts,
                "record_ids_sha256": stable_json_hash(training_ids),
                "manifest_hash": training_manifest_hash,
            },
            "split_manifest_hash": split_manifest_hash,
            "experiment_config_hash": config_hash,
            "training_database_fingerprint": database_fingerprint,
            "software": {"xgboost": xgb.__version__, "numpy": np.__version__},
            "round_trip_smoke_test": {
                "status": "passed",
                "probability_wildfire": float(smoke[1]),
            },
        }
        staged_metrics.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        staged_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(staged_model, model_path)
        os.replace(staged_metrics, metrics_path)
        os.replace(staged_manifest, manifest_path)
    return manifest


def export_remoteclip_v2(
    records: list[dict[str, Any]],
    *,
    version: str,
    feature_variant: str,
    classifier_config: dict[str, Any],
    checkpoint_path: Path,
    embedding_config: dict[str, Any],
    training_manifest_hash: str,
    split_manifest_hash: str,
    config_hash: str,
    database_fingerprint: dict[str, Any],
    metrics: dict[str, Any],
    overwrite: bool = False,
) -> dict[str, Any]:
    model_dir = FOREST_MODELS_DIR / "remoteclip_logreg" / version
    artifact_path = model_dir / "classifier.joblib"
    metrics_path = model_dir / "metrics.json"
    manifest_path = model_dir / "manifest.json"
    existing = [path.name for path in (artifact_path, metrics_path, manifest_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"RemoteCLIP-LogReg {version} already contains {existing}")
    model_dir.mkdir(parents=True, exist_ok=True)
    scaler, classifier = fit_classifier(records, classifier_config)
    training_ids = sorted(str(row["event_id"]) for row in records)
    class_counts = dict(sorted(Counter(str(row["label"]) for row in records).items()))
    feature_dimension = int(np.asarray(records[0]["features"]).size)
    payload = {
        "artifact_schema_version": "remoteclip_logreg_artifact_v1",
        "producer_name": "REMOTECLIP_LOGREG",
        "producer_version": version,
        "feature_schema_version": "remoteclip_event_embedding_v1",
        "feature_variant": feature_variant,
        "feature_dimension": feature_dimension,
        "class_order": list(CLASS_ORDER),
        "decision_threshold": 0.5,
        "scaler": scaler,
        "classifier": classifier,
    }
    with tempfile.TemporaryDirectory(prefix=".remoteclip-v2-", dir=model_dir) as temp_dir:
        staging = Path(temp_dir)
        staged_artifact = staging / "classifier.joblib"
        staged_metrics = staging / "metrics.json"
        staged_manifest = staging / "manifest.json"
        joblib.dump(payload, staged_artifact, compress=3)
        reloaded = joblib.load(staged_artifact)
        smoke_features = np.asarray(records[0]["features"], dtype=np.float32).reshape(1, -1)
        smoke_probability = float(
            reloaded["classifier"].predict_proba(
                reloaded["scaler"].transform(smoke_features)
            )[0, 1]
        )
        if not np.isfinite(smoke_probability):
            raise RuntimeError("Reloaded RemoteCLIP v2 artifact failed smoke prediction")
        artifact_hash = file_sha256(staged_artifact)
        metrics_payload = {
            **metrics,
            "training_records": len(records),
            "class_counts": class_counts,
            "smoke_probability_wildfire": smoke_probability,
        }
        manifest = {
            "artifact_schema_version": "remoteclip_logreg_artifact_v1",
            "producer_name": "REMOTECLIP_LOGREG",
            "producer_version": version,
            "recipe_version": "legacy_recipe_event_level_v2",
            "artifact_path": _portable_path(artifact_path),
            "artifact_sha256": artifact_hash,
            "metrics_path": _portable_path(metrics_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "training_database_fingerprint": database_fingerprint,
            "training_cohort": {
                "cohort_id": "FULL_COHORT",
                "records": len(records),
                "samples": len({row["sample_id"] for row in records}),
                "class_counts": class_counts,
                "record_ids_sha256": stable_json_hash(training_ids),
                "manifest_hash": training_manifest_hash,
            },
            "class_order": list(CLASS_ORDER),
            "feature_schema_version": "remoteclip_event_embedding_v1",
            "feature_variant": feature_variant,
            "feature_dimension": feature_dimension,
            "required_periods": ["PRE", "POST"],
            "decision_threshold": 0.5,
            "embedding": {
                "model_name": embedding_config.get("model_name"),
                "preprocessing_version": embedding_config.get("preprocessing_version"),
                "normalization_version": embedding_config.get("normalization_version"),
                "checkpoint_path": _portable_path(checkpoint_path),
                "checkpoint_sha256": file_sha256(checkpoint_path),
            },
            "classifier_config": classifier_config,
            "split_manifest_hash": split_manifest_hash,
            "experiment_config_hash": config_hash,
            "software": {"numpy": np.__version__, "scikit_learn": sklearn.__version__},
            "round_trip_smoke_test": {
                "status": "passed",
                "probability_wildfire": smoke_probability,
            },
        }
        staged_metrics.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        staged_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(staged_artifact, artifact_path)
        os.replace(staged_metrics, metrics_path)
        os.replace(staged_manifest, manifest_path)
    return manifest


def predict_remoteclip_artifact(manifest: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    payload = joblib.load(resolve_storage_path(manifest["artifact_path"]))
    features = np.asarray(record["features"], dtype=np.float32).reshape(1, -1)
    probabilities = np.asarray(
        payload["classifier"].predict_proba(payload["scaler"].transform(features))[0],
        dtype=float,
    )
    index = int(probabilities[1] >= float(payload.get("decision_threshold") or 0.5))
    return {
        "prediction": CLASS_ORDER[index],
        "probability_logging": float(probabilities[0]),
        "probability_wildfire": float(probabilities[1]),
    }


def predict_xgboost_artifact(manifest: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    feature_names = [str(value) for value in manifest["feature_names"]]
    fill_values = {
        str(key): float(value)
        for key, value in (manifest.get("preprocessing") or {}).get("fill_values", {}).items()
    }
    matrix = _xgboost_matrix([record], feature_names, fill_values)
    model = xgb.XGBClassifier()
    model.load_model(resolve_storage_path(manifest["artifact_path"]))
    probabilities = np.asarray(model.predict_proba(matrix)[0], dtype=float)
    index = int(np.argmax(probabilities))
    return {
        "prediction": CLASS_ORDER[index],
        "probability_logging": float(probabilities[0]),
        "probability_wildfire": float(probabilities[1]),
    }

