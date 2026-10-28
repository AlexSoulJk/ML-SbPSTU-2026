from __future__ import annotations

import hashlib
import json
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn

from ResearchTool.backend.config import FOREST_MODELS_DIR

from .config import ExperimentConfig, PROJECT_ROOT, stable_json_hash
from .experiments import INT_TO_LABEL, fit_classifier, records_for_variant
from .reporting import json_value


PRODUCER_NAME = "REMOTECLIP_LOGREG"
ARTIFACT_SCHEMA_VERSION = "remoteclip_logreg_artifact_v1"
FEATURE_SCHEMA_VERSION = "remoteclip_event_embedding_v1"
SUPPORTED_TRAINING_MODES = {
    "MANUAL_ONLY",
    "MANUAL_PLUS_WEAK",
    "WEAK_ONLY_TO_MANUAL",
}
REQUIRED_PERIODS_BY_VARIANT = {
    "PRE": ["PRE"],
    "POST": ["POST"],
    "DELTA": ["PRE", "POST"],
    "PRE_POST": ["PRE", "POST"],
    "POST_DELTA": ["PRE", "POST"],
    "PRE_POST_DELTA": ["PRE", "POST"],
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: str | Path) -> str:
    resolved = Path(path).resolve(strict=False)
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _safe_version(version: str) -> str:
    value = version.strip()
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"Invalid RemoteCLIP artifact version: {version}")
    return value


def _metric_number(row: dict[str, Any], key: str) -> float:
    try:
        value = float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("-inf")
    return value if np.isfinite(value) else float("-inf")


def select_grouped_cv_metric(
    baseline_metrics: list[dict[str, Any]],
    *,
    training_mode: str = "MANUAL_ONLY",
    feature_variant: str | None = None,
) -> dict[str, Any]:
    if training_mode not in SUPPORTED_TRAINING_MODES:
        raise ValueError(
            f"Unsupported production training mode {training_mode}. "
            f"Expected one of {sorted(SUPPORTED_TRAINING_MODES)}."
        )
    candidates = [
        row
        for row in baseline_metrics
        if row.get("mode") == training_mode
        and row.get("split_strategy") == "stratified_group"
        and (feature_variant is None or row.get("feature_variant") == feature_variant)
        and _metric_number(row, "balanced_accuracy_mean") != float("-inf")
    ]
    if not candidates:
        requested = feature_variant or "the available variants"
        raise ValueError(
            f"No grouped-CV metric is available for {training_mode} and {requested}."
        )
    selected = max(
        candidates,
        key=lambda row: (
            _metric_number(row, "balanced_accuracy_mean"),
            _metric_number(row, "roc_auc_mean"),
            _metric_number(row, "f1_mean"),
            str(row.get("feature_variant") or ""),
        ),
    )
    return dict(selected)


def plan_remoteclip_export(
    baseline_metrics: list[dict[str, Any]],
    *,
    version: str = "v1",
    training_mode: str = "MANUAL_ONLY",
    feature_variant: str | None = None,
) -> dict[str, Any]:
    safe_version = _safe_version(version)
    selected = select_grouped_cv_metric(
        baseline_metrics,
        training_mode=training_mode,
        feature_variant=feature_variant,
    )
    model_dir = FOREST_MODELS_DIR / "remoteclip_logreg" / safe_version
    return {
        "producer_name": PRODUCER_NAME,
        "producer_version": safe_version,
        "training_mode": training_mode,
        "feature_variant": selected["feature_variant"],
        "selection_metric": "balanced_accuracy_mean",
        "selected_grouped_cv_metrics": selected,
        "model_dir": _portable_path(model_dir),
        "artifact_path": _portable_path(model_dir / "classifier.joblib"),
        "manifest_path": _portable_path(model_dir / "manifest.json"),
        "metrics_path": _portable_path(model_dir / "metrics.json"),
    }


def _training_records(
    event_records: list[dict[str, Any]],
    *,
    feature_variant: str,
    training_mode: str,
) -> list[dict[str, Any]]:
    records = records_for_variant(event_records, feature_variant)
    manual = [row for row in records if row["validation_cohort"] == "MANUAL_VALID"]
    weak = [row for row in records if row["validation_cohort"] == "NO_MANUAL_WEAK"]
    if training_mode == "MANUAL_ONLY":
        return manual
    if training_mode == "MANUAL_PLUS_WEAK":
        return manual + weak
    if training_mode == "WEAK_ONLY_TO_MANUAL":
        return weak
    raise ValueError(f"Unsupported production training mode: {training_mode}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(json_value(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def export_remoteclip_classifier(
    event_records: list[dict[str, Any]],
    baseline_metrics: list[dict[str, Any]],
    config: ExperimentConfig,
    database_fingerprint: dict[str, Any],
    *,
    version: str = "v1",
    training_mode: str = "MANUAL_ONLY",
    feature_variant: str | None = None,
    decision_threshold: float = 0.5,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not 0.0 < decision_threshold < 1.0:
        raise ValueError("decision_threshold must be between 0 and 1.")
    plan = plan_remoteclip_export(
        baseline_metrics,
        version=version,
        training_mode=training_mode,
        feature_variant=feature_variant,
    )
    selected_metric = plan["selected_grouped_cv_metrics"]
    selected_variant = str(plan["feature_variant"])
    if selected_variant not in REQUIRED_PERIODS_BY_VARIANT:
        raise ValueError(f"Unsupported RemoteCLIP feature variant: {selected_variant}")
    records = _training_records(
        event_records,
        feature_variant=selected_variant,
        training_mode=training_mode,
    )
    if not records:
        raise ValueError("The selected RemoteCLIP production cohort is empty.")
    label_counts = Counter(str(row["label"]) for row in records)
    if len(label_counts) < 2:
        raise ValueError(f"The selected production cohort has one class: {dict(label_counts)}")
    dimensions = {int(np.asarray(row["features"]).shape[0]) for row in records}
    if len(dimensions) != 1:
        raise ValueError(f"Inconsistent RemoteCLIP feature dimensions: {sorted(dimensions)}")

    checkpoint_path = config.checkpoint_path().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"RemoteCLIP checkpoint not found: {checkpoint_path}")
    scaler, classifier = fit_classifier(records, dict(config.data.get("classifier", {})))
    class_order = [INT_TO_LABEL[int(value)] for value in classifier.classes_.tolist()]
    model_dir = FOREST_MODELS_DIR / "remoteclip_logreg" / str(plan["producer_version"])
    artifact_path = model_dir / "classifier.joblib"
    manifest_path = model_dir / "manifest.json"
    metrics_path = model_dir / "metrics.json"
    existing = [path.name for path in (artifact_path, manifest_path, metrics_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"RemoteCLIP artifact version {plan['producer_version']} already contains {existing}. "
            "Choose another version or explicitly enable overwrite."
        )
    model_dir.mkdir(parents=True, exist_ok=True)

    training_record_ids = sorted(str(row["record_id"]) for row in records)
    training_summary = {
        "records": len(records),
        "samples": len({str(row["sample_id"]) for row in records}),
        "class_counts": dict(sorted(label_counts.items())),
        "record_ids_sha256": stable_json_hash(training_record_ids),
    }
    classifier_payload = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "producer_name": PRODUCER_NAME,
        "producer_version": plan["producer_version"],
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_variant": selected_variant,
        "feature_dimension": next(iter(dimensions)),
        "class_order": class_order,
        "decision_threshold": float(decision_threshold),
        "scaler": scaler,
        "classifier": classifier,
    }
    grouped_rows = [
        dict(row)
        for row in baseline_metrics
        if row.get("feature_variant") == selected_variant
        and row.get("mode") == training_mode
        and row.get("split_strategy") == "stratified_group"
    ]

    with tempfile.TemporaryDirectory(prefix=".remoteclip-export-", dir=model_dir) as temp_dir:
        staging = Path(temp_dir)
        staged_artifact = staging / "classifier.joblib"
        staged_metrics = staging / "metrics.json"
        staged_manifest = staging / "manifest.json"
        joblib.dump(classifier_payload, staged_artifact, compress=3)
        artifact_sha256 = file_sha256(staged_artifact)

        reloaded = joblib.load(staged_artifact)
        smoke_features = np.asarray(records[0]["features"], dtype=np.float32).reshape(1, -1)
        smoke_probability = float(
            reloaded["classifier"].predict_proba(
                reloaded["scaler"].transform(smoke_features)
            )[0, 1]
        )
        if not np.isfinite(smoke_probability):
            raise RuntimeError("Reloaded RemoteCLIP artifact returned a non-finite probability.")

        metrics_payload = {
            "producer_name": PRODUCER_NAME,
            "producer_version": plan["producer_version"],
            "selected_grouped_cv_metrics": selected_metric,
            "grouped_cv_metrics": grouped_rows,
            "training": training_summary,
            "smoke_probability_wildfire": smoke_probability,
        }
        manifest_payload = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "producer_name": PRODUCER_NAME,
            "producer_version": plan["producer_version"],
            "artifact_path": _portable_path(artifact_path),
            "artifact_sha256": artifact_sha256,
            "metrics_path": _portable_path(metrics_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "training_database_fingerprint": database_fingerprint,
            "training_cohort": {
                "mode": training_mode,
                **training_summary,
            },
            "class_order": class_order,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_variant": selected_variant,
            "feature_dimension": next(iter(dimensions)),
            "required_periods": REQUIRED_PERIODS_BY_VARIANT[selected_variant],
            "decision_threshold": float(decision_threshold),
            "selection": {
                "strategy": "best grouped-CV balanced_accuracy within configured training mode",
                "metrics": selected_metric,
            },
            "embedding": {
                "model_name": config.data.get("embedding", {}).get("model_name"),
                "preprocessing_version": config.data.get("embedding", {}).get(
                    "preprocessing_version"
                ),
                "normalization_version": config.data.get("embedding", {}).get(
                    "normalization_version"
                ),
                "checkpoint_path": _portable_path(checkpoint_path),
                "checkpoint_sha256": file_sha256(checkpoint_path),
            },
            "classifier_config": dict(config.data.get("classifier", {})),
            "experiment_config_fingerprint": config.fingerprint,
            "software": {
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "round_trip_smoke_test": {
                "status": "passed",
                "probability_wildfire": smoke_probability,
            },
        }
        _write_json(staged_metrics, metrics_payload)
        _write_json(staged_manifest, manifest_payload)
        staged_artifact.replace(artifact_path)
        staged_metrics.replace(metrics_path)
        staged_manifest.replace(manifest_path)

    return {
        **plan,
        "artifact_sha256": artifact_sha256,
        "training": training_summary,
        "class_order": class_order,
        "feature_dimension": next(iter(dimensions)),
        "round_trip_smoke_test": "passed",
        "smoke_probability_wildfire": smoke_probability,
    }
