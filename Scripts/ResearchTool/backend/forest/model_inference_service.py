from __future__ import annotations

import csv
import hashlib
import json
import os
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import FOREST_CACHE_DIR, FOREST_EXPORTS_DIR
from .ml.spectral_features import (
    median_feature_rows,
    pre_post_features_from_stats,
    stats_for_tif,
)
from .model_labeling_service import build_model_labeling_plan
from .models import (
    DerivedPreview,
    EventLabel,
    LabelRun,
    Sample,
    SampleEvent,
    SentinelDownload,
    utc_now,
)
from .storage_paths import resolve_storage_path, to_storage_path


MODEL_NAMES = ("REMOTECLIP_LOGREG", "XGBOOST")
ProgressCallback = Callable[[str, str, int, int], None]
CancelCallback = Callable[[], bool]


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Cannot normalize a zero or non-finite embedding")
    return value / norm


@dataclass(frozen=True)
class InferenceEvent:
    event: SampleEvent
    sample: Sample
    downloads_by_period: dict[str, list[SentinelDownload]]
    previews: dict[str, DerivedPreview]
    sample_events: list[SampleEvent]


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]
    input_fingerprint: str
    metadata: dict[str, Any]


class ModelRunner(Protocol):
    source_name: str
    source_version: str

    def predict(self, item: InferenceEvent, should_cancel: CancelCallback | None = None) -> Prediction: ...


class ImageEmbedder(Protocol):
    fingerprint: str

    def encode(self, image_path: str | Path) -> np.ndarray: ...


class EmbeddingCache:
    def __init__(self, root: Path, embedder_fingerprint: str):
        self.root = root / embedder_fingerprint[:16]
        self.root.mkdir(parents=True, exist_ok=True)
        self.embedder_fingerprint = embedder_fingerprint

    def key_for(self, download_id: str, source_fingerprint: str) -> str:
        return stable_hash(
            {
                "download_id": download_id,
                "view": "rgb",
                "source_fingerprint": source_fingerprint,
                "embedder_fingerprint": self.embedder_fingerprint,
            }
        )

    def read(self, key: str) -> np.ndarray | None:
        path = self.root / key[:2] / f"{key}.npz"
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as payload:
                vector = np.asarray(payload["embedding"], dtype=np.float32)
                metadata = json.loads(str(payload["metadata"].item()))
            if (
                metadata.get("key") != key
                or metadata.get("embedder_fingerprint") != self.embedder_fingerprint
                or vector.ndim != 1
                or not np.isfinite(vector).all()
            ):
                return None
            return vector
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return None

    def write(self, key: str, vector: np.ndarray, metadata: dict[str, Any]) -> None:
        path = self.root / key[:2] / f"{key}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                embedding=np.asarray(vector, dtype=np.float32),
                metadata=np.array(
                    json.dumps(
                        {
                            **metadata,
                            "key": key,
                            "embedder_fingerprint": self.embedder_fingerprint,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                ),
            )
        os.replace(temporary, path)


class RuntimeRemoteClipEmbedder:
    def __init__(self, manifest: dict[str, Any]):
        embedding = manifest.get("embedding") or {}
        self.model_name = str(embedding.get("model_name") or "ViT-B-32")
        configured = os.getenv("REMOTECLIP_CHECKPOINT") or embedding.get("checkpoint_path")
        if not configured:
            raise ValueError("RemoteCLIP checkpoint is not configured in manifest or REMOTECLIP_CHECKPOINT")
        self.checkpoint_path = resolve_storage_path(str(configured))
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"RemoteCLIP checkpoint is missing: {self.checkpoint_path}")
        checkpoint_hash = file_sha256(self.checkpoint_path)
        expected_hash = embedding.get("checkpoint_sha256")
        if expected_hash and checkpoint_hash != expected_hash:
            raise ValueError("RemoteCLIP checkpoint SHA-256 does not match manifest")
        self.device_name = str(embedding.get("device") or "auto")
        self.fingerprint = stable_hash(
            {
                "model_name": self.model_name,
                "checkpoint_sha256": checkpoint_hash,
                "preprocessing_version": embedding.get("preprocessing_version"),
                "normalization_version": embedding.get("normalization_version"),
            }
        )
        self._model = None
        self._preprocess = None
        self._device = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "RemoteCLIP dependencies are missing; install ResearchScripts/ml_tests/requirements.txt"
            ) from exc
        if self.device_name == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.device_name)
        model, _, preprocess = open_clip.create_model_and_transforms(self.model_name)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        if isinstance(checkpoint, dict) and checkpoint:
            first_key = next(iter(checkpoint))
            if first_key.startswith("module."):
                checkpoint = {
                    key.removeprefix("module."): value for key, value in checkpoint.items()
                }
        model.load_state_dict(checkpoint)
        self._model = model.to(device).eval()
        self._preprocess = preprocess
        self._device = device

    def encode(self, image_path: str | Path) -> np.ndarray:
        self._load()
        import torch

        with Image.open(image_path) as image:
            tensor = self._preprocess(image.convert("RGB")).unsqueeze(0).to(self._device)
        with torch.no_grad():
            features = self._model.encode_image(tensor)
        return l2_normalize(features.detach().cpu().numpy()[0])


def remoteclip_feature(vectors: dict[str, np.ndarray], variant: str) -> np.ndarray:
    pre = vectors.get("PRE")
    post = vectors.get("POST")
    if variant == "PRE" and pre is not None:
        return pre
    if variant == "POST" and post is not None:
        return post
    if pre is None or post is None:
        raise ValueError(f"RemoteCLIP feature {variant} requires both PRE and POST")
    delta = post - pre
    variants = {
        "DELTA": delta,
        "PRE_POST": np.concatenate([pre, post]),
        "POST_DELTA": np.concatenate([post, delta]),
        "PRE_POST_DELTA": np.concatenate([pre, post, delta]),
    }
    if variant not in variants:
        raise ValueError(f"Unsupported RemoteCLIP feature variant: {variant}")
    return variants[variant]


class RemoteClipRunner:
    source_name = "REMOTECLIP_LOGREG"

    def __init__(
        self,
        artifact: dict[str, Any],
        *,
        embedder: ImageEmbedder | None = None,
        cache_root: Path | None = None,
    ):
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError("joblib is required for RemoteCLIP inference") from exc
        self.manifest = artifact["manifest"]
        self.payload = joblib.load(resolve_storage_path(artifact["artifact_path"]))
        self.source_version = str(self.manifest.get("producer_version") or artifact["producer_version"])
        self.feature_variant = str(
            self.payload.get("feature_variant") or self.manifest.get("feature_variant") or ""
        )
        self.class_order = list(
            self.payload.get("class_order") or self.manifest.get("class_order") or []
        )
        if self.class_order != ["Logging", "Wildfire"]:
            raise ValueError(f"Unsupported RemoteCLIP class order: {self.class_order}")
        self.threshold = float(
            self.payload.get("decision_threshold")
            or self.manifest.get("decision_threshold")
            or 0.5
        )
        self.embedder = embedder or RuntimeRemoteClipEmbedder(self.manifest)
        self.cache = EmbeddingCache(
            cache_root or FOREST_CACHE_DIR / "ml_embeddings",
            self.embedder.fingerprint,
        )

    def _embedding(self, download_id: str, preview: DerivedPreview) -> tuple[np.ndarray, dict[str, Any]]:
        path = resolve_storage_path(preview.png_path)
        source_hash = file_sha256(path)
        source_fingerprint = stable_hash({"size": path.stat().st_size, "sha256": source_hash})
        key = self.cache.key_for(download_id, source_fingerprint)
        cached = self.cache.read(key)
        if cached is not None:
            return cached, {"download_id": download_id, "sha256": source_hash, "cache": "hit"}
        vector = l2_normalize(self.embedder.encode(path))
        self.cache.write(
            key,
            vector,
            {
                "download_id": download_id,
                "view": "rgb",
                "source_path": to_storage_path(path) or str(path),
                "source_fingerprint": source_fingerprint,
            },
        )
        return vector, {"download_id": download_id, "sha256": source_hash, "cache": "computed"}

    def predict(self, item: InferenceEvent, should_cancel: CancelCallback | None = None) -> Prediction:
        vectors: dict[str, np.ndarray] = {}
        sources: list[dict[str, Any]] = []
        for period in ("PRE", "POST"):
            period_vectors = []
            for download in item.downloads_by_period.get(period, []):
                if should_cancel and should_cancel():
                    raise RuntimeError("Model labeling cancelled")
                preview = item.previews.get(download.download_id)
                if preview is None:
                    continue
                vector, source = self._embedding(download.download_id, preview)
                period_vectors.append(vector)
                sources.append({"period": period, **source})
            if period_vectors:
                vectors[period] = l2_normalize(np.stack(period_vectors).mean(axis=0))
        features = np.asarray(remoteclip_feature(vectors, self.feature_variant), dtype=np.float32)
        expected_dimension = int(
            self.payload.get("feature_dimension") or self.manifest.get("feature_dimension") or 0
        )
        if expected_dimension and features.size != expected_dimension:
            raise ValueError(
                f"RemoteCLIP feature dimension {features.size} does not match {expected_dimension}"
            )
        scaled = self.payload["scaler"].transform(features.reshape(1, -1))
        probabilities = np.asarray(self.payload["classifier"].predict_proba(scaled)[0], dtype=float)
        wildfire_probability = float(probabilities[1])
        predicted_index = 1 if wildfire_probability >= self.threshold else 0
        probability_map = {
            label: float(probabilities[index]) for index, label in enumerate(self.class_order)
        }
        fingerprint = stable_hash(
            {
                "event_id": item.event.event_id,
                "source_name": self.source_name,
                "source_version": self.source_version,
                "artifact_sha256": self.manifest.get("artifact_sha256"),
                "feature_variant": self.feature_variant,
                "embedder_fingerprint": self.embedder.fingerprint,
                "sources": sorted(sources, key=lambda row: (row["period"], row["download_id"])),
            }
        )
        label = self.class_order[predicted_index]
        return Prediction(
            label=label,
            confidence=probability_map[label],
            probabilities=probability_map,
            input_fingerprint=fingerprint,
            metadata={"feature_variant": self.feature_variant, "inputs": sources},
        )


def _sample_hansen_features(item: InferenceEvent) -> dict[str, float]:
    try:
        raw = json.loads(item.sample.raw_properties_json or "{}")
    except json.JSONDecodeError:
        raw = {}
    metrics = raw.get("candidate_metrics") or {}
    total_area = float(metrics.get("total_loss_area_ha") or 0.0)
    total_fraction = float(metrics.get("post_2016_loss_aoi_percent") or 0.0) / 100.0
    event_areas = [float(event.loss_area_ha or 0.0) for event in item.sample_events]
    dominant_share = max(event_areas, default=0.0) / total_area if total_area else 0.0
    max_year = max((event.event_year for event in item.sample_events), default=item.event.event_year)
    recent_area = sum(
        float(event.loss_area_ha or 0.0)
        for event in item.sample_events
        if event.event_year > max_year - 5
    )
    recent_fraction = total_fraction * recent_area / total_area if total_area else 0.0
    return {
        "hansen_loss_fraction": total_fraction,
        "hansen_recent_loss_5y": recent_fraction,
        "hansen_dominant_share": dominant_share,
    }


class XGBoostRunner:
    source_name = "XGBOOST"

    def __init__(self, artifact: dict[str, Any]):
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise RuntimeError("xgboost is required for XGBoost inference") from exc
        self.manifest = artifact["manifest"]
        self.source_version = str(self.manifest.get("producer_version") or artifact["producer_version"])
        self.feature_names = [str(value) for value in self.manifest.get("feature_names") or []]
        self.class_order = [str(value) for value in self.manifest.get("class_order") or []]
        if not self.feature_names:
            raise ValueError("XGBoost manifest has no feature_names")
        if self.class_order != ["Logging", "Wildfire"]:
            raise ValueError(f"Unsupported XGBoost class order: {self.class_order}")
        preprocessing = self.manifest.get("preprocessing") or {}
        self.fill_values = {
            str(key): float(value) for key, value in (preprocessing.get("fill_values") or {}).items()
        }
        missing_fill = [name for name in self.feature_names if name not in self.fill_values]
        if missing_fill:
            raise ValueError(f"XGBoost manifest has no fill values for: {missing_fill}")
        self.model = xgb.XGBClassifier()
        self.model.load_model(resolve_storage_path(artifact["artifact_path"]))
        self._sha_cache: dict[Path, str] = {}
        self._stats_cache: dict[Path, dict[str, float]] = {}

    def _file_hash(self, path: Path) -> str:
        if path not in self._sha_cache:
            self._sha_cache[path] = file_sha256(path)
        return self._sha_cache[path]

    def _spectral_stats(self, path: Path) -> dict[str, float]:
        if path not in self._stats_cache:
            self._stats_cache[path] = stats_for_tif(path)
        return self._stats_cache[path]

    def predict(self, item: InferenceEvent, should_cancel: CancelCallback | None = None) -> Prediction:
        if should_cancel and should_cancel():
            raise RuntimeError("Model labeling cancelled")
        sources = []
        paths_by_period: dict[str, list[Path]] = {"PRE": [], "POST": []}
        for period in paths_by_period:
            for download in item.downloads_by_period.get(period, []):
                path = resolve_storage_path(download.local_path)
                paths_by_period[period].append(path)
                sources.append(
                    {
                        "period": period,
                        "download_id": download.download_id,
                        "sha256": self._file_hash(path),
                    }
                )
        values = pre_post_features_from_stats(
            median_feature_rows(self._spectral_stats(path) for path in paths_by_period["PRE"]),
            median_feature_rows(self._spectral_stats(path) for path in paths_by_period["POST"]),
        )
        values.update(_sample_hansen_features(item))
        feature_values = [
            float(values.get(name, self.fill_values[name]))
            if np.isfinite(float(values.get(name, self.fill_values[name])))
            else self.fill_values[name]
            for name in self.feature_names
        ]
        probabilities = np.asarray(
            self.model.predict_proba(np.asarray(feature_values, dtype=np.float32).reshape(1, -1))[0],
            dtype=float,
        )
        predicted_index = int(np.argmax(probabilities))
        probability_map = {
            label: float(probabilities[index]) for index, label in enumerate(self.class_order)
        }
        fingerprint = stable_hash(
            {
                "event_id": item.event.event_id,
                "source_name": self.source_name,
                "source_version": self.source_version,
                "artifact_sha256": self.manifest.get("artifact_sha256"),
                "feature_names": self.feature_names,
                "feature_values": feature_values,
                "sources": sorted(sources, key=lambda row: (row["period"], row["download_id"])),
            }
        )
        label = self.class_order[predicted_index]
        return Prediction(
            label=label,
            confidence=probability_map[label],
            probabilities=probability_map,
            input_fingerprint=fingerprint,
            metadata={
                "feature_schema_version": self.manifest.get("feature_schema_version"),
                "feature_names": self.feature_names,
                "feature_values": feature_values,
                "inputs": sources,
            },
        )


RunnerFactory = Callable[[dict[str, Any]], ModelRunner]


def _event_inputs(
    db: Session,
    plan: dict[str, Any],
    selected_event_ids: set[str] | None,
) -> list[InferenceEvent]:
    rows = [
        row
        for row in plan["events"]
        if row["input_status"] == "READY"
        and (selected_event_ids is None or row["event_id"] in selected_event_ids)
    ]
    event_ids = [row["event_id"] for row in rows]
    events = {
        event.event_id: event
        for event in db.scalars(select(SampleEvent).where(SampleEvent.event_id.in_(event_ids)))
    } if event_ids else {}
    sample_ids = sorted({event.sample_id for event in events.values()})
    samples = {
        sample.sample_id: sample
        for sample in db.scalars(select(Sample).where(Sample.sample_id.in_(sample_ids)))
    } if sample_ids else {}
    sample_events: dict[str, list[SampleEvent]] = {}
    for event in db.scalars(select(SampleEvent).where(SampleEvent.sample_id.in_(sample_ids))):
        sample_events.setdefault(event.sample_id, []).append(event)
    download_ids = sorted(
        {
            download_id
            for row in rows
            for ids in row["usable_download_ids"].values()
            for download_id in ids
        }
    )
    downloads = {
        download.download_id: download
        for download in db.scalars(
            select(SentinelDownload).where(SentinelDownload.download_id.in_(download_ids))
        )
    } if download_ids else {}
    previews = {
        preview.download_id: preview
        for preview in db.scalars(
            select(DerivedPreview).where(
                DerivedPreview.download_id.in_(download_ids),
                DerivedPreview.view == "rgb",
            )
        )
    } if download_ids else {}
    result = []
    for row in rows:
        event = events[row["event_id"]]
        result.append(
            InferenceEvent(
                event=event,
                sample=samples[event.sample_id],
                downloads_by_period={
                    period: [downloads[item] for item in ids if item in downloads]
                    for period, ids in row["usable_download_ids"].items()
                },
                previews=previews,
                sample_events=sample_events.get(event.sample_id, []),
            )
        )
    return result


def _create_label_run(
    db: Session,
    source_name: str,
    artifact: dict[str, Any],
    mining_run_id: str,
) -> LabelRun:
    manifest = artifact.get("manifest") or {}
    run = LabelRun(
        label_run_id=uuid.uuid4().hex,
        producer_type="MODEL",
        producer_name=source_name,
        producer_version=str(manifest.get("producer_version") or artifact["producer_version"]),
        model_artifact_path=artifact.get("artifact_path"),
        model_artifact_sha256=artifact.get("artifact_sha256"),
        feature_schema_version=manifest.get("feature_schema_version"),
        config_json=json.dumps(
            {"mining_run_id": mining_run_id, "manifest": manifest},
            ensure_ascii=False,
            sort_keys=True,
        ),
        status="RUNNING",
    )
    db.add(run)
    db.commit()
    return run


def _write_report(directory: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "predictions.csv"
    json_path = directory / "summary.json"
    fieldnames = sorted({key for row in rows for key in row}) or ["status"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "artifact_dir": to_storage_path(directory) or str(directory),
        "operations_csv": to_storage_path(csv_path) or str(csv_path),
        "summary_json": to_storage_path(json_path) or str(json_path),
    }


def execute_model_labeling(
    db: Session,
    *,
    mining_run_id: str,
    remoteclip_version: str = "v1",
    xgboost_version: str = "v1",
    event_ids: list[str] | None = None,
    execution_id: str | None = None,
    progress: ProgressCallback | None = None,
    should_cancel: CancelCallback | None = None,
    runner_factories: dict[str, RunnerFactory] | None = None,
) -> dict[str, Any]:
    plan = build_model_labeling_plan(
        db,
        mining_run_id=mining_run_id,
        remoteclip_version=remoteclip_version,
        xgboost_version=xgboost_version,
    )
    selected = set(event_ids) if event_ids else None
    inputs = _event_inputs(db, plan, selected)
    ready_models = [name for name in MODEL_NAMES if plan["models"][name]["ready"]]
    if not inputs:
        raise ValueError("No events have usable PRE and POST inputs")
    if not ready_models:
        raise ValueError("No model artifacts are ready for inference")

    factories = runner_factories or {
        "REMOTECLIP_LOGREG": lambda artifact: RemoteClipRunner(artifact),
        "XGBOOST": lambda artifact: XGBoostRunner(artifact),
    }
    total = len(inputs) * len(ready_models)
    done = 0
    report_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    model_summaries: dict[str, Any] = {}
    cancelled = False

    for source_name in ready_models:
        artifact = plan["models"][source_name]
        label_run = _create_label_run(db, source_name, artifact, mining_run_id)
        model_errors = 0
        try:
            runner = factories[source_name](artifact)
        except Exception as exc:
            label_run.status = "FAILED"
            label_run.finished_at = utc_now()
            db.commit()
            counts["failed"] += len(inputs)
            model_summaries[source_name] = {"status": "FAILED", "error": str(exc)[:500]}
            for item in inputs:
                report_rows.append(
                    {
                        "event_id": item.event.event_id,
                        "sample_id": item.event.sample_id,
                        "source_name": source_name,
                        "status": "failed_model_load",
                        "error": str(exc)[:500],
                    }
                )
            done += len(inputs)
            if progress:
                progress("model_load_failed", f"{source_name}: {str(exc)[:160]}", done, total)
            continue

        for item in inputs:
            if should_cancel and should_cancel():
                cancelled = True
                break
            stage = "remoteclip_inference" if source_name == "REMOTECLIP_LOGREG" else "xgboost_inference"
            if progress:
                progress(
                    stage,
                    f"{source_name}: {item.event.sample_id[:8]} / {item.event.event_year}",
                    done,
                    total,
                )
            try:
                prediction = runner.predict(item, should_cancel)
                existing = db.scalar(
                    select(EventLabel).where(
                        EventLabel.event_id == item.event.event_id,
                        EventLabel.source_name == source_name,
                        EventLabel.source_version == runner.source_version,
                        EventLabel.input_fingerprint == prediction.input_fingerprint,
                    )
                )
                if existing is None:
                    label = EventLabel(
                        label_id=uuid.uuid4().hex,
                        sample_id=item.event.sample_id,
                        event_id=item.event.event_id,
                        label_run_id=label_run.label_run_id,
                        source_type="MODEL",
                        source_name=source_name,
                        source_version=runner.source_version,
                        label=prediction.label,
                        confidence=prediction.confidence,
                        probabilities_json=json.dumps(
                            prediction.probabilities,
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        input_fingerprint=prediction.input_fingerprint,
                        metadata_json=json.dumps(
                            prediction.metadata,
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    )
                    db.add(label)
                    db.commit()
                    action = "created"
                else:
                    action = "reused"
                counts[action] += 1
                report_rows.append(
                    {
                        "event_id": item.event.event_id,
                        "sample_id": item.event.sample_id,
                        "event_year": item.event.event_year,
                        "source_name": source_name,
                        "source_version": runner.source_version,
                        "status": action,
                        "label": prediction.label,
                        "confidence": round(prediction.confidence, 8),
                        "input_fingerprint": prediction.input_fingerprint,
                    }
                )
            except Exception as exc:
                db.rollback()
                model_errors += 1
                counts["failed"] += 1
                report_rows.append(
                    {
                        "event_id": item.event.event_id,
                        "sample_id": item.event.sample_id,
                        "event_year": item.event.event_year,
                        "source_name": source_name,
                        "source_version": runner.source_version,
                        "status": "failed",
                        "error": str(exc)[:500],
                    }
                )
            done += 1
            if progress:
                progress(stage, f"Processed {done}/{total} model-event inputs", done, total)

        current_run = db.get(LabelRun, label_run.label_run_id)
        if current_run is not None:
            current_run.status = "CANCELLED" if cancelled else "PARTIAL" if model_errors else "READY"
            current_run.finished_at = utc_now()
            db.commit()
        model_summaries[source_name] = {
            "status": "CANCELLED" if cancelled else "PARTIAL" if model_errors else "READY",
            "failed": model_errors,
            "label_run_id": label_run.label_run_id,
        }
        if cancelled:
            break

    summary = {
        "execution_id": execution_id,
        "mining_run_id": mining_run_id,
        "status": "CANCELLED" if cancelled else "COMPLETED_WITH_ERRORS" if counts["failed"] else "COMPLETED",
        "events_ready": len(inputs),
        "models_ready": ready_models,
        "created": counts["created"],
        "reused": counts["reused"],
        "failed": counts["failed"],
        "model_runs": model_summaries,
    }
    artifact_id = execution_id or uuid.uuid4().hex
    summary.update(
        _write_report(
            FOREST_EXPORTS_DIR / "model_labeling" / artifact_id,
            report_rows,
            summary,
        )
    )
    return summary
