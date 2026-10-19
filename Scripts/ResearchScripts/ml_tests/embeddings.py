from __future__ import annotations

import json
import os
import uuid
from collections import Counter
from pathlib import Path
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
from PIL import Image

from .config import ExperimentConfig, stable_json_hash
from .dataset import file_sha256, resolve_manifest_path
from .init_model import load_remoteclip_model


class ImageEmbedder(Protocol):
    fingerprint: str

    def encode(self, image_path: str | Path) -> np.ndarray: ...


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    if norm == 0:
        raise ValueError("Cannot normalize a zero embedding.")
    return value / norm


class RemoteClipEmbedder:
    def __init__(self, config: ExperimentConfig):
        settings = config.data.get("embedding", {})
        self.model_name = str(settings.get("model_name") or "ViT-B-32")
        self.checkpoint_path = config.checkpoint_path().resolve()
        self.device_name = str(settings.get("device") or "auto")
        self.preprocessing_version = str(
            settings.get("preprocessing_version") or "open_clip_default_v1"
        )
        self.normalization_version = str(settings.get("normalization_version") or "l2_v1")
        self.checkpoint_sha256 = file_sha256(self.checkpoint_path)
        self.fingerprint = stable_json_hash(
            {
                "model_name": self.model_name,
                "checkpoint_sha256": self.checkpoint_sha256,
                "preprocessing_version": self.preprocessing_version,
                "normalization_version": self.normalization_version,
            }
        )
        self._model = None
        self._preprocess = None
        self._device = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        self._model, self._preprocess, self._device = load_remoteclip_model(
            model_name=self.model_name,
            checkpoint_path=self.checkpoint_path,
            device=self.device_name,
        )

    def encode(self, image_path: str | Path) -> np.ndarray:
        self._ensure_loaded()
        import torch

        with Image.open(image_path) as image:
            tensor = self._preprocess(image.convert("RGB")).unsqueeze(0).to(self._device)
        with torch.no_grad():
            features = self._model.encode_image(tensor)
        return l2_normalize(features.detach().cpu().numpy()[0])


class EmbeddingCache:
    POLICIES = {"read_write", "read_only", "refresh", "off"}

    def __init__(
        self,
        root: str | Path,
        *,
        embedder_fingerprint: str,
        policy: str = "read_write",
    ):
        if policy not in self.POLICIES:
            raise ValueError(f"Unsupported embedding cache policy: {policy}")
        self.root = Path(root)
        self.embedder_fingerprint = embedder_fingerprint
        self.policy = policy
        self.namespace = self.root / embedder_fingerprint[:16]
        if policy in {"read_write", "refresh"}:
            self.namespace.mkdir(parents=True, exist_ok=True)

    def source_fingerprint(self, image_path: str | Path) -> str:
        path = Path(image_path)
        stat = path.stat()
        return stable_json_hash({"size": stat.st_size, "sha256": file_sha256(path)})

    def key_for(self, row: dict[str, Any], source_fingerprint: str) -> str:
        return stable_json_hash(
            {
                "download_id": row["download_id"],
                "view": row["view"],
                "source_fingerprint": source_fingerprint,
                "embedder_fingerprint": self.embedder_fingerprint,
            }
        )

    def path_for(self, key: str) -> Path:
        return self.namespace / key[:2] / f"{key}.npz"

    def read(self, key: str) -> np.ndarray | None:
        if self.policy in {"refresh", "off"}:
            return None
        path = self.path_for(key)
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as payload:
                embedding = np.asarray(payload["embedding"], dtype=np.float32)
                metadata = json.loads(str(payload["metadata"].item()))
            if metadata.get("key") != key:
                return None
            if metadata.get("embedder_fingerprint") != self.embedder_fingerprint:
                return None
            if embedding.ndim != 1 or not np.isfinite(embedding).all():
                return None
            return embedding
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return None

    def write(self, key: str, embedding: np.ndarray, metadata: dict[str, Any]) -> Path | None:
        if self.policy in {"read_only", "off"}:
            return None
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        payload = {
            **metadata,
            "key": key,
            "embedder_fingerprint": self.embedder_fingerprint,
        }
        with temp.open("wb") as handle:
            np.savez_compressed(
                handle,
                embedding=np.asarray(embedding, dtype=np.float32),
                metadata=np.array(json.dumps(payload, ensure_ascii=False, sort_keys=True)),
            )
        os.replace(temp, path)
        return path


def compute_manifest_embeddings(
    manifest: list[dict[str, Any]],
    config: ExperimentConfig,
    *,
    embedder: ImageEmbedder | None = None,
    progress: Callable[[int, int, dict[str, int]], None] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    embedder = embedder or RemoteClipEmbedder(config)
    settings = config.data.get("embedding", {})
    policy = str(settings.get("cache_policy") or "read_write")
    cache = EmbeddingCache(
        config.path_value("embedding_cache"),
        embedder_fingerprint=embedder.fingerprint,
        policy=policy,
    )
    embeddings: dict[str, np.ndarray] = {}
    report_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    candidates = {
        f"{row['download_id']}:{row['view']}"
        for row in manifest
        if row.get("eligible")
    }
    total = len(candidates)

    for row in manifest:
        if not row.get("eligible"):
            continue
        image_id = f"{row['download_id']}:{row['view']}"
        if image_id in seen:
            continue
        seen.add(image_id)
        source = resolve_manifest_path(row["png_path"])
        if not source.exists():
            counts["failed"] += 1
            report_rows.append({"image_id": image_id, "status": "failed", "error": "missing_file"})
            if progress:
                progress(len(seen), total, dict(counts))
            continue
        try:
            source_fingerprint = cache.source_fingerprint(source)
            key = cache.key_for(row, source_fingerprint)
            cached = cache.read(key)
            if cached is not None:
                embeddings[image_id] = cached
                counts["hits"] += 1
                report_rows.append({"image_id": image_id, "status": "hit", "cache_key": key})
                if progress:
                    progress(len(seen), total, dict(counts))
                continue
            counts["misses"] += 1
            if policy == "read_only":
                report_rows.append(
                    {"image_id": image_id, "status": "missing_read_only", "cache_key": key}
                )
                if progress:
                    progress(len(seen), total, dict(counts))
                continue
            embedding = l2_normalize(embedder.encode(source))
            embeddings[image_id] = embedding
            cache.write(
                key,
                embedding,
                {
                    "download_id": row["download_id"],
                    "view": row["view"],
                    "source_path": row["png_path"],
                    "source_fingerprint": source_fingerprint,
                },
            )
            counts["computed"] += 1
            report_rows.append({"image_id": image_id, "status": "computed", "cache_key": key})
        except Exception as exc:
            counts["failed"] += 1
            report_rows.append(
                {"image_id": image_id, "status": "failed", "error": str(exc)[:500]}
            )
        if progress:
            progress(len(seen), total, dict(counts))

    return embeddings, {
        "policy": policy,
        "cache_path": str(cache.namespace),
        "embedder_fingerprint": embedder.fingerprint,
        "counts": dict(counts),
        "rows": report_rows,
    }

