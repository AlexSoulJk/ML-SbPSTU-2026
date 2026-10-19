from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import DEFAULT_CONFIG_PATH, load_config
from .embeddings import RemoteClipEmbedder, l2_normalize


_default_embedder: RemoteClipEmbedder | None = None


def get_embedding(image_path: str | Path) -> np.ndarray:
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = RemoteClipEmbedder(load_config(DEFAULT_CONFIG_PATH))
    return _default_embedder.encode(image_path)


def embedding_consistency_loo(embeddings: list[np.ndarray]) -> np.ndarray:
    values = np.stack(embeddings)
    result: list[float] = []
    for index in range(len(values)):
        if len(values) == 1:
            result.append(np.nan)
            continue
        center = l2_normalize(np.delete(values, index, axis=0).mean(axis=0))
        result.append(float(np.dot(values[index], center)))
    return np.asarray(result)


def aggregate_embeddings_by_sample_and_period(
    embeddings_for_period: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    aggregate = l2_normalize(np.stack(embeddings_for_period).mean(axis=0))
    return aggregate, embedding_consistency_loo(embeddings_for_period)


def calculate_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    return float(np.dot(l2_normalize(embedding1), l2_normalize(embedding2)))


def get_butch_embeddings(
    images_info_grouped: dict[str, dict[str, list[dict[str, Any]]]],
):
    embeddings_pre: list[np.ndarray] = []
    embeddings_post: list[np.ndarray] = []
    labels_pre: list[str] = []
    labels_post: list[str] = []
    statistic_info: dict[str, Any] = {"cosine_similarity_at_one_point": {}}
    for sample_id, periods in images_info_grouped.items():
        statistic_info["cosine_similarity_at_one_point"][sample_id] = {}
        for period, images in periods.items():
            period_embeddings = [get_embedding(image["path"]) for image in images]
            aggregate, similarities = aggregate_embeddings_by_sample_and_period(period_embeddings)
            if period == "PRE":
                embeddings_pre.append(aggregate)
                labels_pre.append(images[0]["label"])
            elif period == "POST":
                embeddings_post.append(aggregate)
                labels_post.append(images[0]["label"])
            statistic_info["cosine_similarity_at_one_point"][sample_id][period] = (
                similarities.tolist(),
                [image["image_id"] for image in images],
            )
    return embeddings_pre, labels_pre, embeddings_post, labels_post, statistic_info
