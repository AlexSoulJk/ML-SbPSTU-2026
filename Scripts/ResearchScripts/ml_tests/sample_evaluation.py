from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Any, Iterable

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

from .experiments import aggregate_event_embeddings


def _finite_float(value: float | np.floating | None) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def normalize_rows(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2 or not len(values):
        raise ValueError("Expected a non-empty 2D embedding matrix.")
    if not np.isfinite(values).all():
        raise ValueError("Embedding matrix contains NaN or infinite values.")
    before = np.linalg.norm(values, axis=1)
    if np.any(before <= 0):
        raise ValueError("Embedding matrix contains zero vectors.")
    normalized = values / before[:, None]
    after = np.linalg.norm(normalized, axis=1)
    return normalized.astype(np.float32), before, after


def valid_post_event_matrix(
    manifest: list[dict[str, Any]],
    embeddings: dict[str, np.ndarray],
) -> dict[str, Any]:
    event_records = aggregate_event_embeddings(manifest, embeddings)
    records = [
        record
        for record in event_records
        if record.get("validation_cohort") == "MANUAL_VALID"
        and record.get("vectors", {}).get("POST") is not None
    ]
    if len(records) < 3:
        raise ValueError(
            "At least three MANUAL_VALID events with POST embeddings are required; "
            f"found {len(records)}."
        )
    matrix, norms_before, norms_after = normalize_rows(
        np.stack([record["vectors"]["POST"] for record in records])
    )
    metadata = [
        {
            "record_index": index,
            "sample_id": record["sample_id"],
            "event_id": record["event_id"],
            "event_year": record["event_year"],
            "label": record["label"],
            "validation_cohort": record["validation_cohort"],
            "post_image_count": int(record.get("image_counts", {}).get("POST", 0)),
            "event_loss_area_ha": record.get("event_loss_area_ha"),
            "event_loss_share": record.get("event_loss_share"),
            "event_is_dominant": bool(record.get("event_is_dominant")),
            "is_multi_event_area": bool(record.get("is_multi_event_area")),
        }
        for index, record in enumerate(records)
    ]
    image_counts = [row["post_image_count"] for row in metadata]
    return {
        "records": records,
        "metadata": metadata,
        "matrix": matrix,
        "norms_before": norms_before,
        "norms_after": norms_after,
        "summary": {
            "validation_cohort": "MANUAL_VALID",
            "period": "POST",
            "events": len(records),
            "samples": len({record["sample_id"] for record in records}),
            "embedding_dimensions": int(matrix.shape[1]),
            "post_images": int(sum(image_counts)),
            "post_images_per_event_min": int(min(image_counts)),
            "post_images_per_event_median": float(np.median(image_counts)),
            "post_images_per_event_max": int(max(image_counts)),
            "class_counts": dict(sorted(Counter(record["label"] for record in records).items())),
            "l2_norm_before_min": float(norms_before.min()),
            "l2_norm_before_max": float(norms_before.max()),
            "l2_norm_after_min": float(norms_after.min()),
            "l2_norm_after_max": float(norms_after.max()),
        },
    }


def fit_pca(
    matrix: np.ndarray,
    *,
    variance_levels: Iterable[float] = (0.90, 0.95),
    selected_variance: float = 0.95,
) -> dict[str, Any]:
    if not 0.0 < selected_variance <= 1.0:
        raise ValueError("selected_variance must be in (0, 1].")
    model = PCA(svd_solver="full")
    full_coordinates = model.fit_transform(matrix)
    cumulative = np.cumsum(model.explained_variance_ratio_)

    def component_count(level: float) -> int:
        return min(len(cumulative), int(np.searchsorted(cumulative, level, side="left") + 1))

    levels = sorted({float(level) for level in variance_levels} | {float(selected_variance)})
    rows = [
        {
            "target_variance": level,
            "components": component_count(level),
            "explained_variance": float(cumulative[component_count(level) - 1]),
        }
        for level in levels
    ]
    selected_components = component_count(selected_variance)
    coordinates = full_coordinates[:, :selected_components].astype(np.float32)
    plot_coordinates = np.zeros((len(matrix), 2), dtype=np.float32)
    plot_coordinates[:, : min(2, full_coordinates.shape[1])] = full_coordinates[:, :2]
    return {
        "model": model,
        "coordinates": coordinates,
        "plot_coordinates": plot_coordinates,
        "explained_variance_ratio": model.explained_variance_ratio_,
        "cumulative_variance": cumulative,
        "rows": rows,
        "selected_variance": selected_variance,
        "selected_components": selected_components,
    }


def _valid_k_values(sample_count: int, k_min: int, k_max: int) -> list[int]:
    return list(range(max(2, k_min), min(k_max, sample_count - 1) + 1))


def evaluate_kmeans(
    coordinates: np.ndarray,
    *,
    k_min: int = 2,
    k_max: int = 10,
    random_state: int = 42,
    n_init: int = 30,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    labels_by_k: dict[int, np.ndarray] = {}
    models_by_k: dict[int, KMeans] = {}
    for k in _valid_k_values(len(coordinates), k_min, k_max):
        model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = model.fit_predict(coordinates)
        labels_by_k[k] = labels
        models_by_k[k] = model
        rows.append(
            {
                "k": k,
                "silhouette": float(silhouette_score(coordinates, labels)),
                "davies_bouldin": float(davies_bouldin_score(coordinates, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(coordinates, labels)),
                "inertia": float(model.inertia_),
            }
        )
    if not rows:
        raise ValueError("The valid k range is empty for this sample count.")
    return {"rows": rows, "labels_by_k": labels_by_k, "models_by_k": models_by_k}


def evaluate_cluster_stability(
    coordinates: np.ndarray,
    reference_labels: dict[int, np.ndarray],
    *,
    runs: int = 20,
    sample_fraction: float = 0.80,
    random_state: int = 42,
    n_init: int = 20,
) -> dict[str, Any]:
    if not 0.0 < sample_fraction <= 1.0:
        raise ValueError("sample_fraction must be in (0, 1].")
    generator = np.random.default_rng(random_state)
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    sample_count = len(coordinates)
    for k, reference in sorted(reference_labels.items()):
        subset_size = min(sample_count, max(k + 1, int(round(sample_count * sample_fraction))))
        scores: list[float] = []
        for run_index in range(runs):
            indices = generator.choice(sample_count, size=subset_size, replace=False)
            model = KMeans(
                n_clusters=k,
                random_state=random_state + run_index + k * 1000,
                n_init=n_init,
            ).fit(coordinates[indices])
            predicted = model.predict(coordinates)
            score = float(adjusted_rand_score(reference, predicted))
            scores.append(score)
            details.append({"k": k, "run": run_index, "adjusted_rand_index": score})
        rows.append(
            {
                "k": k,
                "runs": runs,
                "sample_fraction": sample_fraction,
                "subset_size": subset_size,
                "ari_mean": float(np.mean(scores)),
                "ari_std": float(np.std(scores)),
                "ari_min": float(np.min(scores)),
                "ari_max": float(np.max(scores)),
            }
        )
    return {"rows": rows, "details": details}


def evaluate_gmm(
    coordinates: np.ndarray,
    *,
    k_min: int = 1,
    k_max: int = 10,
    random_state: int = 42,
    covariance_type: str = "full",
    n_init: int = 5,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    labels_by_k: dict[int, np.ndarray] = {}
    upper = min(k_max, len(coordinates) - 1)
    for k in range(max(1, k_min), upper + 1):
        model = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            random_state=random_state,
            n_init=n_init,
            reg_covar=1e-6,
        ).fit(coordinates)
        labels_by_k[k] = model.predict(coordinates)
        rows.append(
            {
                "k": k,
                "bic": float(model.bic(coordinates)),
                "aic": float(model.aic(coordinates)),
                "converged": bool(model.converged_),
                "iterations": int(model.n_iter_),
            }
        )
    best = min(rows, key=lambda row: row["bic"])
    return {"rows": rows, "labels_by_k": labels_by_k, "best_k": int(best["k"])}


def _hdbscan_grid(sample_count: int) -> list[tuple[int, int | None]]:
    sizes = {
        max(3, int(round(sample_count * fraction)))
        for fraction in (0.05, 0.08, 0.10, 0.15, 0.20)
    }
    sizes.update(size for size in (3, 5, 10, 15, 20) if size < sample_count)
    sizes = {min(sample_count - 1, size) for size in sizes if size < sample_count}
    grid: list[tuple[int, int | None]] = []
    for size in sorted(sizes):
        candidates = {None, 1, 2, 3, 5, min(10, size)}
        for min_samples in sorted(candidates, key=lambda value: -1 if value is None else value):
            if min_samples is None or min_samples <= size:
                grid.append((size, min_samples))
    return grid


def evaluate_hdbscan(
    coordinates: np.ndarray,
    *,
    parameter_grid: Iterable[tuple[int, int | None]] | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    labels_by_key: dict[str, np.ndarray] = {}
    for min_cluster_size, min_samples in parameter_grid or _hdbscan_grid(len(coordinates)):
        model = HDBSCAN(
            min_cluster_size=int(min_cluster_size),
            min_samples=min_samples,
            metric="euclidean",
            allow_single_cluster=False,
            copy=True,
        )
        labels = model.fit_predict(coordinates)
        key = f"mcs={min_cluster_size};ms={min_samples}"
        labels_by_key[key] = labels
        clustered = labels >= 0
        cluster_count = len(set(labels[clustered].tolist()))
        coverage = float(clustered.mean())
        silhouette = None
        if cluster_count >= 2 and int(clustered.sum()) > cluster_count:
            silhouette = float(silhouette_score(coordinates[clustered], labels[clustered]))
        membership = getattr(model, "probabilities_", np.ones(len(labels), dtype=float))
        membership_mean = float(np.mean(membership[clustered])) if clustered.any() else 0.0
        rows.append(
            {
                "key": key,
                "min_cluster_size": int(min_cluster_size),
                "min_samples": min_samples,
                "clusters": cluster_count,
                "noise_points": int((~clustered).sum()),
                "noise_fraction": float((~clustered).mean()),
                "coverage": coverage,
                "silhouette_clustered": silhouette,
                "membership_mean": membership_mean,
                "selection_score": (silhouette * coverage) if silhouette is not None else -1.0,
            }
        )
    if not rows:
        raise ValueError("HDBSCAN parameter grid is empty.")
    selected = max(
        rows,
        key=lambda row: (
            row["selection_score"],
            row["coverage"],
            row["clusters"],
            -row["min_cluster_size"],
        ),
    )
    return {
        "rows": rows,
        "selected": selected,
        "labels": labels_by_key[selected["key"]],
    }


def _metric_winner(rows: list[dict[str, Any]], key: str, *, maximize: bool) -> int:
    valid = [row for row in rows if row.get(key) is not None]
    selected = (max if maximize else min)(valid, key=lambda row: row[key])
    return int(selected["k"])


def cluster_composition(
    metadata: list[dict[str, Any]],
    labels: np.ndarray,
    *,
    method: str,
) -> list[dict[str, Any]]:
    counts = Counter((int(cluster), row["label"]) for cluster, row in zip(labels, metadata, strict=True))
    totals = Counter(int(cluster) for cluster in labels)
    return [
        {
            "method": method,
            "cluster": cluster,
            "cluster_name": "noise" if cluster == -1 else str(cluster),
            "label": label,
            "events": count,
            "cluster_events": totals[cluster],
            "cluster_share": count / totals[cluster],
        }
        for (cluster, label), count in sorted(counts.items())
    ]


def run_valid_post_clustering(
    manifest: list[dict[str, Any]],
    embeddings: dict[str, np.ndarray],
    *,
    pca_variance: float = 0.95,
    k_min: int = 2,
    k_max: int = 10,
    stability_runs: int = 20,
    stability_sample_fraction: float = 0.80,
    random_state: int = 42,
    hdbscan_parameter_grid: Iterable[tuple[int, int | None]] | None = None,
) -> dict[str, Any]:
    dataset = valid_post_event_matrix(manifest, embeddings)
    pca = fit_pca(dataset["matrix"], selected_variance=pca_variance)
    kmeans = evaluate_kmeans(
        pca["coordinates"],
        k_min=k_min,
        k_max=k_max,
        random_state=random_state,
    )
    stability = evaluate_cluster_stability(
        pca["coordinates"],
        kmeans["labels_by_k"],
        runs=stability_runs,
        sample_fraction=stability_sample_fraction,
        random_state=random_state,
    )
    gmm = evaluate_gmm(
        pca["coordinates"],
        k_min=1,
        k_max=k_max,
        random_state=random_state,
    )
    hdbscan = evaluate_hdbscan(
        pca["coordinates"],
        parameter_grid=hdbscan_parameter_grid,
    )

    winners = {
        "silhouette_k": _metric_winner(kmeans["rows"], "silhouette", maximize=True),
        "davies_bouldin_k": _metric_winner(kmeans["rows"], "davies_bouldin", maximize=False),
        "calinski_harabasz_k": _metric_winner(kmeans["rows"], "calinski_harabasz", maximize=True),
        "stability_k": _metric_winner(stability["rows"], "ari_mean", maximize=True),
        "gmm_bic_k": int(gmm["best_k"]),
    }
    kmeans_votes = [
        winners["silhouette_k"],
        winners["davies_bouldin_k"],
        winners["calinski_harabasz_k"],
        winners["stability_k"],
    ]
    reference_k = min(kmeans["labels_by_k"], key=lambda k: abs(k - median(kmeans_votes)))
    reference_labels = kmeans["labels_by_k"][reference_k]
    best_silhouette = max(row["silhouette"] for row in kmeans["rows"])
    best_stability = max(row["ari_mean"] for row in stability["rows"])
    if best_silhouette < 0.10:
        structure_signal = "weak separation"
    elif best_silhouette < 0.25:
        structure_signal = "limited separation"
    elif best_silhouette < 0.50:
        structure_signal = "moderate separation"
    else:
        structure_signal = "strong separation"
    kmeans_composition = cluster_composition(
        dataset["metadata"], reference_labels, method=f"KMeans k={reference_k}"
    )
    hdbscan_composition = cluster_composition(
        dataset["metadata"], hdbscan["labels"], method="HDBSCAN"
    )
    event_rows = [
        {
            **row,
            "kmeans_cluster": int(kmeans_cluster),
            "hdbscan_cluster": int(hdbscan_cluster),
        }
        for row, kmeans_cluster, hdbscan_cluster in zip(
            dataset["metadata"], reference_labels, hdbscan["labels"], strict=True
        )
    ]
    summary = {
        **dataset["summary"],
        "pca_target_variance": pca_variance,
        "pca_components": pca["selected_components"],
        **winners,
        "reference_kmeans_k": int(reference_k),
        "best_silhouette": float(best_silhouette),
        "best_stability_ari": float(best_stability),
        "structure_signal": structure_signal,
        "hdbscan_selected": hdbscan["selected"],
        "interpretation_note": (
            "Clusters are unsupervised. Logging/Wildfire is shown only after fitting "
            "to describe cluster composition."
        ),
    }
    return {
        "summary": summary,
        "dataset": dataset,
        "pca": pca,
        "kmeans": kmeans,
        "stability": stability,
        "gmm": gmm,
        "hdbscan": hdbscan,
        "event_rows": event_rows,
        "kmeans_composition": kmeans_composition,
        "hdbscan_composition": hdbscan_composition,
    }


def serializable_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary": result["summary"],
        "pca": {
            "rows": result["pca"]["rows"],
            "selected_variance": result["pca"]["selected_variance"],
            "selected_components": result["pca"]["selected_components"],
            "explained_variance_ratio": result["pca"]["explained_variance_ratio"],
            "cumulative_variance": result["pca"]["cumulative_variance"],
        },
        "kmeans": result["kmeans"]["rows"],
        "stability": result["stability"]["rows"],
        "gmm": result["gmm"]["rows"],
        "hdbscan": {
            "selected": result["hdbscan"]["selected"],
            "rows": result["hdbscan"]["rows"],
        },
        "kmeans_composition": result["kmeans_composition"],
        "hdbscan_composition": result["hdbscan_composition"],
    }


def plot_l2_normalization(result: dict[str, Any]):
    import matplotlib.pyplot as plt

    before = result["dataset"]["norms_before"]
    after = result["dataset"]["norms_after"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    before_bins = 1 if float(np.ptp(before)) < 1e-6 else min(20, max(5, len(before) // 2))
    after_bins = 1 if float(np.ptp(after)) < 1e-6 else min(20, max(5, len(after) // 2))
    axes[0].hist(before, bins=before_bins, color="#4776a8", alpha=0.85)
    axes[0].set_title("L2 norms before normalization")
    axes[1].hist(after, bins=after_bins, color="#4b9b73", alpha=0.85)
    axes[1].set_title("L2 norms after normalization")
    for axis in axes:
        axis.set_xlabel("Norm")
        axis.set_ylabel("Events")
        axis.grid(alpha=0.2)
    return figure


def plot_pca_variance(result: dict[str, Any]):
    import matplotlib.pyplot as plt

    cumulative = result["pca"]["cumulative_variance"]
    components = np.arange(1, len(cumulative) + 1)
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    axis.plot(components, cumulative, marker="o", markersize=3, color="#4776a8")
    for level, color in ((0.90, "#b67a2f"), (0.95, "#a84343")):
        axis.axhline(level, color=color, linestyle="--", label=f"{level:.0%} variance")
    axis.axvline(result["pca"]["selected_components"], color="#4b9b73", linestyle=":")
    axis.set(title="PCA cumulative explained variance", xlabel="Components", ylabel="Cumulative variance")
    axis.set_ylim(0.0, 1.01)
    axis.grid(alpha=0.25)
    axis.legend()
    return figure


def plot_kmeans_metrics(result: dict[str, Any]):
    import matplotlib.pyplot as plt

    rows = result["kmeans"]["rows"]
    k = [row["k"] for row in rows]
    specs = [
        ("silhouette", "Silhouette (higher is better)", "#4776a8"),
        ("davies_bouldin", "Davies-Bouldin (lower is better)", "#a84343"),
        ("calinski_harabasz", "Calinski-Harabasz (higher is better)", "#4b9b73"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    for axis, (key, title, color) in zip(axes, specs, strict=True):
        axis.plot(k, [row[key] for row in rows], marker="o", color=color)
        axis.set(title=title, xlabel="k")
        axis.set_xticks(k)
        axis.grid(alpha=0.25)
    return figure


def plot_kmeans_metric(result: dict[str, Any], metric: str):
    import matplotlib.pyplot as plt

    specs = {
        "silhouette": ("Silhouette score", "higher is better", "#4776a8"),
        "davies_bouldin": ("Davies-Bouldin index", "lower is better", "#a84343"),
        "calinski_harabasz": ("Calinski-Harabasz index", "higher is better", "#4b9b73"),
    }
    if metric not in specs:
        raise ValueError(f"Unsupported K-Means metric: {metric}")
    title, direction, color = specs[metric]
    rows = result["kmeans"]["rows"]
    figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    axis.plot(
        [row["k"] for row in rows],
        [row[metric] for row in rows],
        marker="o",
        color=color,
    )
    axis.set(title=f"{title} ({direction})", xlabel="k", ylabel=title)
    axis.set_xticks([row["k"] for row in rows])
    axis.grid(alpha=0.25)
    return figure


def plot_cluster_stability(result: dict[str, Any]):
    import matplotlib.pyplot as plt

    rows = result["stability"]["rows"]
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    axis.errorbar(
        [row["k"] for row in rows],
        [row["ari_mean"] for row in rows],
        yerr=[row["ari_std"] for row in rows],
        marker="o",
        capsize=4,
        color="#6d58a5",
    )
    axis.set(title="Cluster stability under 80% event subsampling", xlabel="k", ylabel="Adjusted Rand index")
    axis.set_ylim(-0.05, 1.05)
    axis.set_xticks([row["k"] for row in rows])
    axis.grid(alpha=0.25)
    return figure


def plot_gmm_bic(result: dict[str, Any]):
    import matplotlib.pyplot as plt

    rows = result["gmm"]["rows"]
    k = [row["k"] for row in rows]
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    axis.plot(k, [row["bic"] for row in rows], marker="o", label="BIC", color="#a84343")
    axis.plot(k, [row["aic"] for row in rows], marker="s", label="AIC", color="#4776a8", alpha=0.75)
    axis.axvline(result["gmm"]["best_k"], linestyle="--", color="#4b9b73", label="Best BIC")
    axis.set(title="Gaussian mixture model selection", xlabel="Components", ylabel="Information criterion (lower is better)")
    axis.set_xticks(k)
    axis.grid(alpha=0.25)
    axis.legend()
    return figure


def plot_hdbscan(result: dict[str, Any]):
    import matplotlib.pyplot as plt

    coordinates = result["pca"]["plot_coordinates"]
    labels = result["hdbscan"]["labels"]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for cluster in sorted(set(labels.tolist())):
        selected = labels == cluster
        color = "#9aa0a6" if cluster == -1 else None
        axes[0].scatter(
            coordinates[selected, 0],
            coordinates[selected, 1],
            s=40,
            alpha=0.8,
            color=color,
            label="noise" if cluster == -1 else f"cluster {cluster}",
        )
    axes[0].set(title="Selected HDBSCAN solution in PCA space", xlabel="PC1", ylabel="PC2")
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    rows = [row for row in result["hdbscan"]["rows"] if row["selection_score"] >= 0]
    if not rows:
        rows = result["hdbscan"]["rows"]
    axes[1].scatter(
        [row["noise_fraction"] for row in rows],
        [row["selection_score"] for row in rows],
        c=[row["clusters"] for row in rows],
        s=[40 + row["min_cluster_size"] * 3 for row in rows],
        cmap="viridis",
        alpha=0.8,
    )
    axes[1].set(
        title="HDBSCAN parameter sweep",
        xlabel="Noise fraction",
        ylabel="Silhouette x coverage",
    )
    axes[1].grid(alpha=0.2)
    return figure


def plot_cluster_composition(result: dict[str, Any]):
    import matplotlib.pyplot as plt

    rows = result["kmeans_composition"]
    clusters = sorted({int(row["cluster"]) for row in rows})
    labels = sorted({str(row["label"]) for row in rows})
    lookup = {(int(row["cluster"]), str(row["label"])): int(row["events"]) for row in rows}
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    bottom = np.zeros(len(clusters), dtype=float)
    for label in labels:
        values = np.asarray([lookup.get((cluster, label), 0) for cluster in clusters], dtype=float)
        axis.bar([str(cluster) for cluster in clusters], values, bottom=bottom, label=label)
        bottom += values
    axis.set(
        title=f"Reference K-Means composition (k={result['summary']['reference_kmeans_k']})",
        xlabel="Cluster",
        ylabel="MANUAL_VALID events",
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    return figure


def evaluation_figures(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "l2_normalization": plot_l2_normalization(result),
        "pca_variance": plot_pca_variance(result),
        "kmeans_metrics": plot_kmeans_metrics(result),
        "cluster_stability": plot_cluster_stability(result),
        "gmm_bic": plot_gmm_bic(result),
        "hdbscan": plot_hdbscan(result),
        "cluster_composition": plot_cluster_composition(result),
    }
