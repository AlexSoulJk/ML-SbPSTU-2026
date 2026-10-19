from __future__ import annotations

import csv
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_value(item) for item in value]
    return value


def create_run_directory(output_root: str | Path, run_id: str | None = None) -> Path:
    resolved = Path(output_root)
    resolved.mkdir(parents=True, exist_ok=True)
    name = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = resolved / name
    suffix = 1
    while run_dir.exists():
        run_dir = resolved / f"{name}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True)
    (run_dir / "plots").mkdir()
    return run_dir


def write_json(path: str | Path, value: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(json_value(value), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return target


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set, np.ndarray)):
        return json.dumps(json_value(value), ensure_ascii=False, sort_keys=True)
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    values = list(rows)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in values:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        if not fieldnames:
            handle.write("")
            return target
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in values:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})
    return target


def read_csv(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    if not source.exists() or source.stat().st_size == 0:
        return []
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _metric_text(row: dict[str, Any], metric: str) -> str:
    mean_value = row.get(f"{metric}_mean", "")
    std_value = row.get(f"{metric}_std", "")
    try:
        return f"{float(mean_value):.3f} +/- {float(std_value):.3f}"
    except (TypeError, ValueError):
        return "-"


def baseline_metrics_html(rows: Iterable[dict[str, Any]]) -> str:
    values = list(rows)
    headers = [
        "Features",
        "Mode",
        "Split",
        "Runs",
        "Accuracy",
        "Balanced accuracy",
        "F1",
        "ROC-AUC",
    ]
    body: list[list[str]] = []
    for row in values:
        body.append(
            [
                str(row.get("feature_variant", "")),
                str(row.get("mode", "")),
                str(row.get("split_strategy", "")),
                str(row.get("fold_runs", "")),
                _metric_text(row, "accuracy"),
                _metric_text(row, "balanced_accuracy"),
                _metric_text(row, "f1"),
                _metric_text(row, "roc_auc"),
            ]
        )
    header_html = "".join(f"<th>{html.escape(value)}</th>" for value in headers)
    rows_html = "".join(
        "<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in row) + "</tr>"
        for row in body
    )
    return (
        "<div style='overflow-x:auto'>"
        "<table style='border-collapse:collapse;width:100%;font-size:13px'>"
        f"<thead><tr>{header_html}</tr></thead><tbody>{rows_html}</tbody></table></div>"
        "<style>table th,table td{border:1px solid #d7dce2;padding:6px 8px;text-align:left;}"
        "table th{background:#f3f5f7;position:sticky;top:0;}</style>"
    )


def review_queue_html(rows: Iterable[dict[str, Any]], *, limit: int = 20) -> str:
    values = list(rows)[:limit]
    headers = [
        "sample_id",
        "event_year",
        "weak_label",
        "predicted_class",
        "probability_wildfire",
        "agreement_with_weak_label",
        "review_priority",
    ]
    header_html = "".join(f"<th>{html.escape(value)}</th>" for value in headers)
    rows_html = ""
    for row in values:
        cells: list[str] = []
        for key in headers:
            value = row.get(key, "")
            if key in {"probability_wildfire", "review_priority"}:
                try:
                    value = f"{float(value):.3f}"
                except (TypeError, ValueError):
                    pass
            cells.append(f"<td>{html.escape(str(value))}</td>")
        rows_html += "<tr>" + "".join(cells) + "</tr>"
    return (
        "<div style='overflow-x:auto'>"
        "<table style='border-collapse:collapse;width:100%;font-size:13px'>"
        f"<thead><tr>{header_html}</tr></thead><tbody>{rows_html}</tbody></table></div>"
    )


def plot_baseline_metrics(rows: Iterable[dict[str, Any]]):
    import matplotlib.pyplot as plt

    values = list(rows)
    metrics = ["accuracy", "balanced_accuracy", "f1", "roc_auc"]
    variants = list(dict.fromkeys(str(row.get("feature_variant", "")) for row in values))
    series = list(
        dict.fromkeys(
            f"{row.get('mode', '')} / {row.get('split_strategy', '')}" for row in values
        )
    )
    lookup = {
        (
            str(row.get("feature_variant", "")),
            f"{row.get('mode', '')} / {row.get('split_strategy', '')}",
        ): row
        for row in values
    }
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    for axis, metric in zip(axes.flat, metrics, strict=True):
        for name in series:
            y_values: list[float] = []
            errors: list[float] = []
            for variant in variants:
                row = lookup.get((variant, name), {})
                try:
                    y_values.append(float(row[f"{metric}_mean"]))
                    errors.append(float(row[f"{metric}_std"]))
                except (KeyError, TypeError, ValueError):
                    y_values.append(float("nan"))
                    errors.append(0.0)
            axis.errorbar(variants, y_values, yerr=errors, marker="o", capsize=3, label=name)
        axis.set_title(metric.replace("_", " ").title())
        axis.set_ylim(0.0, 1.05)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=30)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=8)
    return figure

