from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sqlalchemy import func, select
from sqlalchemy.orm import Session, aliased

from ResearchTool.backend.forest.models import EventLabel, EventReview, Sample, SampleEvent

from .reporting import create_run_directory, write_csv, write_json


DEFAULT_MODEL_SOURCES = ("REMOTECLIP_LOGREG", "XGBOOST")
TARGET_LABELS = ("Logging", "Wildfire")


def _iso(value: Any) -> str | None:
    return value.isoformat() if value is not None else None


def _json_object(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _model_prefix(source_name: str) -> str:
    return {
        "REMOTECLIP_LOGREG": "remoteclip",
        "XGBOOST": "xgboost",
    }.get(source_name, source_name.lower())


def _latest_model_labels(
    db: Session,
    event_ids: Sequence[str],
    model_sources: Sequence[str],
) -> dict[tuple[str, str], EventLabel]:
    if not event_ids:
        return {}
    labels = db.scalars(
        select(EventLabel)
        .where(
            EventLabel.event_id.in_(event_ids),
            EventLabel.source_type == "MODEL",
            EventLabel.source_name.in_(model_sources),
        )
        .order_by(
            EventLabel.event_id,
            EventLabel.source_name,
            EventLabel.created_at,
            EventLabel.label_id,
        )
    ).all()
    latest: dict[tuple[str, str], EventLabel] = {}
    for label in labels:
        latest[(label.event_id, label.source_name)] = label
    return latest


def load_reviewer_validation(
    db: Session,
    *,
    reviewer_name: str,
    model_sources: Sequence[str] = DEFAULT_MODEL_SOURCES,
) -> list[dict[str, Any]]:
    """Load the current event reviews and latest model labels for one reviewer."""

    human = aliased(EventLabel)
    reviewer_key = reviewer_name.strip().casefold()
    records = db.execute(
        select(EventReview, SampleEvent, Sample, human)
        .join(SampleEvent, SampleEvent.event_id == EventReview.event_id)
        .join(Sample, Sample.sample_id == SampleEvent.sample_id)
        .outerjoin(human, human.label_id == EventReview.human_label_id)
        .where(func.lower(func.trim(EventReview.reviewer_name)) == reviewer_key)
        .order_by(Sample.sample_id, SampleEvent.event_year, SampleEvent.event_id)
    ).all()
    event_ids = [event.event_id for _, event, _, _ in records]
    latest = _latest_model_labels(db, event_ids, model_sources)

    rows: list[dict[str, Any]] = []
    for review, event, sample, human_label in records:
        human_value = human_label.label if human_label is not None else None
        row: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "display_name": sample.display_name,
            "source_id": sample.source_id,
            "source_format": sample.source_format,
            "lat": sample.lat,
            "lon": sample.lon,
            "event_id": event.event_id,
            "event_year": event.event_year,
            "event_source": event.source,
            "event_loss_area_ha": event.loss_area_ha,
            "event_loss_share": event.loss_share,
            "event_is_dominant": bool(event.is_dominant),
            "review_status": review.review_status,
            "human_label_id": review.human_label_id,
            "human_label": human_value,
            "rejection_reason": review.rejection_reason,
            "review_notes": review.notes,
            "reviewer_name": review.reviewer_name,
            "review_created_at": _iso(review.created_at),
            "review_updated_at": _iso(review.updated_at),
        }
        model_values: list[str | None] = []
        for source_name in model_sources:
            prefix = _model_prefix(source_name)
            label = latest.get((event.event_id, source_name))
            value = label.label if label is not None else None
            model_values.append(value)
            row[f"{prefix}_label"] = value
            row[f"{prefix}_confidence"] = label.confidence if label is not None else None
            row[f"{prefix}_version"] = label.source_version if label is not None else None
            row[f"{prefix}_label_id"] = label.label_id if label is not None else None
            row[f"{prefix}_probabilities"] = (
                _json_object(label.probabilities_json) if label is not None else {}
            )
            row[f"{prefix}_correct"] = (
                value == human_value
                if review.review_status == "VALID"
                and human_value in TARGET_LABELS
                and value is not None
                else None
            )

        complete = all(value is not None for value in model_values)
        row["models_complete"] = complete
        row["models_agree"] = (
            len(set(model_values)) == 1 if complete else None
        )
        if review.review_status != "VALID":
            if complete:
                row["joint_outcome"] = (
                    "MODELS_AGREE_REJECTED"
                    if row["models_agree"]
                    else "MODELS_DISAGREE_REJECTED"
                )
            else:
                row["joint_outcome"] = "MISSING_MODEL_LABEL_REJECTED"
        elif human_value not in TARGET_LABELS:
            row["joint_outcome"] = "NON_TARGET_HUMAN_LABEL"
        elif not complete:
            row["joint_outcome"] = "MISSING_MODEL_LABEL"
        else:
            correct = [value == human_value for value in model_values]
            if all(correct):
                row["joint_outcome"] = "BOTH_CORRECT"
            elif not any(correct):
                row["joint_outcome"] = "BOTH_WRONG"
            elif correct[0]:
                row["joint_outcome"] = "REMOTECLIP_ONLY_CORRECT"
            else:
                row["joint_outcome"] = "XGBOOST_ONLY_CORRECT"
        rows.append(row)
    return rows


def _count_rows(
    rows: Iterable[dict[str, Any]],
    keys: Sequence[str],
    *,
    denominator: int | None = None,
) -> list[dict[str, Any]]:
    counts = Counter(tuple(row.get(key) for key in keys) for row in rows)
    total = denominator if denominator is not None else sum(counts.values())
    result: list[dict[str, Any]] = []
    for values, count in sorted(counts.items(), key=lambda item: (-item[1], str(item[0]))):
        record = {key: value for key, value in zip(keys, values, strict=True)}
        record["count"] = count
        record["percent"] = 100.0 * count / total if total else 0.0
        result.append(record)
    return result


def _group_bootstrap_ci(
    rows: Sequence[dict[str, Any]],
    prediction_key: str,
    *,
    bootstrap_runs: int,
    seed: int,
) -> dict[str, float | None]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["sample_id"])].append(row)
    group_ids = sorted(grouped)
    if not group_ids or bootstrap_runs <= 0:
        return {
            "accuracy_ci_low": None,
            "accuracy_ci_high": None,
            "balanced_accuracy_ci_low": None,
            "balanced_accuracy_ci_high": None,
        }
    generator = np.random.default_rng(seed)
    accuracy_values: list[float] = []
    balanced_values: list[float] = []
    for _ in range(bootstrap_runs):
        selected = generator.choice(group_ids, size=len(group_ids), replace=True)
        sampled = [row for group_id in selected for row in grouped[str(group_id)]]
        y_true = [str(row["human_label"]) for row in sampled]
        y_pred = [str(row[prediction_key]) for row in sampled]
        accuracy_values.append(float(accuracy_score(y_true, y_pred)))
        if len(set(y_true)) == len(TARGET_LABELS):
            balanced_values.append(float(balanced_accuracy_score(y_true, y_pred)))

    def interval(values: Sequence[float]) -> tuple[float | None, float | None]:
        if not values:
            return None, None
        low, high = np.quantile(np.asarray(values), [0.025, 0.975])
        return float(low), float(high)

    accuracy_low, accuracy_high = interval(accuracy_values)
    balanced_low, balanced_high = interval(balanced_values)
    return {
        "accuracy_ci_low": accuracy_low,
        "accuracy_ci_high": accuracy_high,
        "balanced_accuracy_ci_low": balanced_low,
        "balanced_accuracy_ci_high": balanced_high,
    }


def _model_metrics(
    rows: Sequence[dict[str, Any]],
    model_sources: Sequence[str],
    *,
    bootstrap_runs: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    valid = [
        row
        for row in rows
        if row["review_status"] == "VALID" and row["human_label"] in TARGET_LABELS
    ]
    metrics: list[dict[str, Any]] = []
    class_metrics: list[dict[str, Any]] = []
    confusion_rows: dict[str, list[dict[str, Any]]] = {}
    for index, source_name in enumerate(model_sources):
        prefix = _model_prefix(source_name)
        prediction_key = f"{prefix}_label"
        evaluated = [row for row in valid if row.get(prediction_key) in TARGET_LABELS]
        y_true = [str(row["human_label"]) for row in evaluated]
        y_pred = [str(row[prediction_key]) for row in evaluated]
        metric: dict[str, Any] = {
            "model": source_name,
            "eligible_events": len(valid),
            "evaluated_events": len(evaluated),
            "missing_predictions": len(valid) - len(evaluated),
            "coverage": len(evaluated) / len(valid) if valid else 0.0,
            "accuracy": None,
            "balanced_accuracy": None,
            "macro_f1": None,
        }
        matrix = np.zeros((len(TARGET_LABELS), len(TARGET_LABELS)), dtype=int)
        if evaluated:
            metric["accuracy"] = float(accuracy_score(y_true, y_pred))
            metric["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
            _, _, macro_f1, _ = precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=list(TARGET_LABELS),
                average="macro",
                zero_division=0,
            )
            metric["macro_f1"] = float(macro_f1)
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=list(TARGET_LABELS),
                zero_division=0,
            )
            for label, p_value, r_value, f_value, count in zip(
                TARGET_LABELS, precision, recall, f1, support, strict=True
            ):
                class_metrics.append(
                    {
                        "model": source_name,
                        "class": label,
                        "precision": float(p_value),
                        "recall": float(r_value),
                        "f1": float(f_value),
                        "support": int(count),
                    }
                )
            matrix = confusion_matrix(y_true, y_pred, labels=list(TARGET_LABELS))
            metric.update(
                _group_bootstrap_ci(
                    evaluated,
                    prediction_key,
                    bootstrap_runs=bootstrap_runs,
                    seed=seed + index,
                )
            )
        else:
            metric.update(
                {
                    "accuracy_ci_low": None,
                    "accuracy_ci_high": None,
                    "balanced_accuracy_ci_low": None,
                    "balanced_accuracy_ci_high": None,
                }
            )
        metrics.append(metric)
        confusion_rows[source_name] = [
            {
                "true_label": true_label,
                **{
                    f"predicted_{predicted_label}": int(matrix[true_index, predicted_index])
                    for predicted_index, predicted_label in enumerate(TARGET_LABELS)
                },
                "total": int(matrix[true_index].sum()),
            }
            for true_index, true_label in enumerate(TARGET_LABELS)
        ]
    return metrics, class_metrics, confusion_rows


def _sample_rows(rows: Sequence[dict[str, Any]], model_sources: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["sample_id"])].append(row)
    result: list[dict[str, Any]] = []
    for sample_id, events in sorted(grouped.items()):
        first = events[0]
        human_labels = sorted(
            {str(row["human_label"]) for row in events if row.get("human_label")}
        )
        record: dict[str, Any] = {
            "sample_id": sample_id,
            "display_name": first.get("display_name"),
            "source_id": first.get("source_id"),
            "source_format": first.get("source_format"),
            "lat": first.get("lat"),
            "lon": first.get("lon"),
            "reviewed_events": len(events),
            "valid_events": sum(row["review_status"] == "VALID" for row in events),
            "rejected_events": sum(row["review_status"] == "REJECTED" for row in events),
            "event_years": sorted({int(row["event_year"]) for row in events}),
            "human_labels": human_labels,
            "human_labels_consistent": len(human_labels) <= 1,
        }
        for source_name in model_sources:
            prefix = _model_prefix(source_name)
            labels = sorted(
                {str(row[f"{prefix}_label"]) for row in events if row.get(f"{prefix}_label")}
            )
            valid_correct = [
                bool(row[f"{prefix}_correct"])
                for row in events
                if row.get(f"{prefix}_correct") is not None
            ]
            record[f"{prefix}_labels"] = labels
            record[f"{prefix}_labels_consistent"] = len(labels) <= 1
            record[f"{prefix}_all_valid_events_correct"] = (
                all(valid_correct) if valid_correct else None
            )
        result.append(record)
    return result


def summarize_reviewer_validation(
    rows: Sequence[dict[str, Any]],
    *,
    model_sources: Sequence[str] = DEFAULT_MODEL_SOURCES,
    bootstrap_runs: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    valid_target = [
        row
        for row in rows
        if row["review_status"] == "VALID" and row["human_label"] in TARGET_LABELS
    ]
    model_metrics, class_metrics, confusion = _model_metrics(
        rows,
        model_sources,
        bootstrap_runs=bootstrap_runs,
        seed=seed,
    )
    sample_rows = _sample_rows(rows, model_sources)
    comparison_keys = [f"{_model_prefix(source)}_label" for source in model_sources]
    both_present = [row for row in rows if all(row.get(key) for key in comparison_keys)]
    valid_both = [
        row
        for row in valid_target
        if all(row.get(key) in TARGET_LABELS for key in comparison_keys)
    ]
    return {
        "summary": {
            "reviewed_events": len(rows),
            "reviewed_samples": len({row["sample_id"] for row in rows}),
            "valid_events": sum(row["review_status"] == "VALID" for row in rows),
            "rejected_events": sum(row["review_status"] == "REJECTED" for row in rows),
            "valid_target_events": len(valid_target),
            "events_with_both_models": len(both_present),
            "valid_events_with_both_models": len(valid_both),
            "models_agree_all_reviewed": sum(row["models_agree"] is True for row in both_present),
            "models_disagree_all_reviewed": sum(
                row["models_agree"] is False for row in both_present
            ),
        },
        "event_rows": list(rows),
        "sample_rows": sample_rows,
        "review_status_counts": _count_rows(rows, ["review_status"]),
        "human_label_counts": _count_rows(
            [row for row in rows if row["review_status"] == "VALID"],
            ["human_label"],
        ),
        "rejection_reason_counts": _count_rows(
            [row for row in rows if row["review_status"] == "REJECTED"],
            ["rejection_reason"],
        ),
        "joint_outcome_counts": _count_rows(valid_target, ["joint_outcome"]),
        "model_agreement_counts": _count_rows(
            both_present,
            comparison_keys,
        ),
        "truth_model_combinations": _count_rows(
            valid_both,
            ["human_label", *comparison_keys, "joint_outcome"],
        ),
        "model_metrics": model_metrics,
        "class_metrics": class_metrics,
        "confusion_matrices": confusion,
        "settings": {
            "model_sources": list(model_sources),
            "target_labels": list(TARGET_LABELS),
            "bootstrap_runs": bootstrap_runs,
            "bootstrap_unit": "sample_id",
            "seed": seed,
        },
    }


def _format_metric(value: Any) -> str:
    return "-" if value is None else f"{float(value):.3f}"


def reviewer_validation_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    reviewer = result["reviewer_name"]
    lines = [
        f"# Model validation on {reviewer}'s reviews",
        "",
        "## Cohort",
        "",
        f"- Reviewed events: **{summary['reviewed_events']}**.",
        f"- Reviewed points: **{summary['reviewed_samples']}**.",
        f"- Valid events: **{summary['valid_events']}**.",
        f"- Rejected events: **{summary['rejected_events']}**.",
        f"- Valid target-class events: **{summary['valid_target_events']}**.",
        "",
        "A rejected event is reported separately and is not treated as a model error. "
        "Human truth is taken only from the label referenced by the current event review.",
        "",
        "## Model metrics on valid events",
        "",
        "| Model | Evaluated / eligible | Accuracy (95% grouped CI) | Balanced accuracy (95% grouped CI) | Macro F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in result["model_metrics"]:
        accuracy_ci = (
            f"{_format_metric(row['accuracy'])} "
            f"[{_format_metric(row['accuracy_ci_low'])}, {_format_metric(row['accuracy_ci_high'])}]"
        )
        balanced_ci = (
            f"{_format_metric(row['balanced_accuracy'])} "
            f"[{_format_metric(row['balanced_accuracy_ci_low'])}, "
            f"{_format_metric(row['balanced_accuracy_ci_high'])}]"
        )
        lines.append(
            f"| {row['model']} | {row['evaluated_events']} / {row['eligible_events']} | "
            f"{accuracy_ci} | {balanced_ci} | {_format_metric(row['macro_f1'])} |"
        )
    lines.extend(
        [
            "",
            "Bootstrap confidence intervals resample complete points (`sample_id`), not individual years.",
            "",
            "## Per-class metrics",
            "",
            "| Model | Human class | Precision | Recall | F1 | Support |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in result["class_metrics"]:
        lines.append(
            f"| {row['model']} | {row['class']} | {_format_metric(row['precision'])} | "
            f"{_format_metric(row['recall'])} | {_format_metric(row['f1'])} | "
            f"{row['support']} |"
        )
    lines.extend(
        [
            "",
            "## Confusion matrices",
            "",
            "Rows are accepted human labels; prediction classes are columns.",
            "",
        ]
    )
    for model_name, rows in result["confusion_matrices"].items():
        lines.extend(
            [
                f"### {model_name}",
                "",
                "| Human class | Predicted Logging | Predicted Wildfire | Total |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['true_label']} | {row['predicted_Logging']} | "
                f"{row['predicted_Wildfire']} | {row['total']} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Model-to-model predictions",
            "",
            "This table includes all reviewed events where both model labels are present.",
            "",
            "| RemoteCLIP | XGBoost | Events | Percent |",
            "|---|---|---:|---:|",
        ]
    )
    for row in result["model_agreement_counts"]:
        lines.append(
            f"| {row.get('remoteclip_label')} | {row.get('xgboost_label')} | "
            f"{row['count']} | {row['percent']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Joint outcomes on valid events",
            "",
            "| Outcome | Events | Percent |",
            "|---|---:|---:|",
        ]
    )
    for row in result["joint_outcome_counts"]:
        lines.append(
            f"| {row['joint_outcome']} | {row['count']} | {row['percent']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Rejected events",
            "",
            "| Rejection reason | Events | Percent |",
            "|---|---:|---:|",
        ]
    )
    for row in result["rejection_reason_counts"]:
        lines.append(
            f"| {row['rejection_reason'] or '(not specified)'} | {row['count']} | "
            f"{row['percent']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `event_level.csv`: every reviewed event with human truth and both predictions.",
            "- `reviewed_points.csv`: one row per point reviewed by this reviewer.",
            "- `model_metrics.csv` and `class_metrics.csv`: aggregate and per-class metrics.",
            "- `model_agreement.csv` and `truth_model_combinations.csv`: comparative tables.",
            "- `confusion_*.csv`: confusion matrices with human labels in rows.",
            "- `rejected_events.csv`: invalid events kept outside accuracy metrics.",
            "",
        ]
    )
    return "\n".join(lines)


def validation_figures(result: dict[str, Any]) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    figures: dict[str, Any] = {}
    model_metrics = result["model_metrics"]
    if model_metrics:
        names = [str(row["model"]).replace("_LOGREG", "") for row in model_metrics]
        values = [float(row["accuracy"] or 0.0) for row in model_metrics]
        lower = []
        upper = []
        for value, row in zip(values, model_metrics, strict=True):
            low = row["accuracy_ci_low"]
            high = row["accuracy_ci_high"]
            lower.append(max(0.0, value - float(low if low is not None else value)))
            upper.append(max(0.0, float(high if high is not None else value) - value))
        figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
        axis.bar(names, values, yerr=np.asarray([lower, upper]), capsize=5, color=["#3569a8", "#dd7b2d"])
        axis.set_ylim(0.0, 1.05)
        axis.set_ylabel("Accuracy")
        axis.set_title("Accuracy on valid reviewed events (95% grouped bootstrap CI)")
        axis.grid(axis="y", alpha=0.25)
        figures["model_accuracy"] = figure

    confusion = result["confusion_matrices"]
    if confusion:
        figure, axes = plt.subplots(1, len(confusion), figsize=(6 * len(confusion), 4.8), constrained_layout=True)
        axes_values = np.atleast_1d(axes)
        for axis, (model, rows) in zip(axes_values, confusion.items(), strict=True):
            matrix = np.asarray(
                [
                    [row[f"predicted_{label}"] for label in TARGET_LABELS]
                    for row in rows
                ],
                dtype=int,
            )
            image = axis.imshow(matrix, cmap="Blues")
            for row_index in range(matrix.shape[0]):
                for column_index in range(matrix.shape[1]):
                    axis.text(column_index, row_index, str(matrix[row_index, column_index]), ha="center", va="center")
            axis.set_xticks(range(len(TARGET_LABELS)), TARGET_LABELS)
            axis.set_yticks(range(len(TARGET_LABELS)), TARGET_LABELS)
            axis.set_xlabel("Predicted label")
            axis.set_ylabel("Human label")
            axis.set_title(model)
            figure.colorbar(image, ax=axis, shrink=0.8)
        figures["confusion_matrices"] = figure
    return figures


def export_reviewer_validation(
    result: dict[str, Any],
    *,
    output_root: str | Path,
    run_id: str | None = None,
) -> Path:
    run_dir = create_run_directory(output_root, run_id)
    write_csv(run_dir / "event_level.csv", result["event_rows"])
    write_csv(run_dir / "reviewed_points.csv", result["sample_rows"])
    write_csv(run_dir / "model_metrics.csv", result["model_metrics"])
    write_csv(run_dir / "class_metrics.csv", result["class_metrics"])
    write_csv(run_dir / "review_status_counts.csv", result["review_status_counts"])
    write_csv(run_dir / "human_label_counts.csv", result["human_label_counts"])
    write_csv(run_dir / "rejection_reasons.csv", result["rejection_reason_counts"])
    write_csv(run_dir / "joint_outcomes.csv", result["joint_outcome_counts"])
    write_csv(run_dir / "model_agreement.csv", result["model_agreement_counts"])
    write_csv(run_dir / "truth_model_combinations.csv", result["truth_model_combinations"])
    write_csv(
        run_dir / "rejected_events.csv",
        [row for row in result["event_rows"] if row["review_status"] == "REJECTED"],
    )
    for source_name, rows in result["confusion_matrices"].items():
        write_csv(run_dir / f"confusion_{source_name.lower()}.csv", rows)
    serializable = {
        key: value
        for key, value in result.items()
        if key not in {"event_rows", "sample_rows"}
    }
    write_json(run_dir / "summary.json", serializable)
    (run_dir / "REPORT.md").write_text(reviewer_validation_report(result), encoding="utf-8")
    figures = validation_figures(result)
    for name, figure in figures.items():
        figure.savefig(run_dir / "plots" / f"{name}.png", dpi=160, bbox_inches="tight")
    result["run_dir"] = str(run_dir)
    result["report_path"] = str(run_dir / "REPORT.md")
    return run_dir


def run_reviewer_validation(
    db: Session,
    *,
    reviewer_name: str,
    model_sources: Sequence[str] = DEFAULT_MODEL_SOURCES,
    bootstrap_runs: int = 2000,
    seed: int = 42,
    output_root: str | Path | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    rows = load_reviewer_validation(
        db,
        reviewer_name=reviewer_name,
        model_sources=model_sources,
    )
    result = summarize_reviewer_validation(
        rows,
        model_sources=model_sources,
        bootstrap_runs=bootstrap_runs,
        seed=seed,
    )
    result["reviewer_name"] = reviewer_name
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    if output_root is not None:
        export_reviewer_validation(result, output_root=output_root, run_id=run_id)
    return result
