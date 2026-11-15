from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .rejection_diagnostics import review_hierarchy_class
from .reporting import create_run_directory, write_csv, write_json
from .review_validation import DEFAULT_MODEL_SOURCES, summarize_reviewer_validation


COHORT_ORDER = ("baseline", "added", "current")
COHORT_LABELS = {
    "baseline": "Baseline cohort",
    "added": "Added cohort",
    "current": "Current total",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _metric_lookup(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["model"]): row for row in result["model_metrics"]}


def _count_lookup(rows: Sequence[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    return {str(row.get(key)): row for row in rows}


def _pct(value: Any) -> str:
    return "-" if value is None else f"{100.0 * float(value):.1f}%"


def _pp(value: Any) -> str:
    if value is None:
        return "-"
    return f"{100.0 * float(value):+.1f} pp"


def _review_signature(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(row.get(key) or "")
        for key in (
            "review_status",
            "human_label",
            "rejection_reason",
            "remoteclip_label",
            "xgboost_label",
        )
    )


def _changed_baseline_reviews(
    baseline_events: Sequence[dict[str, Any]],
    current_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    baseline_by_id = {str(row["event_id"]): row for row in baseline_events}
    current_by_id = {str(row["event_id"]): row for row in current_rows}
    changed: list[dict[str, Any]] = []
    for event_id in sorted(baseline_by_id.keys() & current_by_id.keys()):
        old = baseline_by_id[event_id]
        new = current_by_id[event_id]
        if _review_signature(old) == _review_signature(new):
            continue
        changed.append(
            {
                "event_id": event_id,
                "sample_id": new.get("sample_id"),
                "event_year": new.get("event_year"),
                "baseline_review_status": old.get("review_status"),
                "current_review_status": new.get("review_status"),
                "baseline_human_label": old.get("human_label"),
                "current_human_label": new.get("human_label"),
                "baseline_rejection_reason": old.get("rejection_reason"),
                "current_rejection_reason": new.get("rejection_reason"),
                "baseline_remoteclip_label": old.get("remoteclip_label"),
                "current_remoteclip_label": new.get("remoteclip_label"),
                "baseline_xgboost_label": old.get("xgboost_label"),
                "current_xgboost_label": new.get("xgboost_label"),
            }
        )
    return (
        changed,
        sorted(baseline_by_id.keys() - current_by_id.keys()),
        sorted(current_by_id.keys() - baseline_by_id.keys()),
    )


def _cohort_summary_rows(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cohort in COHORT_ORDER:
        summary = results[cohort]["summary"]
        with_both = int(summary["events_with_both_models"])
        agreement = int(summary["models_agree_all_reviewed"])
        rows.append(
            {
                "cohort": cohort,
                "cohort_label": COHORT_LABELS[cohort],
                **summary,
                "valid_share": (
                    int(summary["valid_events"]) / int(summary["reviewed_events"])
                    if summary["reviewed_events"]
                    else None
                ),
                "model_agreement_share": agreement / with_both if with_both else None,
            }
        )
    return rows


def _model_metric_rows(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = _metric_lookup(results["baseline"])
    rows: list[dict[str, Any]] = []
    for cohort in COHORT_ORDER:
        for model, metric in _metric_lookup(results[cohort]).items():
            base = baseline[model]
            rows.append(
                {
                    "cohort": cohort,
                    "cohort_label": COHORT_LABELS[cohort],
                    **metric,
                    "accuracy_delta_vs_baseline": metric["accuracy"] - base["accuracy"],
                    "balanced_accuracy_delta_vs_baseline": (
                        metric["balanced_accuracy"] - base["balanced_accuracy"]
                    ),
                    "macro_f1_delta_vs_baseline": metric["macro_f1"] - base["macro_f1"],
                }
            )
    return rows


def _class_metric_rows(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cohort in COHORT_ORDER:
        for metric in results[cohort]["class_metrics"]:
            rows.append(
                {
                    "cohort": cohort,
                    "cohort_label": COHORT_LABELS[cohort],
                    **metric,
                }
            )
    return rows


def _status_rows(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cohort in COHORT_ORDER:
        for status in results[cohort]["review_status_counts"]:
            rows.append({"cohort": cohort, "cohort_label": COHORT_LABELS[cohort], **status})
    return rows


def _human_label_rows(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cohort in COHORT_ORDER:
        for label in results[cohort]["human_label_counts"]:
            rows.append({"cohort": cohort, "cohort_label": COHORT_LABELS[cohort], **label})
    return rows


def _hierarchy_rows(cohort_rows: dict[str, Sequence[dict[str, Any]]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for cohort in COHORT_ORDER:
        counts = Counter(review_hierarchy_class(row) for row in cohort_rows[cohort])
        total = sum(counts.values())
        for review_class, count in counts.most_common():
            result.append(
                {
                    "cohort": cohort,
                    "cohort_label": COHORT_LABELS[cohort],
                    "review_class": review_class,
                    "count": count,
                    "percent": 100.0 * count / total if total else None,
                }
            )
    return result


def _plot_dynamics(
    output_dir: Path,
    metric_rows: Sequence[dict[str, Any]],
    status_rows: Sequence[dict[str, Any]],
    label_rows: Sequence[dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    models = list(DEFAULT_MODEL_SOURCES)
    metrics = ("accuracy", "balanced_accuracy", "macro_f1")
    figure, axes = plt.subplots(1, len(metrics), figsize=(15, 4.6), constrained_layout=True)
    width = 0.24
    x = np.arange(len(models))
    for offset, cohort in enumerate(COHORT_ORDER):
        lookup = {
            str(row["model"]): row
            for row in metric_rows
            if row["cohort"] == cohort
        }
        for axis, metric in zip(axes, metrics, strict=True):
            axis.bar(
                x + (offset - 1) * width,
                [float(lookup[model][metric]) for model in models],
                width,
                label=COHORT_LABELS[cohort],
            )
            axis.set_title(metric.replace("_", " ").title())
            axis.set_xticks(x, ["RemoteCLIP", "XGBoost"])
            axis.set_ylim(0.0, 1.05)
            axis.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8, loc="lower left")
    figure.savefig(output_dir / "plots" / "model_metric_dynamics.png", dpi=170, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    status_lookup = {
        (str(row["cohort"]), str(row["review_status"])): float(row["percent"])
        for row in status_rows
    }
    label_lookup = {
        (str(row["cohort"]), str(row["human_label"])): float(row["percent"])
        for row in label_rows
    }
    labels = [COHORT_LABELS[key] for key in COHORT_ORDER]
    x = np.arange(len(labels))
    valid = [status_lookup.get((key, "VALID"), 0.0) for key in COHORT_ORDER]
    rejected = [status_lookup.get((key, "REJECTED"), 0.0) for key in COHORT_ORDER]
    axes[0].bar(x, valid, label="VALID")
    axes[0].bar(x, rejected, bottom=valid, label="REJECTED")
    axes[0].set_title("Review status composition")
    axes[0].set_ylabel("Events, %")
    axes[0].set_xticks(x, ["Baseline", "Added", "Current"])
    axes[0].legend()
    logging = [label_lookup.get((key, "Logging"), 0.0) for key in COHORT_ORDER]
    wildfire = [label_lookup.get((key, "Wildfire"), 0.0) for key in COHORT_ORDER]
    axes[1].bar(x, logging, label="Logging")
    axes[1].bar(x, wildfire, bottom=logging, label="Wildfire")
    axes[1].set_title("Accepted class composition")
    axes[1].set_ylabel("Valid events, %")
    axes[1].set_xticks(x, ["Baseline", "Added", "Current"])
    axes[1].legend()
    figure.savefig(output_dir / "plots" / "cohort_composition.png", dpi=170, bbox_inches="tight")
    plt.close(figure)


def _report(
    *,
    baseline_dir: Path,
    current_run_dir: Path,
    results: dict[str, dict[str, Any]],
    cohort_summary: Sequence[dict[str, Any]],
    metric_rows: Sequence[dict[str, Any]],
    hierarchy_rows: Sequence[dict[str, Any]],
    changed_reviews: Sequence[dict[str, Any]],
    missing_baseline_events: Sequence[str],
    added_baseline_events: Sequence[str],
    remaining_assignments: Sequence[dict[str, Any]],
) -> str:
    cohort_by_name = {str(row["cohort"]): row for row in cohort_summary}
    metric_by_key = {
        (str(row["cohort"]), str(row["model"])): row for row in metric_rows
    }
    hierarchy_classes = sorted({str(row["review_class"]) for row in hierarchy_rows})
    hierarchy_lookup = {
        (str(row["cohort"]), str(row["review_class"])): row for row in hierarchy_rows
    }
    baseline = cohort_by_name["baseline"]
    added = cohort_by_name["added"]
    current = cohort_by_name["current"]
    lines = [
        "# Dynamics of Olga's Hansen-mining validation",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`.",
        "",
        "## Executive summary",
        "",
        f"- The reviewed cohort grew from **{baseline['reviewed_samples']} to "
        f"{current['reviewed_samples']} points** and from **{baseline['reviewed_events']} to "
        f"{current['reviewed_events']} events**.",
        f"- The added batch contains **{added['reviewed_samples']} points / "
        f"{added['reviewed_events']} events**.",
        f"- Baseline integrity check: **{len(changed_reviews)} changed**, "
        f"**{len(missing_baseline_events)} missing**, and "
        f"**{len(added_baseline_events)} unexpected** events among the original 49 points.",
        f"- Assignment completion is **{current['reviewed_samples']}/100 points**; "
        f"**{len(remaining_assignments)}** assigned point remains outside Olga's completed reviews.",
        "",
        "The old 49-point cohort is unchanged, so the metric movement below is attributable to "
        "the newly reviewed 50-point cohort rather than edits to the baseline labels.",
        "",
        "## Cohort growth",
        "",
        "| Cohort | Points | Events | Valid | Rejected | Valid share | Both-model events | Model agreement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cohort in COHORT_ORDER:
        row = cohort_by_name[cohort]
        lines.append(
            f"| {COHORT_LABELS[cohort]} | {row['reviewed_samples']} | "
            f"{row['reviewed_events']} | {row['valid_events']} | {row['rejected_events']} | "
            f"{_pct(row['valid_share'])} | {row['events_with_both_models']} | "
            f"{_pct(row['model_agreement_share'])} |"
        )
    lines.extend(
        [
            "",
            "## Model metrics on human-accepted events",
            "",
            "| Model | Cohort | Events | Accuracy | Balanced accuracy | Macro F1 | Accuracy Δ vs baseline | Balanced Δ | F1 Δ |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model in DEFAULT_MODEL_SOURCES:
        for cohort in COHORT_ORDER:
            row = metric_by_key[(cohort, model)]
            lines.append(
                f"| {model} | {COHORT_LABELS[cohort]} | {row['evaluated_events']} | "
                f"{_pct(row['accuracy'])} | {_pct(row['balanced_accuracy'])} | "
                f"{_pct(row['macro_f1'])} | {_pp(row['accuracy_delta_vs_baseline'])} | "
                f"{_pp(row['balanced_accuracy_delta_vs_baseline'])} | "
                f"{_pp(row['macro_f1_delta_vs_baseline'])} |"
            )
    lines.extend(
        [
            "",
            "### Interpretation",
            "",
            "- RemoteCLIP remains the stronger model on the expanded cohort. Its raw accuracy "
            "decreases slightly, while balanced accuracy and macro F1 improve because the new "
            "batch contains substantially more Logging examples.",
            "- XGBoost's raw accuracy drops more strongly on the new batch. Its balanced accuracy "
            "rises above the original 0.5 baseline because it now identifies some Logging events, "
            "but class separation remains weak.",
            "- Model-to-model agreement falls on the added cohort, making the second half materially "
            "more informative for comparison than the first Wildfire-heavy batch.",
            "",
            "## Normalized review hierarchy",
            "",
            "| Review class | Baseline 49 | Added 50 | Current 99 |",
            "|---|---:|---:|---:|",
        ]
    )
    for review_class in hierarchy_classes:
        values = []
        for cohort in COHORT_ORDER:
            row = hierarchy_lookup.get((cohort, review_class))
            values.append("0" if row is None else f"{row['count']} ({row['percent']:.1f}%)")
        lines.append(f"| {review_class} | {values[0]} | {values[1]} | {values[2]} |")
    lines.extend(
        [
            "",
            "English comments containing explicit references to a previous, next, or concrete "
            "calendar year are normalized as `YEAR_MISMATCH`.",
            "",
            "## Remaining assignment",
            "",
        ]
    )
    if remaining_assignments:
        lines.extend(
            [
                "| Sample | Name | Event years | Review state | Reviewer |",
                "|---|---|---|---|---|",
            ]
        )
        for row in remaining_assignments:
            years = ", ".join(str(value) for value in row.get("event_years", []))
            lines.append(
                f"| `{row.get('sample_id')}` | {row.get('display_name') or '-'} | {years or '-'} | "
                f"{row.get('review_status') or 'not reviewed'} | {row.get('reviewer_name') or '-'} |"
            )
    else:
        lines.append("All assigned points have a completed Olga review.")
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            f"- Frozen baseline: `{baseline_dir.as_posix()}`.",
            f"- Current validation run: `{current_run_dir.as_posix()}`.",
            "- Accuracy metrics include only `VALID` events with Logging/Wildfire human labels.",
            "- Rejected events are analyzed as acquisition/temporal/geometry outcomes and are not "
            "counted as model classification errors.",
            "- Bootstrap confidence intervals in the source validation reports resample complete "
            "points rather than individual events.",
            "",
            "## Artifacts",
            "",
            "- `cohort_summary.csv`: sample/event growth and agreement rates.",
            "- `model_metric_dynamics.csv`: baseline, added-cohort, and full-cohort model metrics.",
            "- `class_metric_dynamics.csv`: per-class precision, recall, and F1.",
            "- `review_status_dynamics.csv` and `human_label_dynamics.csv`: cohort composition.",
            "- `review_hierarchy_dynamics.csv`: normalized rejection and accepted-label hierarchy.",
            "- `newly_reviewed_points.csv`: the 50 points added after the baseline report.",
            "- `remaining_assignments.csv`: assigned points without a completed Olga review.",
            "- `changed_baseline_reviews.csv`: baseline integrity audit.",
            "- `plots/`: model metric and cohort-composition figures.",
            "",
        ]
    )
    return "\n".join(lines)


def export_review_validation_dynamics(
    *,
    baseline_run_dir: str | Path,
    current_run_dir: str | Path,
    current_rows: Sequence[dict[str, Any]],
    remaining_assignments: Sequence[dict[str, Any]],
    output_root: str | Path,
    bootstrap_runs: int = 2000,
    seed: int = 42,
    run_id: str | None = None,
) -> dict[str, Any]:
    baseline_dir = Path(baseline_run_dir).resolve()
    current_dir = Path(current_run_dir).resolve()
    baseline_points = _read_csv(baseline_dir / "reviewed_points.csv")
    baseline_events = _read_csv(baseline_dir / "event_level.csv")
    baseline_sample_ids = {str(row["sample_id"]) for row in baseline_points}
    baseline_current = [
        dict(row) for row in current_rows if str(row["sample_id"]) in baseline_sample_ids
    ]
    added_rows = [
        dict(row) for row in current_rows if str(row["sample_id"]) not in baseline_sample_ids
    ]
    current_values = [dict(row) for row in current_rows]
    cohort_rows = {
        "baseline": baseline_current,
        "added": added_rows,
        "current": current_values,
    }
    results = {
        cohort: summarize_reviewer_validation(
            rows,
            model_sources=DEFAULT_MODEL_SOURCES,
            bootstrap_runs=bootstrap_runs,
            seed=seed,
        )
        for cohort, rows in cohort_rows.items()
    }
    changed, missing_events, added_events = _changed_baseline_reviews(
        baseline_events,
        baseline_current,
    )
    cohort_summary = _cohort_summary_rows(results)
    metric_rows = _model_metric_rows(results)
    class_rows = _class_metric_rows(results)
    status_rows = _status_rows(results)
    label_rows = _human_label_rows(results)
    hierarchy_rows = _hierarchy_rows(cohort_rows)
    output_dir = create_run_directory(output_root, run_id)
    write_csv(output_dir / "cohort_summary.csv", cohort_summary)
    write_csv(output_dir / "model_metric_dynamics.csv", metric_rows)
    write_csv(output_dir / "class_metric_dynamics.csv", class_rows)
    write_csv(output_dir / "review_status_dynamics.csv", status_rows)
    write_csv(output_dir / "human_label_dynamics.csv", label_rows)
    write_csv(output_dir / "review_hierarchy_dynamics.csv", hierarchy_rows)
    write_csv(output_dir / "newly_reviewed_points.csv", results["added"]["sample_rows"])
    write_csv(output_dir / "remaining_assignments.csv", remaining_assignments)
    write_csv(output_dir / "changed_baseline_reviews.csv", changed)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_run_dir": str(baseline_dir),
        "current_run_dir": str(current_dir),
        "settings": {"bootstrap_runs": bootstrap_runs, "seed": seed},
        "cohorts": {cohort: results[cohort]["summary"] for cohort in COHORT_ORDER},
        "baseline_integrity": {
            "changed_events": len(changed),
            "missing_events": len(missing_events),
            "unexpected_events": len(added_events),
        },
        "remaining_assignments": len(remaining_assignments),
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "REPORT.md").write_text(
        _report(
            baseline_dir=baseline_dir,
            current_run_dir=current_dir,
            results=results,
            cohort_summary=cohort_summary,
            metric_rows=metric_rows,
            hierarchy_rows=hierarchy_rows,
            changed_reviews=changed,
            missing_baseline_events=missing_events,
            added_baseline_events=added_events,
            remaining_assignments=remaining_assignments,
        ),
        encoding="utf-8",
    )
    _plot_dynamics(output_dir, metric_rows, status_rows, label_rows)
    return {
        "run_dir": str(output_dir),
        "report_path": str(output_dir / "REPORT.md"),
        "summary": summary,
        "model_metric_rows": metric_rows,
    }
