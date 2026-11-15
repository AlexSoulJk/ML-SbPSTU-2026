from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, func, inspect, select, text
from sqlalchemy.orm import sessionmaker

from ResearchTool.backend.forest.models import EventReview

from .config import ExperimentConfig, PROJECT_ROOT, stable_json_hash
from .dataset import database_fingerprint
from .embeddings import compute_manifest_embeddings
from .event_links import plan_dominant_event_backfill, virtual_backfill_rows
from .experiments import aggregate_event_embeddings
from .reporting import write_csv, write_json
from .splits import assert_no_group_leakage, build_constrained_grouped_splits
from .stage_a_dataset import (
    build_current_event_manifest,
    load_frozen_remoteclip_legacy_cohort,
    load_frozen_xgboost_legacy_cohort,
)
from .stage_a_models import (
    build_xgboost_event_records,
    export_remoteclip_v2,
    export_xgboost_v2,
    predict_remoteclip_artifact,
    predict_xgboost_artifact,
    restrict_remoteclip_records,
    restrict_xgboost_records,
    run_cohort_size_experiment,
)


DEFAULT_STAGE_A_CONFIG = Path(__file__).resolve().parent / "configs" / "pipeline_v2_stage_a.json"
TASKS = {
    "T00": "baseline",
    "T01": "frozen_cohorts",
    "T02": "grouped_splits",
    "T03": "xgboost_v2",
    "T04": "remoteclip_v2",
    "T05": "four_artifact_smoke",
}


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _task_dir(output_root: Path, task_id: str) -> Path:
    path = output_root / "checkpoints" / f"{task_id}_{TASKS[task_id]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _project_path(value: str | Path) -> str:
    path = Path(value)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve(strict=False).relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _write_task_summary(
    directory: Path,
    *,
    task_id: str,
    title: str,
    result: str,
    tests: list[str],
    artifacts: list[str],
    code_changes: list[str],
    manual_check: list[str],
    limitations: list[str] | None = None,
) -> dict[str, Any]:
    portable_artifacts = [_project_path(item) for item in artifacts]
    payload = {
        "task_id": task_id,
        "title": title,
        "status": "PASS",
        "result": result,
        "tests": tests,
        "artifacts": portable_artifacts,
        "code_changes": code_changes,
        "manual_check": manual_check,
        "limitations": limitations or [],
        "completed_at": _now(),
    }
    write_json(directory / "checkpoint.json", payload)
    lines = [
        f"# {task_id}. {title}",
        "",
        "Status: **PASS**",
        "",
        "## Result",
        "",
        result,
        "",
        "## Self-checks",
        "",
        *[f"- {item}" for item in tests],
        "",
        "## Artifacts",
        "",
        *[f"- `{item}`" for item in portable_artifacts],
        "",
        "## Code / schema changes",
        "",
        *[f"- `{item}`" for item in code_changes],
        "",
        "## Manual validation",
        "",
        *[f"{index}. {item}" for index, item in enumerate(manual_check, start=1)],
    ]
    if limitations:
        lines.extend(["", "## Limitations", "", *[f"- {item}" for item in limitations]])
    directory.joinpath("SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def _reuse_compatible_artifact(
    manifest_path: Path,
    *,
    producer_name: str,
    producer_version: str,
    training_manifest_hash: str,
    split_manifest_hash: str,
    config_hash: str,
    database_sha256: str,
    checkpoint_path: Path | None = None,
) -> dict[str, Any] | None:
    """Reuse an immutable v2 artifact only when every identity field still matches."""

    if not manifest_path.exists():
        return None
    from .stage_a_models import file_sha256

    manifest = _load_json(manifest_path)
    checks = {
        "producer_name": manifest.get("producer_name") == producer_name,
        "producer_version": manifest.get("producer_version") == producer_version,
        "training_manifest_hash": (
            (manifest.get("training_cohort") or {}).get("manifest_hash")
            == training_manifest_hash
        ),
        "split_manifest_hash": manifest.get("split_manifest_hash") == split_manifest_hash,
        "config_hash": manifest.get("experiment_config_hash") == config_hash,
        "database_sha256": (
            (manifest.get("training_database_fingerprint") or {}).get("sha256")
            == database_sha256
        ),
    }
    artifact_path = _resolve(str(manifest.get("artifact_path") or ""))
    checks["artifact_exists"] = artifact_path.is_file()
    checks["artifact_sha256"] = bool(checks["artifact_exists"]) and (
        file_sha256(artifact_path) == manifest.get("artifact_sha256")
    )
    if checkpoint_path is not None:
        checks["checkpoint_sha256"] = (
            file_sha256(checkpoint_path)
            == (manifest.get("embedding") or {}).get("checkpoint_sha256")
        )
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise FileExistsError(
            f"Existing {producer_name}/{producer_version} artifact is incompatible: {failed}. "
            "Use a new artifact version or --overwrite-artifacts after review."
        )
    return manifest


def _table_inventory(engine) -> dict[str, int]:
    names = sorted(name for name in inspect(engine).get_table_names() if name.startswith("forest_"))
    with engine.connect() as connection:
        return {
            name: int(connection.execute(text(f'SELECT COUNT(*) FROM "{name}"')).scalar_one())
            for name in names
        }


def _artifact_inventory(paths: list[Path]) -> list[dict[str, Any]]:
    from .stage_a_models import file_sha256

    rows: list[dict[str, Any]] = []
    for manifest_path in paths:
        manifest = _load_json(manifest_path)
        artifact_path = _resolve(manifest["artifact_path"])
        rows.append(
            {
                "producer_name": manifest.get("producer_name"),
                "producer_version": manifest.get("producer_version"),
                "manifest_path": manifest_path.relative_to(PROJECT_ROOT).as_posix(),
                "manifest_sha256": file_sha256(manifest_path),
                "artifact_path": str(manifest.get("artifact_path") or ""),
                "artifact_exists": artifact_path.exists(),
                "artifact_sha256_expected": manifest.get("artifact_sha256"),
                "artifact_sha256_actual": file_sha256(artifact_path) if artifact_path.exists() else "",
            }
        )
    return rows


def _experiment_config(stage_config: dict[str, Any], profile: str, config_path: Path) -> ExperimentConfig:
    remoteclip = dict(stage_config["remoteclip"])
    paths = stage_config["paths"]
    data = {
        "version": 1,
        "paths": {
            "database": paths["database"],
            "output_root": paths["output_root"],
            "embedding_cache": paths["embedding_cache"],
            "checkpoint": paths["remoteclip_checkpoint"],
        },
        "embedding": {
            "model_name": remoteclip["model_name"],
            "device": remoteclip["device"],
            "cache_policy": remoteclip["cache_policy"],
            "preprocessing_version": remoteclip["preprocessing_version"],
            "normalization_version": remoteclip["normalization_version"],
        },
        "classifier": remoteclip["classifier"],
        "profiles": stage_config["profiles"],
    }
    return ExperimentConfig(path=config_path.resolve(), data=data, profile_name=profile)


def _flat_event_rows(event_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in event_rows:
        result.append(
            {
                key: value
                for key, value in row.items()
                if key not in {"inputs", "period_counts"}
            }
            | {
                "xgboost_pre": row["period_counts"]["xgboost"]["PRE"],
                "xgboost_post": row["period_counts"]["xgboost"]["POST"],
                "remoteclip_pre": row["period_counts"]["remoteclip"]["PRE"],
                "remoteclip_post": row["period_counts"]["remoteclip"]["POST"],
                "download_ids": json.dumps(
                    [item["download_id"] for item in row["inputs"]],
                    ensure_ascii=False,
                ),
            }
        )
    return result


def _fold_summary(saved_splits: dict[str, Any], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(row["record_id"]): row for row in records}
    rows = []
    for split in saved_splits["rows"]:
        train = [by_id[item] for item in split["train_record_ids"]]
        test = [by_id[item] for item in split["test_record_ids"]]
        rows.append(
            {
                "requested_seed": split.get("requested_seed", split["seed"]),
                "effective_seed": split.get("effective_seed", split["seed"]),
                "seed": split["seed"],
                "fold": split["fold"],
                "train_events": len(train),
                "train_samples": len({row["sample_id"] for row in train}),
                "train_logging": sum(row["label"] == "Logging" for row in train),
                "train_wildfire": sum(row["label"] == "Wildfire" for row in train),
                "test_events": len(test),
                "test_samples": len({row["sample_id"] for row in test}),
                "test_logging": sum(row["label"] == "Logging" for row in test),
                "test_wildfire": sum(row["label"] == "Wildfire" for row in test),
                "group_overlap": len(set(split["train_groups"]) & set(split["test_groups"])),
            }
        )
    return rows


def _plot_model_comparison(output_path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = ("logging_recall_mean", "balanced_accuracy_mean", "macro_f1_mean")
    labels = ("Recall Logging", "Balanced accuracy", "Macro F1")
    groups = [(row["model"], row["cohort_mode"]) for row in rows]
    x = np.arange(len(metrics))
    width = 0.8 / max(len(rows), 1)
    figure, axis = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    for index, row in enumerate(rows):
        values = [float(row[metric]) for metric in metrics]
        axis.bar(x + (index - (len(rows) - 1) / 2) * width, values, width, label=" / ".join(groups[index]))
    axis.set_xticks(x, labels)
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title("Pipeline v2 Stage A: cohort-size comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def _select_smoke_ids(records: list[dict[str, Any]], per_class: int) -> list[str]:
    ordered = sorted(records, key=lambda item: (item["sample_id"], item["event_id"]))
    logging = [row for row in ordered if row["label"] == "Logging"][:per_class]
    wildfire = [row for row in ordered if row["label"] == "Wildfire"][:per_class]
    return [str(row["event_id"]) for row in logging + wildfire]


def _stage_report(
    *,
    stage_dir: Path,
    tasks: list[dict[str, Any]],
    baseline: dict[str, Any],
    cohort_audit: dict[str, Any],
    split_hash: str,
    split_hashes: dict[str, str],
    model_summaries: list[dict[str, Any]],
    smoke_rows: list[dict[str, Any]],
    before_db: dict[str, Any],
    after_db: dict[str, Any],
    run_dir: Path,
) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    warning = cohort_audit["ground_truth_union_note"]
    lines = [
        "# Pipeline v2 — Stage A report",
        "",
        "Status: **PASS — ready for user validation**",
        "",
        "## Outcome",
        "",
        "The complete ML stage was executed read-only against the forest database. "
        "Two versioned v2 artifacts were trained and four-artifact inference was smoke-tested.",
        "",
        "## Task status",
        "",
        "| Task | Status | Result | Evidence |",
        "|---|---:|---|---|",
    ]
    for task in tasks:
        lines.append(
            f"| {task['task_id']} | {task['status']} | {task['result']} | "
            f"`{Path(task['artifacts'][0]).as_posix()}` |"
        )
    lines.extend(
        [
            "",
            "## Cohort audit",
            "",
            f"- Current human-VALID events: **{cohort_audit['current_valid_events']}**.",
            f"- Current human-VALID samples: **{cohort_audit['current_valid_samples']}**.",
            f"- Common PRE/POST-ready events: **{cohort_audit['common_ready_events']}**.",
            f"- Ground-truth sources: `{json.dumps(cohort_audit['ground_truth_source_counts'], ensure_ascii=False, sort_keys=True)}`.",
            f"- Event-download links used: `{json.dumps(cohort_audit['event_link_counts'], ensure_ascii=False, sort_keys=True)}`.",
            f"- Exact RemoteCLIP old events mapped/current: **{cohort_audit['legacy_current_valid_overlap']['remoteclip']}**; model-ready on the common input set: **{cohort_audit['remoteclip_old_events']}**.",
            f"- Exact XGBoost old events mapped/current: **{cohort_audit['legacy_current_valid_overlap']['xgboost']}**; model-ready on the common input set: **{cohort_audit['xgboost_old_events']}**.",
            f"- Frozen quick split hash: `{split_hashes['quick']}`.",
            f"- Frozen final split hash: `{split_hashes['final']}`.",
            f"- Selected split hash: `{split_hash}`.",
            f"- Database SHA before/after: `{before_db['sha256']}` / `{after_db['sha256']}`.",
            "",
            "> [!IMPORTANT]",
            f"> {warning}",
            "",
            "## Model metrics",
            "",
            "| Model | Cohort | Recall Logging | Balanced accuracy | Macro F1 | Fold runs |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in model_summaries:
        lines.append(
            f"| {row['model']} | {row['cohort_mode']} | "
            f"{float(row['logging_recall_mean']):.3f} | "
            f"{float(row['balanced_accuracy_mean']):.3f} | "
            f"{float(row['macro_f1_mean']):.3f} | {row['fold_runs']} |"
        )
    lines.extend(
        [
            "",
            "## Self-checks",
            "",
            "- No `sample_id` occurs in train and test within a grouped fold.",
            "- Every test fold contains both target classes.",
            "- OLD and FULL modes use the same frozen test events.",
            "- Legacy artifacts were not overwritten.",
            "- Both v2 artifacts passed reload and finite-probability smoke tests.",
            f"- Four-artifact smoke table contains **{len(smoke_rows)}** event rows.",
            "- Forest SQLite SHA-256 is unchanged.",
            "",
            "## User validation — 5 focused checks",
            "",
            "1. Open `Scripts/ResearchScripts/Outputs/pipeline_v2/checkpoints/T01_frozen_cohorts/full_cohort.csv` and verify several human labels/review timestamps; inspect `virtual_event_backfill_plan.json` to confirm the legacy mapping stayed read-only.",
            "2. Open `Scripts/ResearchScripts/Outputs/pipeline_v2/checkpoints/T02_grouped_splits/fold_summary.csv`; `group_overlap` must be zero and both test class counts must be positive.",
            "3. Compare OLD/FULL rows in `Scripts/ResearchScripts/Outputs/pipeline_v2/checkpoints/T03_xgboost_v2/summary_metrics.csv` and `Scripts/ResearchScripts/Outputs/pipeline_v2/checkpoints/T04_remoteclip_v2/summary_metrics.csv`.",
            "4. Inspect `Scripts/ResearchScripts/Outputs/pipeline_v2/checkpoints/T05_four_artifact_smoke/smoke_predictions.csv`; every event has four finite predictions.",
            "5. Inspect the v2 manifests under `Scripts/ResearchTool/Data/forest/models/*/v2/` and confirm v1 files are untouched.",
            "",
            "## Artifacts",
            "",
            f"- Run directory: `{run_dir.relative_to(PROJECT_ROOT).as_posix()}`.",
            f"- Plot: `{(run_dir / 'model_comparison.png').relative_to(PROJECT_ROOT).as_posix()}`.",
            f"- Machine-readable summary: `{(stage_dir / 'stage_summary.json').relative_to(PROJECT_ROOT).as_posix()}`.",
            "",
            "## Known limitation / refactoring candidate",
            "",
            "- `EventReviewRevision` does not exist before Stage B, therefore the frozen manifest stores a deterministic `review_snapshot_id` derived from the current review row and `updated_at`.",
            "- Older sample-level labels are inherited by linked events because event-level reviews did not exist for that historical cohort. Explicit event reviews always override that inherited label.",
        ]
    )
    stage_dir.joinpath("STAGE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        stage_dir / "stage_summary.json",
        {
            "stage": "A",
            "status": "PASS",
            "completed_at": _now(),
            "tasks": tasks,
            "baseline": baseline,
            "cohort_audit": cohort_audit,
            "split_manifest_hash": split_hash,
            "split_profile_hashes": split_hashes,
            "model_summaries": model_summaries,
            "smoke_rows": len(smoke_rows),
            "database_before": before_db,
            "database_after": after_db,
        },
    )


def run_stage_a(
    *,
    config_path: str | Path = DEFAULT_STAGE_A_CONFIG,
    profile: str = "final",
    overwrite_artifacts: bool = False,
) -> dict[str, Any]:
    config_path = _resolve(config_path).resolve()
    config = _load_json(config_path)
    if config.get("stage") != "A":
        raise ValueError(f"Expected Stage A config, got {config.get('stage')!r}")
    if profile not in config["profiles"]:
        raise KeyError(f"Unknown Stage A profile: {profile}")
    config_hash = stable_json_hash({"config": config, "profile": profile})
    output_root = _resolve(config["paths"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / "stage_a_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    database_path = _resolve(config["paths"]["database"]).resolve()
    engine = create_engine(
        f"sqlite:///{database_path.as_posix()}",
        connect_args={"check_same_thread": False},
        future=True,
    )
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    tasks: list[dict[str, Any]] = []

    # T00 — baseline inventory.
    t00 = _task_dir(output_root, "T00")
    before_db = database_fingerprint(database_path)
    table_counts = _table_inventory(engine)
    legacy_manifest_paths = [
        _resolve(config["paths"]["remoteclip_legacy_manifest"]),
        _resolve(config["paths"]["xgboost_legacy_manifest"]),
    ]
    artifact_inventory = _artifact_inventory(legacy_manifest_paths)
    with SessionLocal() as db:
        review_counts = dict(
            db.execute(
                select(EventReview.review_status, func.count())
                .group_by(EventReview.review_status)
                .order_by(EventReview.review_status)
            ).all()
        )
    baseline = {
        "database": before_db,
        "table_counts": table_counts,
        "review_status_counts": review_counts,
        "legacy_artifacts": artifact_inventory,
        "config_hash": config_hash,
        "profile": profile,
    }
    write_json(t00 / "baseline_inventory.json", baseline)
    write_csv(t00 / "legacy_artifacts.csv", artifact_inventory)
    tasks.append(
        _write_task_summary(
            t00,
            task_id="T00",
            title="Baseline current state",
            result="Frozen database, table-count and legacy-artifact inventory.",
            tests=[
                "Both legacy manifests and artifact files exist.",
                "Artifact hashes match their manifests.",
                "Database fingerprint was recorded before any model work.",
            ],
            artifacts=[str(t00 / "baseline_inventory.json"), str(t00 / "legacy_artifacts.csv")],
            code_changes=[
                "Scripts/ResearchScripts/ml_tests/stage_a.py",
                "Scripts/ResearchScripts/ml_tests/configs/pipeline_v2_stage_a.json",
            ],
            manual_check=[
                "Open baseline_inventory.json and verify review/table counts and both v1 hashes.",
                "Confirm the database SHA is the expected working database before continuing.",
            ],
        )
    )
    if not all(
        row["artifact_exists"]
        and row["artifact_sha256_expected"] == row["artifact_sha256_actual"]
        for row in artifact_inventory
    ):
        raise RuntimeError("Legacy artifact inventory failed hash verification")

    # T01 — immutable event manifest and cohort reconstruction.
    t01 = _task_dir(output_root, "T01")
    remoteclip_legacy_manifest = _load_json(legacy_manifest_paths[0])
    xgboost_legacy_manifest = _load_json(legacy_manifest_paths[1])
    legacy_cohort_provenance = _load_json(
        _resolve(config["paths"]["legacy_cohort_provenance"])
    )
    with SessionLocal() as db:
        if bool(
            config.get("manifest", {}).get(
                "include_virtual_dominant_backfill",
                True,
            )
        ):
            backfill_plan = plan_dominant_event_backfill(db)
            virtual_links = virtual_backfill_rows(backfill_plan)
        else:
            backfill_plan = {"dry_run": True, "counts": {}, "rows": []}
            virtual_links = []
        event_rows, image_rows, manifest_audit = build_current_event_manifest(
            db,
            database_fingerprint=before_db,
            target_classes=config["target_classes"],
            virtual_links=virtual_links,
        )
        rc_legacy = load_frozen_remoteclip_legacy_cohort(
            db,
            cohort_path=_resolve(config["paths"]["remoteclip_legacy_cohort"]),
            legacy_manifest=remoteclip_legacy_manifest,
            provenance=legacy_cohort_provenance,
        )
        xgb_legacy = load_frozen_xgboost_legacy_cohort(
            db,
            cohort_path=_resolve(config["paths"]["xgboost_legacy_cohort"]),
            legacy_manifest=xgboost_legacy_manifest,
            provenance=legacy_cohort_provenance,
        )
    current_valid_ids = {str(row["event_id"]) for row in event_rows}
    rc_legacy_current = {
        str(row["current_event_id"])
        for row in rc_legacy["rows"]
        if row["mapping_status"] == "EXACT"
    }
    xgb_legacy_current = {
        str(row["current_event_id"])
        for row in xgb_legacy["rows"]
        if row["mapping_status"] == "EXACT"
    }
    legacy_overlap = {
        "remoteclip": len(rc_legacy_current & current_valid_ids),
        "xgboost": len(xgb_legacy_current & current_valid_ids),
    }
    if not rc_legacy["hash_matches_artifact"] or not xgb_legacy["hash_matches_artifact"]:
        raise RuntimeError("Legacy training cohort reconstruction does not match artifact manifest")
    write_json(
        t01 / "virtual_event_backfill_plan.json",
        {
            "dry_run": True,
            "counts": backfill_plan.get("counts", {}),
            "used_by_manifest": manifest_audit.get("virtual_dominant_backfill_links", 0),
        },
    )
    write_csv(t01 / "virtual_event_backfill_plan.csv", backfill_plan.get("rows", []))
    write_json(t01 / "event_manifest.json", {"audit": manifest_audit, "events": event_rows})
    write_csv(t01 / "full_cohort.csv", _flat_event_rows(event_rows))
    write_csv(t01 / "remoteclip_image_manifest.csv", image_rows)
    write_json(t01 / "remoteclip_legacy_cohort.json", rc_legacy)
    write_csv(t01 / "remoteclip_legacy_cohort.csv", rc_legacy["rows"])
    write_json(t01 / "xgboost_legacy_cohort.json", xgb_legacy)
    write_csv(t01 / "xgboost_legacy_cohort.csv", xgb_legacy["rows"])
    # Feature readiness is part of the immutable model-specific cohort snapshot.
    experiment_config = _experiment_config(config, profile, config_path)
    embeddings, embedding_report = compute_manifest_embeddings(image_rows, experiment_config)
    write_json(
        t01 / "embedding_cache_report.json",
        {key: value for key, value in embedding_report.items() if key != "rows"},
    )
    write_csv(t01 / "embedding_cache_rows.csv", embedding_report["rows"])
    rc_event_records = aggregate_event_embeddings(image_rows, embeddings)
    rc_ready_ids = {
        str(row["event_id"])
        for row in rc_event_records
        if {"PRE", "POST"} <= set(row.get("vectors", {}))
    }
    feature_names = [str(value) for value in config["xgboost"]["feature_names"]]
    xgb_records_all, xgb_feature_audit = build_xgboost_event_records(
        event_rows,
        feature_names=feature_names,
    )
    xgb_ready_ids = {str(row["event_id"]) for row in xgb_records_all}
    common_ids = rc_ready_ids & xgb_ready_ids
    if not common_ids:
        raise RuntimeError("No common human-VALID events are ready for both models")
    remoteclip_records = restrict_remoteclip_records(
        rc_event_records,
        common_ids,
        feature_variant=str(config["remoteclip"]["feature_variant"]),
    )
    xgboost_records = restrict_xgboost_records(xgb_records_all, common_ids)
    remoteclip_old_ids = rc_legacy_current & common_ids
    xgboost_old_ids = xgb_legacy_current & common_ids
    cohort_audit = {
        "current_valid_events": len(event_rows),
        "current_valid_samples": len({row["sample_id"] for row in event_rows}),
        "current_class_counts": dict(sorted(Counter(row["human_label"] for row in event_rows).items())),
        "ground_truth_source_counts": manifest_audit["ground_truth_source_counts"],
        "ground_truth_source_sample_counts": manifest_audit[
            "ground_truth_source_sample_counts"
        ],
        "event_link_counts": {
            "explicit": manifest_audit["explicit_event_download_links"],
            "virtual_dominant_backfill": manifest_audit[
                "virtual_dominant_backfill_links"
            ],
        },
        "virtual_backfill_plan_counts": backfill_plan.get("counts", {}),
        "remoteclip_ready_events": len(rc_ready_ids),
        "xgboost_ready_events": len(xgb_ready_ids),
        "common_ready_events": len(common_ids),
        "common_class_counts": dict(
            sorted(Counter(row["label"] for row in remoteclip_records).items())
        ),
        "remoteclip_old_events": len(remoteclip_old_ids),
        "remoteclip_old_class_counts": dict(
            sorted(
                Counter(
                    row["label"]
                    for row in remoteclip_records
                    if row["event_id"] in remoteclip_old_ids
                ).items()
            )
        ),
        "xgboost_old_events": len(xgboost_old_ids),
        "xgboost_old_class_counts": dict(
            sorted(
                Counter(
                    row["label"]
                    for row in xgboost_records
                    if row["event_id"] in xgboost_old_ids
                ).items()
            )
        ),
        "legacy_current_valid_overlap": legacy_overlap,
        "manifest_hash": manifest_audit["manifest_hash"],
        "qc_snapshot_hash": manifest_audit["qc_snapshot_hash"],
        "remoteclip_embedding_cache": embedding_report["counts"],
        "xgboost_feature_audit": {
            key: value for key, value in xgb_feature_audit.items() if key != "failure_rows"
        },
        "ground_truth_union_note": (
            "FULL_COHORT combines authoritative event-level VALID reviews with older "
            "sample-level ManualValidation=Valid labels. An explicit EventReview always "
            "takes precedence, including exclusion of DRAFT/REJECTED events from the legacy source."
        ),
    }
    write_json(t01 / "cohort_audit.json", cohort_audit)
    write_json(
        t01 / "model_ready_cohorts.json",
        {
            "full_event_ids": sorted(common_ids),
            "full_event_ids_sha256": stable_json_hash(sorted(common_ids)),
            "remoteclip_old_event_ids": sorted(remoteclip_old_ids),
            "remoteclip_old_event_ids_sha256": stable_json_hash(sorted(remoteclip_old_ids)),
            "xgboost_old_event_ids": sorted(xgboost_old_ids),
            "xgboost_old_event_ids_sha256": stable_json_hash(sorted(xgboost_old_ids)),
        },
    )
    for model_name, records, old_ids in (
        ("RemoteCLIP", remoteclip_records, remoteclip_old_ids),
        ("XGBoost", xgboost_records, xgboost_old_ids),
    ):
        if len(old_ids) < 2 or len(
            {row["label"] for row in records if row["event_id"] in old_ids}
        ) < 2:
            raise RuntimeError(f"{model_name} exact legacy cohort is too small or one-class")
    tasks.append(
        _write_task_summary(
            t01,
            task_id="T01",
            title="Event-level manifest and frozen cohorts",
            result=(
                f"Frozen {len(event_rows)} human-VALID events; {len(common_ids)} are ready for "
                f"both models; their model-ready exact-old subsets contain "
                f"{len(remoteclip_old_ids)} RemoteCLIP and {len(xgboost_old_ids)} "
                f"XGBoost events."
            ),
            tests=[
                "FULL ground truth contains both historical sample-level VALID labels and newer VALID event reviews.",
                "An explicit EventReview is authoritative for its event; DRAFT/REJECTED events are not inherited from the sample label.",
                "Legacy RemoteCLIP record-id hash matches its v1 manifest.",
                "Legacy XGBoost dataset hash matches its v1 manifest.",
                "Current and model-specific exact legacy cohort hashes were persisted.",
                "Legacy unlinked downloads were assigned virtually to the single dominant event without modifying SQLite.",
            ],
            artifacts=[
                str(t01 / "full_cohort.csv"),
                str(t01 / "cohort_audit.json"),
                str(t01 / "model_ready_cohorts.json"),
                str(t01 / "virtual_event_backfill_plan.json"),
            ],
            code_changes=[
                "Scripts/ResearchScripts/ml_tests/stage_a_dataset.py",
                "Scripts/ResearchScripts/ml_tests/stage_a_models.py",
                "Scripts/ResearchScripts/ml_tests/pipeline_v2_stage_a.ipynb",
            ],
            manual_check=[
                "Open pipeline_v2_stage_a.ipynb with RUN_STAGE_A=False.",
                "Inspect the cohort audit, excluded/not-ready events, duplicate audit and OLD/FULL membership tables.",
            ],
            limitations=[cohort_audit["ground_truth_union_note"]],
        )
    )

    # T02 — one frozen split manifest reused by both models and both cohort modes.
    t02 = _task_dir(output_root, "T02")
    splits_by_profile: dict[str, dict[str, Any]] = {}
    split_hashes: dict[str, str] = {}
    for profile_name in ("quick", "final"):
        profile_config = config["profiles"][profile_name]
        profile_splits = build_constrained_grouped_splits(
            remoteclip_records,
            old_record_ids=remoteclip_old_ids,
            required_old_record_id_sets={
                "remoteclip": remoteclip_old_ids,
                "xgboost": xgboost_old_ids,
            },
            requested_n_splits=int(profile_config["n_splits"]),
            seeds=[int(value) for value in profile_config["seeds"]],
        )
        assert_no_group_leakage(profile_splits)
        profile_summary = _fold_summary(profile_splits, remoteclip_records)
        if any(row["group_overlap"] for row in profile_summary):
            raise AssertionError(f"Grouped split leakage detected in {profile_name}")
        if any(
            row["test_logging"] == 0 or row["test_wildfire"] == 0
            for row in profile_summary
        ):
            raise RuntimeError(f"At least one {profile_name} test fold lacks a target class")
        splits_by_profile[profile_name] = profile_splits
        split_hashes[profile_name] = stable_json_hash(profile_splits)
        write_json(t02 / f"group_splits_{profile_name}.json", profile_splits)
        write_csv(t02 / f"fold_summary_{profile_name}.csv", profile_summary)

    saved_splits = splits_by_profile[profile]
    assert_no_group_leakage(saved_splits)
    fold_summary = _fold_summary(saved_splits, remoteclip_records)
    if any(row["group_overlap"] for row in fold_summary):
        raise AssertionError("Grouped split leakage detected")
    if any(row["test_logging"] == 0 or row["test_wildfire"] == 0 for row in fold_summary):
        raise RuntimeError("At least one grouped test fold lacks a target class")
    split_hash = stable_json_hash(saved_splits)
    write_json(t02 / "group_splits.json", saved_splits)
    write_json(
        t02 / "split_manifest.json",
        {
            "selected_profile": profile,
            "split_manifest_hash": split_hash,
            "profiles": {
                name: {
                    "split_manifest_hash": split_hashes[name],
                    **config["profiles"][name],
                }
                for name in ("quick", "final")
            },
        },
    )
    write_csv(t02 / "fold_summary.csv", fold_summary)
    tasks.append(
        _write_task_summary(
            t02,
            task_id="T02",
            title="Saved grouped outer splits",
            result=f"Saved {len(fold_summary)} grouped folds with hash {split_hash}.",
            tests=[
                "assert_no_group_leakage passed.",
                "Every test fold contains Logging and Wildfire.",
                "Both model experiments consume this exact split manifest.",
                "Quick and final profile manifests were both persisted.",
            ],
            artifacts=[
                str(t02 / "group_splits_final.json"),
                str(t02 / "group_splits_quick.json"),
                str(t02 / "fold_summary.csv"),
            ],
            code_changes=[
                "Scripts/ResearchScripts/ml_tests/splits.py",
                "Scripts/ResearchScripts/ml_tests/stage_a.py",
            ],
            manual_check=[
                "Open fold_summary_quick.csv and fold_summary_final.csv.",
                "Verify group_overlap=0 and positive Logging/Wildfire counts in every test fold.",
                "Re-run with the same config and confirm the hashes in split_manifest.json do not change.",
            ],
        )
    )

    # T03 — XGBoost old/full CV and v2 artifact.
    t03 = _task_dir(output_root, "T03")
    xgb_result = run_cohort_size_experiment(
        xgboost_records,
        old_event_ids=xgboost_old_ids,
        saved_splits=saved_splits,
        model_name="XGBOOST",
        xgboost_feature_names=feature_names,
        xgboost_parameters=dict(config["xgboost"]["parameters"]),
    )
    write_csv(t03 / "fold_metrics.csv", xgb_result["fold_metrics"])
    write_csv(t03 / "summary_metrics.csv", xgb_result["summary_metrics"])
    xgb_version = str(config["artifact_versions"]["xgboost"])
    xgb_manifest_path = (
        PROJECT_ROOT
        / "Scripts"
        / "ResearchTool"
        / "Data"
        / "forest"
        / "models"
        / "xgboost"
        / xgb_version
        / "manifest.json"
    )
    xgb_v2_manifest = None if overwrite_artifacts else _reuse_compatible_artifact(
        xgb_manifest_path,
        producer_name="XGBOOST",
        producer_version=xgb_version,
        training_manifest_hash=manifest_audit["manifest_hash"],
        split_manifest_hash=split_hash,
        config_hash=config_hash,
        database_sha256=before_db["sha256"],
    )
    if xgb_v2_manifest is None:
        xgb_v2_manifest = export_xgboost_v2(
            xgboost_records,
            version=xgb_version,
            feature_names=feature_names,
            parameters=dict(config["xgboost"]["parameters"]),
            feature_schema_version=str(config["xgboost"]["feature_schema_version"]),
            training_manifest_hash=manifest_audit["manifest_hash"],
            split_manifest_hash=split_hash,
            config_hash=config_hash,
            database_fingerprint=before_db,
            metrics=xgb_result,
            overwrite=overwrite_artifacts,
        )
    write_json(t03 / "artifact_manifest_snapshot.json", xgb_v2_manifest)
    tasks.append(
        _write_task_summary(
            t03,
            task_id="T03",
            title="XGBoost cohort-size experiment and v2 artifact",
            result="Completed OLD/FULL CV on frozen folds and exported XGBoost v2.",
            tests=[
                "All OLD/FULL fold runs completed.",
                "No test event appears in OLD train.",
                "Reloaded v2 artifact returned finite probabilities.",
                "v1 artifact hash remained unchanged.",
            ],
            artifacts=[
                str(t03 / "summary_metrics.csv"),
                str(t03 / "fold_metrics.csv"),
                str(_resolve(xgb_v2_manifest["artifact_path"])),
            ],
            code_changes=[
                "Scripts/ResearchScripts/ml_tests/stage_a_models.py",
                "Scripts/ResearchScripts/ml_tests/stage_a.py",
            ],
            manual_check=[
                "Compare OLD/FULL rows in summary_metrics.csv and inspect fold_metrics.csv.",
                "Open the XGBoost v2 manifest and verify cohort, split and artifact hashes.",
            ],
        )
    )

    # T04 — RemoteCLIP old/full CV and v2 artifact.
    t04 = _task_dir(output_root, "T04")
    rc_result = run_cohort_size_experiment(
        remoteclip_records,
        old_event_ids=remoteclip_old_ids,
        saved_splits=saved_splits,
        model_name="REMOTECLIP_LOGREG",
        remoteclip_classifier_config=dict(config["remoteclip"]["classifier"]),
    )
    write_csv(t04 / "fold_metrics.csv", rc_result["fold_metrics"])
    write_csv(t04 / "summary_metrics.csv", rc_result["summary_metrics"])
    rc_version = str(config["artifact_versions"]["remoteclip"])
    rc_checkpoint = _resolve(config["paths"]["remoteclip_checkpoint"])
    rc_manifest_path = (
        PROJECT_ROOT
        / "Scripts"
        / "ResearchTool"
        / "Data"
        / "forest"
        / "models"
        / "remoteclip_logreg"
        / rc_version
        / "manifest.json"
    )
    rc_v2_manifest = None if overwrite_artifacts else _reuse_compatible_artifact(
        rc_manifest_path,
        producer_name="REMOTECLIP_LOGREG",
        producer_version=rc_version,
        training_manifest_hash=manifest_audit["manifest_hash"],
        split_manifest_hash=split_hash,
        config_hash=config_hash,
        database_sha256=before_db["sha256"],
        checkpoint_path=rc_checkpoint,
    )
    if rc_v2_manifest is None:
        rc_v2_manifest = export_remoteclip_v2(
            remoteclip_records,
            version=rc_version,
            feature_variant=str(config["remoteclip"]["feature_variant"]),
            classifier_config=dict(config["remoteclip"]["classifier"]),
            checkpoint_path=rc_checkpoint,
            embedding_config=dict(config["remoteclip"]),
            training_manifest_hash=manifest_audit["manifest_hash"],
            split_manifest_hash=split_hash,
            config_hash=config_hash,
            database_fingerprint=before_db,
            metrics=rc_result,
            overwrite=overwrite_artifacts,
        )
    write_json(t04 / "artifact_manifest_snapshot.json", rc_v2_manifest)
    tasks.append(
        _write_task_summary(
            t04,
            task_id="T04",
            title="RemoteCLIP cohort-size experiment and v2 artifact",
            result="Completed OLD/FULL CV using cached embeddings and exported RemoteCLIP-LogReg v2.",
            tests=[
                "All OLD/FULL fold runs completed.",
                "RemoteCLIP encoder was not fine-tuned.",
                "Embedding cache usage was recorded.",
                "Reloaded v2 artifact returned a finite probability.",
            ],
            artifacts=[
                str(t04 / "summary_metrics.csv"),
                str(t04 / "fold_metrics.csv"),
                str(_resolve(rc_v2_manifest["artifact_path"])),
            ],
            code_changes=[
                "Scripts/ResearchScripts/ml_tests/stage_a_models.py",
                "Scripts/ResearchScripts/ml_tests/stage_a.py",
            ],
            manual_check=[
                "Compare OLD/FULL rows in summary_metrics.csv and inspect embedding_cache_report.json.",
                "Open the RemoteCLIP v2 manifest and verify cohort, checkpoint and artifact hashes.",
            ],
        )
    )

    # T05 — deterministic four-artifact smoke prediction table.
    t05 = _task_dir(output_root, "T05")
    smoke_ids = _select_smoke_ids(
        remoteclip_records,
        int(config.get("smoke_events_per_class") or 5),
    )
    rc_by_id = {str(row["event_id"]): row for row in remoteclip_records}
    xgb_by_id = {str(row["event_id"]): row for row in xgboost_records}
    manifests = {
        "remoteclip_legacy_v1": remoteclip_legacy_manifest,
        "remoteclip_v2": rc_v2_manifest,
        "xgboost_legacy_v1": xgboost_legacy_manifest,
        "xgboost_v2": xgb_v2_manifest,
    }
    smoke_rows: list[dict[str, Any]] = []
    for event_id in smoke_ids:
        rc_record = rc_by_id[event_id]
        xgb_record = xgb_by_id[event_id]
        predictions = {
            "remoteclip_legacy_v1": predict_remoteclip_artifact(
                manifests["remoteclip_legacy_v1"], rc_record
            ),
            "remoteclip_v2": predict_remoteclip_artifact(manifests["remoteclip_v2"], rc_record),
            "xgboost_legacy_v1": predict_xgboost_artifact(
                manifests["xgboost_legacy_v1"], xgb_record
            ),
            "xgboost_v2": predict_xgboost_artifact(manifests["xgboost_v2"], xgb_record),
        }
        row: dict[str, Any] = {
            "event_id": event_id,
            "sample_id": rc_record["sample_id"],
            "event_year": rc_record["event_year"],
            "human_label": rc_record["label"],
        }
        for name, prediction in predictions.items():
            probability_logging = float(prediction["probability_logging"])
            probability_wildfire = float(prediction["probability_wildfire"])
            if not (
                math.isfinite(probability_logging)
                and math.isfinite(probability_wildfire)
                and abs(probability_logging + probability_wildfire - 1.0) <= 1e-6
                and prediction["prediction"] in {"Logging", "Wildfire"}
            ):
                raise AssertionError(f"Invalid smoke prediction from {name}: {prediction}")
            row[f"{name}_prediction"] = prediction["prediction"]
            row[f"{name}_probability_logging"] = probability_logging
            row[f"{name}_probability_wildfire"] = probability_wildfire
        smoke_rows.append(row)
    expected_columns = 4 * 3
    if len(smoke_rows) != len(smoke_ids) or any(
        sum(key.endswith(("_prediction", "_probability_logging", "_probability_wildfire")) for key in row)
        != expected_columns
        for row in smoke_rows
    ):
        raise AssertionError("Four-artifact smoke table is incomplete")
    write_csv(t05 / "smoke_predictions.csv", smoke_rows)
    write_json(
        t05 / "smoke_summary.json",
        {
            "events": len(smoke_rows),
            "artifacts": {
                name: {
                    "producer_name": manifest.get("producer_name"),
                    "producer_version": manifest.get("producer_version"),
                    "artifact_sha256": manifest.get("artifact_sha256"),
                }
                for name, manifest in manifests.items()
            },
        },
    )
    tasks.append(
        _write_task_summary(
            t05,
            task_id="T05",
            title="Four-artifact inference smoke-test",
            result=f"Produced four independent predictions for {len(smoke_rows)} frozen events.",
            tests=[
                "Each event has legacy_v1 and v2 predictions for both model families.",
                "All probabilities are finite and sum to one per artifact.",
                "Production labels were not written to SQLite.",
            ],
            artifacts=[str(t05 / "smoke_predictions.csv"), str(t05 / "smoke_summary.json")],
            code_changes=[
                "Scripts/ResearchScripts/ml_tests/stage_a.py",
                "Scripts/ResearchScripts/ml_tests/stage_a_models.py",
            ],
            manual_check=[
                "Open smoke_predictions.csv and verify four model outputs per event.",
                "Confirm probabilities are finite, sum to one and no production labels were written.",
            ],
        )
    )

    model_summaries = xgb_result["summary_metrics"] + rc_result["summary_metrics"]
    write_csv(run_dir / "model_summary.csv", model_summaries)
    _plot_model_comparison(run_dir / "model_comparison.png", model_summaries)
    after_db = database_fingerprint(database_path)
    if before_db["sha256"] != after_db["sha256"]:
        raise RuntimeError("Forest database changed during read-only Stage A")
    after_legacy_artifacts = _artifact_inventory(legacy_manifest_paths)
    before_legacy_hashes = {
        row["producer_name"]: row["artifact_sha256_actual"] for row in artifact_inventory
    }
    after_legacy_hashes = {
        row["producer_name"]: row["artifact_sha256_actual"]
        for row in after_legacy_artifacts
    }
    if before_legacy_hashes != after_legacy_hashes:
        raise RuntimeError("A legacy v1 artifact changed during Stage A")
    stage_dir = output_root / "stages" / "A"
    _stage_report(
        stage_dir=stage_dir,
        tasks=tasks,
        baseline=baseline,
        cohort_audit=cohort_audit,
        split_hash=split_hash,
        split_hashes=split_hashes,
        model_summaries=model_summaries,
        smoke_rows=smoke_rows,
        before_db=before_db,
        after_db=after_db,
        run_dir=run_dir,
    )
    summary = {
        "stage": "A",
        "status": "PASS",
        "profile": profile,
        "run_id": run_id,
        "run_dir": run_dir.relative_to(PROJECT_ROOT).as_posix(),
        "stage_report": (stage_dir / "STAGE_REPORT.md").relative_to(PROJECT_ROOT).as_posix(),
        "tasks": tasks,
        "cohort_audit": cohort_audit,
        "split_manifest_hash": split_hash,
        "split_profile_hashes": split_hashes,
        "model_summaries": model_summaries,
        "database_unchanged": True,
    }
    write_json(run_dir / "run_summary.json", summary)
    return summary

