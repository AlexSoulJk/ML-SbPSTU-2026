from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .artifacts import export_remoteclip_classifier, plan_remoteclip_export
from .config import ExperimentConfig, PROJECT_ROOT
from .dataset import build_event_manifest, database_fingerprint
from .embeddings import ImageEmbedder, compute_manifest_embeddings
from .event_links import (
    apply_dominant_event_backfill,
    backup_sqlite_database,
    plan_dominant_event_backfill,
    virtual_backfill_rows,
)
from .experiments import (
    aggregate_event_embeddings,
    build_nomanual_review_queue,
    run_baseline_experiments,
)
from .reporting import (
    create_run_directory,
    plot_baseline_metrics,
    read_csv,
    write_csv,
    write_json,
)
from .latent_diagnostics import (
    diagnostics_report_markdown,
    diagnostic_figures,
    run_latent_diagnostics as compute_latent_diagnostics,
)
from .sample_evaluation import (
    evaluation_figures,
    run_valid_post_clustering,
    serializable_result,
)


class ForestMLPipeline:
    def __init__(self, config: ExperimentConfig, *, run_id: str | None = None):
        self.config = config
        self.database_path = config.path_value("database").resolve()
        self.output_root = config.path_value("output_root").resolve()
        self.run_dir = create_run_directory(self.output_root, run_id)
        self.engine = create_engine(
            f"sqlite:///{self.database_path.as_posix()}",
            connect_args={"check_same_thread": False},
            future=True,
        )
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autoflush=False,
            autocommit=False,
            future=True,
        )
        self.database_snapshot = database_fingerprint(self.database_path)
        self.summary: dict[str, Any] = {
            "status": "initialized",
            "run_dir": self._display_path(self.run_dir),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "profile": config.profile_name,
        }
        self._write_config_snapshot()

    def _display_path(self, path: str | Path) -> str:
        resolved = Path(path).resolve(strict=False)
        try:
            return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
        except ValueError:
            return str(resolved)

    def _write_config_snapshot(self) -> None:
        payload = {
            "source_config": self._display_path(self.config.path),
            "config_fingerprint": self.config.fingerprint,
            "profile": self.config.profile_name,
            "profile_settings": self.config.profile,
            "config": self.config.data,
            "database_fingerprint": self.database_snapshot,
            "label_source": "Sample.driver_primary",
            "label_semantics": "WRI weak label; MANUAL_VALID confirms event existence only",
            "event_assignment_policy": "explicit plus optional legacy dominant virtual/apply backfill",
            "sample_weight_policy": "1 / eligible_events_in_sample per feature variant",
        }
        write_json(self.run_dir / "config.json", payload)

    def session(self) -> Session:
        return self.SessionLocal()

    def plan_event_backfill(self) -> dict[str, Any]:
        with self.session() as db:
            plan = plan_dominant_event_backfill(db)
        write_csv(self.run_dir / "event_backfill_plan.csv", plan["rows"])
        write_json(
            self.run_dir / "event_backfill_plan.json",
            {key: value for key, value in plan.items() if key != "rows"},
        )
        self.summary["event_backfill_plan"] = plan["counts"]
        self._save_summary()
        return plan

    def apply_event_backfill(self, plan: dict[str, Any]) -> dict[str, Any]:
        backup = backup_sqlite_database(
            self.database_path,
            backup_dir=self.run_dir / "database_backups",
        )
        try:
            with self.session() as db:
                result = apply_dominant_event_backfill(db, plan.get("rows", []))
                db.commit()
        except Exception:
            self.summary["event_backfill_apply"] = "failed"
            self.summary["database_backup"] = self._display_path(backup)
            self._save_summary()
            raise
        result["database_backup"] = self._display_path(backup)
        write_csv(self.run_dir / "event_backfill_report.csv", result["rows"])
        write_json(
            self.run_dir / "event_backfill_report.json",
            {key: value for key, value in result.items() if key != "rows"},
        )
        self.summary["event_backfill_apply"] = result["counts"]
        self.summary["database_backup"] = self._display_path(backup)
        self.database_snapshot = database_fingerprint(self.database_path)
        self._write_config_snapshot()
        self._save_summary()
        return result

    def build_manifest(
        self,
        *,
        backfill_plan: dict[str, Any] | None = None,
        include_hansen_mining: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        include_virtual = bool(
            self.config.data.get("dataset", {}).get("include_virtual_backfill", True)
        )
        virtual = virtual_backfill_rows(backfill_plan or {}) if include_virtual else []
        dataset_config = dict(self.config.data.get("dataset", {}))
        dataset_config["include_hansen_mining"] = bool(include_hansen_mining)
        with self.session() as db:
            manifest, audit = build_event_manifest(
                db,
                dataset_config=dataset_config,
                virtual_links=virtual,
            )
        audit["database_fingerprint"] = database_fingerprint(self.database_path)
        audit["virtual_backfill_links"] = len(virtual)
        audit["include_hansen_mining"] = bool(include_hansen_mining)
        write_csv(self.run_dir / "dataset_manifest.csv", manifest)
        write_json(self.run_dir / "dataset_audit.json", audit)
        cohort_rows = [
            {
                "validation_cohort": cohort,
                "eligible_samples": audit.get("sample_counts_by_cohort", {}).get(cohort, 0),
                "eligible_events": audit.get("event_counts_by_cohort", {}).get(cohort, 0),
                "eligible_images": count,
            }
            for cohort, count in audit.get("cohort_counts", {}).items()
        ]
        write_csv(self.run_dir / "cohort_distribution.csv", cohort_rows)
        self.summary["dataset"] = {
            "rows": audit["rows"],
            "eligible_rows": audit["eligible_rows"],
            "eligible_samples": audit["eligible_samples"],
            "eligible_events": audit["eligible_events"],
        }
        self._save_summary()
        return manifest, audit

    def compute_embeddings(
        self,
        manifest: list[dict[str, Any]],
        *,
        embedder: ImageEmbedder | None = None,
        show_progress: bool = True,
    ):
        progress_every = int(self.config.data.get("embedding", {}).get("progress_every", 25))

        def report_progress(done: int, total: int, counts: dict[str, int]) -> None:
            if show_progress and (done == total or done == 1 or done % progress_every == 0):
                print(f"Embeddings {done}/{total}: {counts}", flush=True)

        embeddings, report = compute_manifest_embeddings(
            manifest,
            self.config,
            embedder=embedder,
            progress=report_progress,
        )
        write_csv(self.run_dir / "embedding_cache_rows.csv", report["rows"])
        write_json(
            self.run_dir / "embedding_cache_report.json",
            {key: value for key, value in report.items() if key != "rows"},
        )
        self.summary["embedding_cache"] = report["counts"]
        self._save_summary()
        return embeddings, report

    def run_baseline(
        self,
        manifest: list[dict[str, Any]],
        embeddings,
    ) -> dict[str, Any]:
        event_records = aggregate_event_embeddings(manifest, embeddings)
        result = run_baseline_experiments(event_records, self.config)
        queue = build_nomanual_review_queue(event_records, self.config)
        write_csv(self.run_dir / "baseline_metrics.csv", result["baseline_metrics"])
        write_csv(self.run_dir / "fold_metrics.csv", result["fold_metrics"])
        write_json(self.run_dir / "group_splits.json", result["saved_splits"])
        write_csv(self.run_dir / "nomanual_review_queue.csv", queue)
        figure = plot_baseline_metrics(result["baseline_metrics"])
        figure.savefig(self.run_dir / "plots" / "baseline_metrics.png", dpi=160)
        import matplotlib.pyplot as plt

        plt.close(figure)
        self.summary["baseline"] = {
            "event_records": len(event_records),
            "metric_rows": len(result["baseline_metrics"]),
            "fold_rows": len(result["fold_metrics"]),
            "nomanual_review_rows": len(queue),
        }
        self.summary["status"] = "completed"
        self._save_summary()
        return {**result, "nomanual_review_queue": queue, "event_records": event_records}

    def run_valid_post_sample_evaluation(
        self,
        manifest: list[dict[str, Any]],
        embeddings,
        *,
        pca_variance: float = 0.95,
        k_min: int = 2,
        k_max: int = 10,
        stability_runs: int = 20,
        stability_sample_fraction: float = 0.80,
        random_state: int = 42,
    ) -> dict[str, Any]:
        if not embeddings:
            raise ValueError("Sample evaluation requires cached or computed embeddings.")
        result = run_valid_post_clustering(
            manifest,
            embeddings,
            pca_variance=pca_variance,
            k_min=k_min,
            k_max=k_max,
            stability_runs=stability_runs,
            stability_sample_fraction=stability_sample_fraction,
            random_state=random_state,
        )
        write_json(self.run_dir / "valid_post_sample_evaluation.json", serializable_result(result))
        write_csv(self.run_dir / "valid_post_event_clusters.csv", result["event_rows"])
        write_csv(self.run_dir / "valid_post_pca_summary.csv", result["pca"]["rows"])
        write_csv(self.run_dir / "valid_post_kmeans_metrics.csv", result["kmeans"]["rows"])
        write_csv(self.run_dir / "valid_post_cluster_stability.csv", result["stability"]["rows"])
        write_csv(self.run_dir / "valid_post_cluster_stability_runs.csv", result["stability"]["details"])
        write_csv(self.run_dir / "valid_post_gmm_metrics.csv", result["gmm"]["rows"])
        write_csv(self.run_dir / "valid_post_hdbscan_sweep.csv", result["hdbscan"]["rows"])
        write_csv(self.run_dir / "valid_post_kmeans_composition.csv", result["kmeans_composition"])
        write_csv(self.run_dir / "valid_post_hdbscan_composition.csv", result["hdbscan_composition"])

        import matplotlib.pyplot as plt

        for name, figure in evaluation_figures(result).items():
            figure.savefig(self.run_dir / "plots" / f"valid_post_{name}.png", dpi=160)
            plt.close(figure)
        self.summary["valid_post_sample_evaluation"] = result["summary"]
        self._save_summary()
        return result

    def run_latent_diagnostics(
        self,
        manifest: list[dict[str, Any]],
        embeddings,
        evaluation: dict[str, Any],
        *,
        remoteclip_version: str = "v1",
        reference_k: int = 3,
        membership_runs: int = 40,
        membership_sample_fraction: float = 0.80,
        random_state: int = 42,
    ) -> dict[str, Any]:
        if not embeddings:
            raise ValueError("Latent diagnostics require cached or computed embeddings.")
        with self.session() as db:
            result = compute_latent_diagnostics(
                db,
                manifest,
                embeddings,
                evaluation,
                remoteclip_version=remoteclip_version,
                reference_k=reference_k,
                membership_runs=membership_runs,
                membership_sample_fraction=membership_sample_fraction,
                random_state=random_state,
            )

        tables = {
            "latent_manual_events.csv": result["manual_rows"],
            "latent_hansen_events.csv": result["hansen_rows"],
            "latent_event_level_table.csv": result["final_rows"],
            "latent_cluster_roles.csv": result["cluster_roles"],
            "latent_cluster_label_season.csv": result["cluster_label_season"],
            "latent_cluster_label_multi_event.csv": result["cluster_label_multi_event"],
            "latent_mode_comparison.csv": result["mode_comparison"],
            "latent_mode_seasons.csv": result["mode_seasons"],
            "latent_loo_knownness_quantiles.csv": result["loo_quantiles"],
            "latent_probability_by_cluster.csv": result["probability_by_cluster"],
            "latent_probability_correlations.csv": result["probability_correlations"],
            "latent_ambiguity_reject.csv": result["ambiguity_reject"],
            "latent_knownness_reject.csv": result["knownness_reject"],
            "latent_combined_reject.csv": result["combined_reject"],
            "latent_train_hansen_quantiles.csv": result["train_hansen_quantiles"],
            "latent_domain_shift.csv": result["domain_shift"],
            "latent_hansen_model_interaction.csv": result["hansen_group_summary"],
            "latent_hdbscan_noise.csv": result["hdbscan_noise_summary"],
            "latent_knn_rank_stability.csv": result["knn_rank_stability"],
            "latent_knn_unknown_overlap.csv": result["knn_unknown_overlap"],
            "latent_membership_stability.csv": result["membership_stability"],
            "latent_membership_stability_summary.csv": result["membership_stability_summary"],
            "latent_membership_stability_runs.csv": result["membership_stability_runs"],
        }
        for filename, rows in tables.items():
            write_csv(self.run_dir / filename, rows)
        write_json(self.run_dir / "latent_diagnostics_summary.json", result["summary"])
        report_path = self.run_dir / "REPORT.md"
        report_path.write_text(
            diagnostics_report_markdown(result),
            encoding="utf-8",
        )
        result["report_path"] = report_path

        import matplotlib.pyplot as plt

        for name, figure in diagnostic_figures(result).items():
            figure.savefig(self.run_dir / "plots" / f"latent_{name}.png", dpi=160)
            plt.close(figure)
        self.summary["latent_diagnostics"] = {
            **result["summary"],
            "report_path": self._display_path(report_path),
        }
        self._save_summary()
        return result

    def load_previous_baseline(self) -> dict[str, Any] | None:
        candidates = sorted(
            (
                path
                for path in self.output_root.iterdir()
                if path.is_dir() and path.resolve() != self.run_dir.resolve()
            ),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        for candidate in candidates:
            config_path = candidate / "config.json"
            metrics_path = candidate / "baseline_metrics.csv"
            if not config_path.exists() or not metrics_path.exists():
                continue
            try:
                saved_config = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if saved_config.get("config_fingerprint") != self.config.fingerprint:
                continue
            saved_database = saved_config.get("database_fingerprint") or {}
            if saved_database.get("sha256") != self.database_snapshot.get("sha256"):
                continue
            result = {
                "baseline_metrics": read_csv(metrics_path),
                "fold_metrics": read_csv(candidate / "fold_metrics.csv"),
                "nomanual_review_queue": read_csv(candidate / "nomanual_review_queue.csv"),
                "saved_run_dir": self._display_path(candidate),
                "reused": True,
            }
            self.summary["reused_baseline"] = result["saved_run_dir"]
            self._save_summary()
            return result
        return None

    def plan_remoteclip_model_export(
        self,
        baseline_result: dict[str, Any],
        *,
        version: str = "v1",
        training_mode: str = "MANUAL_ONLY",
        feature_variant: str | None = None,
    ) -> dict[str, Any]:
        plan = plan_remoteclip_export(
            baseline_result["baseline_metrics"],
            version=version,
            training_mode=training_mode,
            feature_variant=feature_variant,
        )
        self.summary["remoteclip_export_plan"] = plan
        self._save_summary()
        return plan

    def export_remoteclip_model(
        self,
        manifest: list[dict[str, Any]],
        embeddings,
        baseline_result: dict[str, Any],
        *,
        version: str = "v1",
        training_mode: str = "MANUAL_ONLY",
        feature_variant: str | None = None,
        decision_threshold: float = 0.5,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        if not embeddings:
            raise ValueError("RemoteCLIP export requires embeddings loaded from or written to cache.")
        event_records = aggregate_event_embeddings(manifest, embeddings)
        result = export_remoteclip_classifier(
            event_records,
            baseline_result["baseline_metrics"],
            self.config,
            self.database_snapshot,
            version=version,
            training_mode=training_mode,
            feature_variant=feature_variant,
            decision_threshold=decision_threshold,
            overwrite=overwrite,
        )
        self.summary["remoteclip_artifact"] = result
        self._save_summary()
        return result

    def _save_summary(self) -> None:
        self.summary["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json(self.run_dir / "run_summary.json", self.summary)

