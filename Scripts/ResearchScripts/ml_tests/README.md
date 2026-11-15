# Forest ML experiments

The primary research interface is `forest_ml_experiments.ipynb`. The notebook keeps orchestration and compact outputs in cells while database access, RemoteCLIP inference, caching, splits, and metrics live in Python modules.

`valid_post_sample_evaluation.ipynb` is the companion sample-structure report. Its first half uses only eligible `MANUAL_VALID` POST-event embeddings for PCA, K-Means, GMM, HDBSCAN, and stability. The diagnostic half reconstructs the exact frozen `REMOTECLIP_LOGREG/v1` training cohort, adds leave-one-out knownness, class-conditional similarity, reject curves, train-vs-Hansen domain-shift checks, RemoteCLIP/XGBoost interaction, and a combined event-level export. Existing embeddings are reused from the fingerprinted cache.

`olga_model_validation.ipynb` is the human-review audit. It explicitly selects Olga's current event reviews, compares the latest stored RemoteCLIP and XGBoost labels with accepted human labels, keeps rejected events outside accuracy metrics, and exports event-level, point-level, per-class, agreement, and confusion-matrix tables. It reads stored database inference only and does not recompute embeddings or predictions.

`olga_rejection_diagnostics.ipynb` extends that audit into a hierarchical VALID/REJECTED analysis. It normalizes free-text rejection reasons without modifying the database, builds Logging/Wildfire/Rejected closed-set tables, compares probabilities and knownness, audits HDBSCAN noise and model disagreement, aggregates multi-event point histories, evaluates knownness rejection coverage, and runs exploratory grouped-CV validity pre-filters. It also joins materialized Hansen event area/pixels, AOI loss share, linked PRE/EVENT/POST downloads, local QC flags, and manual scene exclusions. Dedicated sections search for an `EVENT_TOO_SMALL` deterministic threshold, audit `YEAR_MISMATCH` temporal evidence, treat `NO_PRE/QC_FAILED` as acquisition-pipeline losses, run reason-specific one-vs-VALID diagnostics, and inspect point-level reason co-occurrence. It joins the current reviews with the latest saved `latent_event_level_table.csv` and reports missing latent coverage explicitly. Expert-visible years remain free text and FIRMS is not currently integrated, so those gaps are reported rather than imputed.

The focused `YEAR_MISMATCH` branch builds a `Y-2/Y-1/Y/Y+1` timeline for every affected point and evaluates whether geometry, model probabilities, acquisition quality, or a leave-one-point-out ranking model can select the expert-indicated physical event year. Ambiguous points are preserved in the timeline but excluded from target-year metrics.

`multi_event_hansen_research.ipynb` runs the full-cohort multi-event study. It uses `sample_id` as the grouping unit, performs a paired clustered-bootstrap comparison of RemoteCLIP and XGBoost, validates the frozen `EVENT_TOO_SMALL` thresholds only on the later review batch, compares dominant-year ranking strategies, and simulates adjacent-year grouping with AOI Hansen-mask connectivity. The notebook writes a standalone `REPORT.md`, CSV tables, and plots under `Outputs/ml_experiments/multi_event_research/`. It is read-only with respect to the database.

Install notebook dependencies from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r Scripts\ResearchScripts\ml_tests\requirements.txt
```

Start Jupyter Lab:

```powershell
.\.venv\Scripts\python.exe -m jupyter lab Scripts\ResearchScripts\ml_tests\forest_ml_experiments.ipynb
```

For the sample-structure evaluation:

```powershell
.\.venv\Scripts\python.exe -m jupyter lab Scripts\ResearchScripts\ml_tests\valid_post_sample_evaluation.ipynb
```

For the reviewer-specific model audit:

```powershell
.\.venv\Scripts\python.exe -m jupyter lab Scripts\ResearchScripts\ml_tests\olga_model_validation.ipynb
```

For hierarchical rejection diagnostics:

```powershell
.\.venv\Scripts\python.exe -m jupyter lab Scripts\ResearchScripts\ml_tests\olga_rejection_diagnostics.ipynb
```

For the complete multi-event Hansen study:

```powershell
.\.venv\Scripts\python.exe -m jupyter lab Scripts\ResearchScripts\ml_tests\multi_event_hansen_research.ipynb
```

The notebook defaults to:

- `quick` profile;
- event-link dry-run with virtual links in the manifest;
- `REUSE_SAVED_BASELINE = True`, which reloads the latest compatible metrics;
- explicit switches for RemoteCLIP cache loading and baseline training.

Inspect the backfill and dataset audit first, then set `RUN_EMBEDDINGS = True` and `RUN_BASELINE = True`. Set `APPLY_EVENT_BACKFILL = True` only after reviewing the plan; the pipeline creates a SQLite backup before applying links.

After a kernel restart, RemoteCLIP embeddings are read from the fingerprinted disk cache. They are recomputed only when an image, checkpoint, or preprocessing fingerprint changes. Saved baseline metrics are reused only when both the experiment config fingerprint and database SHA match.

`APPLY_EVENT_BACKFILL` writes the planned legacy download-to-dominant-event links to `forest_sample_event_downloads`. It does not move image files or change Hansen analyses, manual validation, QC reviews, or WRI labels. With the flag disabled, the same planned links are used only virtually for the current manifest.

For a headless baseline run:

```powershell
.\.venv\Scripts\python.exe Scripts\ResearchScripts\run_forest_ml_experiments.py --profile quick --experiment baseline
```

Embeddings are stored under the configured project-relative cache directory and are not embedded into the notebook or committed to Git.

The notebook also contains an opt-in production artifact section. It first shows the best grouped-CV RemoteCLIP configuration and writes nothing. After review, set `SAVE_REMOTECLIP_ARTIFACT = True` to train the final scaler/classifier and save the versioned bundle under:

```text
Scripts/ResearchTool/Data/forest/models/remoteclip_logreg/<version>/
  classifier.joblib
  manifest.json
  metrics.json
```

Existing versions are protected unless `OVERWRITE_REMOTECLIP_ARTIFACT = True` is set explicitly. The manifest includes the artifact hash, database fingerprint, training cohort, feature variant, class order, decision threshold, RemoteCLIP checkpoint hash, and a round-trip smoke-test result.

The latent diagnostic section is read-only. It audits the reconstructed
`MANUAL_VALID + PRE_POST` event-id hash against the production model manifest and
surfaces a mismatch without silently retraining or replacing the frozen artifact.
The default `UNKNOWN` (`manual LOO knownness k=5, p05`) and `AMBIGUOUS`
(`P(Wildfire) in 0.40..0.60`) flags are diagnostic exports only; they are not fitted
on Hansen-mined events. HDBSCAN noise is reported separately and never implies
`UNKNOWN` automatically.

Each completed latent-diagnostic run also writes `REPORT.md` into its run
directory. It is a human-readable summary assembled from the same in-memory
tables as the CSV exports and includes cohort audit, cluster structure,
knownness/reject diagnostics, Hansen domain shift, model interaction, and
stability results.

## Pipeline v2 Stage A

`pipeline_v2_stage_a.ipynb` is the thin inspection interface for the complete
T00-T05 ML stage. It shows the frozen event-level cohort, excluded/not-ready
events, duplicate audit, quick/final grouped splits, OLD/FULL metrics,
aggregate confusion matrices, and four-artifact smoke predictions. Keep
`RUN_STAGE_A = False` when reopening it to inspect existing results without
recomputing embeddings or retraining models.

The FULL human-training cohort is the union of the historical sample-level
`ManualValidation=Valid + driver_primary` labels and the newer event-level
`EventReview=VALID + human EventLabel` labels. An explicit event review is
authoritative: reviewed DRAFT/REJECTED events are not reintroduced through the
historical sample label, and a VALID event label overrides `driver_primary`.
Legacy Sentinel downloads without an explicit event link are assigned virtually
to the sample's single dominant event for Stage A. This reproduces the existing
experiment pipeline without writing backfill rows to SQLite.

The complete headless run is:

```powershell
.\.venv\Scripts\python.exe Scripts\ResearchScripts\run_pipeline_v2_stage_a.py --profile final
```

Compatible existing v2 artifacts are verified and reused. A conflicting v2
artifact is never overwritten implicitly; use a new version or the explicit
`--overwrite-artifacts` switch after reviewing the manifests. The consolidated
acceptance report is written to:

```text
Scripts/ResearchScripts/Outputs/pipeline_v2/stages/A/STAGE_REPORT.md
```
