# Forest ML experiments

The primary research interface is `forest_ml_experiments.ipynb`. The notebook keeps orchestration and compact outputs in cells while database access, RemoteCLIP inference, caching, splits, and metrics live in Python modules.

`valid_post_sample_evaluation.ipynb` is the companion sample-structure report. Its first half uses only eligible `MANUAL_VALID` POST-event embeddings for PCA, K-Means, GMM, HDBSCAN, and stability. The diagnostic half reconstructs the exact frozen `REMOTECLIP_LOGREG/v1` training cohort, adds leave-one-out knownness, class-conditional similarity, reject curves, train-vs-Hansen domain-shift checks, RemoteCLIP/XGBoost interaction, and a combined event-level export. Existing embeddings are reused from the fingerprinted cache.

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
