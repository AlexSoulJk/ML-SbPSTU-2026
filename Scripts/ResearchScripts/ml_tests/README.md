# Forest ML experiments

The primary research interface is `forest_ml_experiments.ipynb`. The notebook keeps orchestration and compact outputs in cells while database access, RemoteCLIP inference, caching, splits, and metrics live in Python modules.

Install notebook dependencies from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r Scripts\ResearchScripts\ml_tests\requirements.txt
```

Start Jupyter Lab:

```powershell
.\.venv\Scripts\python.exe -m jupyter lab Scripts\ResearchScripts\ml_tests\forest_ml_experiments.ipynb
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
