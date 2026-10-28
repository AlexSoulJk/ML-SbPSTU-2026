# Graph Report - ML-SbPSTU-2026  (2026-10-28)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 1643 nodes · 5713 edges · 104 communities (50 shown, 54 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 338 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `6186d4fe`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- candidate_service.py
- sentinel_service.py
- hansen_service.py
- main.py
- SentinelDownload
- db_sync_service.py
- ui.js
- app.js
- Research Tool Python Dependencies
- samples.js
- csv
- generate_ml_data.py
- Session
- fetch_cdse_init_data.py
- escapeHtml
- analyze_sentinel_darkness_metrics.py
- renderDetail
- numpy
- data_operations.js
- compare_forest_databases.py
- Sample
- backend/geometry.py
- typing
- routes.py
- import_service.py
- experiments.py
- ForestMLPipeline
- renderCompareView
- bindEvents
- threading
- json
- pathlib
- map.js
- test_pipeline_units.py
- core.py
- Any
- compute_manifest_embeddings
- bindSentinelDownloadControls
- sys
- build_training_dataset.py
- reporting.py
- build_event_manifest
- build_change_maps.py
- rebuild_missing_hansen_masks.py
- jobs.py
- renderSentinelDownloads
- model_labeling_service.py
- run_sentinel_batch_job
- save_sentinel_review
- find_matched_pairs.py
- check_sentinel_preview_cache.py
- raster_palette.js
- state.js

## God Nodes (most connected - your core abstractions)
1. `SentinelDownload` - 77 edges
2. `Sample` - 51 edges
3. `resolve_storage_path()` - 49 edges
4. `to_storage_path()` - 43 edges
5. `analyze_sample_hansen()` - 39 edges
6. `renderDetail()` - 37 edges
7. `utc_now()` - 33 edges
8. `mine_hansen_candidates()` - 33 edges
9. `execute_sentinel_batch()` - 33 edges
10. `SampleEvent` - 32 edges

## Surprising Connections (you probably didn't know these)
- `build_model_labeling_plan()` --uses--> `EventLabel`  [INFERRED]
  Scripts/ResearchTool/backend/forest/model_labeling_service.py → Scripts/ResearchTool/backend/forest/models.py
- `DataOperationsTest` --uses--> `SentinelBatchRequest`  [INFERRED]
  Scripts/ResearchTool/tests/test_data_operations.py → Scripts/ResearchTool/backend/forest/routes.py
- `EventBackfillTests` --uses--> `Base`  [INFERRED]
  Scripts/ResearchScripts/ml_tests/tests/test_pipeline_units.py → Scripts/ResearchTool/backend/forest/models.py
- `init_forest_database()` --uses--> `Base`  [INFERRED]
  Scripts/ResearchTool/backend/forest/database.py → Scripts/ResearchTool/backend/forest/models.py
- `existing_tile_record()` --uses--> `HansenTile`  [INFERRED]
  Scripts/ResearchScripts/prefetch_hansen_tiles.py → Scripts/ResearchTool/backend/forest/models.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Forest Sample Review Workflow** — scripts_researchtool_frontend_index_forest_mode, scripts_researchtool_frontend_index_forest_sample_import_filters, scripts_researchtool_frontend_index_batch_hansen_analysis, scripts_researchtool_frontend_index_forest_database_sync, scripts_researchtool_frontend_index_pre_post_compare, scripts_researchtool_frontend_index_forest_layer_controls [EXTRACTED 1.00]
- **Pipeline Geometry and AOI Workflow** — scripts_researchtool_readme_kml_protection_zone, scripts_researchtool_readme_closed_linestring_polygon, scripts_researchtool_readme_raster_medial_axis_centerline, scripts_researchtool_readme_centerline_segments, scripts_researchtool_readme_square_aoi, scripts_researchtool_frontend_index_pipeline_kml_aoi_controls [INFERRED 0.85]
- **Sentinel-2 Search, Fetch, and Preview Workflow** — scripts_researchtool_readme_cdse_stac_search, scripts_researchtool_readme_sentinel_2_raster_fetch, scripts_researchtool_readme_multispectral_previews, scripts_researchtool_frontend_index_sentinel_2_search_controls, scripts_researchtool_frontend_index_pipeline_observations_drawer, scripts_researchtool_frontend_index_raster_preview_controls [INFERRED 0.85]

## Communities (104 total, 54 thin omitted)

### Community 0 - "candidate_service.py"
Cohesion: 0.06
Nodes (76): _active_exclusions(), _aoi_histogram(), _candidate_identity(), candidate_run_payload(), _candidate_score(), CandidateMiningOptions, fill_candidate_run(), hansen_tile_inventory() (+68 more)

### Community 1 - "sentinel_service.py"
Cohesion: 0.06
Nodes (74): classify_review_action(), clean_reason_codes(), main(), manual_validation_payload(), now_tag(), print_reason_stats(), print_valid_sample_conflicts(), replace_review_reasons() (+66 more)

### Community 2 - "hansen_service.py"
Cohesion: 0.07
Nodes (59): existing_tile_record(), main(), now_tag(), planned_tile_rows(), progress(), unique_sample_tiles(), upsert_tile_record(), write_report() (+51 more)

### Community 3 - "main.py"
Cohesion: 0.05
Nodes (57): bbox_to_web_mercator(), best_s2_observations(), cdse_post(), cdse_session(), cdse_uses_env_proxy(), cloud_cover(), configured_proxy_keys(), env_flag() (+49 more)

### Community 4 - "SentinelDownload"
Cohesion: 0.08
Nodes (66): main(), selected_bundle_path(), build_process_request(), can_download(), decode_json(), display_path(), download_raster(), expected_preview_dir() (+58 more)

### Community 5 - "db_sync_service.py"
Cohesion: 0.12
Nodes (63): add_summary(), allowed_bases_dir(), append_note(), apply_db_sync_plan(), apply_operation(), bool_value(), build_db_sync_plan(), candidate_databases() (+55 more)

### Community 6 - "ui.js"
Cohesion: 0.08
Nodes (46): applyCompareHansenFilter(), applyCompareImageFilter(), applyMainRasterFilters(), applyVisibleRasterFilters(), bboxToBounds(), colorKey(), compareDownloadById(), compareDownloadOptions() (+38 more)

### Community 7 - "app.js"
Cohesion: 0.13
Nodes (45): api(), bboxKey(), bboxToLeafletBounds(), bindEvents(), buildSegments(), checkedObservations(), clearGeometryLayers(), clearLayer() (+37 more)

### Community 8 - "Research Tool Python Dependencies"
Cohesion: 0.08
Nodes (38): Pipeline Application Script, Batch Hansen Analysis, Forest Database Synchronization, Forest Map Layer Controls, Forest Mode, Forest Sample Import and Filtering, Forest UI Module, Leaflet 1.9.4 (+30 more)

### Community 9 - "samples.js"
Cohesion: 0.05
Nodes (4): cleanFilters(), exportForestSamples(), listForestSamples(), startHansenBatch()

### Community 10 - "csv"
Cohesion: 0.15
Nodes (25): add_filter_columns(), built_in_russia_geometry(), extract_csv(), extract_geojson(), feature_matches_russia(), is_russia_point(), iter_input_files(), load_russia_geometry() (+17 more)

### Community 11 - "generate_ml_data.py"
Cohesion: 0.15
Nodes (26): create_metadata_for_images(), derived_relative_path(), display_path(), get_images_from_db(), get_original_image_path(), get_path_to_image(), main(), resolve_project_path() (+18 more)

### Community 12 - "Session"
Cohesion: 0.10
Nodes (31): analyze_hansen(), analyze_hansen_batch(), cancel_job(), db_sync_apply(), db_sync_compare(), DbSyncApplyRequest, DbSyncCompareRequest, event_backfill_plan() (+23 more)

### Community 13 - "fetch_cdse_init_data.py"
Cohesion: 0.14
Nodes (22): Area, build_catalog_request(), build_process_request(), catalog_summary(), CollectionSpec, download_chip(), fetch_token(), import_requests() (+14 more)

### Community 14 - "escapeHtml"
Cohesion: 0.11
Nodes (33): cancelDbSyncSentinelCacheJob(), dbSyncSentinelCache(), dbSyncSentinelItems(), dbSyncStatus(), escapeHtml(), hansenStageLabel(), isHansenJobRunning(), isTerminalJobStatus() (+25 more)

### Community 15 - "analyze_sentinel_darkness_metrics.py"
Cohesion: 0.19
Nodes (28): candidate_stats(), combined_or_rule_stats(), combined_or_rule_sweep_rows(), contrast_payload(), finite_number(), finite_values(), fraction_below(), histogram() (+20 more)

### Community 16 - "renderDetail"
Cohesion: 0.16
Nodes (32): cancelBatchHansen(), cancelHansenAnalysis(), canCompareSentinel(), clearSelectedSample(), downloadCheckedSentinelScenes(), ensureActiveSentinelPreviewForSample(), fitMapToHansenTile(), loadSelectedHansenTilePreview() (+24 more)

### Community 17 - "numpy"
Cohesion: 0.20
Nodes (13): main(), train_features(), main(), train_features(), main(), train_features(), main(), run_block() (+5 more)

### Community 18 - "data_operations.js"
Cohesion: 0.20
Nodes (29): activeJobHost(), artifactUrl(), checkedValues(), dataOperationsRunning(), downloadPayload(), ensureMiningMap(), escapeHtml(), humanBytes() (+21 more)

### Community 19 - "compare_forest_databases.py"
Cohesion: 0.20
Nodes (25): main(), column_names(), compare_hansen_analyses(), compare_hansen_tiles(), compare_manual(), compare_statuses(), connect(), count_by() (+17 more)

### Community 20 - "Sample"
Cohesion: 0.14
Nodes (17): iter_hansen_ready_unvalidated_samples(), iter_hansen_too_old_samples(), iter_unvalidated_samples(), upsert_manual_validation(), append_manual_note(), hansen_comment(), latest_analyses(), main() (+9 more)

### Community 21 - "backend/geometry.py"
Cohesion: 0.16
Nodes (19): bbox_from_polygon(), build_segments(), geometry_payload(), interpolate_line(), line_to_geojson(), longest_path_in_skeleton(), component(), dijkstra() (+11 more)

### Community 22 - "typing"
Cohesion: 0.17
Nodes (8): ExperimentConfig, load_config(), stable_json_hash(), database_fingerprint(), file_sha256(), project_relative_path(), resolve_manifest_path(), backup_sqlite_database()

### Community 23 - "routes.py"
Cohesion: 0.16
Nodes (20): get_db(), BundlePlanRequest, create_bundle_plan(), data_operation_input_path(), db_sync_candidates(), forest_context(), get_bundle_artifact(), get_data_operation_artifact() (+12 more)

### Community 24 - "import_service.py"
Cohesion: 0.21
Nodes (18): clean_text(), detect_format(), geometry_point(), import_file(), lookup(), normalize_key(), normalize_sample(), NormalizedSample (+10 more)

### Community 25 - "experiments.py"
Cohesion: 0.18
Nodes (13): aggregate_event_embeddings(), build_nomanual_review_queue(), _evaluate_model(), feature_vector(), _fit_model(), records_for_variant(), run_baseline_experiments(), _run_fold() (+5 more)

### Community 26 - "ForestMLPipeline"
Cohesion: 0.22
Nodes (7): ForestMLPipeline, create_run_directory(), read_csv(), write_csv(), write_json(), main(), parse_args()

### Community 27 - "renderCompareView"
Cohesion: 0.15
Nodes (23): availableCompareViews(), bestCompareDownload(), bindCompareControls(), compareDownloadMeta(), compareDownloads(), compareHansenOverlay(), compareViewOptions(), currentHansenLegendOverlay() (+15 more)

### Community 28 - "bindEvents"
Cohesion: 0.13
Nodes (23): bindEvents(), closeCompareView(), closeDbSyncView(), copyDebugEvents(), createExport(), currentFilters(), debugEventsText(), enhanceSidebarPanels() (+15 more)

### Community 29 - "threading"
Cohesion: 0.14
Nodes (16): BundleExportJobRequest, BundleImportRequest, create_bundle_import_plan(), HansenCandidateMiningRequest, job_cancel_requested(), run_bundle_export_job(), run_bundle_import_job(), run_event_backfill_job() (+8 more)

### Community 30 - "json"
Cohesion: 0.18
Nodes (11): inspect_one(), main(), resample_hansen_to_sentinel(), save_png(), analyze_sample(), main(), plot_and_save(), resample_hansen() (+3 more)

### Community 31 - "pathlib"
Cohesion: 0.16
Nodes (8): evaluate_linear_probe(), group_by_samples_and_period(), load_metadata_from_folder(), class_similarity(), permutation_test(), permutation_test_post_vs_pre(), plot_pca(), similarity_gap()

### Community 32 - "map.js"
Cohesion: 0.23
Nodes (17): addCircle(), addVisiblePoint(), attachMapCaptureDebug(), bboxToBounds(), clearForestLayers(), debugMap(), ensureForestPanes(), eventTargetSummary() (+9 more)

### Community 33 - "test_pipeline_units.py"
Cohesion: 0.26
Nodes (7): apply_dominant_event_backfill(), plan_dominant_event_backfill(), virtual_backfill_rows(), EventBackfillTests, sample(), SampleEvent, SampleEventDownload

### Community 34 - "core.py"
Cohesion: 0.20
Nodes (9): aggregate_embeddings_by_sample_and_period(), calculate_cosine_similarity(), embedding_consistency_loo(), get_butch_embeddings(), get_embedding(), l2_normalize(), RemoteClipEmbedder, load_remoteclip_model() (+1 more)

### Community 35 - "Any"
Cohesion: 0.24
Nodes (16): create_export(), dt_iso(), export_row(), export_row_empty(), ExportRequest, get_sample(), json_or_none(), list_samples() (+8 more)

### Community 36 - "compute_manifest_embeddings"
Cohesion: 0.29
Nodes (4): compute_manifest_embeddings(), EmbeddingCache, ImageEmbedder, EmbeddingCacheTests

### Community 37 - "bindSentinelDownloadControls"
Cohesion: 0.33
Nodes (15): baseSentinelReview(), bindSentinelDownloadControls(), cancelSentinelReviewDraft(), effectiveSentinelReview(), findSentinelDownload(), renderSentinelDownloadsPanel(), restoreSentinelReview(), saveSentinelReviewDraft() (+7 more)

### Community 39 - "build_training_dataset.py"
Cohesion: 0.27
Nodes (9): collect_change_features(), collect_hansen(), collect_hansen_shape(), collect_spectral(), load_valid_ids(), main(), stats_for_tif(), loss_shape_features() (+1 more)

### Community 40 - "reporting.py"
Cohesion: 0.30
Nodes (7): baseline_metrics_html(), _csv_value(), json_value(), _metric_text(), plot_baseline_metrics(), review_queue_html(), NotebookTests

### Community 41 - "build_event_manifest"
Cohesion: 0.23
Nodes (7): build_event_manifest(), parse_scene_datetime(), _preview_map(), _reason_map(), season_for_datetime(), validation_cohort(), DatasetHelpersTests

### Community 42 - "build_change_maps.py"
Cohesion: 0.30
Nodes (7): aggregate_period(), is_strictly_excluded(), load_valid_ids(), main(), nbr_and_mask(), read_bands(), scene_datetime()

### Community 43 - "rebuild_missing_hansen_masks.py"
Cohesion: 0.35
Nodes (8): display_path(), latest_analysis(), main(), masks_for_analysis(), now_tag(), sample_rebuild_reason(), selected_samples(), write_report()

### Community 44 - "jobs.py"
Cohesion: 0.31
Nodes (3): BackgroundJob, InMemoryJobStore, utc_iso()

### Community 45 - "renderSentinelDownloads"
Cohesion: 0.22
Nodes (11): allDownloadedPreviews(), chooseDefaultSentinelPreview(), downloadedPreviews(), eventScopedSentinelDownloads(), filteredSentinelDownloads(), findPreviewByUrl(), renderImageIdSearch(), renderSentinelDownloads() (+3 more)

### Community 46 - "model_labeling_service.py"
Cohesion: 0.33
Nodes (5): build_model_labeling_plan(), _download_usable(), file_sha256(), model_artifact_status(), _safe_version()

### Community 47 - "run_sentinel_batch_job"
Cohesion: 0.50
Nodes (6): batch_options_from_request(), run_sentinel_batch_job(), selected_batch_samples(), sentinel_batch_plan(), SentinelBatchRequest, start_sentinel_batch_job()

### Community 48 - "save_sentinel_review"
Cohesion: 0.38
Nodes (6): clean_optional_text(), clean_reason_codes(), EventReviewRequest, save_sample_event_review(), save_sentinel_review(), SentinelReviewRequest

### Community 49 - "find_matched_pairs.py"
Cohesion: 0.48
Nodes (4): count_pairs(), haversine_km(), load_data(), main()

### Community 50 - "check_sentinel_preview_cache.py"
Cohesion: 0.53
Nodes (4): main(), now_tag(), preview_rows(), write_report()

## Knowledge Gaps
- **5 isolated node(s):** `Batch Hansen Analysis`, `Frontend Stylesheet`, `CDSE_USE_ENV_PROXY Configuration`, `Sentinel Hub Process API`, `Pydantic >=2.11`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 212 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **54 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `SentinelDownload` connect `SentinelDownload` to `candidate_service.py`, `sentinel_service.py`, `test_pipeline_units.py`, `db_sync_service.py`, `build_training_dataset.py`, `build_event_manifest`, `build_change_maps.py`, `generate_ml_data.py`, `model_labeling_service.py`, `analyze_sentinel_darkness_metrics.py`, `check_sentinel_preview_cache.py`, `typing`, `routes.py`?**
  _High betweenness centrality (0.021) - this node is a cross-community bridge._
- **Are the 54 inferred relationships involving `SentinelDownload` (e.g. with `selected_downloads()` and `main()`) actually correct?**
  _`SentinelDownload` has 54 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Batch Hansen Analysis`, `Frontend Stylesheet`, `CDSE_USE_ENV_PROXY Configuration` to the rest of the system?**
  _5 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `candidate_service.py` be split into smaller, more focused modules?**
  _Cohesion score 0.0560625814863103 - nodes in this community are weakly interconnected._
- **Why does `Sample` connect `Sample` to `candidate_service.py`, `test_pipeline_units.py`, `hansen_service.py`, `sentinel_service.py`, `db_sync_service.py`, `build_training_dataset.py`, `build_event_manifest`, `generate_ml_data.py`, `rebuild_missing_hansen_masks.py`, `find_matched_pairs.py`, `typing`, `routes.py`, `import_service.py`?**
  _High betweenness centrality (0.013) - this node is a cross-community bridge._
- **Should `sentinel_service.py` be split into smaller, more focused modules?**
  _Cohesion score 0.06118421052631579 - nodes in this community are weakly interconnected._
- **Why does `analyze_sample_hansen()` connect `hansen_service.py` to `candidate_service.py`, `build_training_dataset.py`, `rebuild_missing_hansen_masks.py`, `Session`, `numpy`, `Sample`, `routes.py`, `import_service.py`, `threading`, `json`?**
  _High betweenness centrality (0.011) - this node is a cross-community bridge._