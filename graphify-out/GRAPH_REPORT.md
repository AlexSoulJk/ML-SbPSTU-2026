# Graph Report - ML-SbPSTU-2026  (2026-10-28)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 1757 nodes · 6060 edges · 125 communities (59 shown, 66 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 298 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `2d16f13c`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- json
- main.py
- db_sync_service.py
- analyze_sentinel_darkness_metrics.py
- csv
- ui.js
- models.py
- resolve_storage_path
- app.js
- Research Tool Python Dependencies
- candidate_service.py
- model_inference_service.py
- samples.js
- pipeline.py
- sentinel_batch_service.py
- data_operations.js
- hansen_service.py
- generate_ml_data.py
- test_pipeline_units.py
- routes.py
- fetch_cdse_init_data.py
- sys
- escapeHtml
- renderDetail
- compare_forest_databases.py
- Any
- experiments.py
- typing
- rasterio
- import_service.py
- build_training_dataset.py
- renderCompareView
- bindEvents
- threading
- pathlib
- map.js
- event_review_service.py
- core.py
- ExperimentConfig
- DataOperationsTest
- artifacts.py
- numpy
- backend/config.py
- compute_manifest_embeddings
- bindSentinelDownloadControls
- model_labeling_service.py
- sample_geometries
- build_change_maps.py
- jobs.py
- renderSentinelDownloads
- post
- hansen_provider.py
- resolve_bundle_download_ids
- run_sentinel_batch_job
- _tile_proposals
- save_sentinel_review
- inspect_change_map.py
- find_matched_pairs.py
- start_bundle_import_job
- NotebookTests
- raster_palette.js
- state.js

## God Nodes (most connected - your core abstractions)
1. `SentinelDownload` - 75 edges
2. `resolve_storage_path()` - 57 edges
3. `Sample` - 50 edges
4. `to_storage_path()` - 49 edges
5. `analyze_sample_hansen()` - 39 edges
6. `renderDetail()` - 37 edges
7. `utc_now()` - 35 edges
8. `mine_hansen_candidates()` - 33 edges
9. `execute_sentinel_batch()` - 33 edges
10. `SampleEvent` - 32 edges

## Surprising Connections (you probably didn't know these)
- `selected_downloads()` --uses--> `SentinelDownload`  [INFERRED]
  Scripts/ResearchScripts/analyze_sentinel_darkness_metrics.py → Scripts/ResearchTool/backend/forest/models.py
- `main()` --uses--> `SentinelDownload`  [INFERRED]
  Scripts/ResearchScripts/check_sentinel_preview_cache.py → Scripts/ResearchTool/backend/forest/models.py
- `preview_rows()` --uses--> `SentinelDownload`  [INFERRED]
  Scripts/ResearchScripts/check_sentinel_preview_cache.py → Scripts/ResearchTool/backend/forest/models.py
- `get_images_from_db()` --uses--> `SentinelDownload`  [INFERRED]
  Scripts/ResearchScripts/generate_ml_data.py → Scripts/ResearchTool/backend/forest/models.py
- `build_event_manifest()` --uses--> `SentinelDownload`  [INFERRED]
  Scripts/ResearchScripts/ml_tests/dataset.py → Scripts/ResearchTool/backend/forest/models.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Forest Sample Review Workflow** — scripts_researchtool_frontend_index_forest_mode, scripts_researchtool_frontend_index_forest_sample_import_filters, scripts_researchtool_frontend_index_batch_hansen_analysis, scripts_researchtool_frontend_index_forest_database_sync, scripts_researchtool_frontend_index_pre_post_compare, scripts_researchtool_frontend_index_forest_layer_controls [EXTRACTED 1.00]
- **Pipeline Geometry and AOI Workflow** — scripts_researchtool_readme_kml_protection_zone, scripts_researchtool_readme_closed_linestring_polygon, scripts_researchtool_readme_raster_medial_axis_centerline, scripts_researchtool_readme_centerline_segments, scripts_researchtool_readme_square_aoi, scripts_researchtool_frontend_index_pipeline_kml_aoi_controls [INFERRED 0.85]
- **Sentinel-2 Search, Fetch, and Preview Workflow** — scripts_researchtool_readme_cdse_stac_search, scripts_researchtool_readme_sentinel_2_raster_fetch, scripts_researchtool_readme_multispectral_previews, scripts_researchtool_frontend_index_sentinel_2_search_controls, scripts_researchtool_frontend_index_pipeline_observations_drawer, scripts_researchtool_frontend_index_raster_preview_controls [INFERRED 0.85]

## Communities (125 total, 66 thin omitted)

### Community 0 - "json"
Cohesion: 0.06
Nodes (84): classify_review_action(), clean_reason_codes(), main(), manual_validation_payload(), now_tag(), print_reason_stats(), print_valid_sample_conflicts(), replace_review_reasons() (+76 more)

### Community 1 - "main.py"
Cohesion: 0.06
Nodes (55): bbox_to_web_mercator(), best_s2_observations(), cdse_post(), cdse_session(), cdse_uses_env_proxy(), cloud_cover(), configured_proxy_keys(), env_flag() (+47 more)

### Community 2 - "db_sync_service.py"
Cohesion: 0.12
Nodes (60): add_summary(), allowed_bases_dir(), append_note(), apply_db_sync_plan(), bool_value(), build_db_sync_plan(), candidate_databases(), compare_cache_diagnostics() (+52 more)

### Community 3 - "analyze_sentinel_darkness_metrics.py"
Cohesion: 0.09
Nodes (47): candidate_stats(), combined_or_rule_stats(), combined_or_rule_sweep_rows(), contrast_payload(), finite_number(), finite_values(), fraction_below(), histogram() (+39 more)

### Community 4 - "csv"
Cohesion: 0.09
Nodes (37): add_filter_columns(), built_in_russia_geometry(), extract_csv(), extract_geojson(), feature_matches_russia(), is_russia_point(), iter_input_files(), load_russia_geometry() (+29 more)

### Community 5 - "ui.js"
Cohesion: 0.08
Nodes (46): applyCompareHansenFilter(), applyCompareImageFilter(), applyMainRasterFilters(), applyVisibleRasterFilters(), bboxToBounds(), colorKey(), compareDownloadById(), compareDownloadOptions() (+38 more)

### Community 6 - "models.py"
Cohesion: 0.12
Nodes (31): iter_hansen_ready_unvalidated_samples(), iter_hansen_too_old_samples(), iter_unvalidated_samples(), upsert_manual_validation(), append_manual_note(), hansen_comment(), latest_analyses(), main() (+23 more)

### Community 7 - "resolve_storage_path"
Cohesion: 0.13
Nodes (32): main(), selected_download_ids(), main(), selected_bundle_path(), add_bundle_file(), bundle_candidates(), bundle_member_exists(), bundle_member_sha256() (+24 more)

### Community 8 - "app.js"
Cohesion: 0.13
Nodes (45): api(), bboxKey(), bboxToLeafletBounds(), bindEvents(), buildSegments(), checkedObservations(), clearGeometryLayers(), clearLayer() (+37 more)

### Community 9 - "Research Tool Python Dependencies"
Cohesion: 0.08
Nodes (38): Pipeline Application Script, Batch Hansen Analysis, Forest Database Synchronization, Forest Map Layer Controls, Forest Mode, Forest Sample Import and Filtering, Forest UI Module, Leaflet 1.9.4 (+30 more)

### Community 10 - "candidate_service.py"
Cohesion: 0.12
Nodes (27): _active_exclusions(), candidate_hansen_preview_payload(), _candidate_identity(), _candidate_lossyear_sources(), candidate_run_payload(), _candidate_score(), CandidateMiningOptions, fill_candidate_run() (+19 more)

### Community 11 - "model_inference_service.py"
Cohesion: 0.11
Nodes (18): _create_label_run(), EmbeddingCache, _event_inputs(), execute_model_labeling(), file_sha256(), ImageEmbedder, InferenceEvent, l2_normalize() (+10 more)

### Community 12 - "samples.js"
Cohesion: 0.05
Nodes (4): cleanFilters(), exportForestSamples(), listForestSamples(), startHansenBatch()

### Community 13 - "pipeline.py"
Cohesion: 0.13
Nodes (14): database_fingerprint(), backup_sqlite_database(), aggregate_event_embeddings(), ForestMLPipeline, baseline_metrics_html(), create_run_directory(), _csv_value(), json_value() (+6 more)

### Community 14 - "sentinel_batch_service.py"
Cohesion: 0.14
Nodes (25): main(), selected_samples(), BatchOptions, build_operation_rows(), calendar_year_for_period(), date_gap_ok(), decode_json(), download_datetime() (+17 more)

### Community 15 - "data_operations.js"
Cohesion: 0.17
Nodes (35): activeJobHost(), activeMiningCandidate(), artifactUrl(), checkedValues(), dataOperationsRunning(), downloadPayload(), ensureMiningMap(), escapeHtml() (+27 more)

### Community 16 - "hansen_service.py"
Cohesion: 0.13
Nodes (23): analyze_sample_hansen(), apply_sample_multiple_event_flag(), cache_url(), existing_hansen_analysis(), existing_lossyear_tile(), hansen_tile_preview_payload(), hex_to_rgb(), latest_hansen_payload() (+15 more)

### Community 17 - "generate_ml_data.py"
Cohesion: 0.15
Nodes (26): create_metadata_for_images(), derived_relative_path(), display_path(), get_images_from_db(), get_original_image_path(), get_path_to_image(), main(), resolve_project_path() (+18 more)

### Community 18 - "test_pipeline_units.py"
Cohesion: 0.15
Nodes (19): apply_dominant_event_backfill(), plan_dominant_event_backfill(), virtual_backfill_rows(), EventBackfillTests, sample(), backfill_hansen_events(), decode_histogram(), deterministic_id() (+11 more)

### Community 19 - "routes.py"
Cohesion: 0.11
Nodes (32): get_db(), configured_annotator_name(), ForestExport, analyze_hansen_batch(), create_export(), db_sync_apply(), db_sync_compare(), DbSyncApplyRequest (+24 more)

### Community 20 - "fetch_cdse_init_data.py"
Cohesion: 0.14
Nodes (22): Area, build_catalog_request(), build_process_request(), catalog_summary(), CollectionSpec, download_chip(), fetch_token(), import_requests() (+14 more)

### Community 21 - "sys"
Cohesion: 0.17
Nodes (14): main(), train_features(), main(), train_features(), main(), train_features(), main(), run_block() (+6 more)

### Community 22 - "escapeHtml"
Cohesion: 0.11
Nodes (33): cancelDbSyncSentinelCacheJob(), dbSyncSentinelCache(), dbSyncSentinelItems(), dbSyncStatus(), escapeHtml(), hansenStageLabel(), isHansenJobRunning(), isTerminalJobStatus() (+25 more)

### Community 23 - "renderDetail"
Cohesion: 0.16
Nodes (32): cancelBatchHansen(), cancelHansenAnalysis(), canCompareSentinel(), clearSelectedSample(), downloadCheckedSentinelScenes(), ensureActiveSentinelPreviewForSample(), fitMapToHansenTile(), loadSelectedHansenTilePreview() (+24 more)

### Community 24 - "compare_forest_databases.py"
Cohesion: 0.20
Nodes (25): main(), column_names(), compare_hansen_analyses(), compare_hansen_tiles(), compare_manual(), compare_statuses(), connect(), count_by() (+17 more)

### Community 25 - "Any"
Cohesion: 0.19
Nodes (22): db_sync_candidates(), dt_iso(), export_row(), get_export(), get_hansen_candidate_preview(), get_hansen_candidate_run(), get_hansen_candidate_runs(), get_hansen_tile_preview() (+14 more)

### Community 26 - "experiments.py"
Cohesion: 0.18
Nodes (13): build_nomanual_review_queue(), _evaluate_model(), feature_vector(), fit_classifier(), _fit_model(), records_for_variant(), run_baseline_experiments(), _run_fold() (+5 more)

### Community 27 - "typing"
Cohesion: 0.16
Nodes (10): build_event_manifest(), file_sha256(), parse_scene_datetime(), _preview_map(), project_relative_path(), _reason_map(), resolve_manifest_path(), season_for_datetime() (+2 more)

### Community 28 - "rasterio"
Cohesion: 0.20
Nodes (16): main(), refresh_mask_png(), build_existing_mask_index(), display_path(), find_existing_tif(), main(), now_tag(), png_for_mask() (+8 more)

### Community 29 - "import_service.py"
Cohesion: 0.22
Nodes (18): clean_text(), detect_format(), geometry_point(), import_file(), lookup(), normalize_key(), normalize_sample(), NormalizedSample (+10 more)

### Community 30 - "build_training_dataset.py"
Cohesion: 0.16
Nodes (14): init_forest_database(), migrate_forest_database(), collect_change_features(), collect_hansen(), collect_hansen_shape(), collect_spectral(), load_valid_ids(), main() (+6 more)

### Community 31 - "renderCompareView"
Cohesion: 0.15
Nodes (23): availableCompareViews(), bestCompareDownload(), bindCompareControls(), compareDownloadMeta(), compareDownloads(), compareHansenOverlay(), compareViewOptions(), currentHansenLegendOverlay() (+15 more)

### Community 32 - "bindEvents"
Cohesion: 0.13
Nodes (23): bindEvents(), closeCompareView(), closeDbSyncView(), copyDebugEvents(), createExport(), currentFilters(), debugEventsText(), enhanceSidebarPanels() (+15 more)

### Community 33 - "threading"
Cohesion: 0.14
Nodes (16): BundleExportJobRequest, HansenCandidateMiningRequest, job_cancel_requested(), ModelLabelingJobRequest, run_bundle_export_job(), run_event_backfill_job(), run_hansen_candidate_mining_job(), run_hansen_sample_job() (+8 more)

### Community 34 - "pathlib"
Cohesion: 0.16
Nodes (8): evaluate_linear_probe(), group_by_samples_and_period(), load_metadata_from_folder(), class_similarity(), permutation_test(), permutation_test_post_vs_pre(), plot_pca(), similarity_gap()

### Community 35 - "map.js"
Cohesion: 0.23
Nodes (17): addCircle(), addVisiblePoint(), attachMapCaptureDebug(), bboxToBounds(), clearForestLayers(), debugMap(), ensureForestPanes(), eventTargetSummary() (+9 more)

### Community 36 - "event_review_service.py"
Cohesion: 0.26
Nodes (13): create_draft_review(), ensure_human_label_run(), event_review_payload(), _human_run_id(), _label_payload(), model_agreement(), review_completion(), roll_up_sample_reviews() (+5 more)

### Community 37 - "core.py"
Cohesion: 0.20
Nodes (9): aggregate_embeddings_by_sample_and_period(), calculate_cosine_similarity(), embedding_consistency_loo(), get_butch_embeddings(), get_embedding(), l2_normalize(), RemoteClipEmbedder, load_remoteclip_model() (+1 more)

### Community 38 - "ExperimentConfig"
Cohesion: 0.18
Nodes (6): ExperimentConfig, load_config(), stable_json_hash(), ModelArtifactTests, main(), parse_args()

### Community 39 - "DataOperationsTest"
Cohesion: 0.20
Nodes (4): HansenTile, analysis(), DataOperationsTest, sample()

### Community 40 - "artifacts.py"
Cohesion: 0.28
Nodes (9): export_remoteclip_classifier(), file_sha256(), _metric_number(), plan_remoteclip_export(), _portable_path(), _safe_version(), select_grouped_cv_metric(), _training_records() (+1 more)

### Community 41 - "numpy"
Cohesion: 0.24
Nodes (8): analyze_sample(), load_hansen_mask(), main(), plot_and_save(), resample_hansen(), main(), process_sample(), resample_hansen()

### Community 42 - "backend/config.py"
Cohesion: 0.23
Nodes (11): display_path(), latest_analysis(), main(), masks_for_analysis(), now_tag(), sample_rebuild_reason(), selected_samples(), write_report() (+3 more)

### Community 43 - "compute_manifest_embeddings"
Cohesion: 0.31
Nodes (4): compute_manifest_embeddings(), EmbeddingCache, ImageEmbedder, EmbeddingCacheTests

### Community 44 - "bindSentinelDownloadControls"
Cohesion: 0.33
Nodes (15): baseSentinelReview(), bindSentinelDownloadControls(), cancelSentinelReviewDraft(), effectiveSentinelReview(), findSentinelDownload(), renderSentinelDownloadsPanel(), restoreSentinelReview(), saveSentinelReviewDraft() (+7 more)

### Community 45 - "model_labeling_service.py"
Cohesion: 0.26
Nodes (5): build_model_labeling_plan(), _download_usable(), file_sha256(), model_artifact_status(), _safe_version()

### Community 46 - "sample_geometries"
Cohesion: 0.26
Nodes (7): _metric_geometry(), _overlap(), haversine_m(), sample_geometries(), square_around_point(), transformers_for_lonlat(), utm_epsg_for_lonlat()

### Community 47 - "build_change_maps.py"
Cohesion: 0.30
Nodes (7): aggregate_period(), is_strictly_excluded(), load_valid_ids(), main(), nbr_and_mask(), read_bands(), scene_datetime()

### Community 48 - "jobs.py"
Cohesion: 0.31
Nodes (3): BackgroundJob, InMemoryJobStore, utc_iso()

### Community 49 - "renderSentinelDownloads"
Cohesion: 0.22
Nodes (11): allDownloadedPreviews(), chooseDefaultSentinelPreview(), downloadedPreviews(), eventScopedSentinelDownloads(), filteredSentinelDownloads(), findPreviewByUrl(), renderImageIdSearch(), renderSentinelDownloads() (+3 more)

### Community 50 - "post"
Cohesion: 0.24
Nodes (8): analyze_hansen(), cancel_job(), fill_hansen_candidate_rows(), HansenAnalysisRequest, import_samples(), restore_hansen_candidate_exclusion(), start_hansen_sample_analysis(), upload_data_operation_source_file()

### Community 51 - "hansen_provider.py"
Cohesion: 0.27
Nodes (5): download_hansen_tile(), format_mb(), hansen_tile_path(), hansen_tile_url(), tile_model()

### Community 52 - "resolve_bundle_download_ids"
Cohesion: 0.31
Nodes (7): BundlePlanRequest, create_bundle_plan(), data_operation_input_path(), get_bundle_artifact(), get_data_operation_artifact(), get_missing_geotiff_artifact(), resolve_bundle_download_ids()

### Community 53 - "run_sentinel_batch_job"
Cohesion: 0.50
Nodes (6): batch_options_from_request(), run_sentinel_batch_job(), selected_batch_samples(), sentinel_batch_plan(), SentinelBatchRequest, start_sentinel_batch_job()

### Community 54 - "_tile_proposals"
Cohesion: 0.29
Nodes (7): _aoi_histogram(), _native_pixel_size_m(), _required_tile_ids(), _tile_proposals(), TileInventoryItem, tile_id_for_lonlat(), pixel_area_ha()

### Community 55 - "save_sentinel_review"
Cohesion: 0.38
Nodes (6): clean_optional_text(), clean_reason_codes(), EventReviewRequest, save_sample_event_review(), save_sentinel_review(), SentinelReviewRequest

### Community 56 - "inspect_change_map.py"
Cohesion: 0.62
Nodes (5): inspect_one(), load_hansen_mask(), main(), resample_hansen_to_sentinel(), save_png()

### Community 57 - "find_matched_pairs.py"
Cohesion: 0.48
Nodes (4): count_pairs(), haversine_km(), load_data(), main()

### Community 58 - "start_bundle_import_job"
Cohesion: 0.50
Nodes (4): BundleImportRequest, create_bundle_import_plan(), run_bundle_import_job(), start_bundle_import_job()

## Knowledge Gaps
- **5 isolated node(s):** `Batch Hansen Analysis`, `Frontend Stylesheet`, `CDSE_USE_ENV_PROXY Configuration`, `Sentinel Hub Process API`, `Pydantic >=2.11`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 240 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **66 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `SentinelDownload` connect `json` to `db_sync_service.py`, `analyze_sentinel_darkness_metrics.py`, `csv`, `event_review_service.py`, `models.py`, `resolve_storage_path`, `DataOperationsTest`, `model_inference_service.py`, `model_labeling_service.py`, `sentinel_batch_service.py`, `build_change_maps.py`, `generate_ml_data.py`, `test_pipeline_units.py`, `routes.py`, `typing`, `build_training_dataset.py`?**
  _High betweenness centrality (0.034) - this node is a cross-community bridge._
- **Are the 50 inferred relationships involving `SentinelDownload` (e.g. with `selected_downloads()` and `main()`) actually correct?**
  _`SentinelDownload` has 50 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Batch Hansen Analysis`, `Frontend Stylesheet`, `CDSE_USE_ENV_PROXY Configuration` to the rest of the system?**
  _5 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `json` be split into smaller, more focused modules?**
  _Cohesion score 0.06161314413741598 - nodes in this community are weakly interconnected._
- **Why does `resolve_storage_path()` connect `resolve_storage_path` to `json`, `analyze_sentinel_darkness_metrics.py`, `csv`, `numpy`, `backend/config.py`, `candidate_service.py`, `model_inference_service.py`, `model_labeling_service.py`, `sentinel_batch_service.py`, `build_change_maps.py`, `hansen_service.py`, `routes.py`, `inspect_change_map.py`, `Any`, `typing`, `rasterio`, `build_training_dataset.py`?**
  _High betweenness centrality (0.022) - this node is a cross-community bridge._
- **Should `main.py` be split into smaller, more focused modules?**
  _Cohesion score 0.059018367961457395 - nodes in this community are weakly interconnected._
- **Why does `SearchPeriod` connect `json` to `sentinel_batch_service.py`?**
  _High betweenness centrality (0.016) - this node is a cross-community bridge._