# Graph Report - ML-SbPSTU-2026  (2026-10-29)

## Corpus Check
- 111 files · ~128,622 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 34 file(s) not represented in the graph (top: .csv 16, .sqlite 5, .geojson 4)

## Summary
- 1816 nodes · 6475 edges · 68 communities (57 shown, 11 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 464 edges (avg confidence: 0.93)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `d397bffa`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- routes.py
- sentinel_service.py
- main.py
- db_sync_service.py
- generate_ml_data.py
- analyze_sentinel_darkness_metrics.py
- SentinelDownload
- model_inference_service.py
- ui.js
- json
- app.js
- test_pipeline_units.py
- Research Tool Python Dependencies
- candidate_service.py
- samples.js
- hansen_service.py
- reporting.py
- typing
- data_operations.js
- fetch_cdse_init_data.py
- compare_forest_databases.py
- bundle_service.py
- xgboost/v1/manifest.json
- escapeHtml
- resolve_storage_path
- renderDetail
- pathlib
- hashlib
- embeddings.py
- numpy
- auto_review_sentinel_quality.py
- renderCompareView
- bindEvents
- backend/config.py
- open_clip_test.py
- dataset.py
- pandas
- map.js
- ExperimentConfig
- metrics
- remoteclip_logreg/v1/manifest.json
- bindSentinelDownloadControls
- HansenTile
- prefetch_hansen_tiles.py
- renderSentinelDownloads
- sample_geometries
- to_storage_path
- BackgroundJob
- mark_hansen_multiple_events.py
- localize_changes.py
- training_cohort
- collections
- embedding
- classifier_config
- training_database_fingerprint
- round_trip_smoke_test
- software
- raster_palette.js
- state.js
- inspect_change_map.py
- README.md

## God Nodes (most connected - your core abstractions)
1. `SentinelDownload` - 103 edges
2. `Sample` - 82 edges
3. `resolve_storage_path()` - 58 edges
4. `SampleEvent` - 57 edges
5. `execute_sentinel_batch()` - 52 edges
6. `to_storage_path()` - 51 edges
7. `DataOperationsTest` - 42 edges
8. `analyze_sample_hansen()` - 39 edges
9. `renderDetail()` - 37 edges
10. `utc_now()` - 35 edges

## Surprising Connections (you probably didn't know these)
- `selected_downloads()` --uses--> `SentinelDownload`  [INFERRED]
  Scripts/ResearchScripts/analyze_sentinel_darkness_metrics.py → Scripts/ResearchTool/backend/forest/models.py
- `manual_payload()` --uses--> `ManualValidation`  [INFERRED]
  Scripts/ResearchScripts/analyze_sentinel_darkness_metrics.py → Scripts/ResearchTool/backend/forest/models.py
- `review_payload()` --uses--> `SentinelSceneReview`  [INFERRED]
  Scripts/ResearchScripts/analyze_sentinel_darkness_metrics.py → Scripts/ResearchTool/backend/forest/models.py
- `row_base()` --uses--> `SentinelDownload`  [INFERRED]
  Scripts/ResearchScripts/auto_review_sentinel_quality.py → Scripts/ResearchTool/backend/forest/models.py
- `review_payload()` --uses--> `SentinelSceneReview`  [INFERRED]
  Scripts/ResearchScripts/auto_review_sentinel_quality.py → Scripts/ResearchTool/backend/forest/models.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Forest Sample Review Workflow** — scripts_researchtool_frontend_index_forest_mode, scripts_researchtool_frontend_index_forest_sample_import_filters, scripts_researchtool_frontend_index_batch_hansen_analysis, scripts_researchtool_frontend_index_forest_database_sync, scripts_researchtool_frontend_index_pre_post_compare, scripts_researchtool_frontend_index_forest_layer_controls [EXTRACTED 1.00]
- **Pipeline Geometry and AOI Workflow** — scripts_researchtool_readme_kml_protection_zone, scripts_researchtool_readme_closed_linestring_polygon, scripts_researchtool_readme_raster_medial_axis_centerline, scripts_researchtool_readme_centerline_segments, scripts_researchtool_readme_square_aoi, scripts_researchtool_frontend_index_pipeline_kml_aoi_controls [INFERRED 0.85]
- **Sentinel-2 Search, Fetch, and Preview Workflow** — scripts_researchtool_readme_cdse_stac_search, scripts_researchtool_readme_sentinel_2_raster_fetch, scripts_researchtool_readme_multispectral_previews, scripts_researchtool_frontend_index_sentinel_2_search_controls, scripts_researchtool_frontend_index_pipeline_observations_drawer, scripts_researchtool_frontend_index_raster_preview_controls [INFERRED 0.85]

## Communities (68 total, 11 thin omitted)

### Community 0 - "routes.py"
Cohesion: 0.05
Nodes (106): get_db(), DbSyncError, haversine_m(), ForestExport, Sample, analyze_hansen(), analyze_hansen_batch(), batch_options_from_request() (+98 more)

### Community 1 - "sentinel_service.py"
Cohesion: 0.07
Nodes (65): build_process_request(), can_download(), decode_json(), display_path(), download_raster(), expected_preview_dir(), main(), max_cloud_from_metadata() (+57 more)

### Community 2 - "main.py"
Cohesion: 0.06
Nodes (52): bbox_to_web_mercator(), best_s2_observations(), cdse_post(), cdse_session(), cdse_uses_env_proxy(), cloud_cover(), configured_proxy_keys(), env_flag() (+44 more)

### Community 3 - "db_sync_service.py"
Cohesion: 0.12
Nodes (63): add_summary(), allowed_bases_dir(), append_note(), apply_db_sync_plan(), apply_operation(), bool_value(), build_db_sync_plan(), candidate_databases() (+55 more)

### Community 4 - "generate_ml_data.py"
Cohesion: 0.15
Nodes (26): create_metadata_for_images(), derived_relative_path(), display_path(), get_images_from_db(), get_original_image_path(), get_path_to_image(), main(), resolve_project_path() (+18 more)

### Community 5 - "analyze_sentinel_darkness_metrics.py"
Cohesion: 0.09
Nodes (47): candidate_stats(), combined_or_rule_stats(), combined_or_rule_sweep_rows(), contrast_payload(), finite_number(), finite_values(), fraction_below(), histogram() (+39 more)

### Community 6 - "SentinelDownload"
Cohesion: 0.05
Nodes (88): EventBackfillTests, sample(), configured_annotator_name(), create_draft_review(), ensure_human_label_run(), event_review_payload(), _human_run_id(), _label_payload() (+80 more)

### Community 7 - "model_inference_service.py"
Cohesion: 0.09
Nodes (26): median_feature_rows(), median_stats(), pre_post_features_from_stats(), pre_post_spectral_features(), stats_for_tif(), _create_label_run(), EmbeddingCache, _event_inputs() (+18 more)

### Community 8 - "ui.js"
Cohesion: 0.08
Nodes (46): applyCompareHansenFilter(), applyCompareImageFilter(), applyMainRasterFilters(), applyVisibleRasterFilters(), bboxToBounds(), colorKey(), compareDownloadById(), compareDownloadOptions() (+38 more)

### Community 9 - "json"
Cohesion: 0.09
Nodes (44): add_filter_columns(), built_in_russia_geometry(), extract_csv(), extract_geojson(), feature_matches_russia(), is_russia_point(), iter_input_files(), load_russia_geometry() (+36 more)

### Community 10 - "app.js"
Cohesion: 0.13
Nodes (45): api(), bboxKey(), bboxToLeafletBounds(), bindEvents(), buildSegments(), checkedObservations(), clearGeometryLayers(), clearLayer() (+37 more)

### Community 11 - "test_pipeline_units.py"
Cohesion: 0.20
Nodes (10): export_remoteclip_classifier(), file_sha256(), _metric_number(), plan_remoteclip_export(), _portable_path(), _safe_version(), select_grouped_cv_metric(), _training_records() (+2 more)

### Community 12 - "Research Tool Python Dependencies"
Cohesion: 0.08
Nodes (38): Pipeline Application Script, Batch Hansen Analysis, Forest Database Synchronization, Forest Map Layer Controls, Forest Mode, Forest Sample Import and Filtering, Forest UI Module, Leaflet 1.9.4 (+30 more)

### Community 13 - "candidate_service.py"
Cohesion: 0.15
Nodes (31): _active_exclusions(), candidate_hansen_preview_payload(), _candidate_identity(), _candidate_lossyear_raster(), _candidate_lossyear_sources(), _candidate_mask_ready_sample_ids(), candidate_mask_run_status(), candidate_run_payload() (+23 more)

### Community 14 - "samples.js"
Cohesion: 0.05
Nodes (4): cleanFilters(), exportForestSamples(), listForestSamples(), startHansenBatch()

### Community 15 - "hansen_service.py"
Cohesion: 0.16
Nodes (26): tile_id_for_lonlat(), tile_model(), analyze_sample_hansen(), apply_sample_multiple_event_flag(), cache_url(), existing_hansen_analysis(), existing_lossyear_tile(), hansen_tile_preview_payload() (+18 more)

### Community 16 - "reporting.py"
Cohesion: 0.17
Nodes (12): ForestMLPipeline, baseline_metrics_html(), create_run_directory(), _csv_value(), json_value(), _metric_text(), plot_baseline_metrics(), read_csv() (+4 more)

### Community 17 - "typing"
Cohesion: 0.19
Nodes (6): load_config(), apply_dominant_event_backfill(), backup_sqlite_database(), plan_dominant_event_backfill(), virtual_backfill_rows(), parse_args()

### Community 18 - "data_operations.js"
Cohesion: 0.17
Nodes (35): activeJobHost(), activeMiningCandidate(), artifactUrl(), checkedValues(), dataOperationsRunning(), downloadPayload(), ensureMiningMap(), escapeHtml() (+27 more)

### Community 19 - "fetch_cdse_init_data.py"
Cohesion: 0.14
Nodes (22): Area, build_catalog_request(), build_process_request(), catalog_summary(), CollectionSpec, download_chip(), fetch_token(), import_requests() (+14 more)

### Community 20 - "compare_forest_databases.py"
Cohesion: 0.20
Nodes (25): main(), column_names(), compare_hansen_analyses(), compare_hansen_tiles(), compare_manual(), compare_statuses(), connect(), count_by() (+17 more)

### Community 21 - "bundle_service.py"
Cohesion: 0.16
Nodes (29): main(), selected_bundle_path(), add_bundle_file(), bundle_candidates(), bundle_member_exists(), bundle_member_sha256(), bundle_plan(), collect_bundle_files() (+21 more)

### Community 22 - "xgboost/v1/manifest.json"
Cohesion: 0.06
Nodes (32): artifact_path, artifact_schema_version, artifact_sha256, Logging, Wildfire, class_order, created_at, feature_names (+24 more)

### Community 23 - "escapeHtml"
Cohesion: 0.11
Nodes (33): cancelDbSyncSentinelCacheJob(), dbSyncSentinelCache(), dbSyncSentinelItems(), dbSyncStatus(), escapeHtml(), hansenStageLabel(), isHansenJobRunning(), isTerminalJobStatus() (+25 more)

### Community 24 - "resolve_storage_path"
Cohesion: 0.22
Nodes (18): main(), refresh_mask_png(), build_existing_mask_index(), display_path(), find_existing_tif(), main(), now_tag(), png_for_mask() (+10 more)

### Community 25 - "renderDetail"
Cohesion: 0.16
Nodes (32): cancelBatchHansen(), cancelHansenAnalysis(), canCompareSentinel(), clearSelectedSample(), downloadCheckedSentinelScenes(), ensureActiveSentinelPreviewForSample(), fitMapToHansenTile(), loadSelectedHansenTilePreview() (+24 more)

### Community 26 - "pathlib"
Cohesion: 0.13
Nodes (17): main(), selected_download_ids(), iter_hansen_ready_unvalidated_samples(), iter_hansen_too_old_samples(), iter_unvalidated_samples(), upsert_manual_validation(), aggregate_period(), is_strictly_excluded() (+9 more)

### Community 27 - "hashlib"
Cohesion: 0.10
Nodes (17): evaluate_linear_probe(), group_by_samples_and_period(), load_metadata_from_folder(), main(), train_features(), main(), train_features(), main() (+9 more)

### Community 28 - "embeddings.py"
Cohesion: 0.11
Nodes (15): stable_json_hash(), aggregate_embeddings_by_sample_and_period(), calculate_cosine_similarity(), embedding_consistency_loo(), get_butch_embeddings(), get_embedding(), resolve_manifest_path(), compute_manifest_embeddings() (+7 more)

### Community 29 - "numpy"
Cohesion: 0.16
Nodes (17): aggregate_event_embeddings(), build_nomanual_review_queue(), _evaluate_model(), feature_vector(), fit_classifier(), _fit_model(), records_for_variant(), run_baseline_experiments() (+9 more)

### Community 30 - "auto_review_sentinel_quality.py"
Cohesion: 0.24
Nodes (14): classify_review_action(), clean_reason_codes(), main(), manual_validation_payload(), now_tag(), print_reason_stats(), print_valid_sample_conflicts(), replace_review_reasons() (+6 more)

### Community 31 - "renderCompareView"
Cohesion: 0.15
Nodes (23): availableCompareViews(), bestCompareDownload(), bindCompareControls(), compareDownloadMeta(), compareDownloads(), compareHansenOverlay(), compareViewOptions(), currentHansenLegendOverlay() (+15 more)

### Community 32 - "bindEvents"
Cohesion: 0.13
Nodes (23): bindEvents(), closeCompareView(), closeDbSyncView(), copyDebugEvents(), createExport(), currentFilters(), debugEventsText(), enhanceSidebarPanels() (+15 more)

### Community 33 - "backend/config.py"
Cohesion: 0.16
Nodes (16): main(), selected_samples(), display_path(), latest_analysis(), main(), masks_for_analysis(), now_tag(), sample_rebuild_reason() (+8 more)

### Community 34 - "open_clip_test.py"
Cohesion: 0.27
Nodes (5): class_similarity(), permutation_test(), permutation_test_post_vs_pre(), plot_pca(), similarity_gap()

### Community 35 - "dataset.py"
Cohesion: 0.24
Nodes (9): build_event_manifest(), database_fingerprint(), file_sha256(), _preview_map(), project_relative_path(), _reason_map(), season_for_datetime(), validation_cohort() (+1 more)

### Community 36 - "pandas"
Cohesion: 0.17
Nodes (13): init_forest_database(), migrate_forest_database(), collect_change_features(), collect_hansen(), collect_hansen_shape(), collect_spectral(), load_valid_ids(), main() (+5 more)

### Community 37 - "map.js"
Cohesion: 0.23
Nodes (17): addCircle(), addVisiblePoint(), attachMapCaptureDebug(), bboxToBounds(), clearForestLayers(), debugMap(), ensureForestPanes(), eventTargetSummary() (+9 more)

### Community 39 - "metrics"
Cohesion: 0.12
Nodes (16): accuracy_mean, accuracy_std, balanced_accuracy_mean, balanced_accuracy_std, f1_mean, f1_std, feature_variant, fold_runs (+8 more)

### Community 40 - "remoteclip_logreg/v1/manifest.json"
Cohesion: 0.13
Nodes (14): artifact_path, artifact_schema_version, artifact_sha256, class_order, created_at, decision_threshold, experiment_config_fingerprint, feature_dimension (+6 more)

### Community 41 - "bindSentinelDownloadControls"
Cohesion: 0.33
Nodes (15): baseSentinelReview(), bindSentinelDownloadControls(), cancelSentinelReviewDraft(), effectiveSentinelReview(), findSentinelDownload(), renderSentinelDownloadsPanel(), restoreSentinelReview(), saveSentinelReviewDraft() (+7 more)

### Community 42 - "HansenTile"
Cohesion: 0.17
Nodes (9): _aoi_histogram(), CandidateMiningOptions, hansen_tile_inventory(), _native_pixel_size_m(), _required_tile_ids(), _tile_proposals(), TileInventoryItem, pixel_area_ha() (+1 more)

### Community 43 - "prefetch_hansen_tiles.py"
Cohesion: 0.42
Nodes (8): existing_tile_record(), main(), now_tag(), planned_tile_rows(), progress(), unique_sample_tiles(), upsert_tile_record(), write_report()

### Community 44 - "renderSentinelDownloads"
Cohesion: 0.22
Nodes (11): allDownloadedPreviews(), chooseDefaultSentinelPreview(), downloadedPreviews(), eventScopedSentinelDownloads(), filteredSentinelDownloads(), findPreviewByUrl(), renderImageIdSearch(), renderSentinelDownloads() (+3 more)

### Community 45 - "sample_geometries"
Cohesion: 0.20
Nodes (7): _metric_geometry(), _overlap(), _round_robin_pool(), sample_geometries(), square_around_point(), transformers_for_lonlat(), utm_epsg_for_lonlat()

### Community 46 - "to_storage_path"
Cohesion: 0.23
Nodes (7): download_hansen_tile(), format_mb(), hansen_tile_path(), hansen_tile_url(), forest_cache_url(), _normalize_relative(), to_storage_path()

### Community 47 - "BackgroundJob"
Cohesion: 0.31
Nodes (3): BackgroundJob, InMemoryJobStore, utc_iso()

### Community 48 - "mark_hansen_multiple_events.py"
Cohesion: 0.42
Nodes (7): append_manual_note(), hansen_comment(), latest_analyses(), main(), notes_with_hansen_comment(), now_tag(), write_report()

### Community 49 - "localize_changes.py"
Cohesion: 0.33
Nodes (4): analyze_sample(), main(), plot_and_save(), resample_hansen()

### Community 50 - "training_cohort"
Cohesion: 0.25
Nodes (8): Logging, Wildfire, training_cohort, class_counts, mode, record_ids_sha256, records, samples

### Community 51 - "collections"
Cohesion: 0.43
Nodes (4): main(), now_tag(), preview_rows(), write_report()

### Community 52 - "embedding"
Cohesion: 0.33
Nodes (6): embedding, checkpoint_path, checkpoint_sha256, model_name, normalization_version, preprocessing_version

### Community 53 - "classifier_config"
Cohesion: 0.40
Nodes (5): classifier_config, class_weight, max_iter, random_state, solver

### Community 54 - "training_database_fingerprint"
Cohesion: 0.40
Nodes (5): training_database_fingerprint, mtime_ns, path, sha256, size

### Community 55 - "round_trip_smoke_test"
Cohesion: 0.67
Nodes (3): round_trip_smoke_test, probability_wildfire, status

### Community 56 - "software"
Cohesion: 0.67
Nodes (3): software, numpy, scikit_learn

### Community 62 - "inspect_change_map.py"
Cohesion: 0.67
Nodes (4): inspect_one(), main(), resample_hansen_to_sentinel(), save_png()

## Knowledge Gaps
- **5 isolated node(s):** `Batch Hansen Analysis`, `Frontend Stylesheet`, `CDSE_USE_ENV_PROXY Configuration`, `Sentinel Hub Process API`, `Pydantic >=2.11`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 260 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `SentinelDownload` connect `SentinelDownload` to `routes.py`, `sentinel_service.py`, `dataset.py`, `generate_ml_data.py`, `analyze_sentinel_darkness_metrics.py`, `db_sync_service.py`, `pandas`, `model_inference_service.py`, `test_pipeline_units.py`, `typing`, `collections`, `bundle_service.py`, `pathlib`, `auto_review_sentinel_quality.py`?**
  _High betweenness centrality (0.035) - this node is a cross-community bridge._
- **Are the 74 inferred relationships involving `SentinelDownload` (e.g. with `selected_downloads()` and `main()`) actually correct?**
  _`SentinelDownload` has 74 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Batch Hansen Analysis`, `Frontend Stylesheet`, `CDSE_USE_ENV_PROXY Configuration` to the rest of the system?**
  _5 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `routes.py` be split into smaller, more focused modules?**
  _Cohesion score 0.052993375828021494 - nodes in this community are weakly interconnected._
- **Why does `Sample` connect `routes.py` to `backend/config.py`, `sentinel_service.py`, `dataset.py`, `generate_ml_data.py`, `db_sync_service.py`, `SentinelDownload`, `pandas`, `model_inference_service.py`, `json`, `test_pipeline_units.py`, `prefetch_hansen_tiles.py`, `candidate_service.py`, `hansen_service.py`, `mark_hansen_multiple_events.py`, `pathlib`?**
  _High betweenness centrality (0.024) - this node is a cross-community bridge._
- **Are the 57 inferred relationships involving `Sample` (e.g. with `selected_samples()` and `iter_hansen_ready_unvalidated_samples()`) actually correct?**
  _`Sample` has 57 INFERRED edges - model-reasoned connections that need verification._
- **Should `sentinel_service.py` be split into smaller, more focused modules?**
  _Cohesion score 0.07160493827160494 - nodes in this community are weakly interconnected._