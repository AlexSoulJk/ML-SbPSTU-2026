# Graph Report - ML-SbPSTU-2026  (2026-10-18)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 1330 nodes · 4495 edges · 63 communities (37 shown, 26 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 323 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `f1234926`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- routes.py
- main.py
- hansen_service.py
- db_sync_service.py
- analyze_sentinel_darkness_metrics.py
- csv
- ui.js
- app.js
- Research Tool Python Dependencies
- SentinelDownload
- fetch_cdse_init_data.py
- bundle_service.py
- import_service.py
- renderDetail
- models.py
- json
- generate_ml_data.py
- sentinel_batch_service.py
- pandas
- samples.js
- compare_forest_databases.py
- renderCompareView
- data_operations.js
- pathlib
- database.py
- linear_probe_test.py
- renderDbSyncCacheRepair
- map.js
- ensureActiveSentinelPreviewForSample
- numpy
- SentinelSceneReview
- config.py
- bindSentinelDownloadControls
- escapeHtml
- build_change_maps.py
- forest/geometry.py
- inspect_change_map.py
- SearchPeriod
- state.js

## God Nodes (most connected - your core abstractions)
1. `SentinelDownload` - 76 edges
2. `Sample` - 65 edges
3. `analyze_sample_hansen()` - 39 edges
4. `resolve_storage_path()` - 37 edges
5. `execute_sentinel_batch()` - 36 edges
6. `to_storage_path()` - 36 edges
7. `renderDetail()` - 35 edges
8. `HansenAnalysis` - 30 edges
9. `SentinelSceneReview` - 29 edges
10. `setForestStatus()` - 29 edges

## Surprising Connections (you probably didn't know these)
- `iter_hansen_ready_unvalidated_samples()` --uses--> `Sample`  [INFERRED]
  Scripts/ResearchScripts/filter_to_old_points.py → Scripts/ResearchTool/backend/forest/models.py
- `iter_hansen_too_old_samples()` --uses--> `Sample`  [INFERRED]
  Scripts/ResearchScripts/filter_to_old_points.py → Scripts/ResearchTool/backend/forest/models.py
- `iter_unvalidated_samples()` --uses--> `Sample`  [INFERRED]
  Scripts/ResearchScripts/filter_to_old_points.py → Scripts/ResearchTool/backend/forest/models.py
- `get_images_from_db()` --uses--> `Sample`  [INFERRED]
  Scripts/ResearchScripts/generate_ml_data.py → Scripts/ResearchTool/backend/forest/models.py
- `main()` --uses--> `Sample`  [INFERRED]
  Scripts/ResearchScripts/mark_hansen_multiple_events.py → Scripts/ResearchTool/backend/forest/models.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Forest Sample Review Workflow** — scripts_researchtool_frontend_index_forest_mode, scripts_researchtool_frontend_index_forest_sample_import_filters, scripts_researchtool_frontend_index_batch_hansen_analysis, scripts_researchtool_frontend_index_forest_database_sync, scripts_researchtool_frontend_index_pre_post_compare, scripts_researchtool_frontend_index_forest_layer_controls [EXTRACTED 1.00]
- **Pipeline Geometry and AOI Workflow** — scripts_researchtool_readme_kml_protection_zone, scripts_researchtool_readme_closed_linestring_polygon, scripts_researchtool_readme_raster_medial_axis_centerline, scripts_researchtool_readme_centerline_segments, scripts_researchtool_readme_square_aoi, scripts_researchtool_frontend_index_pipeline_kml_aoi_controls [INFERRED 0.85]
- **Sentinel-2 Search, Fetch, and Preview Workflow** — scripts_researchtool_readme_cdse_stac_search, scripts_researchtool_readme_sentinel_2_raster_fetch, scripts_researchtool_readme_multispectral_previews, scripts_researchtool_frontend_index_sentinel_2_search_controls, scripts_researchtool_frontend_index_pipeline_observations_drawer, scripts_researchtool_frontend_index_raster_preview_controls [INFERRED 0.85]

## Communities (63 total, 26 thin omitted)

### Community 0 - "routes.py"
Cohesion: 0.07
Nodes (82): bundle_candidates(), get_db(), haversine_m(), ForestExport, Sample, analyze_hansen(), analyze_hansen_batch(), batch_options_from_request() (+74 more)

### Community 1 - "main.py"
Cohesion: 0.06
Nodes (55): bbox_to_web_mercator(), best_s2_observations(), cdse_post(), cdse_session(), cdse_uses_env_proxy(), cloud_cover(), configured_proxy_keys(), env_flag() (+47 more)

### Community 2 - "hansen_service.py"
Cohesion: 0.08
Nodes (56): existing_tile_record(), main(), now_tag(), planned_tile_rows(), progress(), unique_sample_tiles(), upsert_tile_record(), write_report() (+48 more)

### Community 3 - "db_sync_service.py"
Cohesion: 0.10
Nodes (65): iter_hansen_ready_unvalidated_samples(), iter_hansen_too_old_samples(), iter_unvalidated_samples(), upsert_manual_validation(), add_summary(), allowed_bases_dir(), append_note(), apply_db_sync_plan() (+57 more)

### Community 4 - "analyze_sentinel_darkness_metrics.py"
Cohesion: 0.09
Nodes (47): candidate_stats(), combined_or_rule_stats(), combined_or_rule_sweep_rows(), contrast_payload(), finite_number(), finite_values(), fraction_below(), histogram() (+39 more)

### Community 5 - "csv"
Cohesion: 0.09
Nodes (36): add_filter_columns(), built_in_russia_geometry(), extract_csv(), extract_geojson(), feature_matches_russia(), is_russia_point(), iter_input_files(), load_russia_geometry() (+28 more)

### Community 6 - "ui.js"
Cohesion: 0.08
Nodes (43): applyCompareHansenFilter(), applyCompareImageFilter(), applyMainRasterFilters(), applyVisibleRasterFilters(), bboxToBounds(), colorKey(), compareDownloadById(), currentHansenOverlay() (+35 more)

### Community 7 - "app.js"
Cohesion: 0.13
Nodes (45): api(), bboxKey(), bboxToLeafletBounds(), bindEvents(), buildSegments(), checkedObservations(), clearGeometryLayers(), clearLayer() (+37 more)

### Community 8 - "Research Tool Python Dependencies"
Cohesion: 0.08
Nodes (38): Pipeline Application Script, Batch Hansen Analysis, Forest Database Synchronization, Forest Map Layer Controls, Forest Mode, Forest Sample Import and Filtering, Forest UI Module, Leaflet 1.9.4 (+30 more)

### Community 9 - "SentinelDownload"
Cohesion: 0.14
Nodes (36): build_process_request(), can_download(), decode_json(), display_path(), download_raster(), expected_preview_dir(), main(), max_cloud_from_metadata() (+28 more)

### Community 10 - "fetch_cdse_init_data.py"
Cohesion: 0.12
Nodes (23): Area, build_catalog_request(), build_process_request(), catalog_summary(), CollectionSpec, download_chip(), fetch_token(), import_requests() (+15 more)

### Community 11 - "bundle_service.py"
Cohesion: 0.17
Nodes (27): main(), selected_bundle_path(), add_bundle_file(), bundle_member_exists(), bundle_member_sha256(), bundle_plan(), collect_bundle_files(), copy_bundle_member() (+19 more)

### Community 12 - "import_service.py"
Cohesion: 0.12
Nodes (21): clean_text(), detect_format(), geometry_point(), import_file(), lookup(), normalize_key(), normalize_sample(), NormalizedSample (+13 more)

### Community 13 - "renderDetail"
Cohesion: 0.15
Nodes (37): bindCompareControls(), bindEvents(), cancelBatchHansen(), cancelHansenAnalysis(), canCompareSentinel(), clearSelectedSample(), closeCompareView(), closeDbSyncView() (+29 more)

### Community 14 - "models.py"
Cohesion: 0.16
Nodes (20): backfill_hansen_events(), decode_histogram(), deterministic_id(), event_download_id_for(), event_id_for(), hansen_event_specs(), latest_analyses(), link_unambiguous_legacy_downloads() (+12 more)

### Community 15 - "json"
Cohesion: 0.18
Nodes (30): SentinelSceneReviewReason, apply_auto_scene_review(), auto_review_reason(), auto_review_reasons(), cache_url(), cloud_metrics(), colorize_index(), derive_previews() (+22 more)

### Community 16 - "generate_ml_data.py"
Cohesion: 0.15
Nodes (26): create_metadata_for_images(), derived_relative_path(), display_path(), get_images_from_db(), get_original_image_path(), get_path_to_image(), main(), resolve_project_path() (+18 more)

### Community 17 - "sentinel_batch_service.py"
Cohesion: 0.15
Nodes (26): main(), selected_samples(), SentinelSearch, BatchOptions, build_operation_rows(), calendar_year_for_period(), date_gap_ok(), decode_json() (+18 more)

### Community 18 - "pandas"
Cohesion: 0.15
Nodes (15): count_pairs(), haversine_km(), load_data(), main(), main(), train_features(), main(), train_features() (+7 more)

### Community 19 - "samples.js"
Cohesion: 0.07
Nodes (4): cleanFilters(), exportForestSamples(), listForestSamples(), startHansenBatch()

### Community 20 - "compare_forest_databases.py"
Cohesion: 0.20
Nodes (25): main(), column_names(), compare_hansen_analyses(), compare_hansen_tiles(), compare_manual(), compare_statuses(), connect(), count_by() (+17 more)

### Community 21 - "renderCompareView"
Cohesion: 0.13
Nodes (24): availableCompareViews(), bestCompareDownload(), compareDownloadMeta(), compareDownloadOptions(), compareDownloads(), compareHansenOverlay(), compareOptionLabel(), compareViewOptions() (+16 more)

### Community 22 - "data_operations.js"
Cohesion: 0.26
Nodes (20): activeJobHost(), artifactUrl(), checkedValues(), dataOperationsRunning(), downloadPayload(), escapeHtml(), humanBytes(), initDataOperations() (+12 more)

### Community 23 - "pathlib"
Cohesion: 0.16
Nodes (3): main(), selected_download_ids(), export()

### Community 24 - "database.py"
Cohesion: 0.16
Nodes (9): init_forest_database(), migrate_forest_database(), collect_hansen(), collect_hansen_shape(), collect_spectral(), load_valid_ids(), main(), stats_for_tif() (+1 more)

### Community 25 - "linear_probe_test.py"
Cohesion: 0.14
Nodes (8): evaluate_linear_probe(), group_by_samples_and_period(), load_metadata_from_folder(), class_similarity(), permutation_test(), permutation_test_post_vs_pre(), plot_pca(), similarity_gap()

### Community 26 - "renderDbSyncCacheRepair"
Cohesion: 0.16
Nodes (20): cancelDbSyncSentinelCacheJob(), dbSyncSentinelCache(), dbSyncSentinelItems(), dbSyncStatus(), isHansenJobRunning(), isTerminalJobStatus(), operationStatusClass(), pollDbSyncSentinelCacheJob() (+12 more)

### Community 27 - "map.js"
Cohesion: 0.23
Nodes (17): addCircle(), addVisiblePoint(), attachMapCaptureDebug(), bboxToBounds(), clearForestLayers(), debugMap(), ensureForestPanes(), eventTargetSummary() (+9 more)

### Community 28 - "ensureActiveSentinelPreviewForSample"
Cohesion: 0.13
Nodes (19): allDownloadedPreviews(), chooseDefaultSentinelPreview(), defaultSelectedSceneKeys(), downloadCheckedSentinelScenes(), downloadedPreviews(), ensureActiveSentinelPreviewForSample(), eventScopedSentinelDownloads(), filteredSentinelDownloads() (+11 more)

### Community 29 - "numpy"
Cohesion: 0.20
Nodes (8): aggregate_embeddings_by_sample_and_period(), calculate_cosine_similarity(), embedding_consistency_loo(), get_butch_embeddings(), get_embedding(), main(), process_sample(), resample_hansen()

### Community 30 - "SentinelSceneReview"
Cohesion: 0.24
Nodes (15): classify_review_action(), clean_reason_codes(), main(), manual_validation_payload(), now_tag(), print_reason_stats(), print_valid_sample_conflicts(), replace_review_reasons() (+7 more)

### Community 31 - "config.py"
Cohesion: 0.23
Nodes (11): display_path(), latest_analysis(), main(), masks_for_analysis(), now_tag(), sample_rebuild_reason(), selected_samples(), write_report() (+3 more)

### Community 32 - "bindSentinelDownloadControls"
Cohesion: 0.31
Nodes (16): baseSentinelReview(), bindSentinelDownloadControls(), cancelSentinelReviewDraft(), effectiveSentinelReview(), findSentinelDownload(), refreshSelectedSampleAfterSentinelReview(), renderSentinelDownloadsPanel(), restoreSentinelReview() (+8 more)

### Community 33 - "escapeHtml"
Cohesion: 0.17
Nodes (16): escapeHtml(), hansenStageLabel(), previewButton(), renderHansenJobStatus(), renderHansenSummary(), renderPointInfo(), renderSamplesList(), renderSentinelDownloadRow() (+8 more)

### Community 34 - "build_change_maps.py"
Cohesion: 0.25
Nodes (8): aggregate_period(), event_reference_date(), is_strictly_excluded(), load_valid_ids(), main(), nbr_and_mask(), read_bands(), scene_datetime()

### Community 35 - "forest/geometry.py"
Cohesion: 0.39
Nodes (4): sample_geometries(), square_around_point(), transformers_for_lonlat(), utm_epsg_for_lonlat()

### Community 36 - "inspect_change_map.py"
Cohesion: 0.62
Nodes (5): inspect_one(), load_hansen_mask(), main(), resample_hansen_to_sentinel(), save_png()

## Knowledge Gaps
- **5 isolated node(s):** `Batch Hansen Analysis`, `Frontend Stylesheet`, `CDSE_USE_ENV_PROXY Configuration`, `Sentinel Hub Process API`, `Pydantic >=2.11`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 163 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **26 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Sample` connect `routes.py` to `hansen_service.py`, `db_sync_service.py`, `csv`, `fetch_cdse_init_data.py`, `import_service.py`, `models.py`, `json`, `generate_ml_data.py`, `sentinel_batch_service.py`, `pandas`, `pathlib`, `database.py`, `config.py`?**
  _High betweenness centrality (0.026) - this node is a cross-community bridge._
- **Are the 60 inferred relationships involving `SentinelDownload` (e.g. with `selected_downloads()` and `main()`) actually correct?**
  _`SentinelDownload` has 60 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Batch Hansen Analysis`, `Frontend Stylesheet`, `CDSE_USE_ENV_PROXY Configuration` to the rest of the system?**
  _5 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `routes.py` be split into smaller, more focused modules?**
  _Cohesion score 0.06720263863121007 - nodes in this community are weakly interconnected._
- **Why does `analyze_sample_hansen()` connect `hansen_service.py` to `routes.py`, `forest/geometry.py`, `fetch_cdse_init_data.py`, `import_service.py`, `models.py`, `json`, `numpy`, `config.py`?**
  _High betweenness centrality (0.022) - this node is a cross-community bridge._
- **Are the 47 inferred relationships involving `Sample` (e.g. with `iter_hansen_ready_unvalidated_samples()` and `iter_hansen_too_old_samples()`) actually correct?**
  _`Sample` has 47 INFERRED edges - model-reasoned connections that need verification._
- **Should `main.py` be split into smaller, more focused modules?**
  _Cohesion score 0.057946069994262765 - nodes in this community are weakly interconnected._