# Graph Report - ML-SbPSTU-2026  (2026-10-18)

## Corpus Check
- 76 files · ~57,393 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 23 file(s) not represented in the graph (top: .csv 9, .sqlite 4, .geojson 4)

## Summary
- 1178 nodes · 3950 edges · 43 communities (35 shown, 8 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 261 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Sentinel Review Workflow
- Forest Sample API
- CDSE Pipeline API
- Database Sync Engine
- Hansen ML Experiments
- Pipeline Frontend Map
- Forest UI Structure
- Sentinel Download Cache
- Forest Data Import
- Compare Raster Rendering
- Geospatial Segmentation
- Russia Data Extraction
- Hansen Analysis Service
- Jobs And Path Repair
- Sentinel Hub Client
- Sentinel QC Calibration
- Compare Layer Controls
- Sentinel Bundle Export
- Sentinel Bundle Import
- Database Comparison Script
- Forest UI Orchestration
- Sync Cache Job UI
- Hansen Tile Prefetch
- Sentinel Batch UI
- Forest Database Schema
- Forest API Client
- Hansen Mask Repair
- RemoteCLIP Evaluation
- Forest Map Interaction
- Sentinel Review UI
- Change Map Inspection
- Change Map Building
- Training Dataset Features
- RemoteCLIP Embeddings
- Hansen Mask Rebuild
- Forest State Store

## God Nodes (most connected - your core abstractions)
1. `Sample` - 59 edges
2. `SentinelDownload` - 59 edges
3. `analyze_sample_hansen()` - 38 edges
4. `to_storage_path()` - 35 edges
5. `resolve_storage_path()` - 33 edges
6. `renderDetail()` - 31 edges
7. `setForestStatus()` - 28 edges
8. `HansenAnalysis` - 25 edges
9. `SentinelSceneReview` - 25 edges
10. `escapeHtml()` - 25 edges

## Surprising Connections (you probably didn't know these)
- `selected_downloads()` --uses--> `SentinelDownload`  [INFERRED]
  Scripts/ResearchScripts/analyze_sentinel_darkness_metrics.py → Scripts/ResearchTool/backend/forest/models.py
- `manual_payload()` --uses--> `ManualValidation`  [INFERRED]
  Scripts/ResearchScripts/analyze_sentinel_darkness_metrics.py → Scripts/ResearchTool/backend/forest/models.py
- `review_payload()` --uses--> `SentinelSceneReview`  [INFERRED]
  Scripts/ResearchScripts/analyze_sentinel_darkness_metrics.py → Scripts/ResearchTool/backend/forest/models.py
- `row_base()` --uses--> `SentinelDownload`  [INFERRED]
  Scripts/ResearchScripts/auto_review_sentinel_quality.py → Scripts/ResearchTool/backend/forest/models.py
- `manual_validation_payload()` --uses--> `ManualValidation`  [INFERRED]
  Scripts/ResearchScripts/auto_review_sentinel_quality.py → Scripts/ResearchTool/backend/forest/models.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Pipeline Geometry and AOI Workflow** — scripts_researchtool_readme_kml_protection_zone, scripts_researchtool_readme_closed_linestring_polygon, scripts_researchtool_readme_raster_medial_axis_centerline, scripts_researchtool_readme_centerline_segments, scripts_researchtool_readme_square_aoi, scripts_researchtool_frontend_index_pipeline_kml_aoi_controls [INFERRED 0.85]
- **Sentinel-2 Search, Fetch, and Preview Workflow** — scripts_researchtool_readme_cdse_stac_search, scripts_researchtool_readme_sentinel_2_raster_fetch, scripts_researchtool_readme_multispectral_previews, scripts_researchtool_frontend_index_sentinel_2_search_controls, scripts_researchtool_frontend_index_pipeline_observations_drawer, scripts_researchtool_frontend_index_raster_preview_controls [INFERRED 0.85]
- **Forest Sample Review Workflow** — scripts_researchtool_frontend_index_forest_mode, scripts_researchtool_frontend_index_forest_sample_import_filters, scripts_researchtool_frontend_index_batch_hansen_analysis, scripts_researchtool_frontend_index_forest_database_sync, scripts_researchtool_frontend_index_pre_post_compare, scripts_researchtool_frontend_index_forest_layer_controls [EXTRACTED 1.00]

## Communities (43 total, 8 thin omitted)

### Community 0 - "Sentinel Review Workflow"
Cohesion: 0.07
Nodes (68): classify_review_action(), clean_reason_codes(), main(), manual_validation_payload(), now_tag(), print_reason_stats(), print_valid_sample_conflicts(), replace_review_reasons() (+60 more)

### Community 1 - "Forest Sample API"
Cohesion: 0.08
Nodes (61): select_target_samples(), iter_hansen_too_old_samples(), iter_unvalidated_samples(), upsert_manual_validation(), get_db(), DbSyncError, haversine_m(), ForestExport (+53 more)

### Community 2 - "CDSE Pipeline API"
Cohesion: 0.07
Nodes (46): bbox_to_web_mercator(), best_s2_observations(), cdse_post(), cdse_session(), cdse_uses_env_proxy(), cloud_cover(), configured_proxy_keys(), env_flag() (+38 more)

### Community 3 - "Database Sync Engine"
Cohesion: 0.13
Nodes (57): add_summary(), allowed_bases_dir(), append_note(), apply_db_sync_plan(), apply_operation(), bool_value(), build_db_sync_plan(), candidate_databases() (+49 more)

### Community 4 - "Hansen ML Experiments"
Cohesion: 0.12
Nodes (17): iter_hansen_ready_unvalidated_samples(), export(), count_pairs(), haversine_km(), load_data(), main(), main(), train_features() (+9 more)

### Community 5 - "Pipeline Frontend Map"
Cohesion: 0.13
Nodes (45): api(), bboxKey(), bboxToLeafletBounds(), bindEvents(), buildSegments(), checkedObservations(), clearGeometryLayers(), clearLayer() (+37 more)

### Community 6 - "Forest UI Structure"
Cohesion: 0.08
Nodes (38): Pipeline Application Script, Batch Hansen Analysis, Forest Database Synchronization, Forest Map Layer Controls, Forest Mode, Forest Sample Import and Filtering, Forest UI Module, Leaflet 1.9.4 (+30 more)

### Community 7 - "Sentinel Download Cache"
Cohesion: 0.14
Nodes (36): build_process_request(), can_download(), decode_json(), display_path(), download_raster(), expected_preview_dir(), main(), max_cloud_from_metadata() (+28 more)

### Community 8 - "Forest Data Import"
Cohesion: 0.11
Nodes (30): clean_text(), detect_format(), geometry_point(), import_file(), lookup(), normalize_key(), normalize_sample(), NormalizedSample (+22 more)

### Community 9 - "Compare Raster Rendering"
Cohesion: 0.09
Nodes (38): applyCompareImageFilter(), bboxToBounds(), colorKey(), compareDownloadMeta(), compareDownloadOptions(), compareOptionLabel(), drawImageToCanvas(), ensureCompareMap() (+30 more)

### Community 10 - "Geospatial Segmentation"
Cohesion: 0.11
Nodes (23): sample_geometries(), square_around_point(), transformers_for_lonlat(), utm_epsg_for_lonlat(), bbox_from_polygon(), build_segments(), geometry_payload(), interpolate_line() (+15 more)

### Community 11 - "Russia Data Extraction"
Cohesion: 0.15
Nodes (24): add_filter_columns(), built_in_russia_geometry(), extract_csv(), extract_geojson(), feature_matches_russia(), is_russia_point(), iter_input_files(), load_russia_geometry() (+16 more)

### Community 12 - "Hansen Analysis Service"
Cohesion: 0.14
Nodes (26): tile_id_for_lonlat(), tile_model(), analyze_sample_hansen(), apply_sample_multiple_event_flag(), cache_url(), existing_hansen_analysis(), hex_to_rgb(), latest_hansen_payload() (+18 more)

### Community 13 - "Jobs And Path Repair"
Cohesion: 0.11
Nodes (18): main(), action_for(), is_absolute_path(), main(), normalize_relative(), now_tag(), output_dir(), PathField (+10 more)

### Community 14 - "Sentinel Hub Client"
Cohesion: 0.14
Nodes (22): Area, build_catalog_request(), build_process_request(), catalog_summary(), CollectionSpec, download_chip(), fetch_token(), import_requests() (+14 more)

### Community 15 - "Sentinel QC Calibration"
Cohesion: 0.19
Nodes (28): candidate_stats(), combined_or_rule_stats(), combined_or_rule_sweep_rows(), contrast_payload(), finite_number(), finite_values(), fraction_below(), histogram() (+20 more)

### Community 16 - "Compare Layer Controls"
Cohesion: 0.11
Nodes (32): applyCompareHansenFilter(), applyMainRasterFilters(), applyVisibleRasterFilters(), availableCompareViews(), bestCompareDownload(), bindCompareControls(), canCompareSentinel(), compareDownloadById() (+24 more)

### Community 17 - "Sentinel Bundle Export"
Cohesion: 0.20
Nodes (23): active_bundle_options(), add_file(), collect_bundle_files(), copy_file_to_bundle(), display_path(), main(), normalize_relative(), now_tag() (+15 more)

### Community 18 - "Sentinel Bundle Import"
Cohesion: 0.18
Nodes (22): copy_bundle_file_from_dir(), copy_bundle_file_from_zip(), display_path(), is_zip_bundle(), latest_bundle_path(), load_bundle_rows(), main(), now_tag() (+14 more)

### Community 19 - "Database Comparison Script"
Cohesion: 0.23
Nodes (24): column_names(), compare_hansen_analyses(), compare_hansen_tiles(), compare_manual(), compare_statuses(), connect(), count_by(), csv_safe() (+16 more)

### Community 20 - "Forest UI Orchestration"
Cohesion: 0.11
Nodes (27): bindEvents(), clearSelectedSample(), closeCompareView(), closeDbSyncView(), copyDebugEvents(), createExport(), currentFilters(), debugEventsText() (+19 more)

### Community 21 - "Sync Cache Job UI"
Cohesion: 0.14
Nodes (26): cancelDbSyncSentinelCacheJob(), dbSyncSentinelCache(), dbSyncSentinelItems(), dbSyncStatus(), escapeHtml(), hansenStageLabel(), isHansenJobRunning(), isTerminalJobStatus() (+18 more)

### Community 22 - "Hansen Tile Prefetch"
Cohesion: 0.17
Nodes (16): existing_tile_record(), main(), now_tag(), planned_tile_rows(), progress(), unique_sample_tiles(), upsert_tile_record(), write_report() (+8 more)

### Community 23 - "Sentinel Batch UI"
Cohesion: 0.15
Nodes (23): allDownloadedPreviews(), cancelBatchHansen(), cancelHansenAnalysis(), chooseDefaultSentinelPreview(), defaultSelectedSceneKeys(), downloadCheckedSentinelScenes(), ensureActiveSentinelPreviewForSample(), findPreviewByUrl() (+15 more)

### Community 24 - "Forest Database Schema"
Cohesion: 0.16
Nodes (9): main(), now_tag(), preview_rows(), write_report(), init_forest_database(), migrate_forest_database(), Base, HansenTile (+1 more)

### Community 25 - "Forest API Client"
Cohesion: 0.11
Nodes (4): cleanFilters(), exportForestSamples(), listForestSamples(), startHansenBatch()

### Community 26 - "Hansen Mask Repair"
Cohesion: 0.25
Nodes (15): main(), refresh_mask_png(), build_existing_mask_index(), display_path(), find_existing_tif(), main(), now_tag(), png_for_mask() (+7 more)

### Community 27 - "RemoteCLIP Evaluation"
Cohesion: 0.15
Nodes (8): evaluate_linear_probe(), group_by_samples_and_period(), load_metadata_from_folder(), class_similarity(), permutation_test(), permutation_test_post_vs_pre(), plot_pca(), similarity_gap()

### Community 28 - "Forest Map Interaction"
Cohesion: 0.23
Nodes (17): addCircle(), addVisiblePoint(), attachMapCaptureDebug(), bboxToBounds(), clearForestLayers(), debugMap(), ensureForestPanes(), eventTargetSummary() (+9 more)

### Community 29 - "Sentinel Review UI"
Cohesion: 0.32
Nodes (17): baseSentinelReview(), bindSentinelDownloadControls(), cancelSentinelReviewDraft(), effectiveSentinelReview(), findSentinelDownload(), refreshSelectedSampleAfterSentinelReview(), renderSentinelDownloadsPanel(), restoreSentinelReview() (+9 more)

### Community 30 - "Change Map Inspection"
Cohesion: 0.24
Nodes (8): inspect_one(), load_hansen_mask(), main(), resample_hansen_to_sentinel(), save_png(), main(), process_sample(), resample_hansen()

### Community 31 - "Change Map Building"
Cohesion: 0.25
Nodes (8): aggregate_period(), event_reference_date(), is_strictly_excluded(), load_valid_ids(), main(), nbr_and_mask(), read_bands(), scene_datetime()

### Community 32 - "Training Dataset Features"
Cohesion: 0.26
Nodes (7): collect_hansen(), collect_hansen_shape(), collect_spectral(), load_valid_ids(), main(), stats_for_tif(), loss_shape_features()

### Community 33 - "RemoteCLIP Embeddings"
Cohesion: 0.27
Nodes (5): aggregate_embeddings_by_sample_and_period(), calculate_cosine_similarity(), embedding_consistency_loo(), get_butch_embeddings(), get_embedding()

### Community 34 - "Hansen Mask Rebuild"
Cohesion: 0.35
Nodes (8): display_path(), latest_analysis(), main(), masks_for_analysis(), now_tag(), sample_rebuild_reason(), selected_samples(), write_report()

## Knowledge Gaps
- **5 isolated node(s):** `CDSE_USE_ENV_PROXY Configuration`, `Sentinel Hub Process API`, `Frontend Stylesheet`, `Batch Hansen Analysis`, `Pydantic >=2.11`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 119 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **8 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `SentinelDownload` connect `Sentinel Download Cache` to `Sentinel Review Workflow`, `Training Dataset Features`, `Forest Sample API`, `Database Sync Engine`, `Sentinel QC Calibration`, `Sentinel Bundle Export`, `Sentinel Bundle Import`, `Forest Database Schema`, `Change Map Building`?**
  _High betweenness centrality (0.027) - this node is a cross-community bridge._
- **Are the 45 inferred relationships involving `Sample` (e.g. with `event_year_for_sample()` and `process_period()`) actually correct?**
  _`Sample` has 45 INFERRED edges - model-reasoned connections that need verification._
- **What connects `CDSE_USE_ENV_PROXY Configuration`, `Sentinel Hub Process API`, `Frontend Stylesheet` to the rest of the system?**
  _5 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Sentinel Review Workflow` be split into smaller, more focused modules?**
  _Cohesion score 0.07140758154569497 - nodes in this community are weakly interconnected._
- **Why does `to_storage_path()` connect `Hansen Tile Prefetch` to `Sentinel Review Workflow`, `Forest Sample API`, `Database Sync Engine`, `Sentinel Download Cache`, `Forest Data Import`, `Hansen Analysis Service`, `Sentinel Bundle Export`, `Hansen Mask Repair`?**
  _High betweenness centrality (0.018) - this node is a cross-community bridge._
- **Are the 49 inferred relationships involving `SentinelDownload` (e.g. with `selected_downloads()` and `main()`) actually correct?**
  _`SentinelDownload` has 49 INFERRED edges - model-reasoned connections that need verification._
- **Should `Forest Sample API` be split into smaller, more focused modules?**
  _Cohesion score 0.08441318030359127 - nodes in this community are weakly interconnected._