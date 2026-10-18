# Graph Report - ML-SbPSTU-2026  (2026-10-19)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 1334 nodes · 4502 edges · 62 communities (33 shown, 29 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 324 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `12773bee`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- routes.py
- hansen_service.py
- numpy
- main.py
- json
- db_sync_service.py
- analyze_sentinel_darkness_metrics.py
- csv
- bundle_service.py
- app.js
- Research Tool Python Dependencies
- ui.js
- renderDetail
- generate_ml_data.py
- fetch_cdse_init_data.py
- models.py
- samples.js
- compare_forest_databases.py
- SentinelDownload
- escapeHtml
- import_service.py
- renderCompareView
- data_operations.js
- prefetch_sentinel_downloads.py
- applyVisibleRasterFilters
- map.js
- event_service.py
- SentinelSceneReview
- build_change_maps.py
- test_data_operations.py
- bindSentinelDownloadControls
- copyDebugEvents
- renderSentinelResults
- state.js

## God Nodes (most connected - your core abstractions)
1. `SentinelDownload` - 76 edges
2. `Sample` - 65 edges
3. `analyze_sample_hansen()` - 39 edges
4. `resolve_storage_path()` - 37 edges
5. `to_storage_path()` - 36 edges
6. `renderDetail()` - 36 edges
7. `execute_sentinel_batch()` - 36 edges
8. `HansenAnalysis` - 30 edges
9. `SentinelSceneReview` - 29 edges
10. `bindEvents()` - 29 edges

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

## Communities (62 total, 29 thin omitted)

### Community 0 - "routes.py"
Cohesion: 0.06
Nodes (84): get_db(), haversine_m(), BackgroundJob, InMemoryJobStore, utc_iso(), ForestExport, Sample, analyze_hansen() (+76 more)

### Community 1 - "hansen_service.py"
Cohesion: 0.06
Nodes (64): existing_tile_record(), main(), now_tag(), planned_tile_rows(), progress(), unique_sample_tiles(), upsert_tile_record(), write_report() (+56 more)

### Community 2 - "numpy"
Cohesion: 0.05
Nodes (38): evaluate_linear_probe(), aggregate_embeddings_by_sample_and_period(), calculate_cosine_similarity(), embedding_consistency_loo(), get_butch_embeddings(), get_embedding(), group_by_samples_and_period(), load_metadata_from_folder() (+30 more)

### Community 3 - "main.py"
Cohesion: 0.06
Nodes (58): bbox_to_web_mercator(), best_s2_observations(), cdse_post(), cdse_session(), cdse_uses_env_proxy(), cloud_cover(), configured_proxy_keys(), env_flag() (+50 more)

### Community 4 - "json"
Cohesion: 0.08
Nodes (51): action_counts(), bbox_for_download(), db_preview_views(), decode_json(), display_path(), expected_raster_path(), find_raster(), index_rasters() (+43 more)

### Community 5 - "db_sync_service.py"
Cohesion: 0.12
Nodes (58): add_summary(), allowed_bases_dir(), append_note(), apply_db_sync_plan(), apply_operation(), bool_value(), build_db_sync_plan(), candidate_databases() (+50 more)

### Community 6 - "analyze_sentinel_darkness_metrics.py"
Cohesion: 0.09
Nodes (47): candidate_stats(), combined_or_rule_stats(), combined_or_rule_sweep_rows(), contrast_payload(), finite_number(), finite_values(), fraction_below(), histogram() (+39 more)

### Community 7 - "csv"
Cohesion: 0.08
Nodes (37): add_filter_columns(), built_in_russia_geometry(), extract_csv(), extract_geojson(), feature_matches_russia(), is_russia_point(), iter_input_files(), load_russia_geometry() (+29 more)

### Community 8 - "bundle_service.py"
Cohesion: 0.11
Nodes (32): selected_samples(), main(), selected_download_ids(), main(), selected_bundle_path(), add_bundle_file(), bundle_candidates(), bundle_member_exists() (+24 more)

### Community 9 - "app.js"
Cohesion: 0.13
Nodes (45): api(), bboxKey(), bboxToLeafletBounds(), bindEvents(), buildSegments(), checkedObservations(), clearGeometryLayers(), clearLayer() (+37 more)

### Community 10 - "Research Tool Python Dependencies"
Cohesion: 0.08
Nodes (38): Pipeline Application Script, Batch Hansen Analysis, Forest Database Synchronization, Forest Map Layer Controls, Forest Mode, Forest Sample Import and Filtering, Forest UI Module, Leaflet 1.9.4 (+30 more)

### Community 11 - "ui.js"
Cohesion: 0.08
Nodes (40): allDownloadedPreviews(), chooseDefaultSentinelPreview(), compareDownloadOptions(), compareOptionLabel(), downloadedPreviews(), ensureCompareMap(), eventScopedSentinelDownloads(), filteredSentinelDownloads() (+32 more)

### Community 12 - "renderDetail"
Cohesion: 0.15
Nodes (38): bindEvents(), cancelBatchHansen(), cancelHansenAnalysis(), canCompareSentinel(), clearSelectedSample(), closeCompareView(), closeDbSyncView(), createExport() (+30 more)

### Community 13 - "generate_ml_data.py"
Cohesion: 0.15
Nodes (26): create_metadata_for_images(), derived_relative_path(), display_path(), get_images_from_db(), get_original_image_path(), get_path_to_image(), main(), resolve_project_path() (+18 more)

### Community 14 - "fetch_cdse_init_data.py"
Cohesion: 0.12
Nodes (22): Area, build_catalog_request(), build_process_request(), catalog_summary(), CollectionSpec, download_chip(), fetch_token(), import_requests() (+14 more)

### Community 15 - "models.py"
Cohesion: 0.13
Nodes (17): iter_hansen_ready_unvalidated_samples(), iter_hansen_too_old_samples(), iter_unvalidated_samples(), upsert_manual_validation(), append_manual_note(), hansen_comment(), latest_analyses(), main() (+9 more)

### Community 16 - "samples.js"
Cohesion: 0.07
Nodes (4): cleanFilters(), exportForestSamples(), listForestSamples(), startHansenBatch()

### Community 17 - "compare_forest_databases.py"
Cohesion: 0.20
Nodes (25): main(), column_names(), compare_hansen_analyses(), compare_hansen_tiles(), compare_manual(), compare_statuses(), connect(), count_by() (+17 more)

### Community 18 - "SentinelDownload"
Cohesion: 0.20
Nodes (23): SampleEvent, SentinelDownload, SentinelSearch, build_operation_rows(), calendar_year_for_period(), date_gap_ok(), decode_json(), download_datetime() (+15 more)

### Community 19 - "escapeHtml"
Cohesion: 0.13
Nodes (29): cancelDbSyncSentinelCacheJob(), dbSyncSentinelCache(), dbSyncSentinelItems(), dbSyncStatus(), escapeHtml(), hansenStageLabel(), isHansenJobRunning(), isTerminalJobStatus() (+21 more)

### Community 20 - "import_service.py"
Cohesion: 0.21
Nodes (18): clean_text(), detect_format(), geometry_point(), import_file(), lookup(), normalize_key(), normalize_sample(), NormalizedSample (+10 more)

### Community 21 - "renderCompareView"
Cohesion: 0.14
Nodes (23): availableCompareViews(), bestCompareDownload(), bindCompareControls(), compareDownloadMeta(), compareDownloads(), compareHansenOverlay(), compareViewOptions(), currentSclLegendPreviews() (+15 more)

### Community 22 - "data_operations.js"
Cohesion: 0.26
Nodes (20): activeJobHost(), artifactUrl(), checkedValues(), dataOperationsRunning(), downloadPayload(), escapeHtml(), humanBytes(), initDataOperations() (+12 more)

### Community 23 - "prefetch_sentinel_downloads.py"
Cohesion: 0.27
Nodes (19): build_process_request(), can_download(), decode_json(), display_path(), download_raster(), expected_preview_dir(), main(), max_cloud_from_metadata() (+11 more)

### Community 24 - "applyVisibleRasterFilters"
Cohesion: 0.15
Nodes (21): applyCompareHansenFilter(), applyCompareImageFilter(), applyMainRasterFilters(), applyVisibleRasterFilters(), bboxToBounds(), colorKey(), compareDownloadById(), currentHansenOverlay() (+13 more)

### Community 25 - "map.js"
Cohesion: 0.23
Nodes (17): addCircle(), addVisiblePoint(), attachMapCaptureDebug(), bboxToBounds(), clearForestLayers(), debugMap(), ensureForestPanes(), eventTargetSummary() (+9 more)

### Community 26 - "event_service.py"
Cohesion: 0.29
Nodes (13): backfill_hansen_events(), decode_histogram(), deterministic_id(), event_download_id_for(), event_id_for(), hansen_event_specs(), latest_analyses(), link_unambiguous_legacy_downloads() (+5 more)

### Community 27 - "SentinelSceneReview"
Cohesion: 0.24
Nodes (15): classify_review_action(), clean_reason_codes(), main(), manual_validation_payload(), now_tag(), print_reason_stats(), print_valid_sample_conflicts(), replace_review_reasons() (+7 more)

### Community 28 - "build_change_maps.py"
Cohesion: 0.23
Nodes (9): aggregate_period(), event_reference_date(), is_strictly_excluded(), load_valid_ids(), main(), nbr_and_mask(), read_bands(), scene_datetime() (+1 more)

### Community 29 - "test_data_operations.py"
Cohesion: 0.25
Nodes (7): main(), BatchOptions, plan_sentinel_batch(), sample_is_too_old(), analysis(), DataOperationsTest, sample()

### Community 30 - "bindSentinelDownloadControls"
Cohesion: 0.33
Nodes (15): baseSentinelReview(), bindSentinelDownloadControls(), cancelSentinelReviewDraft(), effectiveSentinelReview(), findSentinelDownload(), renderSentinelDownloadsPanel(), restoreSentinelReview(), saveSentinelReviewDraft() (+7 more)

### Community 31 - "copyDebugEvents"
Cohesion: 0.40
Nodes (5): copyDebugEvents(), debugEventsText(), logUiDebug(), pushDebugEvent(), renderDebugEvents()

### Community 32 - "renderSentinelResults"
Cohesion: 0.50
Nodes (5): defaultSelectedSceneKeys(), renderSentinelResults(), sceneKey(), searchedScenes(), sentinelSearchDownloadState()

## Knowledge Gaps
- **5 isolated node(s):** `Batch Hansen Analysis`, `Frontend Stylesheet`, `CDSE_USE_ENV_PROXY Configuration`, `Sentinel Hub Process API`, `Pydantic >=2.11`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 166 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **29 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Sample` connect `routes.py` to `hansen_service.py`, `numpy`, `json`, `db_sync_service.py`, `csv`, `generate_ml_data.py`, `models.py`, `SentinelDownload`, `import_service.py`, `event_service.py`, `test_data_operations.py`?**
  _High betweenness centrality (0.021) - this node is a cross-community bridge._
- **Are the 60 inferred relationships involving `SentinelDownload` (e.g. with `selected_downloads()` and `main()`) actually correct?**
  _`SentinelDownload` has 60 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Batch Hansen Analysis`, `Frontend Stylesheet`, `CDSE_USE_ENV_PROXY Configuration` to the rest of the system?**
  _5 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `routes.py` be split into smaller, more focused modules?**
  _Cohesion score 0.05788156797331109 - nodes in this community are weakly interconnected._
- **Why does `SentinelDownload` connect `SentinelDownload` to `routes.py`, `numpy`, `json`, `db_sync_service.py`, `analyze_sentinel_darkness_metrics.py`, `csv`, `bundle_service.py`, `generate_ml_data.py`, `models.py`, `prefetch_sentinel_downloads.py`, `event_service.py`, `SentinelSceneReview`, `build_change_maps.py`, `test_data_operations.py`?**
  _High betweenness centrality (0.020) - this node is a cross-community bridge._
- **Are the 47 inferred relationships involving `Sample` (e.g. with `iter_hansen_ready_unvalidated_samples()` and `iter_hansen_too_old_samples()`) actually correct?**
  _`Sample` has 47 INFERRED edges - model-reasoned connections that need verification._
- **Should `hansen_service.py` be split into smaller, more focused modules?**
  _Cohesion score 0.06446886446886448 - nodes in this community are weakly interconnected._