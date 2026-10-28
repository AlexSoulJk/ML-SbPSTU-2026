import {
  cancelForestJob,
  fillHansenCandidateRun,
  getForestJob,
  listHansenCandidateRuns,
  listBundleImportCandidates,
  materializeHansenCandidates,
  planBundleImport,
  planEventBackfill,
  planModelLabeling,
  planSentinelBatch,
  planSentinelBundle,
  startBundleImport,
  startEventBackfill,
  startHansenCandidateMining,
  startMissingGeotiffInventory,
  startSentinelBatch,
  startSentinelBundle,
  rejectHansenCandidates,
  uploadDataOperationSourceFile,
} from "./samples.js?v=hansen-whole-tile-1";
import { forestState } from "./state.js?v=hansen-whole-tile-1";


function $(id) {
  return document.getElementById(id);
}


function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}


function numberValue(id, fallback) {
  const value = Number($(id)?.value);
  return Number.isFinite(value) ? value : fallback;
}


function updateBundleSourceFileControl() {
  const source = $("forestDataOperationsBundleSource").value;
  const label = $("forestDataOperationsInventoryFileLabel");
  const input = $("forestDataOperationsInventoryFile");
  const text = $("forestDataOperationsSourceFileText");
  const needsFile = source === "missing_geotiff_csv" || source === "download_job_result";
  label.hidden = !needsFile;
  if (source === "download_job_result") {
    text.textContent = "Batch selection JSON";
    input.accept = ".json,application/json";
  } else {
    text.textContent = "Inventory CSV";
    input.accept = ".csv,text/csv";
  }
}


function checkedValues(selector) {
  return [...document.querySelectorAll(selector)]
    .filter((input) => input.checked)
    .map((input) => input.value);
}


function humanBytes(value) {
  const bytes = Number(value || 0);
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KiB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GiB`;
}


function artifactUrl(path) {
  return `/api/forest/data-operations/artifact?path=${encodeURIComponent(path)}`;
}


function metricsHtml(metrics) {
  return Object.entries(metrics || {}).map(([label, value]) => `
    <div class="forestDataOperationsMetric">
      <span>${escapeHtml(label.replaceAll("_", " "))}</span>
      <strong>${escapeHtml(value ?? 0)}</strong>
    </div>
  `).join("");
}


function setStatus(message) {
  forestState.dataOperationsStatus = message;
  const host = $("forestDataOperationsStatus");
  if (host) host.textContent = message;
}


function dataOperationsRunning() {
  return ["queued", "running", "cancel_requested"].includes(forestState.dataOperationsJob?.status);
}


function miningPayload() {
  return {
    target_count: numberValue("forestDataOperationsMiningTarget", 100),
    aoi_size_m: numberValue("forestDataOperationsMiningAoi", 2000),
    safety_gap_m: numberValue("forestDataOperationsMiningGap", 200),
    min_dominant_event_aoi_percent: numberValue("forestDataOperationsMiningThreshold", 1),
    min_event_year: numberValue("forestDataOperationsMiningYear", 2016),
    scan_resolution_m: numberValue("forestDataOperationsMiningResolution", 300),
  };
}


function selectedMiningRows() {
  return [...forestState.dataOperationsMiningSelectedIds];
}


function renderMiningRuns() {
  const select = $("forestDataOperationsMiningRun");
  const labeling = $("forestDataOperationsLabelingRun");
  const download = $("forestDataOperationsDownloadMiningRun");
  const labelingValue = labeling?.value || "";
  const downloadValue = download?.value || "";
  const selectedRunId = forestState.dataOperationsMiningRun?.run_id || "";
  const options = forestState.dataOperationsMiningRuns.map((run) => `
    <option value="${escapeHtml(run.run_id)}" ${run.run_id === selectedRunId ? "selected" : ""}>
      ${escapeHtml(run.created_at?.slice(0, 19) || run.run_id.slice(0, 8))} · ${escapeHtml(run.status)} · ${escapeHtml(run.created_sample_count)}/${escapeHtml(run.selected_count)}
    </option>
  `).join("");
  const materializedOptions = forestState.dataOperationsMiningRuns
    .filter((run) => Number(run.created_sample_count || 0) > 0)
    .map((run) => `<option value="${escapeHtml(run.run_id)}">${escapeHtml(run.created_at?.slice(0, 19) || run.run_id.slice(0, 8))} · ${escapeHtml(run.created_sample_count)} samples</option>`)
    .join("");
  if (select) select.innerHTML = `<option value="">New plan</option>${options}`;
  if (labeling) {
    labeling.innerHTML = `<option value="">Select a materialized run</option>${materializedOptions}`;
    labeling.value = labelingValue;
  }
  if (download) {
    download.innerHTML = `<option value="">Select a materialized run</option>${materializedOptions}`;
    download.value = downloadValue;
  }
}


function ensureMiningMap() {
  const host = $("forestDataOperationsMiningMap");
  if (!host || !window.L) return null;
  if (!forestState.dataOperationsMiningMap) {
    const map = window.L.map(host, { preferCanvas: true, zoomControl: true }).setView([60, 90], 3);
    window.L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: "&copy; OpenStreetMap contributors",
    }).addTo(map);
    forestState.dataOperationsMiningMap = map;
    forestState.dataOperationsMiningMapLayers = window.L.layerGroup().addTo(map);
  }
  window.setTimeout(() => forestState.dataOperationsMiningMap?.invalidateSize(), 0);
  return forestState.dataOperationsMiningMap;
}


function renderMiningMap() {
  const map = ensureMiningMap();
  const group = forestState.dataOperationsMiningMapLayers;
  if (!map || !group) return;
  group.clearLayers();
  const run = forestState.dataOperationsMiningRun;
  if (!run) return;
  const bounds = [];
  (run.items || []).forEach((row) => {
    const bbox = row.aoi_bbox || [];
    if (bbox.length !== 4) return;
    const selected = forestState.dataOperationsMiningSelectedIds.has(row.candidate_id);
    const rejected = ["USER_REJECTED", "REJECTED"].includes(row.decision);
    const color = row.created_sample_id
      ? "#2f6b8a"
      : rejected
        ? "#b84936"
        : selected
          ? "#2d6a4f"
          : row.decision === "ELIGIBLE"
            ? "#b1811c"
            : "#71827e";
    const rectangleBounds = [[bbox[1], bbox[0]], [bbox[3], bbox[2]]];
    const layer = window.L.rectangle(rectangleBounds, {
      color,
      weight: selected ? 2 : 1,
      fillColor: color,
      fillOpacity: selected ? 0.28 : 0.12,
    }).addTo(group);
    layer.bindTooltip(
      `#${row.rank || "–"} · ${row.primary_event_year} · ${Number(row.dominant_event_aoi_percent || 0).toFixed(2)}%`,
    );
    if (!row.created_sample_id && !rejected) {
      layer.on("click", () => {
        if (forestState.dataOperationsMiningSelectedIds.has(row.candidate_id)) {
          forestState.dataOperationsMiningSelectedIds.delete(row.candidate_id);
        } else {
          forestState.dataOperationsMiningSelectedIds.add(row.candidate_id);
        }
        renderMiningPlan();
      });
    }
    bounds.push(...rectangleBounds);
  });
  if (bounds.length && forestState.dataOperationsMiningMapRunId !== run.run_id) {
    map.fitBounds(bounds, { padding: [16, 16], maxZoom: 8 });
    forestState.dataOperationsMiningMapRunId = run.run_id;
  }
}


function renderMiningPlan() {
  const run = forestState.dataOperationsMiningRun;
  const summaryHost = $("forestDataOperationsMiningSummary");
  const rowsHost = $("forestDataOperationsMiningRows");
  const totals = $("forestDataOperationsMiningTotals");
  if (!run) {
    summaryHost.innerHTML = metricsHtml({ target: numberValue("forestDataOperationsMiningTarget", 100), plan: "not built" });
    rowsHost.innerHTML = `<div class="meta">Build a plan or select a previous run.</div>`;
    totals.textContent = "Build a plan first.";
    $("forestDataOperationsMiningRejectBtn").disabled = true;
    $("forestDataOperationsMiningFillBtn").disabled = true;
    $("forestDataOperationsMiningApplyBtn").disabled = true;
    renderMiningMap();
    return;
  }
  const rows = run.items || [];
  const selected = selectedMiningRows();
  const sweep = run.summary?.threshold_sweep || {};
  summaryHost.innerHTML = metricsHtml({
    run: run.run_id.slice(0, 8),
    status: run.status,
    selected: run.selected_count,
    reserve: run.summary?.reserve_count || 0,
    created: run.created_sample_count,
    tiles: run.summary?.valid_tiles || 0,
    "sweep 0.5/1/2.5/5/10": [0.5, 1, 2.5, 5, 10].map((key) => sweep[String(key)] || 0).join(" / "),
  });
  if (run.artifacts?.summary_json) {
    summaryHost.insertAdjacentHTML("beforeend", `
      <div class="forestDataOperationsMetric">
        <span>Artifacts</span>
        <strong><a href="${artifactUrl(run.artifacts.summary_json)}">summary</a> · <a href="${artifactUrl(run.artifacts.candidates_csv)}">candidates</a></strong>
      </div>
    `);
  }
  totals.textContent = `${selected.length} checked · ${rows.length} planned rows`;
  rowsHost.innerHTML = rows.map((row) => {
    const selectable = !row.created_sample_id && !["USER_REJECTED", "REJECTED"].includes(row.decision);
    const checked = selectable && forestState.dataOperationsMiningSelectedIds.has(row.candidate_id);
    const className = row.decision === "USER_REJECTED" ? "error" : row.decision === "ELIGIBLE" ? "warning" : "";
    const osmUrl = `https://www.openstreetmap.org/?mlat=${encodeURIComponent(row.lat)}&mlon=${encodeURIComponent(row.lon)}#map=12/${encodeURIComponent(row.lat)}/${encodeURIComponent(row.lon)}`;
    return `
      <article class="forestDataOperationsRow ${className}">
        <input type="checkbox" data-mining-candidate-id="${escapeHtml(row.candidate_id)}" ${checked ? "checked" : ""} ${selectable ? "" : "disabled"} />
        <span>
          <strong>#${escapeHtml(row.rank || "–")} · ${escapeHtml(row.primary_event_year)} · ${escapeHtml(row.candidate_id.slice(0, 8))}</strong>
          <span>${Number(row.lat).toFixed(5)}, ${Number(row.lon).toFixed(5)} · <a href="${osmUrl}" target="_blank" rel="noreferrer">OSM</a></span>
        </span>
        <span>
          <strong>${Number(row.dominant_event_aoi_percent || 0).toFixed(2)}% dominant · ${Number(row.total_loss_area_ha || 0).toFixed(2)} ha total</strong>
          <span>events ${escapeHtml((row.event_years || []).join(", "))} · tiles ${escapeHtml((row.source_tile_ids || []).join(", "))}</span>
        </span>
        <span class="forestDataOperationsBadge">${escapeHtml(row.created_sample_id ? "created" : row.decision)}</span>
      </article>
    `;
  }).join("") || `<div class="meta">No candidates passed the strict filters.</div>`;
  rowsHost.querySelectorAll("[data-mining-candidate-id]").forEach((input) => {
    input.addEventListener("change", (event) => {
      const id = event.target.dataset.miningCandidateId;
      if (event.target.checked) forestState.dataOperationsMiningSelectedIds.add(id);
      else forestState.dataOperationsMiningSelectedIds.delete(id);
      renderMiningPlan();
    });
  });
  const plannedIds = rows.filter((row) => row.decision === "SELECTED" && !row.created_sample_id).map((row) => row.candidate_id);
  $("forestDataOperationsMiningSelectAll").checked = plannedIds.length > 0
    && plannedIds.every((id) => forestState.dataOperationsMiningSelectedIds.has(id));
  const running = dataOperationsRunning();
  $("forestDataOperationsMiningRejectBtn").disabled = running || !selected.length;
  $("forestDataOperationsMiningFillBtn").disabled = running
    || run.selected_count >= run.target_count
    || Number(run.summary?.reserve_count || 0) <= 0;
  $("forestDataOperationsMiningApplyBtn").disabled = running || !selected.length;
  renderMiningMap();
}


async function loadMiningRuns() {
  const payload = await listHansenCandidateRuns(25);
  forestState.dataOperationsMiningRuns = payload.items || [];
  if (forestState.dataOperationsMiningRun) {
    forestState.dataOperationsMiningRun = forestState.dataOperationsMiningRuns.find(
      (run) => run.run_id === forestState.dataOperationsMiningRun.run_id,
    ) || forestState.dataOperationsMiningRun;
  }
  renderMiningRuns();
  renderMiningPlan();
}


function renderLabelingPlan() {
  const plan = forestState.dataOperationsLabelingPlan;
  const summary = $("forestDataOperationsLabelingSummary");
  const rows = $("forestDataOperationsLabelingRows");
  if (!plan) {
    summary.innerHTML = metricsHtml({ plan: "not built", predictions: "manual validation unaffected" });
    rows.innerHTML = `<article class="forestDataOperationsRow warning">
      <strong>Select a materialized mining run</strong>
      <span>The plan checks PRE/POST inputs, local cache and both frozen model artifacts.</span>
      <span>No inference starts automatically.</span>
      <span class="forestDataOperationsBadge">safe</span>
    </article>`;
    return;
  }
  summary.innerHTML = metricsHtml({
    events: plan.summary.events,
    ready_inputs: plan.summary.ready_inputs,
    incomplete_inputs: plan.summary.incomplete_inputs,
    remoteclip: plan.summary.remoteclip_artifact_ready ? "ready" : "missing",
    xgboost: plan.summary.xgboost_artifact_ready ? "ready" : "missing",
  });
  const modelRows = Object.values(plan.models || {}).map((model) => `
    <article class="forestDataOperationsRow ${model.ready ? "" : "error"}">
      <span></span>
      <span><strong>${escapeHtml(model.producer_name)} · ${escapeHtml(model.producer_version)}</strong><span>${escapeHtml(model.artifact_path)}</span></span>
      <span>${escapeHtml(model.ready ? `SHA-256 ${model.artifact_sha256}` : model.errors.join("; "))}</span>
      <span class="forestDataOperationsBadge">${model.ready ? "ready" : "blocked"}</span>
    </article>
  `);
  const eventRows = (plan.events || []).map((event) => `
    <article class="forestDataOperationsRow ${event.input_status === "READY" ? "" : "warning"}">
      <span></span>
      <span><strong>${escapeHtml(event.event_year)} · ${escapeHtml(event.sample_id.slice(0, 8))}</strong><span>${escapeHtml(event.event_id.slice(0, 8))}</span></span>
      <span><strong>PRE ${event.usable_counts.PRE || 0} · POST ${event.usable_counts.POST || 0}</strong><span>${escapeHtml(event.blockers.join(", ") || "inputs ready")}</span></span>
      <span class="forestDataOperationsBadge">${escapeHtml(event.input_status)}</span>
    </article>
  `);
  rows.innerHTML = [...modelRows, ...eventRows].join("") || `<div class="meta">No events are assigned to this mining run.</div>`;
}


function activeJobHost() {
  const kind = forestState.dataOperationsJob?.kind || "";
  if (kind === "hansen_candidate_mining") return $("forestDataOperationsMiningJob");
  if (kind === "sentinel_bundle_export") return $("forestDataOperationsBundleJob");
  if (kind === "sentinel_bundle_import") return $("forestDataOperationsImportJob");
  if (kind === "sentinel_missing_geotiff_inventory") return $("forestDataOperationsBundleJob");
  return $("forestDataOperationsDownloadJob");
}


function renderJob() {
  [
    $("forestDataOperationsMiningJob"),
    $("forestDataOperationsDownloadJob"),
    $("forestDataOperationsBundleJob"),
    $("forestDataOperationsImportJob"),
  ].forEach((host) => {
    if (host) host.innerHTML = "";
  });
  const job = forestState.dataOperationsJob;
  if (!job) return;
  const host = activeJobHost();
  if (!host) return;
  const total = Number(job.total || 0);
  const done = Number(job.done || 0);
  const percent = total > 0 ? Math.round((done / total) * 100) : 0;
  const result = job.result || {};
  const verification = result.verification || null;
  const artifactPath = result.zip_path
    || result.csv_path
    || result.bundle_selection
    || result.operations_csv
    || null;
  host.innerHTML = `
    <strong>${escapeHtml(job.stage || job.status)}</strong>
    <span>${escapeHtml(job.message || "")}</span>
    <div class="forestDataOperationsProgress"><span style="width:${Math.max(0, Math.min(100, percent))}%"></span></div>
    ${verification ? `<div class="forestDataOperationsMetrics">
      ${metricsHtml({
        downloads_ready: `${verification.database_downloads}/${verification.bundle_downloads}`,
        samples: verification.samples,
        no_manual_samples: verification.samples_without_manual_validation,
        preview_files_ready: verification.preview_files_ready,
        missing_previews: verification.missing_preview_files,
        incomplete_downloads: verification.incomplete_preview_downloads,
      })}
    </div>` : ""}
    ${result.backup_path ? `<span class="meta">Database backup: ${escapeHtml(result.backup_path)}</span>` : ""}
    ${artifactPath ? `<a class="forestDataOperationsArtifact" href="${artifactUrl(artifactPath)}">Open artifact</a>` : ""}
  `;

  const running = dataOperationsRunning();
  document.querySelectorAll(
    ".forestDataOperationsSetup input, .forestDataOperationsSetup select, .forestDataOperationsSetup button, [data-data-operations-tab], #forestDataOperationsMiningPlanBtn, #forestDataOperationsLabelingPlanBtn, #forestDataOperationsDownloadPlanBtn, #forestDataOperationsBundlePlanBtn, #forestDataOperationsImportPlanBtn",
  ).forEach((control) => {
    control.disabled = running;
  });
  $("forestDataOperationsEventSyncBtn").disabled = running || !forestState.dataOperationsEventPlan;
  $("forestDataOperationsMiningCancelBtn").disabled = !running;
  $("forestDataOperationsDownloadCancelBtn").disabled = !running;
  $("forestDataOperationsBundleCancelBtn").disabled = !running;
  $("forestDataOperationsImportCancelBtn").disabled = !running;
  $("forestDataOperationsDownloadStartBtn").disabled = running || !forestState.dataOperationsSelectedKeys.size;
  $("forestDataOperationsBundleStartBtn").disabled = running || !forestState.dataOperationsBundleSelectedIds.size;
  $("forestDataOperationsImportStartBtn").disabled = running || !forestState.dataOperationsImportPlan?.can_apply;
  renderMiningPlan();
}


function renderDownloadPlan() {
  const plan = forestState.dataOperationsPlan;
  const summaryHost = $("forestDataOperationsDownloadSummary");
  const rowsHost = $("forestDataOperationsDownloadRows");
  const totals = $("forestDataOperationsDownloadTotals");
  if (!plan) {
    summaryHost.innerHTML = metricsHtml({ samples: 0, actionable: 0, operations: 0 });
    rowsHost.innerHTML = `<div class="emptyState">Build a plan to inspect samples and time windows.</div>`;
    totals.textContent = "Build a plan first.";
    $("forestDataOperationsDownloadStartBtn").disabled = true;
    return;
  }
  const skipCounts = (plan.skipped || []).reduce((counts, row) => {
    counts[`skip_${row.reason}`] = Number(counts[`skip_${row.reason}`] || 0) + 1;
    return counts;
  }, {});
  summaryHost.innerHTML = metricsHtml({ ...plan.summary, ...skipCounts });
  const rows = plan.samples.flatMap((sample) => sample.operations.map((operation) => ({ sample, operation })));
  totals.textContent = `${forestState.dataOperationsSelectedKeys.size}/${rows.length} operations selected`;
  const plannedHtml = rows.map(({ sample, operation }) => {
    const selected = forestState.dataOperationsSelectedKeys.has(operation.operation_key);
    const warning = sample.stop_reason ? " warning" : "";
    return `
      <label class="forestDataOperationsRow${warning}">
        <input type="checkbox" data-data-operations-key="${escapeHtml(operation.operation_key)}" ${selected ? "checked" : ""} ${sample.stop_reason ? "disabled" : ""} />
        <span>
          <strong>${escapeHtml(sample.display_name)}</strong>
          <span>${escapeHtml(sample.short_id)} / ${escapeHtml(sample.driver || "unknown")}</span>
        </span>
        <span>
          <strong>${escapeHtml(operation.event_year)}${operation.is_dominant ? " / dominant" : ""} / ${escapeHtml(operation.period)} / ${escapeHtml(operation.season)}</strong>
          <span>${escapeHtml(operation.start)} .. ${escapeHtml(operation.end)} / QC ${escapeHtml(sample.qc_checked_count)}/${escapeHtml(sample.max_qc_scenes)} / deficit ${escapeHtml(sample.deficit)}</span>
        </span>
        <span class="forestDataOperationsBadge">${escapeHtml(sample.valid_count)}/${escapeHtml(sample.target_valid)} valid</span>
      </label>
    `;
  }).join("");
  const skippedHtml = (plan.skipped || []).map((row) => `
    <div class="forestDataOperationsRow warning">
      <span></span>
      <span><strong>${escapeHtml(row.sample_id)}</strong><span>Skipped sample</span></span>
      <span><strong>${escapeHtml(row.reason)}</strong><span>No operation will be started</span></span>
      <span class="forestDataOperationsBadge">skip</span>
    </div>
  `).join("");
  rowsHost.innerHTML = plannedHtml || skippedHtml
    ? `${plannedHtml}${skippedHtml}`
    : `<div class="emptyState">No operation rows matched the settings.</div>`;
  rowsHost.querySelectorAll("[data-data-operations-key]").forEach((input) => {
    input.addEventListener("change", (event) => {
      const key = event.target.dataset.dataOperationsKey;
      if (event.target.checked) forestState.dataOperationsSelectedKeys.add(key);
      else forestState.dataOperationsSelectedKeys.delete(key);
      renderDownloadPlan();
      renderJob();
    });
  });
  $("forestDataOperationsDownloadSelectAll").checked = Boolean(rows.length)
    && rows.filter(({ sample }) => !sample.stop_reason).every(({ operation }) => (
      forestState.dataOperationsSelectedKeys.has(operation.operation_key)
    ));
  $("forestDataOperationsDownloadStartBtn").disabled = dataOperationsRunning()
    || !forestState.dataOperationsSelectedKeys.size;
}


function downloadPayload(callbacks) {
  const filters = { ...callbacks.currentFilters() };
  const manualValidation = $("forestDataOperationsManualValidation").value;
  const mode = $("forestDataOperationsDownloadMode").value;
  if (manualValidation === "unvalidated") filters.manual_validation = "__no_manual_validation";
  if (manualValidation === "any") delete filters.manual_validation;
  return {
    filters,
    mode,
    mining_run_id: mode === "hansen_mining_run"
      ? $("forestDataOperationsDownloadMiningRun").value || null
      : null,
    limit_samples: numberValue("forestDataOperationsSampleLimit", 100),
    drivers: checkedValues("[data-data-operations-driver]"),
    seasons: checkedValues("[data-data-operations-season]"),
    target_valid: numberValue("forestDataOperationsValidTarget", 10),
    max_qc_scenes: numberValue("forestDataOperationsQcCap", 50),
    min_days: numberValue("forestDataOperationsMinDays", 7),
    stac_max_cloud: numberValue("forestDataOperationsStacCloud", 30),
    local_cloud_threshold: numberValue("forestDataOperationsLocalCloud", 10) / 100,
    search_top_n: 40,
  };
}


function renderBundlePlan() {
  const plan = forestState.dataOperationsBundlePlan;
  const summaryHost = $("forestDataOperationsBundleSummary");
  const rowsHost = $("forestDataOperationsBundleRows");
  const totals = $("forestDataOperationsBundleTotals");
  if (!plan) {
    summaryHost.innerHTML = metricsHtml({ downloads: 0, files: 0, size: "0 B" });
    rowsHost.innerHTML = `<div class="emptyState">Build a plan to inspect bundle contents.</div>`;
    totals.textContent = "Build a plan first.";
    $("forestDataOperationsBundleStartBtn").disabled = true;
    return;
  }
  summaryHost.innerHTML = metricsHtml({
    downloads: plan.summary.unique_downloads,
    files: plan.summary.unique_files,
    size: humanBytes(plan.summary.total_bytes),
    missing: plan.summary.missing_source_files,
    unknown: plan.summary.unknown_download_ids,
    identity_conflicts: plan.summary.identity_conflicts,
    previews_to_rebuild: plan.summary.previews_to_rebuild,
    event_links: plan.summary.event_links,
  });
  const eventById = new Map((plan.events || []).map((event) => [event.event_id, event]));
  const linksByDownload = new Map();
  (plan.event_links || []).forEach((link) => {
    if (!linksByDownload.has(link.download_id)) linksByDownload.set(link.download_id, []);
    linksByDownload.get(link.download_id).push(link);
  });
  const fileCounts = new Map();
  (plan.files || []).forEach((row) => {
    const current = fileCounts.get(row.download_id) || { files: 0, bytes: 0, missing: 0 };
    current.files += 1;
    current.bytes += Number(row.bytes || 0);
    if (!row.exists) current.missing += 1;
    fileCounts.set(row.download_id, current);
  });
  totals.textContent = `${forestState.dataOperationsBundleSelectedIds.size}/${plan.download_ids.length} downloads selected`;
  rowsHost.innerHTML = plan.download_ids.length ? plan.download_ids.map((downloadId) => {
    const item = fileCounts.get(downloadId) || { files: 0, bytes: 0, missing: 0 };
    const eventText = (linksByDownload.get(downloadId) || []).map((link) => {
      const event = eventById.get(link.event_id);
      return `${event?.event_year || link.event_id.slice(0, 8)} ${link.period}`;
    }).join(", ") || "Legacy / unassigned";
    return `
      <label class="forestDataOperationsRow${item.missing ? " warning" : ""}">
        <input type="checkbox" data-bundle-download-id="${escapeHtml(downloadId)}" ${forestState.dataOperationsBundleSelectedIds.has(downloadId) ? "checked" : ""} />
        <span><strong>${escapeHtml(downloadId)}</strong><span>${escapeHtml(item.files)} files</span></span>
        <span><strong>${escapeHtml(humanBytes(item.bytes))}</strong><span>${escapeHtml(item.missing)} missing / ${escapeHtml(eventText)}</span></span>
        <span class="forestDataOperationsBadge">${escapeHtml(plan.profile)}</span>
      </label>
    `;
  }).join("") : `<div class="emptyState">No downloads matched the source.</div>`;
  rowsHost.querySelectorAll("[data-bundle-download-id]").forEach((input) => {
    input.addEventListener("change", (event) => {
      const id = event.target.dataset.bundleDownloadId;
      if (event.target.checked) forestState.dataOperationsBundleSelectedIds.add(id);
      else forestState.dataOperationsBundleSelectedIds.delete(id);
      renderBundlePlan();
      renderJob();
    });
  });
  $("forestDataOperationsBundleSelectAll").checked = Boolean(plan.download_ids.length)
    && plan.download_ids.every((id) => forestState.dataOperationsBundleSelectedIds.has(id));
  $("forestDataOperationsBundleStartBtn").disabled = dataOperationsRunning()
    || !forestState.dataOperationsBundleSelectedIds.size;
}


function renderImportCandidates() {
  const select = $("forestDataOperationsImportCandidate");
  const items = forestState.dataOperationsImportCandidates;
  select.innerHTML = items.length
    ? items.map((item) => `<option value="${escapeHtml(item.path)}">${escapeHtml(item.name)} (${escapeHtml(item.kind)})</option>`).join("")
    : `<option value="">No bundles found</option>`;
}


function renderImportPlan() {
  const plan = forestState.dataOperationsImportPlan;
  const summaryHost = $("forestDataOperationsImportSummary");
  const rowsHost = $("forestDataOperationsImportRows");
  const totals = $("forestDataOperationsImportTotals");
  if (!plan) {
    summaryHost.innerHTML = metricsHtml({ files: 0, copy: 0, conflicts: 0 });
    rowsHost.innerHTML = `<div class="emptyState">Select a bundle and build an import plan.</div>`;
    totals.textContent = "Build a plan first.";
    $("forestDataOperationsImportStartBtn").disabled = true;
    return;
  }
  summaryHost.innerHTML = metricsHtml({
    files: plan.summary.files,
    copy: plan.summary.copy,
    exists: plan.summary.skip_exists,
    size: humanBytes(plan.summary.bytes_to_copy),
    database_not_current: plan.summary.missing_db_downloads,
    identity_conflicts: plan.summary.identity_conflicts,
    checksum_mismatches: plan.summary.checksum_mismatches,
    previews: plan.summary.previews_to_rebuild,
  });
  totals.textContent = plan.can_apply ? "Ready to import" : "Resolve the reported bundle or database conflicts";
  rowsHost.innerHTML = (plan.files || []).map((row) => `
    <div class="forestDataOperationsRow${["database_not_current", "checksum_mismatch"].includes(row.action) ? " error" : row.action.includes("missing") ? " warning" : ""}">
      <span></span>
      <span><strong>${escapeHtml(row.kind)}</strong><span>${escapeHtml(row.download_id)}</span></span>
      <span><strong>${escapeHtml(row.relative_path)}</strong><span>${escapeHtml(humanBytes(row.bytes))}</span></span>
      <span class="forestDataOperationsBadge">${escapeHtml(row.action)}</span>
    </div>
  `).join("") || `<div class="emptyState">The bundle contains no files.</div>`;
  $("forestDataOperationsImportStartBtn").disabled = dataOperationsRunning() || !plan.can_apply;
}


function renderTab() {
  document.querySelectorAll("[data-data-operations-tab]").forEach((button) => {
    button.classList.toggle("active", button.dataset.dataOperationsTab === forestState.dataOperationsTab);
  });
  document.querySelectorAll("[data-data-operations-panel]").forEach((panel) => {
    panel.hidden = panel.dataset.dataOperationsPanel !== forestState.dataOperationsTab;
  });
}


async function pollJob(callbacks) {
  if (!forestState.dataOperationsJobId) return;
  const job = await getForestJob(forestState.dataOperationsJobId);
  forestState.dataOperationsJob = job;
  renderJob();
  setStatus(`${job.kind}: ${job.message || job.stage}`);
  if (["queued", "running", "cancel_requested"].includes(job.status)) {
    forestState.dataOperationsTimer = window.setTimeout(() => callbacks.run(() => pollJob(callbacks)), 900);
    return;
  }
  forestState.dataOperationsJobId = null;
  forestState.dataOperationsTimer = null;
  if (job.kind === "sentinel_missing_geotiff_inventory" && job.result?.csv_path) {
    forestState.dataOperationsInventoryPath = job.result.csv_path;
    const link = $("forestDataOperationsInventoryLink");
    link.href = artifactUrl(job.result.csv_path);
    link.hidden = false;
  }
  if (job.kind === "sentinel_batch_download" && job.result?.bundle_selection) {
    forestState.dataOperationsLastBatchSelectionPath = job.result.bundle_selection;
  }
  if (job.kind === "hansen_candidate_mining" && job.result?.run_id) {
    forestState.dataOperationsMiningRun = job.result;
    forestState.dataOperationsMiningSelectedIds = new Set(
      (job.result.items || [])
        .filter((row) => row.decision === "SELECTED" && !row.created_sample_id)
        .map((row) => row.candidate_id),
    );
    await loadMiningRuns();
  }
  if (["sample_event_backfill", "sentinel_batch_download", "sentinel_bundle_import"].includes(job.kind)) {
    await callbacks.refreshForestData();
  }
  renderJob();
}


function watchJob(job, callbacks) {
  if (forestState.dataOperationsTimer) window.clearTimeout(forestState.dataOperationsTimer);
  forestState.dataOperationsJob = job;
  forestState.dataOperationsJobId = job.job_id;
  renderJob();
  callbacks.run(() => pollJob(callbacks));
}


async function loadImportCandidates() {
  const payload = await listBundleImportCandidates();
  forestState.dataOperationsImportCandidates = payload.items || [];
  renderImportCandidates();
}


export function initDataOperations(callbacks) {
  const storedActor = window.localStorage.getItem("forestAnnotatorName") || "";
  $("forestDataOperationsMiningActor").value = storedActor;
  renderMiningPlan();
  renderLabelingPlan();
  renderDownloadPlan();
  renderBundlePlan();
  renderImportPlan();
  renderTab();
  updateBundleSourceFileControl();

  $("forestDataOperationsOpenBtn").addEventListener("click", () => callbacks.run(async () => {
    callbacks.closeCompareView({ silent: true });
    callbacks.closeDbSyncView({ silent: true });
    forestState.dataOperationsOpen = true;
    $("forestDataOperationsView").hidden = false;
    callbacks.updateWorkspaceFocus();
    setStatus("ready");
    await Promise.all([loadImportCandidates(), loadMiningRuns()]);
  }));
  $("forestDataOperationsCloseBtn").addEventListener("click", () => {
    forestState.dataOperationsOpen = false;
    $("forestDataOperationsView").hidden = true;
    callbacks.updateWorkspaceFocus();
    callbacks.setForestStatus("Data Operations closed");
  });
  document.querySelectorAll("[data-data-operations-tab]").forEach((button) => {
    button.addEventListener("click", () => {
      forestState.dataOperationsTab = button.dataset.dataOperationsTab;
      renderTab();
      if (forestState.dataOperationsTab === "mine") renderMiningMap();
    });
  });

  $("forestDataOperationsMiningRefreshBtn").addEventListener("click", () => callbacks.run(loadMiningRuns));
  $("forestDataOperationsMiningRun").addEventListener("change", (event) => {
    const runId = event.target.value;
    forestState.dataOperationsMiningRun = forestState.dataOperationsMiningRuns.find((run) => run.run_id === runId) || null;
    forestState.dataOperationsMiningSelectedIds = new Set(
      (forestState.dataOperationsMiningRun?.items || [])
        .filter((row) => row.decision === "SELECTED" && !row.created_sample_id)
        .map((row) => row.candidate_id),
    );
    renderMiningPlan();
  });
  $("forestDataOperationsMiningPlanBtn").addEventListener("click", () => callbacks.run(async () => {
    forestState.dataOperationsMiningRun = null;
    forestState.dataOperationsMiningSelectedIds = new Set();
    renderMiningPlan();
    setStatus("starting Hansen candidate mining");
    const job = await startHansenCandidateMining(miningPayload());
    watchJob(job, callbacks);
  }));
  $("forestDataOperationsMiningSelectAll").addEventListener("change", (event) => {
    const rows = forestState.dataOperationsMiningRun?.items || [];
    forestState.dataOperationsMiningSelectedIds = event.target.checked
      ? new Set(rows.filter((row) => row.decision === "SELECTED" && !row.created_sample_id).map((row) => row.candidate_id))
      : new Set();
    renderMiningPlan();
  });
  $("forestDataOperationsMiningRejectBtn").addEventListener("click", () => callbacks.run(async () => {
    const run = forestState.dataOperationsMiningRun;
    if (!run) throw new Error("Build or select a mining run first.");
    const actor = $("forestDataOperationsMiningActor").value.trim();
    if (!actor) throw new Error("Enter an annotator name before excluding candidates.");
    window.localStorage.setItem("forestAnnotatorName", actor);
    const updated = await rejectHansenCandidates(run.run_id, {
      candidate_ids: selectedMiningRows(),
      reason_code: $("forestDataOperationsMiningRejectReason").value,
      reason_text: $("forestDataOperationsMiningRejectNote").value.trim() || null,
      actor_name: actor,
    });
    forestState.dataOperationsMiningRun = updated;
    forestState.dataOperationsMiningSelectedIds = new Set();
    renderMiningPlan();
    setStatus(`${updated.selected_count}/${updated.target_count} candidates remain selected`);
  }));
  $("forestDataOperationsMiningFillBtn").addEventListener("click", () => callbacks.run(async () => {
    const run = forestState.dataOperationsMiningRun;
    if (!run) throw new Error("Build or select a mining run first.");
    const updated = await fillHansenCandidateRun(run.run_id);
    forestState.dataOperationsMiningRun = updated;
    forestState.dataOperationsMiningSelectedIds = new Set(
      (updated.items || []).filter((row) => row.decision === "SELECTED" && !row.created_sample_id).map((row) => row.candidate_id),
    );
    renderMiningPlan();
    setStatus(`${updated.selected_count}/${updated.target_count} candidates selected after fill`);
  }));
  $("forestDataOperationsMiningApplyBtn").addEventListener("click", () => callbacks.run(async () => {
    const run = forestState.dataOperationsMiningRun;
    if (!run) throw new Error("Build or select a mining run first.");
    setStatus("creating Hansen-mined samples and events");
    const result = await materializeHansenCandidates(run.run_id, selectedMiningRows());
    forestState.dataOperationsMiningRun = result.run;
    forestState.dataOperationsMiningSelectedIds = new Set();
    await loadMiningRuns();
    $("forestDataOperationsDownloadMode").value = "hansen_mining_run";
    $("forestDataOperationsDownloadMiningRun").value = run.run_id;
    await callbacks.refreshForestData();
    setStatus(`Created ${result.created} sample(s); ${result.existing} already existed`);
  }));
  $("forestDataOperationsLabelingRun").addEventListener("change", () => {
    forestState.dataOperationsLabelingPlan = null;
    renderLabelingPlan();
  });
  $("forestDataOperationsLabelingPlanBtn").addEventListener("click", () => callbacks.run(async () => {
    const runId = $("forestDataOperationsLabelingRun").value;
    if (!runId) throw new Error("Select a mining run first.");
    setStatus("checking model artifacts and event inputs");
    forestState.dataOperationsLabelingPlan = await planModelLabeling({
      mining_run_id: runId,
      remoteclip_version: $("forestDataOperationsRemoteclipVersion").value.trim() || "v1",
      xgboost_version: $("forestDataOperationsXgboostVersion").value.trim() || "v1",
    });
    renderLabelingPlan();
    setStatus(`${forestState.dataOperationsLabelingPlan.summary.ready_inputs} event input(s) ready for inference`);
  }));

  $("forestDataOperationsEventPlanBtn").addEventListener("click", () => callbacks.run(async () => {
    setStatus("checking Hansen events");
    const result = await planEventBackfill(null);
    forestState.dataOperationsEventPlan = result;
    $("forestDataOperationsEventSyncBtn").disabled = !result.total;
    $("forestDataOperationsDownloadSummary").innerHTML = metricsHtml({
      scope: "all",
      analyses: result.total,
      events: result.event_rows,
      failed: result.failed,
    });
    setStatus(`Event plan: ${result.event_rows} event row(s) across ${result.total} Hansen analyses`);
  }));
  $("forestDataOperationsEventSyncBtn").addEventListener("click", () => callbacks.run(async () => {
    if (!forestState.dataOperationsEventPlan) throw new Error("Check the event plan first.");
    const job = await startEventBackfill(null);
    forestState.dataOperationsEventPlan = null;
    $("forestDataOperationsEventSyncBtn").disabled = true;
    watchJob(job, callbacks);
  }));
  $("forestDataOperationsDownloadPlanBtn").addEventListener("click", () => callbacks.run(async () => {
    setStatus("building Sentinel download plan");
    const plan = await planSentinelBatch(downloadPayload(callbacks));
    forestState.dataOperationsPlan = plan;
    forestState.dataOperationsSelectedKeys = new Set(
      plan.samples.flatMap((sample) => (
        sample.stop_reason ? [] : sample.operations.map((operation) => operation.operation_key)
      )),
    );
    renderDownloadPlan();
    setStatus(`${plan.summary.samples_actionable} actionable sample(s)`);
  }));
  $("forestDataOperationsDownloadSelectAll").addEventListener("change", (event) => {
    const plan = forestState.dataOperationsPlan;
    if (!plan) return;
    forestState.dataOperationsSelectedKeys = event.target.checked
      ? new Set(plan.samples.flatMap((sample) => sample.stop_reason ? [] : sample.operations.map((row) => row.operation_key)))
      : new Set();
    renderDownloadPlan();
    renderJob();
  });
  $("forestDataOperationsDownloadStartBtn").addEventListener("click", () => callbacks.run(async () => {
    const payload = {
      ...downloadPayload(callbacks),
      selected_operation_keys: [...forestState.dataOperationsSelectedKeys],
    };
    const job = await startSentinelBatch(payload);
    watchJob(job, callbacks);
  }));

  $("forestDataOperationsInventoryBtn").addEventListener("click", () => callbacks.run(async () => {
    const job = await startMissingGeotiffInventory();
    watchJob(job, callbacks);
  }));
  $("forestDataOperationsBundleSource").addEventListener("change", (event) => {
    $("forestDataOperationsInventoryFile").value = "";
    updateBundleSourceFileControl();
    forestState.dataOperationsBundlePlan = null;
    forestState.dataOperationsBundleSelectedIds = new Set();
    renderBundlePlan();
  });
  $("forestDataOperationsInventoryFile").addEventListener("change", (event) => callbacks.run(async () => {
    const file = event.target.files?.[0];
    if (!file) return;
    const source = $("forestDataOperationsBundleSource").value;
    const result = await uploadDataOperationSourceFile(file);
    if (source === "download_job_result") {
      forestState.dataOperationsLastBatchSelectionPath = result.path;
      setStatus(`Batch selection ready: ${result.name}`);
    } else {
      forestState.dataOperationsInventoryPath = result.path;
      forestState.dataOperationsInventoryName = result.name;
      setStatus(`Inventory ready: ${result.name}`);
    }
  }));
  $("forestDataOperationsBundlePlanBtn").addEventListener("click", () => callbacks.run(async () => {
    const source = $("forestDataOperationsBundleSource").value;
    const payload = {
      source,
      filters: callbacks.currentFilters(),
      profile: $("forestDataOperationsBundleProfile").value,
      periods: checkedValues("[data-data-operations-export-period]"),
      exclude_reviewed_bad: $("forestDataOperationsExcludeBad").checked,
      limit: $("forestDataOperationsBundleLimit").value
        ? numberValue("forestDataOperationsBundleLimit", null)
        : null,
    };
    if (source === "missing_geotiff_csv") {
      if (!forestState.dataOperationsInventoryPath) throw new Error("Choose the missing GeoTIFF inventory CSV first.");
      payload.inventory_path = forestState.dataOperationsInventoryPath;
    }
    if (source === "download_job_result") {
      if (!forestState.dataOperationsLastBatchSelectionPath) throw new Error("Choose the batch bundle_selection.json first.");
      payload.selection_path = forestState.dataOperationsLastBatchSelectionPath;
    }
    setStatus("building bundle plan");
    const plan = await planSentinelBundle(payload);
    forestState.dataOperationsBundlePlan = plan;
    forestState.dataOperationsBundleSelectedIds = new Set(plan.download_ids);
    renderBundlePlan();
    setStatus(`${plan.summary.unique_downloads} download(s) in bundle plan`);
  }));
  $("forestDataOperationsBundleSelectAll").addEventListener("change", (event) => {
    const plan = forestState.dataOperationsBundlePlan;
    if (!plan) return;
    forestState.dataOperationsBundleSelectedIds = event.target.checked
      ? new Set(plan.download_ids)
      : new Set();
    renderBundlePlan();
    renderJob();
  });
  $("forestDataOperationsBundleStartBtn").addEventListener("click", () => callbacks.run(async () => {
    const plan = forestState.dataOperationsBundlePlan;
    const job = await startSentinelBundle({
      download_ids: [...forestState.dataOperationsBundleSelectedIds],
      profile: $("forestDataOperationsBundleProfile").value,
      exclude_reviewed_bad: $("forestDataOperationsExcludeBad").checked,
      create_zip: $("forestDataOperationsCreateZip").checked,
      source: plan?.source || "selection",
    });
    watchJob(job, callbacks);
  }));

  $("forestDataOperationsImportRefreshBtn").addEventListener("click", () => callbacks.run(loadImportCandidates));
  $("forestDataOperationsImportCandidate").addEventListener("change", () => {
    forestState.dataOperationsImportPlan = null;
    renderImportPlan();
  });
  $("forestDataOperationsImportPlanBtn").addEventListener("click", () => callbacks.run(async () => {
    const bundlePath = $("forestDataOperationsImportCandidate").value;
    if (!bundlePath) throw new Error("Select a bundle first.");
    setStatus("validating bundle");
    const plan = await planBundleImport({
      bundle_path: bundlePath,
      overwrite_existing: $("forestDataOperationsOverwrite").checked,
      rebuild_previews: $("forestDataOperationsRebuildPreviews").checked,
    });
    forestState.dataOperationsImportPlan = plan;
    renderImportPlan();
    setStatus(plan.can_apply ? "bundle ready to import" : "bundle or database conflicts must be resolved");
  }));
  $("forestDataOperationsImportStartBtn").addEventListener("click", () => callbacks.run(async () => {
    const job = await startBundleImport({
      bundle_path: $("forestDataOperationsImportCandidate").value,
      overwrite_existing: $("forestDataOperationsOverwrite").checked,
      rebuild_previews: $("forestDataOperationsRebuildPreviews").checked,
    });
    watchJob(job, callbacks);
  }));

  [
    $("forestDataOperationsMiningCancelBtn"),
    $("forestDataOperationsDownloadCancelBtn"),
    $("forestDataOperationsBundleCancelBtn"),
    $("forestDataOperationsImportCancelBtn"),
  ].forEach((button) => button.addEventListener("click", () => callbacks.run(async () => {
    if (!forestState.dataOperationsJobId) return;
    forestState.dataOperationsJob = await cancelForestJob(forestState.dataOperationsJobId);
    renderJob();
    setStatus("cancellation requested");
  })));
}
