import {
  cancelForestJob,
  getForestJob,
  listBundleImportCandidates,
  planBundleImport,
  planEventBackfill,
  planSentinelBatch,
  planSentinelBundle,
  startBundleImport,
  startEventBackfill,
  startMissingGeotiffInventory,
  startSentinelBatch,
  startSentinelBundle,
  uploadDataOperationSourceFile,
} from "./samples.js?v=forest-bundle-source-1";
import { forestState } from "./state.js?v=forest-event-fix-1";


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


function activeJobHost() {
  const kind = forestState.dataOperationsJob?.kind || "";
  if (kind === "sentinel_bundle_export") return $("forestDataOperationsBundleJob");
  if (kind === "sentinel_bundle_import") return $("forestDataOperationsImportJob");
  if (kind === "sentinel_missing_geotiff_inventory") return $("forestDataOperationsBundleJob");
  return $("forestDataOperationsDownloadJob");
}


function renderJob() {
  [
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
  const artifactPath = result.zip_path
    || result.csv_path
    || result.bundle_selection
    || result.operations_csv
    || null;
  host.innerHTML = `
    <strong>${escapeHtml(job.stage || job.status)}</strong>
    <span>${escapeHtml(job.message || "")}</span>
    <div class="forestDataOperationsProgress"><span style="width:${Math.max(0, Math.min(100, percent))}%"></span></div>
    ${artifactPath ? `<a class="forestDataOperationsArtifact" href="${artifactUrl(artifactPath)}">Open artifact</a>` : ""}
  `;

  const running = dataOperationsRunning();
  document.querySelectorAll(
    ".forestDataOperationsSetup input, .forestDataOperationsSetup select, .forestDataOperationsSetup button, [data-data-operations-tab], #forestDataOperationsDownloadPlanBtn, #forestDataOperationsBundlePlanBtn, #forestDataOperationsImportPlanBtn",
  ).forEach((control) => {
    control.disabled = running;
  });
  $("forestDataOperationsEventSyncBtn").disabled = running || !forestState.dataOperationsEventPlan;
  $("forestDataOperationsDownloadCancelBtn").disabled = !running;
  $("forestDataOperationsBundleCancelBtn").disabled = !running;
  $("forestDataOperationsImportCancelBtn").disabled = !running;
  $("forestDataOperationsDownloadStartBtn").disabled = running || !forestState.dataOperationsSelectedKeys.size;
  $("forestDataOperationsBundleStartBtn").disabled = running || !forestState.dataOperationsBundleSelectedIds.size;
  $("forestDataOperationsImportStartBtn").disabled = running || !forestState.dataOperationsImportPlan?.can_apply;
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
  if (manualValidation === "unvalidated") filters.manual_validation = "__no_manual_validation";
  if (manualValidation === "any") delete filters.manual_validation;
  return {
    filters,
    mode: $("forestDataOperationsDownloadMode").value,
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
    await loadImportCandidates();
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
    });
  });

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
