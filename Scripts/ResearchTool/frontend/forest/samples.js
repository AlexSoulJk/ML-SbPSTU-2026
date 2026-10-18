import { api } from "./api.js?v=forest-data-ops-1";

function cleanFilters(filters) {
  return Object.fromEntries(
    Object.entries(filters).filter(([_key, value]) => value !== null && value !== undefined && value !== ""),
  );
}

export async function importForestFiles(files) {
  const form = new FormData();
  [...files].forEach((file) => form.append("files", file));
  return api("/api/forest/import", { method: "POST", body: form });
}

export async function listForestSamples({ limit, offset, filters, selectedId, nearestN }) {
  const params = new URLSearchParams();
  params.set("limit", String(limit));
  params.set("offset", String(offset));
  params.set("nearest_n", String(nearestN));
  if (selectedId) params.set("selected_id", selectedId);
  Object.entries(cleanFilters(filters)).forEach(([key, value]) => params.set(key, value));
  return api(`/api/forest/samples?${params.toString()}`);
}

export async function getForestSample(sampleId) {
  return api(`/api/forest/samples/${sampleId}`);
}

export async function saveManualValidation(sampleId, validation, notes) {
  return api(`/api/forest/samples/${sampleId}/manual-validation`, {
    method: "POST",
    body: { validation, notes },
  });
}

export async function saveSampleDisplayName(sampleId, displayName) {
  return api(`/api/forest/samples/${sampleId}/display-name`, {
    method: "POST",
    body: { display_name: displayName },
  });
}

export async function saveSampleFlags(sampleId, flags) {
  return api(`/api/forest/samples/${sampleId}/flags`, {
    method: "POST",
    body: flags,
  });
}

export async function analyzeHansen(sampleId, options = {}) {
  return api(`/api/forest/samples/${sampleId}/hansen/analyze-job`, {
    method: "POST",
    body: { include_treecover: Boolean(options.include_treecover) },
  });
}

export async function startHansenBatch(filters, limit, options = {}) {
  return api("/api/forest/hansen/analyze-batch", {
    method: "POST",
    body: {
      filters: cleanFilters(filters),
      limit,
      include_treecover: Boolean(options.include_treecover),
    },
  });
}

export async function getForestJob(jobId) {
  return api(`/api/forest/jobs/${jobId}`);
}

export async function cancelForestJob(jobId) {
  return api(`/api/forest/jobs/${jobId}/cancel`, {
    method: "POST",
    body: {},
  });
}

export async function searchSentinel(sampleId, options) {
  return api(`/api/forest/samples/${sampleId}/sentinel/search`, {
    method: "POST",
    body: options,
  });
}

export async function downloadSentinel(sampleId, options) {
  return api(`/api/forest/samples/${sampleId}/sentinel/download`, {
    method: "POST",
    body: options,
  });
}

export async function saveSentinelReview(downloadId, payload) {
  return api(`/api/forest/sentinel/downloads/${downloadId}/review`, {
    method: "PUT",
    body: payload,
  });
}

export async function exportForestSamples(filters) {
  return api("/api/forest/export", {
    method: "POST",
    body: {
      scope: "current_filtered",
      filters: cleanFilters(filters),
    },
  });
}

export async function listDbSyncCandidates() {
  return api("/api/forest/db-sync/candidates");
}

export async function compareDbSync(incomingPath) {
  return api("/api/forest/db-sync/compare", {
    method: "POST",
    body: { incoming_path: incomingPath },
  });
}

export async function applyDbSync(incomingPath, groups = null) {
  return api("/api/forest/db-sync/apply", {
    method: "POST",
    body: { incoming_path: incomingPath, groups },
  });
}

export async function startSentinelCacheDownload(downloadIds, options = {}) {
  return api("/api/forest/db-sync/cache/sentinel/download-job", {
    method: "POST",
    body: {
      download_ids: downloadIds,
      max_cloud: Number(options.max_cloud ?? 30),
    },
  });
}

export async function planEventBackfill(limit = null) {
  return api("/api/forest/data-operations/events/backfill/plan", {
    method: "POST",
    body: { limit },
  });
}

export async function startEventBackfill(limit = null) {
  return api("/api/forest/data-operations/events/backfill/jobs", {
    method: "POST",
    body: { limit },
  });
}

export async function planSentinelBatch(payload) {
  return api("/api/forest/data-operations/sentinel-download/plan", {
    method: "POST",
    body: payload,
  });
}

export async function startSentinelBatch(payload) {
  return api("/api/forest/data-operations/sentinel-download/jobs", {
    method: "POST",
    body: payload,
  });
}

export async function startMissingGeotiffInventory() {
  return api("/api/forest/data-operations/missing-geotiffs/jobs", {
    method: "POST",
    body: {},
  });
}

export async function uploadMissingGeotiffInventory(file) {
  const form = new FormData();
  form.append("file", file);
  return api("/api/forest/data-operations/inventory/upload", {
    method: "POST",
    body: form,
  });
}

export async function planSentinelBundle(payload) {
  return api("/api/forest/data-operations/bundles/plan", {
    method: "POST",
    body: payload,
  });
}

export async function startSentinelBundle(payload) {
  return api("/api/forest/data-operations/bundles/jobs", {
    method: "POST",
    body: payload,
  });
}

export async function listBundleImportCandidates() {
  return api("/api/forest/data-operations/bundle-import/candidates");
}

export async function planBundleImport(payload) {
  return api("/api/forest/data-operations/bundle-import/plan", {
    method: "POST",
    body: payload,
  });
}

export async function startBundleImport(payload) {
  return api("/api/forest/data-operations/bundle-import/jobs", {
    method: "POST",
    body: payload,
  });
}
