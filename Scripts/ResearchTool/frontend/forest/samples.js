import { api } from "./api.js?v=forest-iter2-24";

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
