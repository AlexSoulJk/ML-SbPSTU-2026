import { clearForestLayers, renderForestMap } from "./map.js?v=forest-event-fix-1";
import { initDataOperations } from "./data_operations.js?v=forest-event-fix-1";
import {
  analyzeHansen,
  applyDbSync,
  cancelForestJob,
  compareDbSync,
  downloadSentinel,
  exportForestSamples,
  getForestSample,
  getForestJob,
  importForestFiles,
  listForestSamples,
  listDbSyncCandidates,
  saveManualValidation,
  saveSampleDisplayName,
  saveSentinelReview,
  searchSentinel,
  startHansenBatch,
  startSentinelCacheDownload,
} from "./samples.js?v=forest-data-ops-1";
import { forestState } from "./state.js?v=forest-event-fix-1";

function $(id) {
  return document.getElementById(id);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function setForestStatus(message) {
  $("forestStatusLine").textContent = message;
}

function pushDebugEvent(entry) {
  forestState.debugEvents.unshift(entry);
  forestState.debugEvents = forestState.debugEvents.slice(0, 30);
  renderDebugEvents();
}

function renderDebugEvents() {
  const host = $("forestMapDebug");
  if (!host) return;
  if (!forestState.debugEvents.length) {
    host.textContent = "waiting for map events";
    return;
  }
  host.innerHTML = forestState.debugEvents
    .map((entry) => {
      const data = Object.entries(entry)
        .filter(([key]) => !["time", "message"].includes(key))
        .map(([key, value]) => `${key}=${String(value)}`)
        .join(" ");
      return `<div><strong>${escapeHtml(entry.time)}</strong> ${escapeHtml(entry.message)}<br /><span>${escapeHtml(data)}</span></div>`;
    })
    .join("");
}

function logUiDebug(message, data = {}) {
  const entry = {
    time: new Date().toLocaleTimeString(),
    message,
    ...data,
  };
  console.debug("[forest-ui]", entry);
  pushDebugEvent(entry);
}

function debugEventsText() {
  return forestState.debugEvents
    .map((entry) => {
      const data = Object.entries(entry)
        .filter(([key]) => !["time", "message"].includes(key))
        .map(([key, value]) => `${key}=${String(value)}`)
        .join(" ");
      return `[${entry.time}] ${entry.message}${data ? ` ${data}` : ""}`;
    })
    .join("\n");
}

async function copyDebugEvents() {
  try {
    await navigator.clipboard.writeText(debugEventsText());
    setForestStatus("map debug copied");
  } catch (error) {
    logUiDebug("copy debug failed", { error: error.message || String(error) });
    setForestStatus("could not copy debug; select text manually");
  }
}

function currentFilters() {
  return {
    q: $("forestQueryInput").value.trim(),
    id_q: $("forestIdInput").value.trim(),
    driver: $("forestDriverInput").value.trim(),
    region: $("forestRegionInput").value.trim(),
    manual_validation: $("forestValidationFilter").value,
  };
}

function nearestN() {
  return Number($("forestNearestInput").value) || 25;
}

function sampleDisplayName(sample) {
  return sample?.display_name || sample?.short_id || "";
}

function statusLabel(status) {
  return status === "TOO_OLD" ? "Too old" : status;
}

const SCL_LEGEND_ITEMS = [
  { code: 0, label: "No data", color: "#000000" },
  { code: 1, label: "Saturated", color: "#b40000" },
  { code: 2, label: "Dark area", color: "#5a5a5a" },
  { code: 3, label: "Cloud shadow", color: "#60361c" },
  { code: 4, label: "Vegetation", color: "#288c43" },
  { code: 5, label: "Bare soil", color: "#dcc462" },
  { code: 6, label: "Water", color: "#326ebe" },
  { code: 7, label: "Unclassified", color: "#a0a0a0" },
  { code: 8, label: "Cloud medium", color: "#e1e1e1" },
  { code: 9, label: "Cloud high", color: "#ffffff" },
  { code: 10, label: "Thin cirrus", color: "#aad2ff" },
  { code: 11, label: "Snow / ice", color: "#d2f5ff" },
];

function hexToRgb(hex) {
  const value = String(hex || "").replace("#", "");
  if (value.length !== 6) return null;
  const red = Number.parseInt(value.slice(0, 2), 16);
  const green = Number.parseInt(value.slice(2, 4), 16);
  const blue = Number.parseInt(value.slice(4, 6), 16);
  if ([red, green, blue].some((channel) => Number.isNaN(channel))) return null;
  return [red, green, blue];
}

function rgbKey(red, green, blue) {
  return `${red},${green},${blue}`;
}

function colorKey(color) {
  const rgb = hexToRgb(color);
  return rgb ? rgbKey(...rgb) : "";
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Could not load image: ${url}`));
    image.src = url;
  });
}

function drawImageToCanvas(image) {
  const width = image.naturalWidth || image.width;
  const height = image.naturalHeight || image.height;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.imageSmoothingEnabled = false;
  context.drawImage(image, 0, 0, width, height);
  return { canvas, context, width, height };
}

async function presentColorsForImage(url) {
  const image = await loadImage(url);
  const { context, width, height } = drawImageToCanvas(image);
  const data = context.getImageData(0, 0, width, height).data;
  const colors = new Set();
  for (let index = 0; index < data.length; index += 4) {
    if (data[index + 3] > 0) {
      colors.add(rgbKey(data[index], data[index + 1], data[index + 2]));
    }
  }
  return colors;
}

async function sclLegendForPreview(preview) {
  if (!preview?.url || preview.view !== "scl") return [];
  if (forestState.sclLegendCache.has(preview.url)) {
    return forestState.sclLegendCache.get(preview.url);
  }
  if (forestState.sclLegendPending.has(preview.url)) {
    return forestState.sclLegendPending.get(preview.url);
  }

  const promise = presentColorsForImage(preview.url)
    .then((colors) => SCL_LEGEND_ITEMS
      .filter((item) => colors.has(colorKey(item.color)))
      .map((item) => ({ ...item, key: String(item.code) })))
    .catch((error) => {
      console.warn("[forest-ui] SCL legend detection failed", error);
      return [];
    })
    .finally(() => forestState.sclLegendPending.delete(preview.url));
  forestState.sclLegendPending.set(preview.url, promise);
  const legend = await promise;
  forestState.sclLegendCache.set(preview.url, legend);
  return legend;
}

function mergeLegendItems(groups, keyField) {
  const byKey = new Map();
  groups.flat().forEach((item) => {
    const key = String(item[keyField]);
    if (!byKey.has(key)) byKey.set(key, item);
  });
  return [...byKey.values()].sort((left, right) => Number(left[keyField]) - Number(right[keyField]));
}

function visibleLegendItems(legend, hiddenKeys, keyField) {
  return legend.filter((item) => !hiddenKeys.has(String(item[keyField])));
}

async function filteredRasterUrl(sourceUrl, legend, hiddenKeys, keyField) {
  if (!sourceUrl || !legend?.length) return sourceUrl;
  const activeHidden = legend
    .map((item) => String(item[keyField]))
    .filter((key) => hiddenKeys.has(key))
    .sort();
  if (!activeHidden.length) return sourceUrl;

  const visibleColors = new Set(visibleLegendItems(legend, hiddenKeys, keyField).map((item) => colorKey(item.color)));
  const cacheKey = `${sourceUrl}::${keyField}::${activeHidden.join(",")}`;
  if (forestState.rasterFilterCache.has(cacheKey)) {
    return forestState.rasterFilterCache.get(cacheKey);
  }

  const image = await loadImage(sourceUrl);
  const { canvas, context, width, height } = drawImageToCanvas(image);
  const imageData = context.getImageData(0, 0, width, height);
  const data = imageData.data;
  for (let index = 0; index < data.length; index += 4) {
    if (data[index + 3] === 0) continue;
    if (!visibleColors.has(rgbKey(data[index], data[index + 1], data[index + 2]))) {
      data[index + 3] = 0;
    }
  }
  context.putImageData(imageData, 0, 0);
  const filteredUrl = canvas.toDataURL("image/png");
  forestState.rasterFilterCache.set(cacheKey, filteredUrl);
  return filteredUrl;
}

function currentHansenOverlay(detail = forestState.selectedDetail) {
  const masks = detail?.hansen?.masks || [];
  const layer = forestState.hansenLayer || "dominant_year";
  const exact =
    masks.find((mask) => mask.mask_type === `aoi_${layer}`)
    || masks.find((mask) => mask.mask_type === layer)
    || null;
  if (layer === "treecover2000") return exact;
  return (
    exact
    || masks.find((mask) => mask.mask_type === "aoi_dominant_year")
    || masks.find((mask) => mask.mask_type === "dominant_year")
    || masks.find((mask) => mask.mask_type === "aoi_all_loss")
    || masks.find((mask) => mask.mask_type === "all_loss")
    || null
  );
}

function renderHansenLegend(hostId = "forestHansenLegend", detail = forestState.selectedDetail) {
  const host = $(hostId);
  if (!host) return;

  const overlay = currentHansenOverlay(detail);
  const legend = overlay?.legend || [];
  if (forestState.hansenLayer !== "all_loss") {
    host.hidden = false;
    host.innerHTML = `<div class="meta">Legend is shown for all loss.</div>`;
    return;
  }
  if (!legend.length) {
    host.hidden = false;
    host.innerHTML = `<div class="meta">No Hansen loss years in current view.</div>`;
    return;
  }

  host.hidden = false;
  host.innerHTML = `
    <div class="forestHansenLegendTitle">Loss year</div>
    <div class="forestHansenLegendItems">
      ${legend
        .map((item) => {
          const key = String(item.year);
          const hidden = forestState.hansenLossHiddenYears.has(key);
          return `
          <button class="forestHansenLegendItem ${hidden ? "muted" : ""}" type="button" data-hansen-loss-year="${escapeHtml(key)}" title="${hidden ? "Show" : "Hide"} ${escapeHtml(item.year)}">
            <i style="background:${escapeHtml(item.color)}"></i>
            <span>${escapeHtml(item.year)}</span>
          </button>
        `;
        })
        .join("")}
    </div>
  `;

  host.querySelectorAll("[data-hansen-loss-year]").forEach((button) => {
    button.addEventListener("click", () => {
      const key = button.dataset.hansenLossYear;
      if (forestState.hansenLossHiddenYears.has(key)) {
        forestState.hansenLossHiddenYears.delete(key);
      } else {
        forestState.hansenLossHiddenYears.add(key);
      }
      renderRasterLegends();
      renderMap();
      if (forestState.compareOpen) renderCompareView();
    });
  });
}

function renderSclLegendItems(host, legend) {
  if (!legend.length) {
    host.hidden = false;
    host.innerHTML = `<div class="meta">No SCL classes in current image.</div>`;
    return;
  }
  host.hidden = false;
  host.innerHTML = `
    <div class="forestHansenLegendTitle">SCL classes</div>
    <div class="forestHansenLegendItems">
      ${legend
        .map((item) => {
          const key = String(item.code);
          const hidden = forestState.sclHiddenClasses.has(key);
          return `
            <button class="forestHansenLegendItem ${hidden ? "muted" : ""}" type="button" data-scl-class="${escapeHtml(key)}" title="${hidden ? "Show" : "Hide"} ${escapeHtml(item.label)}">
              <i style="background:${escapeHtml(item.color)}"></i>
              <span>${escapeHtml(item.code)} ${escapeHtml(item.label)}</span>
            </button>
          `;
        })
        .join("")}
    </div>
  `;
  host.querySelectorAll("[data-scl-class]").forEach((button) => {
    button.addEventListener("click", () => {
      const key = button.dataset.sclClass;
      if (forestState.sclHiddenClasses.has(key)) {
        forestState.sclHiddenClasses.delete(key);
      } else {
        forestState.sclHiddenClasses.add(key);
      }
      renderRasterLegends();
      renderMap();
      if (forestState.compareOpen) renderCompareView();
    });
  });
}

function sclPreviewSignature(previews) {
  return previews
    .filter((preview) => preview?.view === "scl" && previewHasImage(preview))
    .map((preview) => preview.url)
    .sort()
    .join("|");
}

function currentSclLegendPreviews(hostId) {
  if (hostId === "forestCompareSclLegend") {
    return forestState.compareSclPreviews;
  }
  return forestState.activeSentinelPreview ? [forestState.activeSentinelPreview] : [];
}

function renderSclLegend(hostId = "forestSclLegend", previews = []) {
  const host = $(hostId);
  if (!host) return;
  const sclPreviews = previews.filter((preview) => preview?.view === "scl" && previewHasImage(preview));
  if (!sclPreviews.length) {
    host.hidden = true;
    host.innerHTML = "";
    return;
  }

  const ready = sclPreviews.every((preview) => forestState.sclLegendCache.has(preview.url));
  if (!ready) {
    host.hidden = false;
    host.innerHTML = `<div class="meta">Loading SCL legend...</div>`;
    const expectedSignature = sclPreviewSignature(sclPreviews);
    Promise.all(sclPreviews.map((preview) => sclLegendForPreview(preview))).then(() => {
      const currentPreviews = currentSclLegendPreviews(hostId);
      if (sclPreviewSignature(currentPreviews) !== expectedSignature) return;
      renderSclLegend(hostId, currentPreviews);
      applyVisibleRasterFilters();
    });
    return;
  }

  const legend = mergeLegendItems(
    sclPreviews.map((preview) => forestState.sclLegendCache.get(preview.url) || []),
    "code",
  );
  renderSclLegendItems(host, legend);
}

function renderRasterLegends() {
  renderHansenLegend();
  renderSclLegend("forestSclLegend", forestState.activeSentinelPreview ? [forestState.activeSentinelPreview] : []);
  renderHansenLegend("forestCompareHansenLegend", forestState.selectedDetail);
  renderSclLegend("forestCompareSclLegend", forestState.compareSclPreviews);
}

function renderLayerPanelState() {
  const panel = $("mapLayerPanel");
  const openButton = $("mapLayerPanelOpen");
  const closeButton = $("mapLayerPanelClose");
  if (!panel || !openButton || !closeButton) return;
  panel.classList.toggle("collapsed", forestState.layerPanelCollapsed);
  openButton.hidden = !forestState.layerPanelCollapsed;
  closeButton.hidden = forestState.layerPanelCollapsed;
  openButton.setAttribute("aria-expanded", "false");
  closeButton.setAttribute("aria-expanded", "true");
  renderRasterLegends();
}

function setLayerPanelCollapsed(collapsed) {
  forestState.layerPanelCollapsed = collapsed;
  renderLayerPanelState();
  window.pipelineExplorer?.refreshMapLayout();
}

function renderCompareLayerPanelState() {
  const panel = $("forestCompareLayerPanel");
  const openButton = $("forestCompareLayerPanelOpen");
  const closeButton = $("forestCompareLayerPanelClose");
  if (!panel || !openButton || !closeButton) return;
  panel.classList.toggle("collapsed", forestState.compareLayerPanelCollapsed);
  openButton.hidden = !forestState.compareLayerPanelCollapsed;
  closeButton.hidden = forestState.compareLayerPanelCollapsed;
  openButton.setAttribute("aria-expanded", "false");
  closeButton.setAttribute("aria-expanded", "true");
  renderRasterLegends();
}

function setCompareLayerPanelCollapsed(collapsed) {
  forestState.compareLayerPanelCollapsed = collapsed;
  renderCompareLayerPanelState();
  Object.values(forestState.compareMaps).forEach((map) => map?.invalidateSize());
}

function toggleSidebar() {
  const shell = document.querySelector(".shell");
  const button = $("sidebarToggle");
  if (!shell || !button) return;
  shell.classList.toggle("sidebarCollapsed");
  const expanded = !shell.classList.contains("sidebarCollapsed");
  button.setAttribute("aria-expanded", String(expanded));
  button.title = expanded ? "Collapse sidebar" : "Expand sidebar";
  window.setTimeout(() => window.pipelineExplorer?.refreshMapLayout(), 220);
}

function updateWorkspaceFocus() {
  const shell = document.querySelector(".shell");
  if (!shell) return;
  shell.classList.toggle(
    "contextFocus",
    Boolean(forestState.dbSyncOpen || forestState.compareOpen || forestState.dataOperationsOpen),
  );
  window.setTimeout(() => {
    window.pipelineExplorer?.refreshMapLayout();
    Object.values(forestState.compareMaps || {}).forEach((map) => map?.invalidateSize());
  }, 220);
}

function enhanceSidebarPanels() {
  document.querySelectorAll(".sidebar > .panel").forEach((panel) => {
    if (panel.dataset.collapsibleReady) return;
    let header = panel.querySelector(":scope > h2, :scope > .panelTitleRow");
    if (!header) return;

    const body = document.createElement("div");
    body.className = "panelBody";
    [...panel.children].forEach((child) => {
      if (child !== header) body.appendChild(child);
    });
    panel.appendChild(body);

    let titleButton = null;
    if (header.matches("h2")) {
      titleButton = document.createElement("button");
      titleButton.type = "button";
      titleButton.className = "panelCollapseTitle";
      titleButton.textContent = header.textContent;
      header.replaceWith(titleButton);
      header = titleButton;
    } else {
      header.classList.add("panelCollapseRow");
      const title = header.querySelector("h2");
      titleButton = document.createElement("button");
      titleButton.type = "button";
      titleButton.className = "panelCollapseTitle";
      titleButton.textContent = title?.textContent || "Panel";
      title?.replaceWith(titleButton);
    }

    titleButton.setAttribute("aria-expanded", "true");
    titleButton.addEventListener("click", () => {
      const collapsed = panel.classList.toggle("collapsed");
      titleButton.setAttribute("aria-expanded", String(!collapsed));
    });
    panel.dataset.collapsibleReady = "true";
  });
}

async function setMode(mode) {
  document.body.dataset.mode = mode;
  forestState.mode = mode;
  $("forestModeBtn").classList.toggle("active", mode === "forest");
  $("pipelineModeBtn").classList.toggle("active", mode === "pipeline");

  if (mode === "pipeline") {
    closeDbSyncView();
    clearForestLayers();
    if (!window.pipelineExplorer?.state?.kml) {
      await window.pipelineExplorer?.run(window.pipelineExplorer.loadDefaultKml);
    }
  } else {
    window.pipelineExplorer?.clearGeometryLayers();
    window.pipelineExplorer?.clearLayer("raster");
    await refreshSamples({ fit: forestState.samples.length === 0 });
  }
  window.pipelineExplorer?.refreshMapLayout();
}

function renderPageMeta() {
  const start = forestState.total === 0 ? 0 : forestState.offset + 1;
  const end = Math.min(forestState.offset + forestState.limit, forestState.total);
  $("forestPageMeta").textContent = `${start}-${end} / ${forestState.total}`;
  $("forestPrevPageBtn").disabled = forestState.offset <= 0;
  $("forestNextPageBtn").disabled = forestState.offset + forestState.limit >= forestState.total;
}

function sampleCard(sample, options = {}) {
  const hansenRunning = Boolean(isHansenJobRunning(sample));
  const displayName = sampleDisplayName(sample);
  const card = document.createElement("button");
  card.type = "button";
  card.dataset.sampleId = sample.sample_id;
  card.className = [
    "forestSampleCard",
    sample.sample_id === forestState.selectedSampleId ? "active" : "",
    options.pinned ? "pinned" : "",
  ]
    .filter(Boolean)
    .join(" ");
  card.innerHTML = `
    <strong>${escapeHtml(displayName)} ${sample.is_saved ? '<i class="forestSavedShape" aria-label="saved"></i>' : ""}</strong>
    <span>${escapeHtml(sample.driver_primary || "unknown")} / ${escapeHtml(sample.confidence_primary || "n/a")}</span>
    ${sample.manual_validation ? `<span>${escapeHtml(sample.manual_validation)}</span>` : ""}
    ${sample.has_multiple_events ? `<span class="forestSampleBadge multiEvent">multi-event</span>` : ""}
    ${hansenRunning ? `<span class="forestSampleCardJob"><i class="forestSpinner" aria-hidden="true"></i>${escapeHtml(hansenStageLabel(forestState.hansenJob.stage))}</span>` : ""}
  `;
  card.addEventListener("click", () => run(() => selectSample(sample.sample_id, { fit: true })));
  return card;
}

function scrollActiveSampleIntoView() {
  const active = $("forestSamplesList").querySelector(".forestSampleCard.active");
  if (!active) return;
  window.requestAnimationFrame(() => {
    active.scrollIntoView({ block: "nearest", behavior: "smooth" });
  });
}

function renderHansenSummary(sample) {
  if (!sample.hansen) {
    return `<div class="meta">No Hansen data loaded for this sample.</div>`;
  }
  const share = sample.hansen.dominant_year_share ?? 0;
  return `
    <div class="forestMetrics">
      <div><span>Status</span><strong>${escapeHtml(statusLabel(sample.hansen.status))}</strong></div>
      <div><span>Event year</span><strong>${escapeHtml(sample.hansen.event_year || "n/a")}</strong></div>
      <div><span>Total loss</span><strong>${Number(sample.hansen.total_loss_area_ha || 0).toFixed(2)} ha</strong></div>
      <div><span>Dominant share</span><strong>${Math.round(share * 100)}%</strong></div>
    </div>
  `;
}

function sceneKey(period, item) {
  return `${period}|${item.id || item.datetime || "scene"}`;
}

function searchedScenes(sample) {
  return (sample.sentinelSearch?.groups || []).flatMap((group) =>
    (group.items || []).map((item, index) => ({
      period: group.period,
      year: group.year,
      item,
      index,
      key: sceneKey(group.period, item),
    })),
  );
}

function sentinelSearchDownloadState(sample, period, sceneId) {
  const download = (sample.sentinel?.downloads || []).find(
    (item) => item.period === period && item.scene_id === sceneId,
  );
  if (!download) return "";
  return download.review?.is_excluded ? " / excluded" : " / downloaded";
}

function renderSentinelResults(sample) {
  const groups = sample.sentinelSearch?.groups || [];
  if (!groups.length) return `<div class="meta">No Sentinel search results yet.</div>`;
  return `
    <div class="forestPanelToolbar">
      <button id="forestSelectAllScenesBtn" type="button">Select all</button>
      <button id="forestClearScenesBtn" type="button">Clear</button>
      <button id="forestDownloadSentinelBtn" class="primary" type="button">Download checked imagery</button>
    </div>
    <div class="forestSceneGroups">
      ${groups
        .map(
          (group) => `
            <article class="forestSceneGroup">
              <strong>${escapeHtml(group.period)} ${escapeHtml(group.year)}</strong>
              ${(group.items || [])
                .map((item) => {
                  const key = sceneKey(group.period, item);
                  const date = item.datetime ? item.datetime.slice(0, 10) : "unknown";
                  const cloud = item.cloud_cover === null || item.cloud_cover === undefined
                    ? "n/a"
                    : `${Number(item.cloud_cover).toFixed(1)}%`;
                  const downloadState = sentinelSearchDownloadState(sample, group.period, item.id);
                  return `
                    <label class="forestSceneOption">
                      <input class="forestSceneCheck" type="checkbox" data-scene-key="${escapeHtml(key)}" ${forestState.sentinelSelectedSceneKeys.has(key) ? "checked" : ""} />
                      <span>${escapeHtml(date)} / cloud ${escapeHtml(cloud)} / score ${escapeHtml(item.combined_score)}${escapeHtml(downloadState)}</span>
                    </label>
                  `;
                })
                .join("") || "<span>no scenes</span>"}
            </article>
          `,
        )
        .join("")}
    </div>
  `;
}

const SENTINEL_VIEW_ORDER = ["rgb", "false_color", "ndvi", "nbr", "ndmi", "scl"];
const SENTINEL_REVIEW_REASONS = [
  { value: "CLOUDS", label: "Clouds" },
  { value: "CLOUD_SHADOW", label: "Cloud shadow" },
  { value: "HAZE_OR_SMOKE", label: "Haze or smoke" },
  { value: "SNOW_OR_ICE", label: "Snow or ice" },
  { value: "SEASON_MISMATCH", label: "Season mismatch" },
  { value: "DARK_AND_LOW_CONTRAST", label: "Dark + low contrast" },
  { value: "TOO_DARK_OR_LOW_CONTRAST", label: "Too dark / low contrast (legacy)" },
  { value: "NO_DATA_OR_BLACK_PIXELS", label: "No data / black pixels" },
  { value: "AOI_NOT_COVERED", label: "AOI not covered" },
  { value: "GEOREGISTRATION_SHIFT", label: "Georegistration shift" },
  { value: "WRONG_EVENT_WINDOW", label: "Wrong event window" },
  { value: "PREVIEW_OR_PROCESSING_ARTIFACT", label: "Preview artifact" },
  { value: "OTHER", label: "Other" },
];
const SENTINEL_REVIEW_REASON_LABELS = new Map(
  SENTINEL_REVIEW_REASONS.map((reason) => [reason.value, reason.label]),
);

function sentinelViewRank(view) {
  const index = SENTINEL_VIEW_ORDER.indexOf(view);
  return index === -1 ? SENTINEL_VIEW_ORDER.length : index;
}

function sentinelViewLabel(view) {
  return {
    rgb: "RGBA",
    false_color: "NIR",
    ndvi: "NDVI",
    nbr: "NBR",
    ndmi: "NDMI",
    scl: "SCL",
  }[view] || view;
}

function sentinelReviewReasonLabel(reasonCode) {
  return SENTINEL_REVIEW_REASON_LABELS.get(reasonCode) || reasonCode || "No reason";
}

function uniqueSentinelReasonCodes(codes) {
  const result = [];
  (codes || []).forEach((code) => {
    const cleaned = String(code || "").trim();
    if (cleaned && !result.includes(cleaned)) result.push(cleaned);
  });
  return result;
}

function sentinelReviewReasonCodes(review) {
  const codes = uniqueSentinelReasonCodes(review?.reason_codes);
  if (codes.length) return codes;
  return review?.reason_code ? [review.reason_code] : [];
}

function sentinelReviewReasonText(review, reasonCode) {
  const reason = (review?.reasons || []).find((item) => item.reason_code === reasonCode);
  return reason?.reason_text || (reasonCode === "OTHER" ? review?.reason_text : "");
}

function sentinelReviewReasonSummary(review) {
  const reasonCodes = sentinelReviewReasonCodes(review);
  if (!reasonCodes.length) return "Choose reasons before saving.";
  return `Reasons: ${reasonCodes
    .map((code) => {
      const label = sentinelReviewReasonLabel(code);
      const reasonText = sentinelReviewReasonText(review, code);
      if (reasonText) return `${label} / ${reasonText}`;
      return label;
    })
    .join(", ")}`;
}

function baseSentinelReview(download) {
  const review = download.review || {
    is_excluded: false,
    reason_code: null,
    reason_codes: [],
    reasons: [],
    reason_text: null,
    notes: null,
  };
  const reasonCodes = sentinelReviewReasonCodes(review);
  return {
    ...review,
    reason_code: review.reason_code || reasonCodes[0] || null,
    reason_codes: reasonCodes,
  };
}

function effectiveSentinelReview(download) {
  return forestState.sentinelReviewDrafts.get(download.download_id) || baseSentinelReview(download);
}

function sentinelReviewReasonChecks(selectedReasons, downloadId, saving) {
  const selected = new Set(selectedReasons);
  return `
    <div class="forestSentinelReasonGrid">
      ${SENTINEL_REVIEW_REASONS.map(
        (reason) => `
          <label class="forestSentinelReasonOption">
            <input
              class="forestSentinelReasonCheck"
              type="checkbox"
              data-download-id="${escapeHtml(downloadId)}"
              data-reason-code="${escapeHtml(reason.value)}"
              ${selected.has(reason.value) ? "checked" : ""}
              ${saving ? "disabled" : ""}
            />
            <span>${escapeHtml(reason.label)}</span>
          </label>
        `,
      ).join("")}
    </div>
  `;
}

function renderSentinelReviewPanel(download, review, options = {}) {
  if (!review.is_excluded) return "";

  const downloadId = download.download_id || "";
  const reasonCodes = sentinelReviewReasonCodes(review);
  const reasonText = review.reason_text || "";
  const saving = Boolean(options.saving);
  const hasDraft = forestState.sentinelReviewDrafts.has(downloadId);
  const isOther = reasonCodes.includes("OTHER");
  const reasonSummary = sentinelReviewReasonSummary({ ...review, reason_codes: reasonCodes });

  return `
    <div class="forestSentinelReviewPanel">
      <div class="forestSentinelReasonTitle">Reasons</div>
      ${sentinelReviewReasonChecks(reasonCodes, downloadId, saving)}
      ${
        isOther
          ? `<label>
              Other reason
              <input
                class="forestSentinelOtherReasonInput"
                type="text"
                data-download-id="${escapeHtml(downloadId)}"
                value="${escapeHtml(reasonText)}"
                maxlength="500"
                placeholder="What is wrong with this image?"
                ${saving ? "disabled" : ""}
              />
            </label>`
          : ""
      }
      <div class="forestSentinelReviewActions">
        <span class="meta">${escapeHtml(reasonSummary)}</span>
        ${hasDraft ? `<button class="forestSentinelReviewCancelBtn" type="button" data-download-id="${escapeHtml(downloadId)}" ${saving ? "disabled" : ""}>Cancel</button>` : ""}
        <button class="forestSentinelReviewSaveBtn primary" type="button" data-download-id="${escapeHtml(downloadId)}" ${saving ? "disabled" : ""}>
          ${saving ? "Saving" : "Save"}
        </button>
      </div>
    </div>
  `;
}

function sentinelDownloadKey(download) {
  return download.download_id || `${download.period || "S2"}|${download.scene_id || download.datetime || ""}`;
}

function sentinelImageIdMatches(download, query) {
  if (!query) return true;
  const normalized = query.toLowerCase();
  return [
    download.download_id,
    download.scene_id,
  ].some((value) => String(value || "").toLowerCase().includes(normalized));
}

function eventScopedSentinelDownloads(sample) {
  const downloads = sample.sentinel?.downloads || [];
  const selectedEventId = forestState.selectedSampleEventId || "all";
  if (selectedEventId === "all") return downloads;
  if (selectedEventId === "legacy") {
    return downloads.filter((download) => !(download.event_links || []).length);
  }
  return downloads.filter((download) =>
    (download.event_links || []).some((link) => link.event_id === selectedEventId),
  );
}

function filteredSentinelDownloads(sample) {
  const downloads = eventScopedSentinelDownloads(sample);
  const query = forestState.sentinelImageIdQuery.trim();
  if (!query) return downloads;
  return downloads.filter((download) => sentinelImageIdMatches(download, query));
}

function previewHasImage(preview) {
  return Boolean(preview?.url) && preview.cache_exists !== false;
}

function previewButton({ preview, download, className = "", downloadKey = "" }) {
  const hasImage = previewHasImage(preview);
  const active = hasImage && forestState.activeSentinelPreview?.url === preview.url;
  const date = download.datetime ? download.datetime.slice(0, 10) : "";
  const cloud = download.cloud_fraction === null || download.cloud_fraction === undefined
    ? "cloud n/a"
    : `cloud ${Math.round(Number(download.cloud_fraction) * 100)}%`;
  return `
    <button class="forestPreviewCard ${className} ${active ? "active" : ""} ${hasImage ? "" : "missingCache"}" type="button" data-preview-url="${escapeHtml(preview.url || "")}" data-download-key="${escapeHtml(downloadKey)}" ${hasImage ? "" : "disabled"}>
      ${
        hasImage
          ? `<img src="${escapeHtml(preview.url)}" alt="${escapeHtml(sentinelViewLabel(preview.view))}" />`
          : `<span class="forestPreviewMissingThumb">missing</span>`
      }
      <span class="forestPreviewText">
        <strong>${escapeHtml(sentinelViewLabel(preview.view))}</strong>
        <span>${escapeHtml(date)} / ${escapeHtml(cloud)}${hasImage ? "" : " / missing cache"}</span>
      </span>
    </button>
  `;
}

function renderSentinelDownloadRow(download) {
  const previews = [...(download.previews || [])].sort(
    (left, right) => sentinelViewRank(left.view) - sentinelViewRank(right.view),
  );
  const primary = previews.find((preview) => preview.view === "rgb") || previews[0];
  const downloadKey = sentinelDownloadKey(download);
  const expanded = forestState.sentinelExpandedDownloadKeys.has(downloadKey);
  const review = effectiveSentinelReview(download);
  const excluded = Boolean(review.is_excluded);
  const hasDraft = forestState.sentinelReviewDrafts.has(download.download_id);
  const saving = forestState.sentinelReviewSavingIds.has(download.download_id);
  const badgeText = saving ? "Saving" : hasDraft ? "Draft" : excluded ? "Excluded" : "Used";
  const badgeClass = saving ? "saving" : hasDraft ? "draft" : excluded ? "excluded" : "used";
  if (!primary) {
    return `<div class="forestPreviewEmpty">${escapeHtml(download.datetime?.slice(0, 10) || "no date")}</div>`;
  }

  const secondaryPreviews = previews.filter((preview) => preview !== primary);
  return `
    <article class="forestPreviewRow ${excluded ? "excluded" : ""}">
      <div class="forestPreviewReviewBar">
        <label class="forestSentinelReviewCheck">
          <input
            class="forestSentinelExcludeCheck"
            type="checkbox"
            data-download-id="${escapeHtml(download.download_id || "")}"
            ${excluded ? "checked" : ""}
            ${saving ? "disabled" : ""}
          />
          <span>Exclude from use</span>
        </label>
        <span class="forestPreviewBadge ${badgeClass}">${escapeHtml(badgeText)}</span>
      </div>
      ${renderSentinelReviewPanel(download, review, { saving })}
      ${previewButton({
        preview: primary,
        download,
        className: `forestPreviewPrimary ${expanded ? "expanded" : ""}`,
        downloadKey,
      })}
      ${
        expanded && secondaryPreviews.length
          ? `<div class="forestPreviewLayers">
              ${secondaryPreviews
                .map((preview) => previewButton({ preview, download, className: "forestPreviewLayer", downloadKey }))
                .join("")}
            </div>`
          : ""
      }
    </article>
  `;
}

function renderSentinelDownloads(sample) {
  const allDownloads = sample.sentinel?.downloads || [];
  const scopedDownloads = eventScopedSentinelDownloads(sample);
  const downloads = filteredSentinelDownloads(sample);
  if (!downloads.length && scopedDownloads.length) {
    return `<div class="meta">No Sentinel imagery matches this image ID.</div>`;
  }
  if (!downloads.length && allDownloads.length) {
    return `<div class="meta">No Sentinel imagery is linked to this event.</div>`;
  }
  if (!downloads.length) return `<div class="meta">No downloaded Sentinel imagery yet.</div>`;

  const selectedEventId = forestState.selectedSampleEventId || "all";
  const eventGroups = new Map();
  downloads.forEach((download) => {
    let assignments = download.event_links?.length
      ? download.event_links
      : [{ event_id: null, event_year: null, period: download.period || "OTHER" }];
    if (selectedEventId !== "all" && selectedEventId !== "legacy") {
      assignments = assignments.filter((assignment) => assignment.event_id === selectedEventId);
    }
    assignments.forEach((assignment) => {
      const eventKey = assignment.event_id || "legacy";
      if (!eventGroups.has(eventKey)) {
        eventGroups.set(eventKey, {
          label: assignment.event_id
            ? `${assignment.event_year || "Unknown year"}${assignment.is_dominant ? " / dominant" : ""}`
            : "Legacy / unassigned",
          periods: new Map([["PRE", []], ["EVENT", []], ["POST", []]]),
        });
      }
      const eventGroup = eventGroups.get(eventKey);
      const period = assignment.period || download.period || "OTHER";
      if (!eventGroup.periods.has(period)) eventGroup.periods.set(period, []);
      eventGroup.periods.get(period).push(download);
    });
  });

  return `
    <div class="forestPreviewEvents">
      ${[...eventGroups.values()].map((eventGroup) => `
        <section class="forestPreviewEvent">
          <h4>${escapeHtml(eventGroup.label)}</h4>
          <div class="forestPreviewPeriods">
            ${[...eventGroup.periods.entries()]
              .filter(([, periodDownloads]) => periodDownloads.length)
              .map(([period, periodDownloads]) => `
                <section class="forestPreviewPeriod">
                  <h5>${escapeHtml(period)}</h5>
                  <div class="forestPreviewRows">
                    ${periodDownloads
                      .sort((left, right) => String(left.datetime || "").localeCompare(String(right.datetime || "")))
                      .map(renderSentinelDownloadRow)
                      .join("")}
                  </div>
                </section>
              `).join("")}
          </div>
        </section>
      `).join("")}
    </div>
  `;
}

function bboxToBounds(bbox) {
  return [
    [bbox[1], bbox[0]],
    [bbox[3], bbox[2]],
  ];
}

function compareDownloads(sample, period) {
  const selectedEventId = forestState.selectedSampleEventId || "all";
  return (sample?.sentinel?.downloads || [])
    .filter((download) => {
      const links = download.event_links || [];
      let matchesPeriod = download.period === period;
      if (selectedEventId === "legacy") {
        matchesPeriod = !links.length && matchesPeriod;
      } else if (selectedEventId !== "all") {
        matchesPeriod = links.some(
          (link) => link.event_id === selectedEventId && link.period === period,
        );
      } else if (links.length) {
        matchesPeriod = links.some((link) => link.period === period);
      }
      return matchesPeriod && (download.previews || []).some(previewHasImage);
    })
    .sort((left, right) => String(left.datetime || "").localeCompare(String(right.datetime || "")));
}

function previewForView(download, view) {
  return (download?.previews || []).find((preview) => preview.view === view && previewHasImage(preview)) || null;
}

function availableCompareViews(sample) {
  const downloads = [...compareDownloads(sample, "PRE"), ...compareDownloads(sample, "POST")];
  const views = new Set(downloads.flatMap((download) =>
    (download.previews || []).filter(previewHasImage).map((preview) => preview.view),
  ));
  return SENTINEL_VIEW_ORDER.filter((view) => views.has(view));
}

function bestCompareDownload(sample, period, currentId) {
  const downloads = compareDownloads(sample, period);
  if (!downloads.length) return null;
  const current = downloads.find((download) => download.download_id === currentId);
  if (current) return current;
  return downloads.find((download) => !download.review?.is_excluded && previewForView(download, forestState.compareView))
    || downloads.find((download) => !download.review?.is_excluded)
    || downloads.find((download) => previewForView(download, forestState.compareView))
    || downloads[0];
}

function compareDownloadById(sample, downloadId) {
  return (sample?.sentinel?.downloads || []).find((download) => download.download_id === downloadId) || null;
}

function compareOptionLabel(download) {
  const date = download.datetime ? download.datetime.slice(0, 10) : "unknown date";
  const cloud = download.cloud_fraction === null || download.cloud_fraction === undefined
    ? "cloud n/a"
    : `cloud ${Math.round(Number(download.cloud_fraction) * 100)}%`;
  const excluded = download.review?.is_excluded ? " / excluded" : "";
  const scene = download.scene_id ? ` / ${download.scene_id.slice(0, 14)}` : "";
  return `${date} / ${cloud}${excluded}${scene}`;
}

function compareDownloadOptions(downloads, selectedId) {
  if (!downloads.length) return `<option value="">No imagery</option>`;
  return downloads
    .map((download) => `
      <option value="${escapeHtml(download.download_id)}" ${download.download_id === selectedId ? "selected" : ""}>
        ${escapeHtml(compareOptionLabel(download))}
      </option>
    `)
    .join("");
}

function compareViewOptions(sample) {
  const views = availableCompareViews(sample);
  if (!views.length) return `<option value="">No layers</option>`;
  return views
    .map((view) => `
      <option value="${escapeHtml(view)}" ${forestState.compareView === view ? "selected" : ""}>
        ${escapeHtml(sentinelViewLabel(view))}
      </option>
    `)
    .join("");
}

function compareDownloadMeta(download, preview) {
  if (!download) return "No imagery";
  const parts = [
    download.datetime ? download.datetime.slice(0, 10) : "unknown date",
    preview ? sentinelViewLabel(preview.view) : "layer missing",
  ];
  if (download.cloud_fraction !== null && download.cloud_fraction !== undefined) {
    parts.push(`cloud ${Math.round(Number(download.cloud_fraction) * 100)}%`);
  }
  if (download.review?.is_excluded) {
    parts.push(`excluded: ${sentinelReviewReasonSummary(download.review).replace(/^Reasons: /, "")}`);
  }
  return parts.join(" / ");
}

function canCompareSentinel(sample) {
  return compareDownloads(sample, "PRE").length > 0 && compareDownloads(sample, "POST").length > 0;
}

function compareHansenOverlay(sample) {
  return currentHansenOverlay(sample);
}

function ensureCompareSelections(sample) {
  const views = availableCompareViews(sample);
  if (!views.includes(forestState.compareView)) {
    forestState.compareView = views.includes("rgb") ? "rgb" : views[0] || "rgb";
  }

  const pre = bestCompareDownload(sample, "PRE", forestState.comparePreDownloadId);
  const post = bestCompareDownload(sample, "POST", forestState.comparePostDownloadId);
  forestState.comparePreDownloadId = pre?.download_id || null;
  forestState.comparePostDownloadId = post?.download_id || null;
  return { pre, post };
}

function ensureCompareMap(kind) {
  const mapId = kind === "pre" ? "forestComparePreMap" : "forestComparePostMap";
  if (forestState.compareMaps[kind]) return forestState.compareMaps[kind];

  const map = L.map(mapId, {
    zoomControl: true,
    attributionControl: false,
    center: [0, 0],
    zoom: 2,
  });
  map.createPane("compareImagePane");
  map.getPane("compareImagePane").style.zIndex = 300;
  map.createPane("compareHansenPane");
  map.getPane("compareHansenPane").style.zIndex = 420;
  map.on("moveend zoomend", () => syncCompareMaps(kind));
  forestState.compareMaps[kind] = map;
  return map;
}

function syncCompareMaps(sourceKind) {
  if (forestState.compareSyncing) return;
  const source = forestState.compareMaps[sourceKind];
  const target = forestState.compareMaps[sourceKind === "pre" ? "post" : "pre"];
  if (!source || !target) return;
  forestState.compareSyncing = true;
  target.setView(source.getCenter(), source.getZoom(), { animate: false });
  forestState.compareSyncing = false;
}

function removeCompareLayer(name) {
  const kind = name.startsWith("pre") ? "pre" : "post";
  const map = forestState.compareMaps[kind];
  const layer = forestState.compareLayers[name];
  if (map && layer) {
    map.removeLayer(layer);
  }
  forestState.compareLayers[name] = null;
}

function isFilterableHansenOverlay(overlay) {
  return forestState.hansenLayer === "all_loss"
    && ["all_loss", "aoi_all_loss"].includes(overlay?.mask_type)
    && overlay?.png_url
    && (overlay.legend || []).length;
}

async function filteredHansenOverlayUrl(overlay) {
  if (!isFilterableHansenOverlay(overlay)) return overlay?.png_url || null;
  return filteredRasterUrl(overlay.png_url, overlay.legend || [], forestState.hansenLossHiddenYears, "year");
}

async function filteredSclPreviewUrl(preview) {
  if (preview?.view !== "scl" || !preview?.url) return preview?.url || null;
  const legend = await sclLegendForPreview(preview);
  return filteredRasterUrl(preview.url, legend, forestState.sclHiddenClasses, "code");
}

function applyMainRasterFilters() {
  const sentinelLayer = forestState.layers.sentinel;
  const sentinelPreview = forestState.activeSentinelPreview;
  if (sentinelLayer && sentinelPreview?.view === "scl") {
    const layerRef = sentinelLayer;
    filteredSclPreviewUrl(sentinelPreview).then((url) => {
      if (url && forestState.layers.sentinel === layerRef) {
        layerRef.setUrl(url);
      }
    });
  }

  const hansenLayer = forestState.layers.hansen;
  const overlay = currentHansenOverlay();
  if (hansenLayer && isFilterableHansenOverlay(overlay)) {
    const layerRef = hansenLayer;
    filteredHansenOverlayUrl(overlay).then((url) => {
      if (url && forestState.layers.hansen === layerRef) {
        layerRef.setUrl(url);
      }
    });
  }
}

function applyCompareImageFilter(kind, preview) {
  const layerName = `${kind}Image`;
  const layer = forestState.compareLayers[layerName];
  if (!layer || preview?.view !== "scl") return;
  filteredSclPreviewUrl(preview).then((url) => {
    if (url && forestState.compareLayers[layerName] === layer) {
      layer.setUrl(url);
    }
  });
}

function applyCompareHansenFilter(kind, overlay) {
  const layerName = `${kind}Hansen`;
  const layer = forestState.compareLayers[layerName];
  if (!layer || !isFilterableHansenOverlay(overlay)) return;
  filteredHansenOverlayUrl(overlay).then((url) => {
    if (url && forestState.compareLayers[layerName] === layer) {
      layer.setUrl(url);
    }
  });
}

function applyVisibleRasterFilters() {
  applyMainRasterFilters();
  if (!forestState.compareOpen) return;
  const sample = forestState.selectedDetail;
  const pre = compareDownloadById(sample, forestState.comparePreDownloadId);
  const post = compareDownloadById(sample, forestState.comparePostDownloadId);
  const overlay = currentHansenOverlay(sample);
  applyCompareImageFilter("pre", previewForView(pre, forestState.compareView));
  applyCompareImageFilter("post", previewForView(post, forestState.compareView));
  applyCompareHansenFilter("pre", overlay);
  applyCompareHansenFilter("post", overlay);
}

function renderCompareMap(kind, download, preview, hansenOverlay) {
  const map = ensureCompareMap(kind);
  removeCompareLayer(`${kind}Image`);
  removeCompareLayer(`${kind}Hansen`);

  if (preview?.url && preview?.bbox) {
    forestState.compareLayers[`${kind}Image`] = L.imageOverlay(
      preview.url,
      bboxToBounds(preview.bbox),
      { pane: "compareImagePane", opacity: 1, interactive: false },
    ).addTo(map);
  }

  if (forestState.compareShowHansen && hansenOverlay?.png_url && hansenOverlay?.bbox) {
    forestState.compareLayers[`${kind}Hansen`] = L.imageOverlay(
      hansenOverlay.png_url,
      bboxToBounds(hansenOverlay.bbox),
      {
        pane: "compareHansenPane",
        opacity: forestState.compareHansenOpacity,
        interactive: false,
        className: `forestHansenOverlay ${forestState.hansenPixelated ? "pixelated" : ""}`,
      },
    ).addTo(map);
  }

  applyCompareImageFilter(kind, preview);
  applyCompareHansenFilter(kind, hansenOverlay);

  if (preview?.bbox) {
    map.fitBounds(bboxToBounds(preview.bbox), { padding: [20, 20], animate: false });
  } else if (hansenOverlay?.bbox) {
    map.fitBounds(bboxToBounds(hansenOverlay.bbox), { padding: [20, 20], animate: false });
  }
  window.setTimeout(() => map.invalidateSize(), 50);
  window.setTimeout(() => map.invalidateSize(), 240);
}

function renderCompareView() {
  const compare = $("forestCompareView");
  const sample = forestState.selectedDetail;
  if (!compare || !sample || !forestState.compareOpen) return;

  const { pre, post } = ensureCompareSelections(sample);
  const prePreview = previewForView(pre, forestState.compareView);
  const postPreview = previewForView(post, forestState.compareView);
  const hansenOverlay = compareHansenOverlay(sample);
  forestState.compareSclPreviews = forestState.compareView === "scl"
    ? [prePreview, postPreview].filter(Boolean)
    : [];

  $("forestCompareLayerSelect").innerHTML = compareViewOptions(sample);
  $("forestComparePreSelect").innerHTML = compareDownloadOptions(compareDownloads(sample, "PRE"), pre?.download_id);
  $("forestComparePostSelect").innerHTML = compareDownloadOptions(compareDownloads(sample, "POST"), post?.download_id);
  $("forestCompareHansenLayerSelect").value = forestState.hansenLayer;
  $("forestCompareHansenToggle").checked = forestState.compareShowHansen;
  $("forestComparePixelatedToggle").checked = forestState.hansenPixelated;
  $("forestCompareHansenOpacityInput").value = String(Math.round(forestState.compareHansenOpacity * 100));
  $("forestComparePreMeta").textContent = compareDownloadMeta(pre, prePreview);
  $("forestComparePostMeta").textContent = compareDownloadMeta(post, postPreview);

  const selectedName = sampleDisplayName(sample);
  const missingLayer = !prePreview || !postPreview;
  $("forestCompareStatus").textContent = missingLayer
    ? `${selectedName}: selected layer is missing for one side`
    : `${selectedName}: ${sentinelViewLabel(forestState.compareView)} PRE/POST`;

  bindCompareControls();
  renderCompareLayerPanelState();
  renderCompareMap("pre", pre, prePreview, hansenOverlay);
  renderCompareMap("post", post, postPreview, hansenOverlay);
}

function openCompareView() {
  const sample = forestState.selectedDetail;
  if (!sample || !canCompareSentinel(sample)) {
    setForestStatus("Need at least one PRE and one POST Sentinel image");
    return;
  }
  closeDbSyncView({ silent: true });
  forestState.dataOperationsOpen = false;
  $("forestDataOperationsView").hidden = true;
  forestState.compareOpen = true;
  ensureCompareSelections(sample);
  $("forestCompareView").hidden = false;
  renderCompareView();
  updateWorkspaceFocus();
  setForestStatus("PRE/POST compare opened");
}

function closeCompareView(options = {}) {
  forestState.compareOpen = false;
  forestState.compareSclPreviews = [];
  const compare = $("forestCompareView");
  if (compare) compare.hidden = true;
  updateWorkspaceFocus();
  if (!options.silent) setForestStatus("PRE/POST compare closed");
}

function bindCompareControls() {
  $("forestCompareCloseBtn").onclick = () => closeCompareView();
  $("forestCompareLayerPanelOpen").onclick = () => setCompareLayerPanelCollapsed(false);
  $("forestCompareLayerPanelClose").onclick = () => setCompareLayerPanelCollapsed(true);
  $("forestCompareLayerSelect").onchange = (event) => {
    forestState.compareView = event.target.value || "rgb";
    renderCompareView();
  };
  $("forestCompareHansenLayerSelect").onchange = (event) => {
    forestState.hansenLayer = event.target.value || "dominant_year";
    $("forestHansenLayerSelect").value = forestState.hansenLayer;
    renderMap();
    renderCompareView();
  };
  $("forestComparePreSelect").onchange = (event) => {
    forestState.comparePreDownloadId = event.target.value || null;
    renderCompareView();
  };
  $("forestComparePostSelect").onchange = (event) => {
    forestState.comparePostDownloadId = event.target.value || null;
    renderCompareView();
  };
  $("forestCompareHansenToggle").onchange = (event) => {
    forestState.compareShowHansen = event.target.checked;
    renderCompareView();
  };
  $("forestComparePixelatedToggle").onchange = (event) => {
    forestState.hansenPixelated = event.target.checked;
    $("forestHansenPixelatedToggle").checked = forestState.hansenPixelated;
    renderMap();
    renderCompareView();
  };
  $("forestCompareHansenOpacityInput").oninput = (event) => {
    forestState.compareHansenOpacity = Number(event.target.value) / 100;
    ["preHansen", "postHansen"].forEach((name) => {
      forestState.compareLayers[name]?.setOpacity(forestState.compareHansenOpacity);
    });
  };
}

function dbSyncStatus(message) {
  const host = $("forestDbSyncStatus");
  if (host) host.textContent = message;
  forestState.dbSyncStatus = message;
}

function operationStatusClass(operation) {
  if (operation.status === "conflict") return "conflict";
  if (operation.status === "report-only") return "report-only";
  if (operation.safe) return "safe";
  return "";
}

function renderDbSyncCandidates() {
  const select = $("forestDbSyncCandidateSelect");
  if (!select) return;
  select.innerHTML = "";
  if (!forestState.dbSyncCandidates.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No .sqlite files in bases/";
    select.appendChild(option);
    forestState.dbSyncSelectedPath = "";
    return;
  }

  forestState.dbSyncCandidates.forEach((candidate) => {
    const option = document.createElement("option");
    option.value = candidate.path;
    option.textContent = `${candidate.name} (${Math.round((candidate.size_bytes || 0) / 1024)} KiB)`;
    select.appendChild(option);
  });
  if (!forestState.dbSyncSelectedPath) {
    forestState.dbSyncSelectedPath = forestState.dbSyncCandidates[0].path;
  }
  select.value = forestState.dbSyncSelectedPath;
}

function renderDbSyncSummary(plan = forestState.dbSyncPlan) {
  const host = $("forestDbSyncSummary");
  if (!host) return;
  if (!plan) {
    host.innerHTML = `<div class="emptyState">Run compare to see DB differences.</div>`;
    return;
  }

  const rows = Object.entries(plan.summary || {})
    .filter(([entity]) => entity !== "cache_diagnostics");
  if (!rows.length) {
    host.innerHTML = `<div class="emptyState">No DB merge entities found.</div>`;
    return;
  }

  host.innerHTML = rows
    .map(([entity, metrics]) => {
      const metricRows = Object.entries(metrics)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, value]) => `
          <div class="forestDbSyncMetric">
            <span>${escapeHtml(key)}</span>
            <span>${escapeHtml(value)}</span>
          </div>
        `)
        .join("");
      return `
        <article class="forestDbSyncSummaryCard">
          <strong>${escapeHtml(entity.replaceAll("_", " "))}</strong>
          <div class="forestDbSyncMetricGrid">${metricRows}</div>
        </article>
      `;
    })
    .join("");
}

function dbSyncSentinelCache(plan = forestState.dbSyncPlan) {
  return plan?.cache_repair?.sentinel || { items: [] };
}

function dbSyncSentinelItems(plan = forestState.dbSyncPlan) {
  return dbSyncSentinelCache(plan).items || [];
}

function resetDbSyncSentinelSelection(plan = forestState.dbSyncPlan) {
  const ids = dbSyncSentinelItems(plan).map((item) => item.download_id);
  forestState.dbSyncSelectedSentinelDownloadIds = new Set(ids);
  forestState.dbSyncSentinelCacheExpanded = ids.length > 0;
}

function pruneDbSyncSentinelSelection(items) {
  const validIds = new Set(items.map((item) => item.download_id));
  forestState.dbSyncSelectedSentinelDownloadIds = new Set(
    [...forestState.dbSyncSelectedSentinelDownloadIds].filter((id) => validIds.has(id)),
  );
}

function renderDbSyncSentinelJob(job = forestState.dbSyncSentinelJob) {
  if (!job) return "";
  const total = Number(job.total || 0);
  const done = Number(job.done || 0);
  const percent = total ? Math.round((done / total) * 100) : 0;
  const running = !isTerminalJobStatus(job.status);
  return `
    <div class="forestDbSyncCacheJob ${running ? "running" : ""}">
      <div>
        <strong>${escapeHtml(job.status)}</strong>
        <span>${escapeHtml(job.message || "")}</span>
      </div>
      <div class="forestDbSyncProgress" aria-label="Sentinel cache progress">
        <span style="width:${Math.max(0, Math.min(100, percent))}%"></span>
      </div>
      <span>${escapeHtml(done)} / ${escapeHtml(total)}</span>
      ${running ? `<button type="button" data-db-sync-cancel-sentinel-cache>Cancel</button>` : ""}
    </div>
  `;
}

function renderDbSyncCacheRepair(plan = forestState.dbSyncPlan) {
  const host = $("forestDbSyncCacheRepair");
  if (!host) return;
  if (!plan) {
    host.innerHTML = `<div class="emptyState">Run compare to inspect cache repair actions.</div>`;
    return;
  }

  const sentinel = dbSyncSentinelCache(plan);
  const items = dbSyncSentinelItems(plan);
  pruneDbSyncSentinelSelection(items);
  const selectedIds = forestState.dbSyncSelectedSentinelDownloadIds;
  const running = forestState.dbSyncSentinelJob && !isTerminalJobStatus(forestState.dbSyncSentinelJob.status);
  const selectedCount = selectedIds.size;
  const allSelected = items.length > 0 && items.every((item) => selectedIds.has(item.download_id));
  const visibleItems = items.slice(0, 250);
  const hiddenCount = Math.max(0, items.length - visibleItems.length);

  if (!items.length) {
    host.innerHTML = `
      <article class="forestDbSyncCacheCard healthy">
        <div class="forestDbSyncCacheCardHeader">
          <strong>Sentinel cache</strong>
          <span>No missing Sentinel downloads.</span>
        </div>
        ${renderDbSyncSentinelJob()}
      </article>
    `;
    return;
  }

  host.innerHTML = `
    <article class="forestDbSyncCacheCard">
      <button class="forestDbSyncCacheCardHeader" type="button" data-db-sync-toggle-sentinel-cache>
        <strong>${escapeHtml(sentinel.missing_download_count || 0)} Sentinel downloads missing</strong>
        <span>${escapeHtml(sentinel.missing_preview_row_count || 0)} preview rows / ${escapeHtml(items.length)} affected downloads</span>
      </button>
      ${renderDbSyncSentinelJob()}
      ${forestState.dbSyncSentinelCacheExpanded ? `
        <div class="forestDbSyncCacheControls">
          <label>
            <input id="forestDbSyncSentinelSelectAll" type="checkbox" ${allSelected ? "checked" : ""} />
            Select all
          </label>
          <span>${escapeHtml(selectedCount)} selected</span>
          <button id="forestDbSyncDownloadSentinelCacheBtn" class="primary" type="button" ${selectedCount && !running ? "" : "disabled"}>Download Sentinel</button>
        </div>
        <div class="forestDbSyncCacheRows">
          ${visibleItems.map((item) => `
            <label class="forestDbSyncCacheRow">
              <input type="checkbox" data-db-sync-sentinel-download-id="${escapeHtml(item.download_id)}" ${selectedIds.has(item.download_id) ? "checked" : ""} ${running ? "disabled" : ""} />
              <span>
                <strong>${escapeHtml(item.period || "n/a")} / ${escapeHtml((item.datetime || "").slice(0, 10) || "no date")}</strong>
                <span>${escapeHtml(item.sample_name || item.sample_short_id || item.sample_id)} / ${escapeHtml(item.scene_id)}</span>
              </span>
              <span>${item.missing_geotiff ? "GeoTIFF" : "previews"}${item.missing_preview_count ? ` + ${escapeHtml(item.missing_preview_count)} previews` : ""}</span>
            </label>
          `).join("")}
          ${hiddenCount ? `<div class="emptyState">${hiddenCount} more affected downloads hidden.</div>` : ""}
        </div>
      ` : ""}
    </article>
  `;

  host.querySelector("[data-db-sync-toggle-sentinel-cache]")?.addEventListener("click", () => {
    forestState.dbSyncSentinelCacheExpanded = !forestState.dbSyncSentinelCacheExpanded;
    renderDbSyncCacheRepair();
  });
  host.querySelector("#forestDbSyncSentinelSelectAll")?.addEventListener("change", (event) => {
    forestState.dbSyncSelectedSentinelDownloadIds = event.target.checked
      ? new Set(items.map((item) => item.download_id))
      : new Set();
    renderDbSyncCacheRepair();
  });
  host.querySelectorAll("[data-db-sync-sentinel-download-id]").forEach((checkbox) => {
    checkbox.addEventListener("change", (event) => {
      const downloadId = event.target.dataset.dbSyncSentinelDownloadId;
      if (!downloadId) return;
      if (event.target.checked) {
        forestState.dbSyncSelectedSentinelDownloadIds.add(downloadId);
      } else {
        forestState.dbSyncSelectedSentinelDownloadIds.delete(downloadId);
      }
      renderDbSyncCacheRepair();
    });
  });
  host.querySelector("#forestDbSyncDownloadSentinelCacheBtn")?.addEventListener("click", () => run(runDbSyncSentinelCacheDownload));
  host.querySelector("[data-db-sync-cancel-sentinel-cache]")?.addEventListener("click", () => run(cancelDbSyncSentinelCacheJob));
}

function renderDbSyncWarnings(plan = forestState.dbSyncPlan) {
  const host = $("forestDbSyncWarnings");
  if (!host) return;
  const warnings = plan?.warnings || [];
  host.innerHTML = warnings
    .map((warning) => `<div class="forestDbSyncWarning">${escapeHtml(warning)}</div>`)
    .join("");
}

function renderDbSyncOperations(plan = forestState.dbSyncPlan) {
  const host = $("forestDbSyncOperations");
  const totalsHost = $("forestDbSyncTotals");
  const applyButton = $("forestDbSyncApplyBtn");
  if (!host || !totalsHost || !applyButton) return;
  if (!plan) {
    host.innerHTML = `<div class="emptyState">No operations yet.</div>`;
    totalsHost.textContent = "no compare yet";
    applyButton.disabled = true;
    return;
  }

  const totals = plan.totals || {};
  const diagnosticTotals = plan.diagnostic_totals || {};
  totalsHost.textContent =
    `${totals.safe || 0} safe / ${totals.conflicts || 0} conflicts / ${totals.report_only || 0} report-only / ${diagnosticTotals.operations || 0} cache diagnostics`;
  applyButton.disabled = forestState.dbSyncRunning || !Number(totals.safe || 0);

  const operations = plan.operations || [];
  if (!operations.length) {
    host.innerHTML = `<div class="emptyState">No DB merge differences found.</div>`;
    return;
  }

  const visible = operations.slice(0, 350);
  const hiddenCount = Math.max(0, operations.length - visible.length);
  host.innerHTML = visible
    .map((operation) => {
      const statusClass = operationStatusClass(operation);
      const classes = [
        "forestDbSyncOperation",
        statusClass,
        operation.missing_cache ? "missing-cache" : "",
      ]
        .filter(Boolean)
        .join(" ");
      return `
        <article class="${classes}">
          <div>
            <strong>${escapeHtml(operation.entity)}</strong>
            <span>${escapeHtml(operation.action)}</span>
          </div>
          <div>
            <strong>${escapeHtml(operation.key)}</strong>
            <span>${escapeHtml(operation.message)}</span>
            ${operation.field ? `<span>field: ${escapeHtml(operation.field)}</span>` : ""}
            ${operation.missing_cache ? `<span>missing cache</span>` : ""}
          </div>
          <span class="forestDbSyncBadge ${statusClass}">${escapeHtml(operation.status)}</span>
        </article>
      `;
    })
    .join("")
    + (hiddenCount ? `<div class="emptyState">${hiddenCount} more operations hidden in UI; report will contain all.</div>` : "");
}

function renderDbSyncApplyResult(result = forestState.dbSyncApplyResult) {
  const host = $("forestDbSyncApplyResult");
  if (!host) return;
  if (!result) {
    host.innerHTML = "";
    return;
  }
  const reports = result.reports || {};
  host.innerHTML = `
    <strong>Applied ${escapeHtml(result.applied_count || 0)} operations; skipped ${escapeHtml(result.skipped_count || 0)}.</strong>
    <span>Backup: ${escapeHtml(result.backup_path || "n/a")}</span>
    <span>Summary: ${escapeHtml(reports.summary || "n/a")}</span>
    <span>Operations: ${escapeHtml(reports.operations || "n/a")}</span>
    <span>Conflicts: ${escapeHtml(reports.conflicts || "n/a")}</span>
    <span>Diagnostics: ${escapeHtml(reports.diagnostics || "n/a")}</span>
  `;
}

function renderDbSyncView() {
  renderDbSyncCandidates();
  renderDbSyncSummary();
  renderDbSyncCacheRepair();
  renderDbSyncWarnings();
  renderDbSyncOperations();
  renderDbSyncApplyResult();
  dbSyncStatus(forestState.dbSyncStatus || "Select incoming SQLite database.");
}

async function loadDbSyncCandidates() {
  dbSyncStatus("loading DB candidates");
  const payload = await listDbSyncCandidates();
  forestState.dbSyncCandidates = payload.items || [];
  if (
    forestState.dbSyncSelectedPath
    && !forestState.dbSyncCandidates.some((candidate) => candidate.path === forestState.dbSyncSelectedPath)
  ) {
    forestState.dbSyncSelectedPath = "";
  }
  renderDbSyncCandidates();
  dbSyncStatus(`found ${forestState.dbSyncCandidates.length} candidate DB(s)`);
}

async function openDbSyncView() {
  closeCompareView({ silent: true });
  forestState.dataOperationsOpen = false;
  $("forestDataOperationsView").hidden = true;
  forestState.dbSyncOpen = true;
  $("forestDbSyncView").hidden = false;
  updateWorkspaceFocus();
  renderDbSyncView();
  if (!forestState.dbSyncCandidates.length) {
    await loadDbSyncCandidates();
  }
  window.pipelineExplorer?.refreshMapLayout();
}

function closeDbSyncView(options = {}) {
  forestState.dbSyncOpen = false;
  const view = $("forestDbSyncView");
  if (view) view.hidden = true;
  updateWorkspaceFocus();
  if (!options.silent) setForestStatus("Database Sync closed");
}

async function runDbSyncCompare() {
  const path = $("forestDbSyncCandidateSelect").value;
  if (!path) {
    dbSyncStatus("select incoming DB first");
    return;
  }
  forestState.dbSyncSelectedPath = path;
  forestState.dbSyncRunning = true;
  forestState.dbSyncApplyResult = null;
  renderDbSyncOperations();
  dbSyncStatus("comparing databases");
  try {
    forestState.dbSyncPlan = await compareDbSync(path);
    resetDbSyncSentinelSelection(forestState.dbSyncPlan);
    dbSyncStatus(`compare ready: ${forestState.dbSyncPlan.totals.safe || 0} safe operation(s)`);
  } finally {
    forestState.dbSyncRunning = false;
    renderDbSyncView();
  }
}

async function runDbSyncApply() {
  const path = forestState.dbSyncSelectedPath || $("forestDbSyncCandidateSelect").value;
  if (!path) {
    dbSyncStatus("select incoming DB first");
    return;
  }
  forestState.dbSyncRunning = true;
  renderDbSyncOperations();
  dbSyncStatus("applying safe DB sync operations");
  try {
    forestState.dbSyncApplyResult = await applyDbSync(path);
    forestState.dbSyncPlan = await compareDbSync(path);
    dbSyncStatus(`apply ready: ${forestState.dbSyncApplyResult.applied_count || 0} operation(s)`);
    if (forestState.selectedSampleId) {
      forestState.selectedDetail = await getForestSample(forestState.selectedSampleId);
      ensureActiveSentinelPreviewForSample(forestState.selectedDetail);
      renderDetail();
      renderMap();
    }
    await refreshSamples();
  } finally {
    forestState.dbSyncRunning = false;
    renderDbSyncView();
  }
}

async function runDbSyncSentinelCacheDownload() {
  const items = dbSyncSentinelItems();
  const validIds = new Set(items.map((item) => item.download_id));
  const selectedIds = [...forestState.dbSyncSelectedSentinelDownloadIds].filter((id) => validIds.has(id));
  if (!selectedIds.length) {
    dbSyncStatus("select Sentinel downloads first");
    return;
  }
  dbSyncStatus(`queueing ${selectedIds.length} Sentinel cache download(s)`);
  forestState.dbSyncSentinelJob = await startSentinelCacheDownload(selectedIds, { max_cloud: 30 });
  forestState.dbSyncSentinelJobId = forestState.dbSyncSentinelJob.job_id;
  window.clearTimeout(forestState.dbSyncSentinelTimer);
  forestState.dbSyncSentinelTimer = window.setTimeout(() => run(pollDbSyncSentinelCacheJob), 700);
  renderDbSyncCacheRepair();
}

async function pollDbSyncSentinelCacheJob() {
  if (!forestState.dbSyncSentinelJobId) return;
  const job = await getForestJob(forestState.dbSyncSentinelJobId);
  forestState.dbSyncSentinelJob = job;
  dbSyncStatus(job.message || `Sentinel cache ${job.status}`);
  renderDbSyncCacheRepair();
  if (isTerminalJobStatus(job.status)) {
    window.clearTimeout(forestState.dbSyncSentinelTimer);
    forestState.dbSyncSentinelTimer = null;
    forestState.dbSyncSentinelJobId = null;
    if (forestState.dbSyncSelectedPath) {
      forestState.dbSyncPlan = await compareDbSync(forestState.dbSyncSelectedPath);
      resetDbSyncSentinelSelection(forestState.dbSyncPlan);
      renderDbSyncView();
    }
    return;
  }
  forestState.dbSyncSentinelTimer = window.setTimeout(() => run(pollDbSyncSentinelCacheJob), 1500);
}

async function cancelDbSyncSentinelCacheJob() {
  if (!forestState.dbSyncSentinelJobId) return;
  forestState.dbSyncSentinelJob = await cancelForestJob(forestState.dbSyncSentinelJobId);
  renderDbSyncCacheRepair();
}

function downloadedPreviews(downloads) {
  return (downloads || []).flatMap((download) =>
    (download.previews || []).map((preview) => ({
      ...preview,
      download_id: download.download_id,
      download_key: sentinelDownloadKey(download),
      period: download.period,
      datetime: download.datetime,
      scene_id: download.scene_id,
      is_bad_cloud: download.is_bad_cloud,
      is_excluded: Boolean(download.review?.is_excluded),
      review: download.review || { is_excluded: false },
    })),
  );
}

function allDownloadedPreviews(sample) {
  return downloadedPreviews(sample?.sentinel?.downloads || []);
}

function findPreviewByUrl(previewUrl) {
  return allDownloadedPreviews(forestState.selectedDetail).find((preview) => preview.url === previewUrl) || null;
}

function chooseDefaultSentinelPreview(sample) {
  const previews = downloadedPreviews(eventScopedSentinelDownloads(sample));
  const usable = previews.filter((preview) => previewHasImage(preview) && !preview.is_excluded);
  return usable.find((preview) => preview.view === "rgb") || usable[0] || null;
}

function ensureActiveSentinelPreviewForSample(sample) {
  const previews = downloadedPreviews(eventScopedSentinelDownloads(sample));
  if (!previews.length) {
    forestState.activeSentinelPreview = null;
    return;
  }

  const activeUrl = forestState.activeSentinelPreview?.url;
  const activePreview = previews.find((preview) => preview.url === activeUrl);
  if (activePreview && !activePreview.is_excluded) return;
  forestState.activeSentinelPreview = chooseDefaultSentinelPreview(sample);
}

function defaultSelectedSceneKeys(result) {
  const preferred = new Set(["PRE", "EVENT", "POST"]);
  const keys = [];
  (result.groups || []).forEach((group) => {
    if (!preferred.has(group.period)) return;
    const item = (group.items || [])[0];
    if (item) keys.push(sceneKey(group.period, item));
  });

  if (keys.length) return keys;
  return searchedScenes({ sentinelSearch: result })
    .slice(0, 2)
    .map((scene) => scene.key);
}

function renderBatchStatus(job = null) {
  const host = $("forestBatchStatus");
  const cancelButton = $("forestCancelBatchBtn");
  if (!host || !cancelButton) return;

  if (!job) {
    host.textContent = "no batch job";
    cancelButton.disabled = true;
    return;
  }

  const total = Number(job.total || 0);
  const done = Number(job.done || 0);
  host.textContent = `${job.status}: ${done}/${total}${job.message ? ` / ${job.message}` : ""}`;
  cancelButton.disabled = !["queued", "running", "cancel_requested"].includes(job.status);
}

function isTerminalJobStatus(status) {
  return ["completed", "completed_with_errors", "failed", "cancelled"].includes(status);
}

function isHansenJobRunning(sample = forestState.selectedDetail) {
  return forestState.hansenJob
    && sample
    && forestState.hansenJobSampleId === sample.sample_id
    && !isTerminalJobStatus(forestState.hansenJob.status);
}

function hansenStageLabel(stage) {
  return {
    queued: "Queued",
    loading: "Loading",
    cache: "Caching Hansen tiles",
    analysis: "Analyzing rasters",
    done: "Done",
    failed: "Failed",
    cancelled: "Cancelled",
  }[stage] || "Working";
}

function renderHansenJobStatus(sample) {
  const job = forestState.hansenJob;
  if (!sample || forestState.hansenJobSampleId !== sample.sample_id) return "";
  if (!job) return "";

  const running = !isTerminalJobStatus(job.status);
  const canCancel = running && job.status !== "cancel_requested";
  const total = Number(job.total || 0);
  const done = Number(job.done || 0);
  const percent = total ? Math.round((done / total) * 100) : 0;
  return `
    <div class="forestInlineJob ${running ? "running" : ""}">
      <span class="forestSpinner" aria-hidden="true"></span>
      <div>
        <strong>${escapeHtml(hansenStageLabel(job.stage))}</strong>
        <span>${escapeHtml(job.message || job.status)}${total ? ` / ${percent}%` : ""}</span>
      </div>
      <button id="forestCancelHansenBtn" type="button" ${canCancel ? "" : "disabled"}>Cancel</button>
    </div>
  `;
}

function renderSamplesList() {
  const host = $("forestSamplesList");
  host.innerHTML = "";

  if (!forestState.samples.length) {
    const empty = document.createElement("div");
    empty.className = "emptyState";
    empty.textContent = "No samples loaded.";
    host.appendChild(empty);
    return;
  }

  const selectedNotOnPage =
    forestState.selectedDetail &&
    !forestState.samples.some((sample) => sample.sample_id === forestState.selectedDetail.sample_id);

  if (selectedNotOnPage) {
    host.appendChild(sampleCard(forestState.selectedDetail, { pinned: true }));
  }

  forestState.samples.forEach((sample) => {
    host.appendChild(sampleCard(sample));
  });
  scrollActiveSampleIntoView();
}

function renderPointInfo(sample) {
  return `
    <details class="forestPointInfo" open>
      <summary>
        <span class="forestPointInfoTitle">
          <strong>${escapeHtml(sampleDisplayName(sample))}</strong>
          <span>Point info</span>
        </span>
      </summary>
      <div class="forestPointInfoBody">
        <div class="forestNamePanel">
          <label>
            Sample name
            <input id="forestSampleNameInput" type="text" value="${escapeHtml(sample.display_name || "")}" placeholder="${escapeHtml(sample.short_id)}" maxlength="255" />
          </label>
          <button id="forestSaveNameBtn" type="button">Save name</button>
        </div>
        <label class="forestSampleFlagCheck">
          <input id="forestMultipleEventsInput" type="checkbox" ${sample.has_multiple_events ? "checked" : ""} disabled />
          <span>Multiple loss events (derived from Hansen)</span>
        </label>
        <div class="forestDetailGrid">
          <div><span>Driver</span><strong>${escapeHtml(sample.driver_primary || "unknown")}</strong></div>
          <div><span>Confidence</span><strong>${escapeHtml(sample.confidence_primary || "n/a")}</strong></div>
          <div><span>Lat</span><strong>${Number(sample.lat).toFixed(6)}</strong></div>
          <div><span>Lon</span><strong>${Number(sample.lon).toFixed(6)}</strong></div>
        </div>
      </div>
    </details>
  `;
}

function sampleEventLabel(event) {
  const area = event.loss_area_ha === null || event.loss_area_ha === undefined
    ? "n/a"
    : `${Number(event.loss_area_ha).toFixed(2)} ha`;
  const share = event.loss_share === null || event.loss_share === undefined
    ? "share n/a"
    : `${Math.round(Number(event.loss_share) * 100)}%`;
  return `${event.event_year}${event.is_dominant ? " / dominant" : ""} / ${event.source || "unknown"} / ${area} / ${share}`;
}

function sampleEventArea(event) {
  return event.loss_area_ha === null || event.loss_area_ha === undefined
    ? "n/a"
    : `${Number(event.loss_area_ha).toFixed(2)} ha`;
}

function sampleEventShare(event) {
  return event.loss_share === null || event.loss_share === undefined
    ? "n/a"
    : `${Math.round(Number(event.loss_share) * 100)}%`;
}

function unassignedSentinelDownloads(sample) {
  return (sample.sentinel?.downloads || []).filter((download) => !(download.event_links || []));
}

function normalizeSelectedSampleEvent(sample) {
  const selectedEventId = forestState.selectedSampleEventId || "all";
  if (selectedEventId === "all") return;
  if (selectedEventId === "legacy" && unassignedSentinelDownloads(sample).length) return;
  if ((sample.events || []).some((event) => event.event_id === selectedEventId)) return;
  forestState.selectedSampleEventId = "all";
}

function selectedSentinelSearchEvent(sample) {
  const events = sample.events || [];
  return events.find((event) => event.event_id === forestState.selectedSampleEventId)
    || events.find((event) => event.is_dominant)
    || events[0]
    || null;
}

function renderEventFilterRow({ value, title, detail, trailing, event = null }) {
  const selected = (forestState.selectedSampleEventId || "all") === value;
  const counts = event?.download_counts || {};
  return `
    <label class="forestEventRow ${selected ? "active" : ""}">
      <input class="forestEventFilterInput" type="radio" name="forestEventFilter" value="${escapeHtml(value)}" ${selected ? "checked" : ""} />
      <span class="forestEventRowBody">
        <span class="forestEventIdentity">
          <strong>${escapeHtml(title)}</strong>
          <span>${escapeHtml(detail)}</span>
        </span>
        ${event ? `
          <span class="forestEventMetric"><span>Loss area</span><strong>${escapeHtml(sampleEventArea(event))}</strong></span>
          <span class="forestEventMetric"><span>Loss share</span><strong>${escapeHtml(sampleEventShare(event))}</strong></span>
          <span class="forestEventPeriods">
            <span>PRE ${Number(counts.PRE || 0)}</span>
            <span>EVENT ${Number(counts.EVENT || 0)}</span>
            <span>POST ${Number(counts.POST || 0)}</span>
          </span>
        ` : `<span class="forestEventTrailing">${escapeHtml(trailing)}</span>`}
      </span>
    </label>
  `;
}

function renderSampleEvents(sample) {
  const events = sample.events || [];
  const downloads = sample.sentinel?.downloads || [];
  const unassigned = unassignedSentinelDownloads(sample);
  return `
    <article class="forestSampleEvents">
      <div class="forestPanelHeader">
        <h3>Events</h3>
        <span class="meta">${events.length ? `${events.length}${sample.has_multiple_events ? " / multiple" : ""}` : "No derived Hansen events"}</span>
      </div>
      <div class="forestEventRows">
        ${renderEventFilterRow({
          value: "all",
          title: "All events",
          detail: "Show every linked and unassigned image",
          trailing: `${downloads.length} Sentinel downloads`,
        })}
        ${events.map((event) => {
          const detail = `${event.source || "unknown"}${event.is_dominant ? " / dominant by area" : ""}`;
          return renderEventFilterRow({
            value: event.event_id,
            title: String(event.event_year),
            detail,
            event,
          });
        }).join("")}
        ${unassigned.length ? renderEventFilterRow({
          value: "legacy",
          title: "Legacy / unassigned",
          detail: "Images without an unambiguous event link",
          trailing: `${unassigned.length} Sentinel downloads`,
        }) : ""}
      </div>
    </article>
  `;
}

function sentinelEventOptions(sample) {
  const events = sample.events || [];
  if (!events.length) return `<option value="">Legacy / year only</option>`;
  const selectedEvent = selectedSentinelSearchEvent(sample);
  return events.map((event) => `
    <option value="${escapeHtml(event.event_id)}" ${event.event_id === selectedEvent?.event_id ? "selected" : ""}>
      ${escapeHtml(sampleEventLabel(event))}
    </option>
  `).join("");
}

function sentinelDownloadMeta(sample) {
  const total = sample.sentinel?.downloads?.length || 0;
  const scoped = eventScopedSentinelDownloads(sample).length;
  const visible = filteredSentinelDownloads(sample).length;
  if ((forestState.selectedSampleEventId || "all") === "all") {
    return `${visible}/${total} Sentinel downloads`;
  }
  return `${visible}/${scoped} in selection / ${total} total`;
}

function renderImageIdSearch(sample) {
  return `
    <article class="forestImageFilterPanel">
      <label>
        Image ID
        <input id="forestImageIdInput" type="text" value="${escapeHtml(forestState.sentinelImageIdQuery)}" placeholder="download / scene id" />
      </label>
      <span id="forestImageIdMeta" class="meta">${escapeHtml(sentinelDownloadMeta(sample))}</span>
    </article>
  `;
}

function updateImageIdMeta() {
  const host = $("forestImageIdMeta");
  const sample = forestState.selectedDetail;
  if (!host || !sample) return;
  host.textContent = sentinelDownloadMeta(sample);
}

function selectSampleEventFilter(sample, eventId, options = {}) {
  forestState.selectedSampleEventId = eventId || "all";
  normalizeSelectedSampleEvent(sample);
  forestState.comparePreDownloadId = null;
  forestState.comparePostDownloadId = null;

  document.querySelectorAll(".forestEventFilterInput").forEach((input) => {
    input.checked = input.value === forestState.selectedSampleEventId;
    input.closest(".forestEventRow")?.classList.toggle("active", input.checked);
  });

  const selectedEvent = (sample.events || []).find(
    (event) => event.event_id === forestState.selectedSampleEventId,
  );
  if (options.syncSearch !== false) {
    const searchSelect = $("forestSentinelEventSelect");
    if (searchSelect && selectedEvent) searchSelect.value = selectedEvent.event_id;
    if (selectedEvent && $("forestSentinelYearInput")) {
      $("forestSentinelYearInput").value = selectedEvent.event_year;
    }
  }

  ensureActiveSentinelPreviewForSample(sample);
  renderSentinelDownloadsPanel();
  const compareButton = $("forestCompareSentinelBtn");
  if (compareButton) compareButton.disabled = !canCompareSentinel(sample);
  renderMap();
  const label = selectedEvent ? `event ${selectedEvent.event_year}` : forestState.selectedSampleEventId;
  setForestStatus(`showing Sentinel imagery for ${label}`);
}

function renderDetail() {
  const host = $("forestDetails");
  const sample = forestState.selectedDetail;
  if (!sample) {
    host.innerHTML = `<div class="emptyState">Select a sample.</div>`;
    return;
  }
  normalizeSelectedSampleEvent(sample);

  const raw = JSON.stringify(sample.raw_properties || {}, null, 2);
  const hansenBusy = Boolean(isHansenJobRunning(sample));
  host.innerHTML = `
    ${renderPointInfo(sample)}
    ${renderSampleEvents(sample)}
    ${renderImageIdSearch(sample)}
    <article class="forestActionPanel">
      <div class="forestPanelHeader">
        <h3>Hansen</h3>
        <button id="forestAnalyzeHansenBtn" class="primary" type="button" ${hansenBusy ? "disabled" : ""}>Analyze Hansen</button>
      </div>
      <label class="toggle">
        <input id="forestHansenTreecoverInput" type="checkbox" ${forestState.includeHansenTreecover ? "checked" : ""} ${hansenBusy ? "disabled" : ""} />
        <span>treecover 2000</span>
      </label>
      ${renderHansenJobStatus(sample)}
      ${renderHansenSummary(sample)}
    </article>
    <article class="forestActionPanel">
      <div class="forestPanelHeader">
        <h3>Sentinel-2</h3>
        <div class="forestPanelActions">
          <button id="forestCompareSentinelBtn" type="button" ${canCompareSentinel(sample) ? "" : "disabled"}>Compare PRE/POST</button>
          <button id="forestSearchSentinelBtn" class="primary" type="button">Search Sentinel-2</button>
        </div>
      </div>
      <div class="forestSentinelControls">
        <label>
          Event
          <select id="forestSentinelEventSelect">
            ${sentinelEventOptions(sample)}
          </select>
        </label>
        <label>
          Event year
          <input id="forestSentinelYearInput" type="number" min="2001" max="2026" value="${escapeHtml(selectedSentinelSearchEvent(sample)?.event_year || sample.hansen?.event_year || sample.source_event_year || "")}" />
        </label>
        <label>
          Season
          <select id="forestSeasonPreset">
            <option value="full_snow_free">full snow-free</option>
            <option value="spring">spring</option>
            <option value="early_summer">early summer</option>
            <option value="late_summer">late summer</option>
            <option value="autumn">autumn</option>
          </select>
        </label>
        <label>
          Max cloud
          <input id="forestMaxCloudInput" type="number" min="0" max="100" value="30" />
        </label>
      </div>
      <div id="forestSentinelResults">${renderSentinelResults(sample)}</div>
      <div id="forestSentinelDownloads">${renderSentinelDownloads(sample)}</div>
    </article>
    <article class="forestValidation">
      <label>
        Manual validation
        <select id="forestManualValidation">
          <option value="Valid">Valid</option>
          <option value="Unclear">Unclear</option>
          <option value="Wrong">Wrong</option>
          <option value="Too old">Too old</option>
        </select>
      </label>
      <label>
        Notes
        <textarea id="forestManualNotes" rows="3"></textarea>
      </label>
      <button id="forestSaveManualBtn" class="primary" type="button">Save validation</button>
    </article>
    <details class="rawProperties">
      <summary>Raw properties</summary>
      <pre>${escapeHtml(raw)}</pre>
    </details>
  `;

  const suggestedValidation = sample.hansen?.event_year && sample.hansen.event_year < 2016
    ? "Too old"
    : "Unclear";
  $("forestManualValidation").value = sample.manual_validation || suggestedValidation;
  $("forestManualNotes").value = sample.manual_notes || "";
  $("forestSaveNameBtn").addEventListener("click", () => run(saveSampleName));
  $("forestImageIdInput").addEventListener("input", (event) => {
    forestState.sentinelImageIdQuery = event.target.value;
    renderSentinelDownloadsPanel();
  });
  document.querySelectorAll(".forestEventFilterInput").forEach((input) => {
    input.addEventListener("change", () => {
      if (input.checked) selectSampleEventFilter(sample, input.value);
    });
  });
  $("forestHansenTreecoverInput").addEventListener("change", (event) => {
    forestState.includeHansenTreecover = event.target.checked;
  });
  const sentinelEventSelect = $("forestSentinelEventSelect");
  if (sentinelEventSelect) {
    sentinelEventSelect.addEventListener("change", () => {
      const selectedEvent = (sample.events || []).find(
        (event) => event.event_id === sentinelEventSelect.value,
      );
      if (selectedEvent) {
        $("forestSentinelYearInput").value = selectedEvent.event_year;
        selectSampleEventFilter(sample, selectedEvent.event_id, { syncSearch: false });
      }
    });
  }
  $("forestAnalyzeHansenBtn").addEventListener("click", () => run(runHansenAnalysis));
  const cancelHansenButton = $("forestCancelHansenBtn");
  if (cancelHansenButton) {
    cancelHansenButton.addEventListener("click", () => run(cancelHansenAnalysis));
  }
  $("forestSearchSentinelBtn").addEventListener("click", () => run(runSentinelSearch));
  $("forestCompareSentinelBtn").addEventListener("click", () => openCompareView());
  const downloadButton = $("forestDownloadSentinelBtn");
  if (downloadButton) {
    downloadButton.addEventListener("click", () => run(downloadCheckedSentinelScenes));
  }
  const selectAllButton = $("forestSelectAllScenesBtn");
  if (selectAllButton) {
    selectAllButton.addEventListener("click", () => {
      searchedScenes(sample).forEach((scene) => forestState.sentinelSelectedSceneKeys.add(scene.key));
      renderDetail();
      setForestStatus("all Sentinel scenes checked");
    });
  }
  const clearScenesButton = $("forestClearScenesBtn");
  if (clearScenesButton) {
    clearScenesButton.addEventListener("click", () => {
      forestState.sentinelSelectedSceneKeys.clear();
      renderDetail();
      setForestStatus("Sentinel scene checks cleared");
    });
  }
  document.querySelectorAll(".forestSceneCheck").forEach((checkbox) => {
    checkbox.addEventListener("change", (event) => {
      if (event.target.checked) {
        forestState.sentinelSelectedSceneKeys.add(event.target.dataset.sceneKey);
      } else {
        forestState.sentinelSelectedSceneKeys.delete(event.target.dataset.sceneKey);
      }
    });
  });
  bindSentinelDownloadControls();
  $("forestSaveManualBtn").addEventListener("click", () => run(saveValidation));
}

function bindSentinelDownloadControls() {
  document.querySelectorAll(".forestPreviewCard img").forEach((image) => {
    image.addEventListener("error", () => {
      const card = image.closest(".forestPreviewCard");
      if (card) {
        card.classList.add("missingCache");
        card.disabled = true;
      }
      const missing = document.createElement("span");
      missing.className = "forestPreviewMissingThumb";
      missing.textContent = "missing";
      image.replaceWith(missing);
      if (forestState.activeSentinelPreview?.url === image.getAttribute("src")) {
        forestState.activeSentinelPreview = null;
        renderMap();
      }
    }, { once: true });
  });
  document.querySelectorAll(".forestPreviewCard").forEach((button) => {
    button.addEventListener("click", () => {
      const preview = findPreviewByUrl(button.dataset.previewUrl);
      if (!preview) return;
      forestState.activeSentinelPreview = preview;
      if (button.classList.contains("forestPreviewPrimary")) {
        const downloadKey = button.dataset.downloadKey;
        if (downloadKey && forestState.sentinelExpandedDownloadKeys.has(downloadKey)) {
          forestState.sentinelExpandedDownloadKeys.delete(downloadKey);
        } else if (downloadKey) {
          forestState.sentinelExpandedDownloadKeys.add(downloadKey);
        }
      }
      renderDetail();
      renderMap();
      setForestStatus(`showing Sentinel ${sentinelViewLabel(preview.view)}`);
    });
  });
  document.querySelectorAll(".forestSentinelExcludeCheck").forEach((checkbox) => {
    checkbox.addEventListener("click", (event) => event.stopPropagation());
    checkbox.addEventListener("change", (event) => run(() => toggleSentinelReview(event)));
  });
  document.querySelectorAll(".forestSentinelReasonCheck").forEach((checkbox) => {
    checkbox.addEventListener("click", (event) => event.stopPropagation());
    checkbox.addEventListener("change", (event) => updateSentinelReviewReason(event));
  });
  document.querySelectorAll(".forestSentinelOtherReasonInput").forEach((input) => {
    input.addEventListener("click", (event) => event.stopPropagation());
    input.addEventListener("input", (event) => updateSentinelReviewOtherText(event));
  });
  document.querySelectorAll(".forestSentinelReviewSaveBtn").forEach((button) => {
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      run(() => saveSentinelReviewDraft(button.dataset.downloadId));
    });
  });
  document.querySelectorAll(".forestSentinelReviewCancelBtn").forEach((button) => {
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      cancelSentinelReviewDraft(button.dataset.downloadId);
    });
  });
}

function renderSentinelDownloadsPanel() {
  const host = $("forestSentinelDownloads");
  if (!host || !forestState.selectedDetail) return;
  host.innerHTML = renderSentinelDownloads(forestState.selectedDetail);
  bindSentinelDownloadControls();
  updateImageIdMeta();
}

function renderMap(fit = false) {
  renderForestMap({
    samples: forestState.samples,
    nearest: forestState.nearest,
    detail: forestState.selectedDetail,
    visibility: forestState.visibility,
    onSelect: (sampleId) => run(() => selectSample(sampleId)),
    fit,
  });
  renderRasterLegends();
  applyVisibleRasterFilters();
}

function clearSelectedSample() {
  closeCompareView({ silent: true });
  forestState.selectedSampleId = null;
  forestState.selectedDetail = null;
  forestState.activeSentinelPreview = null;
  forestState.sentinelImageIdQuery = "";
  forestState.selectedSampleEventId = "all";
  forestState.nearest = [];
}

async function refreshSamples(options = {}) {
  const previousSelectedId = forestState.selectedSampleId;
  setForestStatus("loading samples");
  const payload = await listForestSamples({
    limit: forestState.limit,
    offset: forestState.offset,
    filters: currentFilters(),
    selectedId: previousSelectedId,
    nearestN: nearestN(),
  });

  const selectedFilteredOut = Boolean(previousSelectedId && !payload.selected);
  if (selectedFilteredOut) {
    clearSelectedSample();
  }

  forestState.samples = payload.items;
  forestState.nearest = selectedFilteredOut ? [] : payload.nearest || [];
  forestState.total = payload.total;
  renderSamplesList();
  renderPageMeta();
  renderMap(Boolean(options.fit));

  if (selectedFilteredOut) {
    renderDetail();
    setForestStatus(`selected sample is outside current filters; samples loaded: ${payload.items.length} of ${payload.total}`);
    return;
  }

  if (!previousSelectedId && forestState.samples.length) {
    await selectSample(forestState.samples[0].sample_id, { fit: true });
    return;
  }

  setForestStatus(`samples loaded: ${payload.items.length} of ${payload.total}`);
}

async function selectSample(sampleId, options = {}) {
  logUiDebug("selectSample start", { sample_id: sampleId });
  if (forestState.selectedSampleId !== sampleId) {
    closeCompareView({ silent: true });
    forestState.sentinelImageIdQuery = "";
    forestState.selectedSampleEventId = "all";
    forestState.comparePreDownloadId = null;
    forestState.comparePostDownloadId = null;
  }
  forestState.selectedSampleId = sampleId;
  forestState.selectedDetail = await getForestSample(sampleId);
  ensureActiveSentinelPreviewForSample(forestState.selectedDetail);
  await refreshSamples({ fit: Boolean(options.fit) });
  renderDetail();
  logUiDebug("selectSample done", {
    sample_id: sampleId,
    active_card: Boolean($("forestSamplesList").querySelector(".forestSampleCard.active")),
    pinned: Boolean($("forestSamplesList").querySelector(".forestSampleCard.pinned")),
    active_preview_view: forestState.activeSentinelPreview?.view || null,
    active_preview_url: forestState.activeSentinelPreview?.url || null,
    active_preview_cache: forestState.activeSentinelPreview?.cache_exists ?? null,
    active_preview_bbox: Boolean(forestState.activeSentinelPreview?.bbox),
  });
  setForestStatus("sample selected");
}

async function refreshForestData() {
  await refreshSamples();
  if (!forestState.selectedSampleId) {
    renderDetail();
    return;
  }
  forestState.selectedDetail = await getForestSample(forestState.selectedSampleId);
  normalizeSelectedSampleEvent(forestState.selectedDetail);
  ensureActiveSentinelPreviewForSample(forestState.selectedDetail);
  renderDetail();
  renderMap();
}

async function runHansenAnalysis() {
  if (!forestState.selectedSampleId) return;
  const includeTreecover = Boolean($("forestHansenTreecoverInput")?.checked);
  forestState.includeHansenTreecover = includeTreecover;
  setForestStatus("starting Hansen analysis");
  const job = await analyzeHansen(forestState.selectedSampleId, {
    include_treecover: includeTreecover,
  });
  forestState.hansenJobId = job.job_id;
  forestState.hansenJobSampleId = forestState.selectedSampleId;
  forestState.hansenJob = job;
  renderDetail();
  window.clearTimeout(forestState.hansenTimer);
  forestState.hansenTimer = window.setTimeout(() => run(pollHansenAnalysis), 300);
  setForestStatus(job.message || "Hansen analysis queued");
}

async function pollHansenAnalysis() {
  if (!forestState.hansenJobId) return;

  const job = await getForestJob(forestState.hansenJobId);
  forestState.hansenJob = job;
  renderDetail();
  setForestStatus(job.message || `Hansen ${job.status}`);

  if (isTerminalJobStatus(job.status)) {
    window.clearTimeout(forestState.hansenTimer);
    forestState.hansenTimer = null;
    forestState.hansenJobId = null;
    const selectedJobSample = forestState.selectedSampleId === forestState.hansenJobSampleId;
    if (forestState.selectedSampleId && selectedJobSample) {
      forestState.selectedDetail = await getForestSample(forestState.selectedSampleId);
      ensureActiveSentinelPreviewForSample(forestState.selectedDetail);
      renderDetail();
      renderMap();
    }
    const hansen = selectedJobSample ? forestState.selectedDetail?.hansen : null;
    setForestStatus(
      hansen
        ? `Hansen ready: ${statusLabel(hansen.status)}, event ${hansen.event_year || "n/a"}`
        : `Hansen ${job.status}: ${job.message || "no result"}`,
    );
    return;
  }

  forestState.hansenTimer = window.setTimeout(() => run(pollHansenAnalysis), 900);
}

async function cancelHansenAnalysis() {
  if (!forestState.hansenJobId) return;
  const job = await cancelForestJob(forestState.hansenJobId);
  forestState.hansenJob = job;
  renderDetail();
  setForestStatus("Hansen cancellation requested");
}

async function runSentinelSearch() {
  if (!forestState.selectedSampleId) return;
  const yearValue = $("forestSentinelYearInput").value;
  const eventYear = yearValue ? Number(yearValue) : null;
  setForestStatus("searching Sentinel-2 for selected sample");
  const result = await searchSentinel(forestState.selectedSampleId, {
    event_id: $("forestSentinelEventSelect")?.value || null,
    event_year: eventYear,
    season_preset: $("forestSeasonPreset").value,
    max_cloud: Number($("forestMaxCloudInput").value),
    top_n: 10,
  });
  forestState.selectedDetail.sentinelSearch = result;
  forestState.sentinelSelectedSceneKeys = new Set(defaultSelectedSceneKeys(result));
  renderDetail();
  const count = result.groups.reduce((sum, group) => sum + (group.items?.length || 0), 0);
  setForestStatus(`Sentinel candidates: ${count}`);
}

async function downloadCheckedSentinelScenes() {
  const sample = forestState.selectedDetail;
  if (!sample) return;

  const scenes = searchedScenes(sample).filter((scene) => forestState.sentinelSelectedSceneKeys.has(scene.key));
  if (!scenes.length) {
    setForestStatus("check at least one Sentinel scene");
    return;
  }

  const previousSearch = sample.sentinelSearch || null;
  const maxCloud = Number($("forestMaxCloudInput")?.value || 30);
  for (const [index, scene] of scenes.entries()) {
    const date = scene.item.datetime ? scene.item.datetime.slice(0, 10) : "unknown";
    setForestStatus(`downloading Sentinel ${index + 1}/${scenes.length}: ${scene.period} ${date}`);
    await downloadSentinel(sample.sample_id, {
      item: scene.item,
      period: scene.period,
      event_id: previousSearch?.event_id || null,
      max_cloud: maxCloud,
    });
  }

  forestState.selectedDetail = await getForestSample(sample.sample_id);
  if (previousSearch) {
    forestState.selectedDetail.sentinelSearch = previousSearch;
  }
  ensureActiveSentinelPreviewForSample(forestState.selectedDetail);
  renderDetail();
  renderMap();
  setForestStatus(`Sentinel imagery downloaded: ${scenes.length}`);
}

function findSentinelDownload(downloadId) {
  return (forestState.selectedDetail?.sentinel?.downloads || []).find(
    (download) => download.download_id === downloadId,
  ) || null;
}

function sentinelReviewDraftFromDownload(download, patch = {}) {
  const current = forestState.sentinelReviewDrafts.get(download.download_id) || baseSentinelReview(download);
  const merged = {
    is_excluded: true,
    reason_codes: sentinelReviewReasonCodes(current),
    reason_code: current.reason_code || "",
    reason_text: current.reason_text || "",
    notes: current.notes || null,
    ...patch,
  };
  const reasonCodes = uniqueSentinelReasonCodes(
    merged.reason_codes?.length ? merged.reason_codes : (merged.reason_code ? [merged.reason_code] : []),
  );
  return {
    ...merged,
    reason_codes: reasonCodes,
    reason_code: reasonCodes[0] || "",
  };
}

function setSentinelReviewDraft(downloadId, patch = {}) {
  const download = findSentinelDownload(downloadId);
  if (!download) return null;
  const draft = sentinelReviewDraftFromDownload(download, patch);
  forestState.sentinelReviewDrafts.set(downloadId, draft);
  return draft;
}

function updateSentinelReviewReason(event) {
  const checkbox = event.target;
  const downloadId = checkbox.dataset.downloadId;
  const reasonCode = checkbox.dataset.reasonCode;
  const download = findSentinelDownload(downloadId);
  if (!download || !reasonCode) return;

  const current = effectiveSentinelReview(download);
  const reasonCodes = sentinelReviewReasonCodes(current);
  const nextReasonCodes = checkbox.checked
    ? uniqueSentinelReasonCodes([...reasonCodes, reasonCode])
    : reasonCodes.filter((code) => code !== reasonCode);
  const draft = setSentinelReviewDraft(downloadId, {
    reason_codes: nextReasonCodes,
    reason_code: nextReasonCodes[0] || "",
    reason_text: nextReasonCodes.includes("OTHER") ? current.reason_text || "" : "",
  });
  renderSentinelDownloadsPanel();
  setForestStatus(draft?.reason_codes?.length ? "Sentinel review reasons selected" : "choose Sentinel review reasons");
}

function updateSentinelReviewOtherText(event) {
  const download = findSentinelDownload(event.target.dataset.downloadId);
  if (!download) return;
  const reasonCodes = uniqueSentinelReasonCodes([
    ...sentinelReviewReasonCodes(effectiveSentinelReview(download)),
    "OTHER",
  ]);
  setSentinelReviewDraft(event.target.dataset.downloadId, {
    reason_codes: reasonCodes,
    reason_code: reasonCodes[0] || "OTHER",
    reason_text: event.target.value,
  });
}

function cancelSentinelReviewDraft(downloadId) {
  if (!downloadId) return;
  forestState.sentinelReviewDrafts.delete(downloadId);
  renderSentinelDownloadsPanel();
  setForestStatus("Sentinel review draft cancelled");
}

async function refreshSelectedSampleAfterSentinelReview(previousSearch) {
  forestState.selectedDetail = await getForestSample(forestState.selectedSampleId);
  if (previousSearch) {
    forestState.selectedDetail.sentinelSearch = previousSearch;
  }
  ensureActiveSentinelPreviewForSample(forestState.selectedDetail);
  renderDetail();
  renderMap();
}

async function restoreSentinelReview(downloadId) {
  const previousSearch = forestState.selectedDetail?.sentinelSearch || null;
  forestState.sentinelReviewDrafts.delete(downloadId);
  forestState.sentinelReviewSavingIds.add(downloadId);
  renderSentinelDownloadsPanel();
  setForestStatus("restoring Sentinel scene");

  let refreshed = false;
  try {
    await saveSentinelReview(downloadId, {
      is_excluded: false,
      reason_codes: [],
      reason_code: null,
      reason_text: null,
      notes: null,
    });
    forestState.sentinelReviewSavingIds.delete(downloadId);
    await refreshSelectedSampleAfterSentinelReview(previousSearch);
    refreshed = true;
    setForestStatus("Sentinel scene restored");
  } finally {
    if (!refreshed) {
      forestState.sentinelReviewSavingIds.delete(downloadId);
      renderSentinelDownloadsPanel();
    }
  }
}

async function toggleSentinelReview(event) {
  const checkbox = event.target;
  const downloadId = checkbox.dataset.downloadId;
  if (!downloadId || !forestState.selectedSampleId) return;

  const download = findSentinelDownload(downloadId);
  if (!download) return;

  if (checkbox.checked) {
    setSentinelReviewDraft(downloadId, { is_excluded: true });
    renderSentinelDownloadsPanel();
    setForestStatus("choose Sentinel exclusion reason");
    return;
  }

  if (download.review?.is_excluded) {
    await restoreSentinelReview(downloadId);
    return;
  }

  forestState.sentinelReviewDrafts.delete(downloadId);
  renderSentinelDownloadsPanel();
  setForestStatus("Sentinel exclusion cancelled");
}

async function saveSentinelReviewDraft(downloadId) {
  if (!downloadId || !forestState.selectedSampleId) return;

  const download = findSentinelDownload(downloadId);
  if (!download) return;
  const review = effectiveSentinelReview(download);
  const reasonCodes = sentinelReviewReasonCodes(review);
  const reasonText = reasonCodes.includes("OTHER") ? String(review.reason_text || "").trim() : null;

  if (!reasonCodes.length) {
    setForestStatus("choose Sentinel exclusion reasons");
    return;
  }
  if (reasonCodes.includes("OTHER") && !reasonText) {
    setForestStatus("fill Other reason before saving");
    return;
  }

  const previousSearch = forestState.selectedDetail?.sentinelSearch || null;
  forestState.sentinelReviewSavingIds.add(downloadId);
  renderSentinelDownloadsPanel();
  setForestStatus("saving Sentinel review");

  let refreshed = false;
  try {
    await saveSentinelReview(downloadId, {
      is_excluded: true,
      reason_codes: reasonCodes,
      reason_code: reasonCodes[0] || null,
      reason_text: reasonText,
      notes: null,
    });
    forestState.sentinelReviewDrafts.delete(downloadId);
    forestState.sentinelReviewSavingIds.delete(downloadId);
    await refreshSelectedSampleAfterSentinelReview(previousSearch);
    refreshed = true;
    setForestStatus("Sentinel scene excluded");
  } finally {
    if (!refreshed) {
      forestState.sentinelReviewSavingIds.delete(downloadId);
      renderSentinelDownloadsPanel();
    }
  }
}

async function importSelectedFiles() {
  const files = $("forestFileInput").files;
  if (!files?.length) {
    setForestStatus("select CSV or GeoJSON first");
    return;
  }
  setForestStatus(`importing ${files.length} file(s)`);
  const result = await importForestFiles(files);
  $("forestImportStatus").textContent =
    `imported ${result.imported_count}, duplicates ${result.duplicate_count}, rejected ${result.rejected_count}`;
  forestState.offset = 0;
  forestState.selectedSampleId = null;
  forestState.selectedDetail = null;
  await refreshSamples({ fit: true });
}

async function saveValidation() {
  if (!forestState.selectedSampleId) return;
  const validation = $("forestManualValidation").value;
  const notes = $("forestManualNotes").value;
  forestState.selectedDetail = await saveManualValidation(
    forestState.selectedSampleId,
    validation,
    notes,
  );
  renderDetail();
  await refreshSamples();
  setForestStatus(`validation saved: ${validation}`);
}

async function saveSampleName() {
  if (!forestState.selectedSampleId) return;
  const displayName = $("forestSampleNameInput").value;
  forestState.selectedDetail = await saveSampleDisplayName(forestState.selectedSampleId, displayName);
  renderDetail();
  await refreshSamples();
  setForestStatus(`sample name saved: ${sampleDisplayName(forestState.selectedDetail)}`);
}

async function createExport() {
  setForestStatus("exporting filtered samples");
  const result = await exportForestSamples(currentFilters());
  $("forestExportLinks").innerHTML = `
    <a href="${result.csv_url}">CSV ${escapeHtml(result.sample_count)}</a>
    <a href="${result.geojson_url}">GeoJSON ${escapeHtml(result.sample_count)}</a>
  `;
  setForestStatus(`export ready: ${result.sample_count} samples`);
}

async function pollBatchHansen() {
  if (!forestState.hansenBatchJobId) return;
  const job = await getForestJob(forestState.hansenBatchJobId);
  renderBatchStatus(job);

  if (isTerminalJobStatus(job.status)) {
    window.clearTimeout(forestState.hansenBatchTimer);
    forestState.hansenBatchTimer = null;
    if (forestState.selectedSampleId) {
      forestState.selectedDetail = await getForestSample(forestState.selectedSampleId);
      ensureActiveSentinelPreviewForSample(forestState.selectedDetail);
      renderDetail();
      renderMap();
    }
    setForestStatus(`Hansen batch ${job.status}`);
    return;
  }

  forestState.hansenBatchTimer = window.setTimeout(() => run(pollBatchHansen), 1500);
}

async function startBatchHansen() {
  const limit = Number($("forestBatchLimitInput").value) || 10;
  const includeTreecover = Boolean($("forestBatchTreecoverInput")?.checked);
  forestState.includeBatchHansenTreecover = includeTreecover;
  setForestStatus("starting Hansen batch");
  const job = await startHansenBatch(currentFilters(), limit, {
    include_treecover: includeTreecover,
  });
  forestState.hansenBatchJobId = job.job_id;
  renderBatchStatus(job);
  window.clearTimeout(forestState.hansenBatchTimer);
  forestState.hansenBatchTimer = window.setTimeout(() => run(pollBatchHansen), 300);
}

async function cancelBatchHansen() {
  if (!forestState.hansenBatchJobId) return;
  const job = await cancelForestJob(forestState.hansenBatchJobId);
  renderBatchStatus(job);
  setForestStatus("Hansen batch cancellation requested");
}

function bindEvents() {
  $("forestModeBtn").addEventListener("click", () => run(() => setMode("forest")));
  $("pipelineModeBtn").addEventListener("click", () => run(() => setMode("pipeline")));
  $("forestImportBtn").addEventListener("click", () => run(importSelectedFiles));
  $("forestFileInput").addEventListener("change", (event) => {
    const count = event.target.files?.length || 0;
    $("forestImportStatus").textContent = count ? `${count} file(s) selected` : "no files selected";
  });
  $("forestReloadBtn").addEventListener("click", () => {
    forestState.offset = 0;
    run(() => refreshSamples({ fit: true }));
  });
  $("forestPrevPageBtn").addEventListener("click", () => {
    forestState.offset = Math.max(0, forestState.offset - forestState.limit);
    run(() => refreshSamples());
  });
  $("forestNextPageBtn").addEventListener("click", () => {
    forestState.offset += forestState.limit;
    run(() => refreshSamples());
  });
  $("forestExportBtn").addEventListener("click", () => run(createExport));
  $("forestDbSyncOpenBtn").addEventListener("click", () => run(openDbSyncView));
  $("forestDbSyncCloseBtn").addEventListener("click", closeDbSyncView);
  $("forestDbSyncRefreshBtn").addEventListener("click", () => run(loadDbSyncCandidates));
  $("forestDbSyncCompareBtn").addEventListener("click", () => run(runDbSyncCompare));
  $("forestDbSyncApplyBtn").addEventListener("click", () => run(runDbSyncApply));
  $("forestDbSyncCandidateSelect").addEventListener("change", (event) => {
    forestState.dbSyncSelectedPath = event.target.value;
    forestState.dbSyncPlan = null;
    forestState.dbSyncApplyResult = null;
    forestState.dbSyncSelectedSentinelDownloadIds = new Set();
    forestState.dbSyncSentinelCacheExpanded = false;
    forestState.dbSyncSentinelJob = null;
    forestState.dbSyncSentinelJobId = null;
    window.clearTimeout(forestState.dbSyncSentinelTimer);
    forestState.dbSyncSentinelTimer = null;
    renderDbSyncView();
  });
  $("forestBatchHansenBtn").addEventListener("click", () => run(startBatchHansen));
  $("forestBatchTreecoverInput").addEventListener("change", (event) => {
    forestState.includeBatchHansenTreecover = event.target.checked;
  });
  $("forestCancelBatchBtn").addEventListener("click", () => run(cancelBatchHansen));
  $("forestClearMapBtn").addEventListener("click", clearForestLayers);
  $("sidebarToggle").addEventListener("click", toggleSidebar);
  $("mapLayerPanelOpen").addEventListener("click", () => setLayerPanelCollapsed(false));
  $("mapLayerPanelClose").addEventListener("click", () => setLayerPanelCollapsed(true));
  $("forestCopyDebugBtn").addEventListener("click", () => run(copyDebugEvents));
  $("forestClearDebugBtn").addEventListener("click", () => {
    forestState.debugEvents = [];
    renderDebugEvents();
  });
  window.addEventListener("forest-map-debug", (event) => pushDebugEvent(event.detail));
  [
    ["forestShowSamplesToggle", "samples"],
    ["forestShowNearestToggle", "nearest"],
    ["forestShowPlotToggle", "plot"],
    ["forestShowAoiToggle", "aoi"],
    ["forestShowHansenToggle", "hansen"],
    ["forestShowSentinelToggle", "sentinel"],
  ].forEach(([id, key]) => {
    $(id).addEventListener("change", (event) => {
      forestState.visibility[key] = event.target.checked;
      renderMap();
    });
  });
  $("forestHansenLayerSelect").addEventListener("change", (event) => {
    forestState.hansenLayer = event.target.value;
    renderMap();
    if (forestState.compareOpen) renderCompareView();
  });
  $("forestHansenOpacityInput").addEventListener("input", (event) => {
    forestState.hansenOpacity = Number(event.target.value) / 100;
    if (forestState.layers.hansen?.setOpacity) {
      forestState.layers.hansen.setOpacity(forestState.hansenOpacity);
    } else {
      renderMap();
    }
  });
  $("forestHansenPixelatedToggle").addEventListener("change", (event) => {
    forestState.hansenPixelated = event.target.checked;
    renderMap();
    if (forestState.compareOpen) renderCompareView();
  });
  $("forestHansenLayerSelect").value = forestState.hansenLayer;
  $("forestHansenOpacityInput").value = String(Math.round(forestState.hansenOpacity * 100));
  $("forestHansenPixelatedToggle").checked = forestState.hansenPixelated;
  renderLayerPanelState();
  initDataOperations({
    currentFilters,
    closeCompareView,
    closeDbSyncView,
    updateWorkspaceFocus,
    setForestStatus,
    refreshForestData,
    run,
  });
}

async function run(task) {
  try {
    await task();
  } catch (error) {
    setForestStatus(error.message || String(error));
    console.error(error);
  }
}

async function initForestUi() {
  await window.pipelineReady;
  enhanceSidebarPanels();
  bindEvents();
  await setMode("forest");
}

initForestUi();
