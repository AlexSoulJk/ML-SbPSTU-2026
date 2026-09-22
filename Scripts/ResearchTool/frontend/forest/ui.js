import { clearForestLayers, renderForestMap } from "./map.js?v=forest-iter2-20";
import {
  analyzeHansen,
  cancelForestJob,
  downloadSentinel,
  exportForestSamples,
  getForestSample,
  getForestJob,
  importForestFiles,
  listForestSamples,
  saveManualValidation,
  saveSampleDisplayName,
  saveSentinelReview,
  searchSentinel,
  startHansenBatch,
} from "./samples.js?v=forest-iter2-20";
import { forestState } from "./state.js?v=forest-iter2-20";

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
    host.innerHTML = `<div class="meta">Legend is shown for all loss.</div>`;
    return;
  }
  if (!legend.length) {
    host.innerHTML = `<div class="meta">No Hansen loss years in current view.</div>`;
    return;
  }

  host.innerHTML = `
    <div class="forestHansenLegendTitle">Loss year</div>
    <div class="forestHansenLegendItems">
      ${legend
        .map((item) => `
          <span class="forestHansenLegendItem">
            <i style="background:${escapeHtml(item.color)}"></i>
            <span>${escapeHtml(item.year)}</span>
          </span>
        `)
        .join("")}
    </div>
  `;
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
  renderHansenLegend();
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
  renderHansenLegend("forestCompareHansenLegend", forestState.selectedDetail);
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
  { value: "TOO_DARK_OR_LOW_CONTRAST", label: "Too dark / low contrast" },
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

function baseSentinelReview(download) {
  return download.review || {
    is_excluded: false,
    reason_code: null,
    reason_text: null,
    notes: null,
  };
}

function effectiveSentinelReview(download) {
  return forestState.sentinelReviewDrafts.get(download.download_id) || baseSentinelReview(download);
}

function sentinelReviewReasonOptions(selectedReason) {
  return [
    `<option value="">Choose reason</option>`,
    ...SENTINEL_REVIEW_REASONS.map(
      (reason) => `
        <option value="${escapeHtml(reason.value)}" ${reason.value === selectedReason ? "selected" : ""}>
          ${escapeHtml(reason.label)}
        </option>
      `,
    ),
  ].join("");
}

function renderSentinelReviewPanel(download, review, options = {}) {
  if (!review.is_excluded) return "";

  const downloadId = download.download_id || "";
  const reasonCode = review.reason_code || "";
  const reasonText = review.reason_text || "";
  const saving = Boolean(options.saving);
  const hasDraft = forestState.sentinelReviewDrafts.has(downloadId);
  const isOther = reasonCode === "OTHER";
  const reasonSummary = reasonCode
    ? `Reason: ${sentinelReviewReasonLabel(reasonCode)}${isOther && reasonText ? ` / ${reasonText}` : ""}`
    : "Choose reason before saving.";

  return `
    <div class="forestSentinelReviewPanel">
      <label>
        Reason
        <select class="forestSentinelReasonSelect" data-download-id="${escapeHtml(downloadId)}" ${saving ? "disabled" : ""}>
          ${sentinelReviewReasonOptions(reasonCode)}
        </select>
      </label>
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

function filteredSentinelDownloads(sample) {
  const downloads = sample.sentinel?.downloads || [];
  const query = forestState.sentinelImageIdQuery.trim();
  if (!query) return downloads;
  return downloads.filter((download) => sentinelImageIdMatches(download, query));
}

function previewButton({ preview, download, className = "", downloadKey = "" }) {
  const active = forestState.activeSentinelPreview?.url === preview.url;
  const date = download.datetime ? download.datetime.slice(0, 10) : "";
  const cloud = download.cloud_fraction === null || download.cloud_fraction === undefined
    ? "cloud n/a"
    : `cloud ${Math.round(Number(download.cloud_fraction) * 100)}%`;
  return `
    <button class="forestPreviewCard ${className} ${active ? "active" : ""}" type="button" data-preview-url="${escapeHtml(preview.url || "")}" data-download-key="${escapeHtml(downloadKey)}">
      <img src="${escapeHtml(preview.url || "")}" alt="${escapeHtml(sentinelViewLabel(preview.view))}" />
      <span class="forestPreviewText">
        <strong>${escapeHtml(sentinelViewLabel(preview.view))}</strong>
        <span>${escapeHtml(date)} / ${escapeHtml(cloud)}</span>
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

  const secondaryPreviews = previews.filter((preview) => preview.url !== primary.url);
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
  const downloads = filteredSentinelDownloads(sample);
  if (!downloads.length && allDownloads.length) {
    return `<div class="meta">No Sentinel imagery matches this image ID.</div>`;
  }
  if (!downloads.length) return `<div class="meta">No downloaded Sentinel imagery yet.</div>`;

  const orderedPeriods = ["PRE", "EVENT", "POST"];
  const periodGroups = new Map(orderedPeriods.map((period) => [period, []]));
  downloads.forEach((download) => {
    const period = download.period || "OTHER";
    if (!periodGroups.has(period)) periodGroups.set(period, []);
    periodGroups.get(period).push(download);
  });

  return `
    <div class="forestPreviewPeriods">
      ${[...periodGroups.entries()]
        .map(([period, periodDownloads]) => `
          <section class="forestPreviewPeriod">
            <h4>${escapeHtml(period)}</h4>
            <div class="forestPreviewRows">
              ${
                periodDownloads.length
                  ? periodDownloads
                    .sort((left, right) => String(left.datetime || "").localeCompare(String(right.datetime || "")))
                    .map(renderSentinelDownloadRow)
                    .join("")
                  : `<div class="forestPreviewEmpty">No imagery</div>`
              }
            </div>
          </section>
        `)
        .join("")}
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
  return (sample?.sentinel?.downloads || [])
    .filter((download) => download.period === period && (download.previews || []).length)
    .sort((left, right) => String(left.datetime || "").localeCompare(String(right.datetime || "")));
}

function previewForView(download, view) {
  return (download?.previews || []).find((preview) => preview.view === view) || null;
}

function availableCompareViews(sample) {
  const downloads = [...compareDownloads(sample, "PRE"), ...compareDownloads(sample, "POST")];
  const views = new Set(downloads.flatMap((download) => (download.previews || []).map((preview) => preview.view)));
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
    const reason = download.review.reason_code
      ? sentinelReviewReasonLabel(download.review.reason_code)
      : "no reason";
    parts.push(`excluded: ${reason}`);
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
  forestState.compareOpen = true;
  ensureCompareSelections(sample);
  $("forestCompareView").hidden = false;
  renderCompareView();
  setForestStatus("PRE/POST compare opened");
}

function closeCompareView(options = {}) {
  forestState.compareOpen = false;
  const compare = $("forestCompareView");
  if (compare) compare.hidden = true;
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

function allDownloadedPreviews(sample) {
  return (sample?.sentinel?.downloads || []).flatMap((download) =>
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

function findPreviewByUrl(previewUrl) {
  return allDownloadedPreviews(forestState.selectedDetail).find((preview) => preview.url === previewUrl) || null;
}

function chooseDefaultSentinelPreview(sample) {
  const previews = allDownloadedPreviews(sample);
  const usable = previews.filter((preview) => !preview.is_excluded);
  return usable.find((preview) => preview.view === "rgb") || usable[0] || null;
}

function ensureActiveSentinelPreviewForSample(sample) {
  const previews = allDownloadedPreviews(sample);
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

function renderImageIdSearch(sample) {
  const total = sample.sentinel?.downloads?.length || 0;
  const visible = filteredSentinelDownloads(sample).length;
  return `
    <article class="forestImageFilterPanel">
      <label>
        Image ID
        <input id="forestImageIdInput" type="text" value="${escapeHtml(forestState.sentinelImageIdQuery)}" placeholder="download / scene id" />
      </label>
      <span id="forestImageIdMeta" class="meta">${visible}/${total} Sentinel downloads</span>
    </article>
  `;
}

function updateImageIdMeta() {
  const host = $("forestImageIdMeta");
  const sample = forestState.selectedDetail;
  if (!host || !sample) return;
  const total = sample.sentinel?.downloads?.length || 0;
  const visible = filteredSentinelDownloads(sample).length;
  host.textContent = `${visible}/${total} Sentinel downloads`;
}

function renderDetail() {
  const host = $("forestDetails");
  const sample = forestState.selectedDetail;
  if (!sample) {
    host.innerHTML = `<div class="emptyState">Select a sample.</div>`;
    return;
  }

  const raw = JSON.stringify(sample.raw_properties || {}, null, 2);
  const hansenBusy = Boolean(isHansenJobRunning(sample));
  host.innerHTML = `
    ${renderPointInfo(sample)}
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
          Event year
          <input id="forestSentinelYearInput" type="number" min="2001" max="2026" value="${escapeHtml(sample.hansen?.event_year || sample.source_event_year || "")}" />
        </label>
        <label>
          Season
          <select id="forestSeasonPreset">
            <option value="full_snow_free">full snow-free</option>
            <option value="early_summer">early summer</option>
            <option value="late_summer">late summer</option>
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
  $("forestHansenTreecoverInput").addEventListener("change", (event) => {
    forestState.includeHansenTreecover = event.target.checked;
  });
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
  document.querySelectorAll(".forestSentinelReasonSelect").forEach((select) => {
    select.addEventListener("click", (event) => event.stopPropagation());
    select.addEventListener("change", (event) => updateSentinelReviewReason(event));
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
  renderHansenLegend();
}

function clearSelectedSample() {
  closeCompareView({ silent: true });
  forestState.selectedSampleId = null;
  forestState.selectedDetail = null;
  forestState.activeSentinelPreview = null;
  forestState.sentinelImageIdQuery = "";
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
  });
  setForestStatus("sample selected");
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
  return {
    is_excluded: true,
    reason_code: current.reason_code || "",
    reason_text: current.reason_text || "",
    notes: current.notes || null,
    ...patch,
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
  const select = event.target;
  const reasonCode = select.value;
  const draft = setSentinelReviewDraft(select.dataset.downloadId, {
    reason_code: reasonCode,
    reason_text: reasonCode === "OTHER" ? undefined : "",
  });
  renderSentinelDownloadsPanel();
  setForestStatus(draft?.reason_code ? "Sentinel review reason selected" : "choose Sentinel review reason");
}

function updateSentinelReviewOtherText(event) {
  setSentinelReviewDraft(event.target.dataset.downloadId, {
    reason_code: "OTHER",
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
  const reasonCode = String(review.reason_code || "").trim();
  const reasonText = reasonCode === "OTHER" ? String(review.reason_text || "").trim() : null;

  if (!reasonCode) {
    setForestStatus("choose Sentinel exclusion reason");
    return;
  }
  if (reasonCode === "OTHER" && !reasonText) {
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
      reason_code: reasonCode,
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
