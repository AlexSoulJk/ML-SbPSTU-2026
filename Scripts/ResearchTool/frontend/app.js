const state = {
  map: null,
  kml: null,
  selectedGeometry: null,
  geometry: null,
  selectedSegment: null,
  observations: [],
  selectedPreviews: [],
  selectedObservationKeys: new Set(),
  previewSets: new Map(),
  activeObservationKey: null,
  batchDownloadRunning: false,
  activeView: "rgb",
  layers: {},
  baseLayers: {},
  activeBaseLayer: "osm",
  boundaryBounds: null,
  overlayVisibility: {
    zone: true,
    centerline: true,
    segments: true,
    aoi: true,
  },
};

const PREVIEW_VIEWS = ["rgb", "false_color", "ndvi", "ndmi"];
const OVERLAY_LAYER_GROUPS = {
  zone: ["boundary", "polygon"],
  centerline: ["centerline"],
  segments: ["segments", "selectedSegment"],
  aoi: ["aoi"],
};
const els = {};

function $(id) {
  return document.getElementById(id);
}

function setStatus(message) {
  $("statusLine").textContent = message;
}

function refreshMapLayout() {
  if (!state.map) return;
  window.requestAnimationFrame(() => state.map.invalidateSize());
  window.setTimeout(() => state.map.invalidateSize(), 250);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let detail = await response.text();
    try {
      detail = JSON.parse(detail).detail || detail;
    } catch (_error) {
      // Keep raw detail.
    }
    throw new Error(detail);
  }
  return response.json();
}

function lonLatToLatLng(coord) {
  return [coord[1], coord[0]];
}

function bboxToLeafletBounds(bbox) {
  return [
    [bbox[1], bbox[0]],
    [bbox[3], bbox[2]],
  ];
}

function geometryToLatLngs(geometry) {
  if (!geometry) return [];
  if (geometry.type === "LineString") {
    return geometry.coordinates.map(lonLatToLatLng);
  }
  if (geometry.type === "Polygon") {
    return geometry.coordinates.map((ring) => ring.map(lonLatToLatLng));
  }
  return [];
}

function clearLayer(name) {
  if (state.layers[name]) {
    state.map.removeLayer(state.layers[name]);
    state.layers[name] = null;
  }
}

function syncOverlayVisibility() {
  Object.entries(OVERLAY_LAYER_GROUPS).forEach(([key, layerNames]) => {
    const visible = state.overlayVisibility[key];
    layerNames.forEach((name) => {
      const layer = state.layers[name];
      if (!layer) return;
      const onMap = state.map.hasLayer(layer);
      if (visible && !onMap) {
        layer.addTo(state.map);
      } else if (!visible && onMap) {
        state.map.removeLayer(layer);
      }
    });
  });
}

function setOverlayVisibility(key, visible) {
  state.overlayVisibility[key] = visible;
  syncOverlayVisibility();
}

function clearGeometryLayers() {
  ["boundary", "polygon", "centerline", "segments", "selectedSegment", "aoi"].forEach(clearLayer);
}

function initMap() {
  state.map = L.map("map", { zoomControl: true }).setView([56.02, 50.33], 10);

  state.map.createPane("rasterPane");
  state.map.getPane("rasterPane").style.zIndex = 300;
  state.map.createPane("aoiPane");
  state.map.getPane("aoiPane").style.zIndex = 430;
  state.map.createPane("boundaryPane");
  state.map.getPane("boundaryPane").style.zIndex = 500;
  state.map.createPane("centerlinePane");
  state.map.getPane("centerlinePane").style.zIndex = 510;
  state.map.createPane("segmentPane");
  state.map.getPane("segmentPane").style.zIndex = 520;

  const osm = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "&copy; OpenStreetMap",
  });
  const blank = L.layerGroup();

  state.baseLayers = { osm, blank };
  osm.addTo(state.map);
  refreshMapLayout();
}

function setBaseLayer(key) {
  if (!state.map || !state.baseLayers[key]) return;
  Object.entries(state.baseLayers).forEach(([layerKey, layer]) => {
    const visible = layerKey === key;
    const onMap = state.map.hasLayer(layer);
    if (visible && !onMap) {
      layer.addTo(state.map);
    } else if (!visible && onMap) {
      state.map.removeLayer(layer);
    }
  });
  state.activeBaseLayer = key;
}

function renderKmlOptions() {
  const select = $("geometrySelect");
  select.innerHTML = "";
  if (!state.kml?.geometries?.length) {
    $("kmlMeta").textContent = "no geometry";
    return;
  }

  state.kml.geometries.forEach((geometry) => {
    const option = document.createElement("option");
    option.value = geometry.id;
    option.textContent = `${geometry.name} - ${geometry.geometry_type}`;
    select.appendChild(option);
  });

  state.selectedGeometry = state.kml.geometries[0];
  select.value = state.selectedGeometry.id;
  renderKml();
}

function renderKml() {
  clearGeometryLayers();
  if (!state.selectedGeometry) return;

  const coords = state.selectedGeometry.coordinates;
  const latLngs = coords.map(lonLatToLatLng);
  const boundaryHalo = L.polyline(latLngs, {
    pane: "boundaryPane",
    color: "#ffffff",
    weight: 8,
    opacity: 0.9,
  });
  const boundaryStroke = L.polyline(latLngs, {
    pane: "boundaryPane",
    color: "#bf4f32",
    weight: 4,
    opacity: 1,
  });
  state.layers.boundary = L.featureGroup([boundaryHalo, boundaryStroke]).addTo(state.map);
  state.boundaryBounds = state.layers.boundary.getBounds();

  if ($("polygonToggle").checked && state.selectedGeometry.closed) {
    state.layers.polygon = L.polygon(latLngs, {
      pane: "aoiPane",
      color: "#bf4f32",
      weight: 1,
      opacity: 0.8,
      fillColor: "#bf4f32",
      fillOpacity: 0.08,
    }).addTo(state.map);
  }

  const bbox = state.kml.bbox?.map((value) => Number(value).toFixed(5)).join(", ");
  $("kmlMeta").textContent =
    `${state.kml.name || "KML"} - ${coords.length} points - bbox ${bbox}`;
  syncOverlayVisibility();
}

async function loadDefaultKml() {
  setStatus("loading KML");
  state.kml = await api("/api/kml/default");
  renderKmlOptions();
  await buildSegments({ initial: true });
}

async function parseUploadedKml(file) {
  const text = await file.text();
  setStatus("parsing KML");
  state.kml = await api("/api/kml/parse", {
    method: "POST",
    body: JSON.stringify({ filename: file.name, text }),
  });
  renderKmlOptions();
  await buildSegments({ initial: true });
}

function selectedContextM() {
  return Number($("contextSelect").value);
}

function selectedSegmentLengthM() {
  return Number($("segmentLengthInput").value);
}

function bboxKey(bbox) {
  return (bbox || [])
    .map((value) => Number(value).toFixed(6))
    .join(",");
}

function previewKeyForItem(item, bbox = state.selectedSegment?.bbox) {
  if (!item?.id || !bbox) return "";
  return `${bboxKey(bbox)}|${item.id}|${item.datetime || ""}`;
}

function observationKey(observation) {
  return previewKeyForItem(observation.best_item);
}

function downloadableObservations() {
  return state.observations.filter((observation) => observation.best_item);
}

function checkedObservations() {
  return downloadableObservations().filter((observation) =>
    state.selectedObservationKeys.has(observationKey(observation)),
  );
}

function updateObservationToolbar() {
  const selectAll = $("selectAllObservations");
  const downloadSelected = $("downloadSelectedBtn");
  if (!selectAll || !downloadSelected) return;

  const downloadable = downloadableObservations();
  const checked = checkedObservations();
  selectAll.disabled = state.batchDownloadRunning || downloadable.length === 0;
  selectAll.checked = downloadable.length > 0 && checked.length === downloadable.length;
  selectAll.indeterminate = checked.length > 0 && checked.length < downloadable.length;

  downloadSelected.disabled = state.batchDownloadRunning || checked.length === 0;
  downloadSelected.textContent = state.batchDownloadRunning
    ? "Loading..."
    : `Load checked (${checked.length})`;
}

function setAllObservationChecks(checked) {
  state.selectedObservationKeys = new Set(
    checked ? downloadableObservations().map(observationKey).filter(Boolean) : [],
  );
  renderObservations();
}

function rememberPreviewResult(item, result) {
  const key = previewKeyForItem(item, result.bbox);
  if (!key) return key;

  state.previewSets.set(key, result);
  state.activeObservationKey = key;
  state.selectedPreviews = result.previews;

  state.observations = state.observations.map((observation) => {
    if (observationKey(observation) !== key) return observation;
    return {
      ...observation,
      preview_cache: result.preview_cache || {
        all_views_cached: true,
        cached_views: result.previews.map((preview) => preview.view),
      },
    };
  });
  return key;
}

function showStoredPreviewSet(item) {
  const key = previewKeyForItem(item);
  const result = state.previewSets.get(key);
  if (!result) return false;

  state.activeObservationKey = key;
  state.selectedPreviews = result.previews;
  renderPreviews();
  showRasterLayer(state.activeView);
  renderObservations();
  setStatus(`shown from memory - ${result.width}x${result.height}`);
  return true;
}

async function buildSegments(options = {}) {
  if (!state.selectedGeometry) return;
  if (!$("polygonToggle").checked || !state.selectedGeometry.closed) {
    throw new Error("Selected geometry must be a closed line treated as polygon.");
  }

  setStatus("building centerline");
  state.geometry = await api("/api/geometry/segments", {
    method: "POST",
    body: JSON.stringify({
      geometry_id: state.selectedGeometry.id,
      coordinates: state.selectedGeometry.coordinates,
      context_m: selectedContextM(),
      segment_length_m: selectedSegmentLengthM(),
      skeleton_pixel_size_m: 25,
    }),
  });

  renderSegments({ fitSelected: options.fitSelected ?? true });
  setStatus(
    options.initial
      ? `ready: ${state.selectedSegment?.id || "segment"} selected`
      : `segments: ${state.geometry.segments.length}`,
  );
}

function renderSegments(options = {}) {
  clearLayer("centerline");
  clearLayer("segments");
  clearLayer("selectedSegment");
  clearLayer("aoi");

  state.layers.centerline = L.geoJSON(state.geometry.centerline, {
    pane: "centerlinePane",
    style: { color: "#25635a", weight: 2, opacity: 0.7, dashArray: "7 7" },
  }).addTo(state.map);

  state.layers.segments = L.geoJSON(
    {
      type: "FeatureCollection",
      features: state.geometry.segments.map((segment) => ({
        type: "Feature",
        properties: { id: segment.id, index: segment.index },
        geometry: segment.line,
      })),
    },
    {
      pane: "segmentPane",
      style: { color: "#e3ad32", weight: 3, opacity: 0.42 },
      onEachFeature: (feature, layer) => {
        layer.on("click", () => selectSegment(feature.properties.id, { fit: true }));
      },
    },
  ).addTo(state.map);

  const select = $("segmentSelect");
  select.innerHTML = "";
  state.geometry.segments.forEach((segment) => {
    const option = document.createElement("option");
    option.value = segment.id;
    option.textContent =
      `${segment.id} - ${(segment.start_m / 1000).toFixed(1)}-${(segment.end_m / 1000).toFixed(1)} km`;
    select.appendChild(option);
  });

  if (state.geometry.segments.length) {
    const initialIndex = Math.floor(state.geometry.segments.length / 2);
    selectSegment(state.geometry.segments[initialIndex].id, { fit: options.fitSelected ?? true });
  }
  syncOverlayVisibility();
}

function selectSegment(segmentId, options = { fit: true }) {
  state.selectedSegment = state.geometry.segments.find((segment) => segment.id === segmentId);
  $("segmentSelect").value = segmentId;
  clearLayer("selectedSegment");
  clearLayer("aoi");
  if (!state.selectedSegment) return;

  const selectedHalo = L.geoJSON(state.selectedSegment.line, {
    pane: "segmentPane",
    style: {
      color: "#ffffff",
      weight: 10,
      opacity: 0.95,
    },
  });
  const selectedStroke = L.geoJSON(state.selectedSegment.line, {
    pane: "segmentPane",
    style: {
      color: "#e3ad32",
      weight: 5,
      opacity: 1,
    },
  });
  state.layers.selectedSegment = L.featureGroup([selectedHalo, selectedStroke]).addTo(state.map);

  state.layers.aoi = L.geoJSON(state.selectedSegment.aoi, {
    pane: "aoiPane",
    style: {
      color: "#1f2a2b",
      weight: 2,
      fillColor: "#e3ad32",
      fillOpacity: 0.16,
    },
  }).addTo(state.map);
  if (options.fit) {
    state.map.fitBounds(state.layers.aoi.getBounds(), { padding: [24, 24] });
  }
  syncOverlayVisibility();
  refreshMapLayout();

  const bbox = state.selectedSegment.bbox.map((value) => Number(value).toFixed(5)).join(", ");
  $("segmentMeta").textContent =
    `${state.selectedSegment.id} - ${state.selectedSegment.length_m.toFixed(0)} m - ${bbox}`;

  state.observations = [];
  state.selectedPreviews = [];
  state.selectedObservationKeys = new Set();
  state.activeObservationKey = null;
  renderObservations();
  renderPreviews();
  clearLayer("raster");
}

function parseIntegerList(value) {
  return value
    .split(",")
    .map((part) => Number(part.trim()))
    .filter((value) => Number.isInteger(value));
}

async function searchSentinel2() {
  if (!state.selectedSegment) {
    throw new Error("Select a segment first.");
  }

  setStatus("searching Sentinel-2");
  const result = await api("/api/sentinel2/observations", {
    method: "POST",
    body: JSON.stringify({
      bbox: state.selectedSegment.bbox,
      years: parseIntegerList($("yearsInput").value),
      months: parseIntegerList($("monthsInput").value),
      max_cloud: Number($("maxCloudInput").value),
    }),
  });
  state.observations = result.observations;
  state.selectedObservationKeys = new Set(
    downloadableObservations().map(observationKey).filter(Boolean),
  );
  state.activeObservationKey = null;
  state.selectedPreviews = [];
  renderObservations();
  renderPreviews();
  clearLayer("raster");
  setStatus(`observations: ${state.observations.filter((item) => item.best_item).length}`);
}

function renderObservations() {
  const host = $("observations");
  host.innerHTML = "";
  if (!state.observations.length) {
    updateObservationToolbar();
    return;
  }

  state.observations.forEach((observation) => {
    const card = document.createElement("article");
    const item = observation.best_item;
    if (!item) {
      card.className = "observation unavailable";
      card.innerHTML = `
        <strong>${observation.period}</strong>
        <span>no item under cloud threshold</span>
      `;
      host.appendChild(card);
      return;
    }

    const key = observationKey(observation);
    const inMemory = state.previewSets.has(key);
    const cached = inMemory || observation.preview_cache?.all_views_cached;
    const active = state.activeObservationKey === key;
    const checked = state.selectedObservationKeys.has(key);
    card.className = [
      "observation",
      cached ? "downloaded" : "",
      active ? "active" : "",
    ].filter(Boolean).join(" ");

    const cloud = item.cloud_cover === null || item.cloud_cover === undefined
      ? "n/a"
      : `${Number(item.cloud_cover).toFixed(1)}%`;
    const date = item.datetime ? item.datetime.slice(0, 10) : "unknown date";
    card.innerHTML = `
      <div class="observationTop">
        <label class="observationCheck">
          <input class="observationSelect" type="checkbox" ${checked ? "checked" : ""} />
        </label>
        <strong>${observation.period} - ${date}</strong>
        <span class="badge">${cached ? "cached" : "remote"}</span>
      </div>
      <span>${cloud} cloud - ${item.platform || "Sentinel-2"}</span>
      <span>${item.id || ""}</span>
      <button class="loadPreviewBtn">${cached ? "Show views" : "Load views"}</button>
    `;
    card.querySelector(".observationSelect").addEventListener("change", (event) => {
      if (event.target.checked) {
        state.selectedObservationKeys.add(key);
      } else {
        state.selectedObservationKeys.delete(key);
      }
      updateObservationToolbar();
    });
    card.querySelector("button").addEventListener("click", () => run(() => openObservation(observation)));
    card.addEventListener("click", (event) => {
      if (event.target.closest("button, input, label")) return;
      run(() => openObservation(observation));
    });
    host.appendChild(card);
  });
  updateObservationToolbar();
}

async function openObservation(observation) {
  const item = observation.best_item;
  if (!item) return;
  if (showStoredPreviewSet(item)) return;
  await loadPreviews(item);
}

async function loadPreviews(item, options = {}) {
  if (!state.selectedSegment) {
    throw new Error("Select a segment first.");
  }

  if (!options.quiet) {
    setStatus("loading raster views");
  }
  const result = await api("/api/sentinel2/previews", {
    method: "POST",
    body: JSON.stringify({
      bbox: state.selectedSegment.bbox,
      item,
      max_cloud: Number($("maxCloudInput").value),
      views: PREVIEW_VIEWS,
    }),
  });
  rememberPreviewResult(item, result);
  renderPreviews();
  showRasterLayer(state.activeView);
  renderObservations();
  if (!options.quiet) {
    setStatus(`views loaded - ${result.width}x${result.height}`);
  }
}

async function downloadCheckedObservations() {
  const checked = checkedObservations();
  if (!checked.length) return;

  state.batchDownloadRunning = true;
  updateObservationToolbar();
  try {
    for (let index = 0; index < checked.length; index += 1) {
      const observation = checked[index];
      setStatus(`loading ${index + 1}/${checked.length}: ${observation.period}`);
      if (!showStoredPreviewSet(observation.best_item)) {
        await loadPreviews(observation.best_item, { quiet: true });
      }
    }
    setStatus(`views ready: ${checked.length}`);
  } finally {
    state.batchDownloadRunning = false;
    renderObservations();
  }
}

function renderPreviews() {
  const host = $("previewGrid");
  host.innerHTML = "";
  state.selectedPreviews.forEach((preview) => {
    const card = document.createElement("article");
    card.className = "preview";
    card.innerHTML = `
      <h3>${preview.view}</h3>
      <img src="${preview.url}" alt="${preview.view}" />
    `;
    card.addEventListener("click", () => showRasterLayer(preview.view));
    host.appendChild(card);
  });
}

function showRasterLayer(view) {
  state.activeView = view;
  clearLayer("raster");
  const preview = state.selectedPreviews.find((item) => item.view === view);
  if (!preview) return;
  state.layers.raster = L.imageOverlay(
    preview.url,
    bboxToLeafletBounds(preview.bbox),
    {
      pane: "rasterPane",
      opacity: Number($("overlayOpacity").value) / 100,
    },
  ).addTo(state.map);
}

function bindEvents() {
  $("loadDefaultBtn").addEventListener("click", () => run(loadDefaultKml));
  $("kmlFileInput").addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    if (file) run(() => parseUploadedKml(file));
  });
  $("geometrySelect").addEventListener("change", (event) => {
    state.selectedGeometry = state.kml.geometries.find(
      (geometry) => geometry.id === event.target.value,
    );
    state.geometry = null;
    state.selectedSegment = null;
    renderKml();
  });
  $("polygonToggle").addEventListener("change", renderKml);
  $("buildSegmentsBtn").addEventListener("click", () => run(buildSegments));
  $("segmentSelect").addEventListener("change", (event) => selectSegment(event.target.value, { fit: true }));
  $("searchS2Btn").addEventListener("click", () => run(searchSentinel2));
  $("selectAllObservations").addEventListener("change", (event) =>
    setAllObservationChecks(event.target.checked),
  );
  $("downloadSelectedBtn").addEventListener("click", () => run(downloadCheckedObservations));
  $("showZoneToggle").addEventListener("change", (event) =>
    setOverlayVisibility("zone", event.target.checked),
  );
  $("showCenterlineToggle").addEventListener("change", (event) =>
    setOverlayVisibility("centerline", event.target.checked),
  );
  $("showSegmentsToggle").addEventListener("change", (event) =>
    setOverlayVisibility("segments", event.target.checked),
  );
  $("showAoiToggle").addEventListener("change", (event) =>
    setOverlayVisibility("aoi", event.target.checked),
  );
  $("clearRasterBtn").addEventListener("click", () => clearLayer("raster"));
  $("overlayOpacity").addEventListener("input", () => {
    if (state.layers.raster) {
      state.layers.raster.setOpacity(Number($("overlayOpacity").value) / 100);
    }
  });
  document.querySelectorAll(".layerBtn").forEach((button) => {
    button.addEventListener("click", () => showRasterLayer(button.dataset.view));
  });
  document.querySelectorAll("input[name='baseLayer']").forEach((input) => {
    input.addEventListener("change", () => {
      if (input.checked) setBaseLayer(input.value);
    });
  });
}

async function run(task) {
  try {
    await task();
  } catch (error) {
    setStatus(error.message || String(error));
    console.error(error);
  }
}

async function init() {
  initMap();
  bindEvents();
  try {
    const health = await api("/api/health");
    $("healthStatus").textContent = health.ok ? "local backend" : "offline";
  } catch (_error) {
    $("healthStatus").textContent = "offline";
  }
  if (document.body.dataset.mode !== "forest") {
    await run(loadDefaultKml);
  } else {
    setStatus("Forest mode ready");
  }
}

window.pipelineExplorer = {
  state,
  api,
  loadDefaultKml,
  clearGeometryLayers,
  clearLayer,
  refreshMapLayout,
  setBaseLayer,
  setStatus,
  run,
};
window.pipelineReady = init();
