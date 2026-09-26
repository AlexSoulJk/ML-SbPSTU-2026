import { forestState } from "./state.js?v=forest-iter2-23";

const FALLBACK_SELECT_RADIUS_PX = 32;

function getMap() {
  return window.pipelineExplorer?.state?.map || null;
}

function removeLayer(name) {
  const map = getMap();
  const layer = forestState.layers[name];
  if (map && layer) {
    map.removeLayer(layer);
  }
  forestState.layers[name] = null;
}

export function clearForestLayers() {
  Object.keys(forestState.layers).forEach(removeLayer);
}

function eventTargetSummary(target) {
  if (!target) return "unknown";
  const tag = target.tagName ? target.tagName.toLowerCase() : "node";
  const className = typeof target.className === "string"
    ? target.className
    : target.className?.baseVal || "";
  return `${tag}${className ? `.${className.trim().replaceAll(/\s+/g, ".")}` : ""}`;
}

function uniqueCandidates(samples, nearest, detail) {
  const byId = new Map();
  [...samples, ...nearest, detail].filter(Boolean).forEach((sample) => {
    byId.set(sample.sample_id, sample);
  });
  return [...byId.values()];
}

function nearestCandidateForEvent(map, event) {
  const candidates = forestState.mapClickCandidates || [];
  if (!candidates.length) return null;

  const clickPoint = map.mouseEventToContainerPoint(event);
  let best = null;
  candidates.forEach((sample) => {
    const samplePoint = map.latLngToContainerPoint([sample.lat, sample.lon]);
    const distancePx = clickPoint.distanceTo(samplePoint);
    if (!best || distancePx < best.distancePx) {
      best = { sample, distancePx };
    }
  });
  return best;
}

function debugMap(message, data = {}) {
  const payload = {
    time: new Date().toLocaleTimeString(),
    message,
    ...data,
  };
  console.debug("[forest-map]", payload);
  window.dispatchEvent(new CustomEvent("forest-map-debug", { detail: payload }));
}

function ensureForestPanes(map) {
  const panes = [
    ["forestSentinelPane", 330, "none"],
    ["forestRasterPane", 360, "none"],
    ["forestShapePane", 560, "none"],
    ["forestMarkerPane", 660, "auto"],
  ];
  panes.forEach(([name, zIndex, pointerEvents]) => {
    if (!map.getPane(name)) {
      map.createPane(name);
    }
    const pane = map.getPane(name);
    pane.style.zIndex = zIndex;
    pane.style.pointerEvents = pointerEvents;
  });
}

function attachMapCaptureDebug(map) {
  if (forestState.mapDebugAttached) return;
  const container = map.getContainer();
  ["pointerdown", "mousedown", "click"].forEach((type) => {
    container.addEventListener(
      type,
      (event) => {
        const latLng = map.mouseEventToLatLng(event);
        debugMap(`container ${type}`, {
          ctrlKey: Boolean(event.ctrlKey),
          button: event.button,
          target: eventTargetSummary(event.target),
          lat: Number(latLng.lat).toFixed(6),
          lon: Number(latLng.lng).toFixed(6),
        });
      },
      true,
    );
  });
  container.addEventListener(
    "click",
    (event) => {
      if (!event.ctrlKey || event.button !== 0) return;

      const best = nearestCandidateForEvent(map, event);
      debugMap("fallback nearest", {
        candidates: forestState.mapClickCandidates.length,
        nearest_sample_id: best?.sample?.sample_id || null,
        nearest_short_id: best?.sample?.short_id || null,
        distance_px: best ? Number(best.distancePx).toFixed(1) : null,
        threshold_px: FALLBACK_SELECT_RADIUS_PX,
      });
      if (!best || best.distancePx > FALLBACK_SELECT_RADIUS_PX) {
        debugMap("fallback ignored: no sample near click");
        return;
      }

      const now = window.performance.now();
      if (now - forestState.lastFallbackSelectAt < 250) return;
      forestState.lastFallbackSelectAt = now;
      debugMap("fallback select requested", {
        sample_id: best.sample.sample_id,
        short_id: best.sample.short_id,
      });
      forestState.mapSelectHandler?.(best.sample.sample_id);
    },
    true,
  );
  forestState.mapDebugAttached = true;
  debugMap("container debug listeners attached");
}

function sampleLabel(sample) {
  return `${sample.display_name || sample.short_id} ${sample.driver_primary || ""}`.trim();
}

function showCtrlHint(hitLayer, sample) {
  hitLayer
    .bindTooltip("Ctrl + LMB to select", {
      direction: "top",
      opacity: 0.95,
      permanent: false,
    })
    .openTooltip();
  window.setTimeout(() => {
    hitLayer.bindTooltip(sampleLabel(sample));
  }, 1200);
}

function stopLeafletEvent(event) {
  if (event.originalEvent) {
    L.DomEvent.stop(event.originalEvent);
  }
}

function addVisiblePoint(sample, options) {
  const latLng = [sample.lat, sample.lon];
  if (!sample.is_saved) {
    return L.circleMarker(latLng, {
      ...options,
      pane: "forestMarkerPane",
      interactive: false,
    });
  }

  const radius = Number(options.radius || 6);
  const side = Math.max(9, radius * 2);
  const weight = Number(options.weight || 1);
  const style = [
    `width:${side}px`,
    `height:${side}px`,
    `background:${options.fillColor || options.color || "#2a9d8f"}`,
    `border:${weight}px solid ${options.color || "#264653"}`,
    `opacity:${options.fillOpacity ?? 1}`,
  ].join(";");
  return L.marker(latLng, {
    pane: "forestMarkerPane",
    interactive: false,
    icon: L.divIcon({
      className: "forestPointIconWrapper",
      html: `<span class="forestPointIcon" style="${style}"></span>`,
      iconSize: [side + weight * 2, side + weight * 2],
      iconAnchor: [(side + weight * 2) / 2, (side + weight * 2) / 2],
    }),
  });
}

function addCircle(sample, options, onSelect) {
  const latLng = [sample.lat, sample.lon];
  const visible = addVisiblePoint(sample, options);
  const hit = L.circleMarker(latLng, {
    pane: "forestMarkerPane",
    radius: Math.max(Number(options.radius || 6) + 12, 18),
    stroke: false,
    fill: true,
    fillColor: "#ffffff",
    fillOpacity: 0.01,
    interactive: true,
    bubblingMouseEvents: false,
  });

  let lastSelectedAt = 0;
  const handleSelect = (event) => {
    stopLeafletEvent(event);
    debugMap(`sample hit ${event.type}`, {
      sample_id: sample.sample_id,
      short_id: sample.short_id,
      ctrlKey: Boolean(event.originalEvent?.ctrlKey),
      button: event.originalEvent?.button,
      target: eventTargetSummary(event.originalEvent?.target),
    });
    if (event.type === "mousedown" && event.originalEvent?.button !== 0) {
      debugMap("sample hit ignored: not left mouse button", {
        sample_id: sample.sample_id,
        button: event.originalEvent?.button,
      });
      return;
    }
    if (!event.originalEvent?.ctrlKey) {
      debugMap("sample hit ignored: Ctrl missing", {
        sample_id: sample.sample_id,
      });
      showCtrlHint(hit, sample);
      return;
    }

    const now = window.performance.now();
    if (now - lastSelectedAt < 250) return;
    lastSelectedAt = now;
    debugMap("sample select requested", {
      sample_id: sample.sample_id,
      short_id: sample.short_id,
    });
    onSelect(sample.sample_id);
  };

  hit.bindTooltip(sampleLabel(sample));
  hit.on("mousedown", handleSelect);
  hit.on("click", handleSelect);
  hit.on("touchstart", stopLeafletEvent);

  return L.layerGroup([visible, hit]);
}

function bboxToBounds(bbox) {
  return [
    [bbox[1], bbox[0]],
    [bbox[3], bbox[2]],
  ];
}

function hansenOverlay(detail) {
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

export function renderForestMap({ samples, nearest, detail, visibility, onSelect, fit = false }) {
  const map = getMap();
  if (!map) return;
  ensureForestPanes(map);
  forestState.mapClickCandidates = uniqueCandidates(samples, nearest, detail);
  forestState.mapSelectHandler = onSelect;
  attachMapCaptureDebug(map);
  clearForestLayers();
  debugMap("render forest map", {
    samples: samples.length,
    nearest: nearest.length,
    candidates: forestState.mapClickCandidates.length,
    selected: detail?.sample_id || null,
    fit,
  });

  if (visibility.samples && samples.length) {
    forestState.layers.samples = L.layerGroup(
      samples.map((sample) =>
        addCircle(
          sample,
          {
            radius: 5,
            color: "#264653",
            weight: 1,
            fillColor: "#2a9d8f",
            fillOpacity: 0.68,
          },
          onSelect,
        ),
      ),
    ).addTo(map);
  }

  if (visibility.nearest && nearest.length) {
    forestState.layers.nearest = L.layerGroup(
      nearest.map((sample) =>
        addCircle(
          sample,
          {
            radius: 6,
            color: "#1f2a2b",
            weight: 1,
            fillColor: "#e9c46a",
            fillOpacity: 0.82,
          },
          onSelect,
        ),
      ),
    ).addTo(map);
  }

  if (detail?.geometry) {
    const sentinel = visibility.sentinel ? forestState.activeSentinelPreview : null;
    if (sentinel?.url && sentinel?.bbox) {
      forestState.layers.sentinel = L.imageOverlay(
        sentinel.url,
        bboxToBounds(sentinel.bbox),
        { pane: "forestSentinelPane", opacity: 0.95, interactive: false },
      ).addTo(map);
    }

    if (visibility.aoi) {
      forestState.layers.aoi = L.geoJSON(detail.geometry.viewer_aoi, {
        pane: "forestShapePane",
        interactive: false,
        style: {
          color: "#2f4858",
          weight: 2,
          fillColor: "#86bbd8",
          fillOpacity: 0.12,
        },
      }).addTo(map);
    }

    if (visibility.plot) {
      forestState.layers.plot = L.geoJSON(detail.geometry.sample_plot, {
        pane: "forestShapePane",
        interactive: false,
        style: {
          color: "#d45d3f",
          weight: 2,
          fillColor: "#d45d3f",
          fillOpacity: 0.18,
        },
      }).addTo(map);
    }

    forestState.layers.selected = addCircle(
      detail,
      {
        radius: 8,
        color: "#ffffff",
        weight: 3,
        fillColor: "#d45d3f",
        fillOpacity: 1,
      },
      onSelect,
    ).addTo(map);

    const overlay = visibility.hansen ? hansenOverlay(detail) : null;
    if (overlay?.png_url && overlay?.bbox) {
      forestState.layers.hansen = L.imageOverlay(
        overlay.png_url,
        bboxToBounds(overlay.bbox),
        {
          pane: "forestRasterPane",
          opacity: forestState.hansenOpacity,
          interactive: false,
          className: `forestHansenOverlay ${forestState.hansenPixelated ? "pixelated" : ""}`,
        },
      ).addTo(map);
    }

    if (fit) {
      map.fitBounds(bboxToBounds(detail.geometry.viewer_aoi_bbox), { padding: [30, 30] });
    }
  } else if (fit && samples.length) {
    const bounds = L.latLngBounds(samples.map((sample) => [sample.lat, sample.lon]));
    map.fitBounds(bounds, { padding: [30, 30] });
  }

  window.pipelineExplorer?.refreshMapLayout();
}
