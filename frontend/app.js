const $ = (id) => document.getElementById(id);

async function safeFetchJSON(url, options = {}) {
  let response;
  try {
    response = await fetch(url, options);
  } catch (err) {
    throw new Error(`Network failure: ${err.message}`);
  }
  const text = await response.text();
  let data;
  try {
    data = text ? JSON.parse(text) : {};
  } catch (err) {
    if (!response.ok) {
      throw new Error(`Backend returned HTTP ${response.status}: ${text.slice(0, 60)}...`);
    }
    throw new Error(`Invalid JSON from server: ${text.slice(0, 60)}...`);
  }
  if (!response.ok) {
    throw new Error(data.detail || data.message || `Backend error HTTP ${response.status}`);
  }
  return data;
}
const state = {
  source: null,
  reference: null,
};

function toast(message) {
  const el = $("toast");
  el.textContent = message;
  el.classList.add("show");
  clearTimeout(window.__toast);
  window.__toast = setTimeout(() => el.classList.remove("show"), 3200);
}

function bindFile(inputId, nameId, dropId, key) {
  const input = $(inputId);
  const name = $(nameId);
  const drop = $(dropId);

  input.addEventListener("change", (e) => {
    if (input.files[0]) {
      state[key] = input.files[0];
      name.textContent = input.files[0].name;
      // Reset input value so same file can be selected again
      e.target.value = '';
    }
  });

  ["dragenter", "dragover"].forEach(evt => drop.addEventListener(evt, e => {
    e.preventDefault(); drop.classList.add("drag");
  }));
  ["dragleave", "drop"].forEach(evt => drop.addEventListener(evt, e => {
    e.preventDefault(); drop.classList.remove("drag");
  }));
  drop.addEventListener("drop", e => {
    const file = e.dataTransfer.files[0];
    if (file) {
      state[key] = file;
      name.textContent = file.name;
    }
  });
}

bindFile("sourceInput", "sourceName", "sourceDrop", "source");
bindFile("referenceInput", "referenceName", "referenceDrop", "reference");

$("ratio").addEventListener("input", e => $("ratioValue").textContent = e.target.value);
$("ransac").addEventListener("input", e => $("ransacValue").textContent = `${e.target.value} px`);

function setWorkflow(activeIndex) {
  document.querySelectorAll(".step").forEach((s, i) => s.classList.toggle("active", i <= activeIndex));
}

function fmt(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(digits);
}

function showResults(data) {
  $("emptyState").classList.add("hidden");
  $("results").classList.remove("hidden");
  $("resultStatus").textContent = "REGISTRATION COMPLETE";
  $("rmse").textContent = fmt(data.metrics.rmse_pixels, 3);
  $("inliers").textContent = data.metrics.inlier_count;
  $("inlierRatio").textContent = `${(data.metrics.inlier_ratio * 100).toFixed(1)}%`;
  $("coverage").textContent = `${(data.metrics.source_spatial_coverage * 100).toFixed(0)}%`;

  const base = window.location.origin;

  $("homography").textContent = data.homography
    .map(row => row.map(v => Number(v).toFixed(6)).join("   "))
    .join("\n");

  $("pipeline").innerHTML = data.pipeline.map(x => `<li>${x}</li>`).join("");

  // Phase 8: Pass REAL API data to Inspector.
  // refUrl: reference image saved by backend (radiometric normalized, used for overlay).
  // regUrl: warped source image (registered output).
  if (window.Inspector && data.job_id) {
    const refUrl = base + "/uploads/" + data.job_id + "_reference.png";
    const regUrl = base + data.outputs.registered_image;
    Inspector.loadData(data, refUrl, regUrl);
  }

  // Populate new Viewer
  const srcOrigUrl = base + "/uploads/" + data.job_id + "_source.png";
  const refOrigUrl = base + "/uploads/" + data.job_id + "_reference.png";
  
  const imgIds = ["viewOrigSrc", "viewOrigRef", "viewFeatSrc", "viewFeatRef", "viewKeySrc", "viewKeyRef", "viewMatches", "viewRegistered"];
  
  // Setup generic error handlers for all images
  imgIds.forEach(id => {
    const img = $(id);
    if (!img) return;
    img.onload = () => { img.style.border = ""; img.title = ""; };
    img.onerror = () => { 
      img.style.border = "2px dashed red"; 
      img.title = `Failed to load: ${img.src}`;
      toast(`Failed to load image: ${img.src}`);
    };
    img.removeAttribute("src"); // Clear previous state
  });

  $("viewOrigSrc").src = srcOrigUrl;
  $("viewOrigRef").src = refOrigUrl;
  
  if (data.outputs.feature_response_source) {
    $("viewFeatSrc").src = base + data.outputs.feature_response_source;
    $("viewFeatRef").src = base + data.outputs.feature_response_reference;
  }
  
  if (data.outputs.keypoints_source) {
    $("viewKeySrc").src = base + data.outputs.keypoints_source;
    $("viewKeyRef").src = base + data.outputs.keypoints_reference;
  }
  
  $("viewMatches").src = base + data.outputs.match_visualization;
  $("viewRegistered").src = base + data.outputs.registered_image;

  const metadata = `DETECTOR: ${data.settings.detector.toUpperCase()} · RAW CORRESPONDENCES: ${data.settings.raw_match_count} · FILTERED CORRESPONDENCES: ${data.settings.post_distribution_match_count} · ECC REFINEMENT: ${data.settings.ecc_used ? 'YES' : 'NO'}`;
  $("viewerMetadata").textContent = metadata;

  // Initialize tabs
  document.querySelectorAll('.viewer-tab').forEach(tab => {
    tab.onclick = () => {
      document.querySelectorAll('.viewer-tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.add('hidden'));
      tab.classList.add('active');
      document.getElementById('tab-' + tab.dataset.tab).classList.remove('hidden');
    };
  });
}

$("runBtn").addEventListener("click", async () => {
  if (!state.source || !state.reference) {
    toast("Please select both a source image and a reference image.");
    return;
  }

  const fd = new FormData();
  fd.append("source", state.source);
  fd.append("reference", state.reference);
  fd.append("detector", $("detector").value);
  fd.append("ratio", $("ratio").value);
  fd.append("ransac_threshold", $("ransac").value);
  fd.append("illumination_normalization", $("illumination").checked);
  fd.append("spatial_distribution", $("distribution").checked);
  fd.append("ecc_refinement", $("ecc").checked);
  fd.append("max_features", "8000");

  const btn = $("runBtn");
  btn.disabled = true;
  btn.querySelector("span").textContent = "PROCESSING…";
  setWorkflow(1);
  $("resultStatus").textContent = "RUNNING PIPELINE";

  try {
    const data = await safeFetchJSON("/api/register", { method: "POST", body: fd });

    setWorkflow(3);
    showResults(data);
    toast(`Registration complete — ${data.metrics.inlier_count} verified inliers.`);
    window.scrollTo({ top: document.querySelector(".results-panel").offsetTop - 30, behavior: "smooth" });
  } catch (err) {
    $("resultStatus").textContent = "ERROR";
    toast(err.message);
    setWorkflow(0);
  } finally {
    btn.disabled = false;
    btn.querySelector("span").textContent = "RUN REGISTRATION";
  }
});

$("demoBtn").addEventListener("click", () => {
  toast("Recommended demo pairs: OHRC ↔ LROC NAC, TMC-2 ↔ LROC WAC, and IIRS ↔ LROC WAC. Use real raster exports for the presentation.");
  document.querySelector(".dataset-note").scrollIntoView({ behavior: "smooth", block: "center" });
});

// Multi-image map builder: kept separate from the established pair-upload state.
const multiState = { files: [], result: null, scale: 1, edgeHits: [] };

document.querySelectorAll(".mode-button").forEach(button => button.addEventListener("click", () => {
  const isMulti = button.dataset.mode === "multi";
  document.querySelectorAll(".mode-button").forEach(item => item.classList.toggle("active", item === button));
  $("pairMode").classList.toggle("hidden", isMulti);
  $("multiMode").classList.toggle("hidden", !isMulti);
}));

function imagePreview(file) {
  return new Promise(resolve => {
    const reader = new FileReader();
    reader.onload = () => { const image = new Image(); image.onload = () => resolve({ url: reader.result, width: image.width, height: image.height }); image.src = reader.result; };
    reader.readAsDataURL(file);
  });
}

async function addMultiFiles(files) {
  for (const file of [...files]) {
    if (multiState.files.length >= 12) { toast("The local prototype accepts up to 12 images per map job."); break; }
    if (!file.type.startsWith("image/") && !/\.(png|jpe?g|bmp|tiff?|webp)$/i.test(file.name)) { toast(`${file.name} is not a supported raster image.`); continue; }
    const preview = await imagePreview(file);
    multiState.files.push({ file, ...preview, sensor: "Unknown / Not provided" });
  }
  renderMultiGallery();
}

function renderMultiGallery() {
  const gallery = $("multiGallery");
  if (!multiState.files.length) { gallery.innerHTML = '<p class="gallery-empty">No images selected. Metadata remains “Unknown / Not provided” unless you add it later.</p>'; return; }
  gallery.innerHTML = multiState.files.map((item, index) => `<article class="image-tile"><img src="${item.url}" alt="${item.file.name}"><div><b>IMAGE ${String(index + 1).padStart(2, "0")}</b><button class="remove-image" data-remove="${index}" title="Remove image">×</button><p>${item.file.name}</p><small>${item.width} × ${item.height} · ${(item.file.name.split(".").pop() || "unknown").toUpperCase()}</small><select data-sensor="${index}"><option>Unknown / Not provided</option><option>OHRC</option><option>TMC / TMC-2</option><option>IIRS</option><option>LROC NAC</option><option>LROC WAC</option><option>SELENE</option></select></div></article>`).join("");
  gallery.querySelectorAll("[data-remove]").forEach(button => button.addEventListener("click", () => { multiState.files.splice(Number(button.dataset.remove), 1); renderMultiGallery(); }));
  gallery.querySelectorAll("[data-sensor]").forEach(select => select.addEventListener("change", () => { multiState.files[Number(select.dataset.sensor)].sensor = select.value; }));
}

$("multiInput").addEventListener("change", event => addMultiFiles(event.target.files));
["dragenter", "dragover"].forEach(eventName => $("multiDrop").addEventListener(eventName, event => { event.preventDefault(); $("multiDrop").classList.add("drag"); }));
$("multiDrop").addEventListener("dragleave", () => $("multiDrop").classList.remove("drag"));
$("multiDrop").addEventListener("drop", event => { event.preventDefault(); $("multiDrop").classList.remove("drag"); addMultiFiles(event.dataTransfer.files); });

function setMultiProgress(message) { $("multiProgress").textContent = message; $("multiStatus").textContent = message; }

$("analyzeSetBtn").addEventListener("click", async () => {
  if (multiState.files.length < 2) { toast("Select at least two images for multi-image registration."); return; }
  const form = new FormData();
  multiState.files.forEach(item => form.append("images", item.file));
  form.append("metadata_json", JSON.stringify(multiState.files.map(item => ({ sensor: item.sensor }))));
  form.append("detector", $("detector").value); form.append("ratio", $("ratio").value); form.append("ransac_threshold", $("ransac").value);
  form.append("illumination_normalization", $("illumination").checked); form.append("spatial_distribution", $("distribution").checked); form.append("ecc_refinement", $("ecc").checked); form.append("max_features", "8000");
  const button = $("analyzeSetBtn"); button.disabled = true; setMultiProgress("ANALYZING COLLECTION");
  try {
    const data = await safeFetchJSON("/api/multi-registration/register", { method: "POST", body: form });
    multiState.result = data; showMultiResults(data); toast(`${data.summary.successful_pair_count} reliable registration relationships accepted.`);
  } catch (error) { setMultiProgress("ERROR"); toast(error.message); }
  finally { button.disabled = false; }
});

function showMultiResults(data) {
  const s = data.summary, base = window.location.origin;
  $("multiResults").classList.remove("hidden"); setMultiProgress(data.status);
  $("multiImages").textContent = s.image_count; $("candidatePairs").textContent = s.candidate_pair_count; $("acceptedPairs").textContent = s.successful_pair_count; $("placedImages").textContent = s.placed_image_count;
  $("mosaicLabel").textContent = data.map_type.toUpperCase();
  if (data.outputs.mosaic) { $("mosaicImage").src = base + data.outputs.mosaic; $("mosaicLink").href = base + data.outputs.mosaic; }
  else { $("mosaicImage").removeAttribute("src"); $("mosaicLink").removeAttribute("href"); }
  $("graphExport").href = `${base}/api/multi-registration/${data.job_id}/export/graph.json`;
  $("metricsExport").href = `${base}/api/multi-registration/${data.job_id}/export/metrics.csv`;
  $("reportExport").href = `${base}/api/multi-registration/${data.job_id}/export/report.json`;
  drawAnalytics(data); drawGraph(data); drawFootprints(data); drawMatchPoints(data); drawPairTable(data); $("explorer3D").classList.add("hidden"); $("multiResults").scrollIntoView({ behavior: "smooth", block: "start" });
}

function drawAnalytics(data) {
  const summary = data.summary, rmse = summary.rmse_statistics_pixels || {}, accepted = data.graph.edges.filter(edge => edge.accepted);
  $("analyticsImages").textContent = summary.image_count;
  $("analyticsRegistered").textContent = `${summary.placed_image_count}/${summary.image_count}`;
  $("analyticsInliers").textContent = summary.average_inliers_per_accepted_pair === null ? "—" : fmt(summary.average_inliers_per_accepted_pair, 1);
  $("analyticsRatio").textContent = summary.average_inlier_ratio === null ? "—" : `${(summary.average_inlier_ratio * 100).toFixed(1)}%`;
  $("analyticsRmse").textContent = rmse.mean === null ? "—" : `${fmt(rmse.mean, 3)} px`;
  $("analyticsCoverage").textContent = summary.average_source_spatial_coverage === null ? "—" : `${(summary.average_source_spatial_coverage * 100).toFixed(0)}%`;
  $("analyticsConnectivity").textContent = summary.graph_connected ? "CONNECTED" : `${summary.graph_component_count} COMPONENTS`;
  $("analyticsTime").textContent = `${fmt(summary.processing_time_seconds, 3)} s`;
  $("analyticsPairs").textContent = `${summary.successful_pair_count}/${summary.processed_pairs + summary.reused_pairs}`;
  $("analyticsCache").textContent = `${summary.cache_hits}/${summary.cache_misses}`;
  $("analyticsDevice").textContent = summary.processing_diagnostics?.device || "CPU";
  $("processingDevice").textContent = `DEVICE: ${summary.processing_diagnostics?.device || "—"}`;
  $("analyticsNote").textContent = summary.processing_diagnostics?.note || summary.scientific_note;
  renderBarChart("outcomeChart", [
    { label: "accepted", value: summary.successful_pair_count, kind: "accepted" },
    { label: "rejected", value: summary.failed_pair_count, kind: "rejected" },
  ]);
  renderBarChart("inlierChart", accepted.map(edge => ({ label: `I${edge.image_a + 1}–I${edge.image_b + 1}`, value: edge.metrics.inlier_count, kind: "inliers" })));
  renderBarChart("rmseChart", accepted.filter(edge => edge.metrics.rmse_pixels !== null).map(edge => ({ label: `I${edge.image_a + 1}–I${edge.image_b + 1}`, value: edge.metrics.rmse_pixels, kind: "rmse" })));
  renderDiagnostics(data);
  renderSensorSummary(data);
}

function renderBarChart(id, values) {
  const chart = $(id), max = Math.max(1, ...values.map(item => item.value));
  chart.innerHTML = values.length ? values.map(item => `<div class="chart-bar ${item.kind}" title="${item.label}: ${Number(item.value).toFixed(3)}"><i style="height:${Math.max(5, item.value / max * 100)}%"></i><small>${item.label}</small><b>${Number(item.value).toFixed(item.kind === "rmse" ? 3 : 0)}</b></div>`).join("") : '<span class="chart-empty">No accepted pair values</span>';
}

function drawGraph(data) {
  const canvas = $("graphCanvas"), ctx = canvas.getContext("2d"), nodes = data.images, edges = data.graph.edges;
  const positions = nodes.map((_, index) => ({ x: 70 + (index % 4) * 165, y: 75 + Math.floor(index / 4) * 135 }));
  ctx.clearRect(0, 0, canvas.width, canvas.height); multiState.edgeHits = [];
  edges.forEach(edge => { const a = positions[edge.image_a], b = positions[edge.image_b]; ctx.strokeStyle = edge.accepted ? "#7cf2bd" : "#6c7486"; ctx.globalAlpha = edge.accepted ? .85 : .25; ctx.lineWidth = edge.accepted ? 2 + Math.min(4, edge.confidence / 30) : 1; ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); multiState.edgeHits.push({ edge, a, b }); });
  ctx.globalAlpha = 1; positions.forEach((point, index) => { ctx.fillStyle = "#111a2b"; ctx.strokeStyle = "#6ee7f9"; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(point.x, point.y, 25, 0, Math.PI * 2); ctx.fill(); ctx.stroke(); ctx.fillStyle = "#edf3ff"; ctx.font = "11px sans-serif"; ctx.textAlign = "center"; ctx.fillText(`IMG ${String(index + 1).padStart(2, "0")}`, point.x, point.y + 4); });
}

$("graphCanvas").addEventListener("click", event => {
  const rect = event.currentTarget.getBoundingClientRect();
  const x = (event.clientX - rect.left) * event.currentTarget.width / rect.width;
  const y = (event.clientY - rect.top) * event.currentTarget.height / rect.height;
  const hit = multiState.edgeHits
    .map(item => ({ item, d: Math.abs((item.b.y - item.a.y) * x - (item.b.x - item.a.x) * y + item.b.x * item.a.y - item.b.y * item.a.x) / Math.max(1, Math.hypot(item.b.y - item.a.y, item.b.x - item.a.x)) }))
    .sort((a,b) => a.d-b.d)[0];
  if (!hit || hit.d > 14) return;
  inspectPair(multiState.result, hit.item.edge);
});

function drawFootprints(data) {
  const overlay = $("footprintsOverlay");
  // Prefer Phase 10 footprints if available
  let footprints = null;
  let w = 0, h = 0;
  if (data.p10_mosaic && data.p10_mosaic.components && data.p10_mosaic.components.length > 0) {
    const comp0 = data.p10_mosaic.components[0];
    footprints = comp0.footprints || [];
    w = comp0.width; h = comp0.height;
  } else if (data.mosaic && data.mosaic.footprints) {
    footprints = data.mosaic.footprints;
    w = data.mosaic.width; h = data.mosaic.height;
  }
  if (!footprints || !footprints.length) { overlay.innerHTML = ""; return; }
  overlay.setAttribute("viewBox", `0 0 ${w} ${h}`);
  overlay.innerHTML = footprints.map(item => {
    const corners = item.corners_mosaic || item.corners || [];
    const pts = corners.map(p => p.join ? p.join(",") : `${p[0]},${p[1]}`).join(" ");
    const label = item.image_id !== undefined ? `IMG${String(item.image_id+1).padStart(2,'0')}` : `IMG${String((item.image_index||0)+1).padStart(2,'0')}`;
    const sensor = item.sensor && item.sensor !== 'Unknown' ? ` [${item.sensor}]` : '';
    const lx = corners[0] ? corners[0][0] : 0;
    const ly = corners[0] ? corners[0][1] + 20 : 20;
    return `<polygon points="${pts}"/><text x="${lx}" y="${ly}">${label}${sensor}</text>`;
  }).join("");
}

function drawMatchPoints(data) {
  const overlay = $("matchPointsOverlay"), info = data.mosaic, points = info?.match_points || [];
  if (!info?.width || !points.length) { overlay.innerHTML = ""; $("matchLegend").hidden = true; return; }
  overlay.setAttribute("viewBox", `0 0 ${info.width} ${info.height}`);
  overlay.innerHTML = points.map(point => `<circle cx="${point.x}" cy="${point.y}" r="2.3"/>`).join("");
  overlay.classList.toggle("hidden", !$("matchPointsToggle").checked); $("matchLegend").hidden = false;
}

function drawPairTable(data) {
  const rows = data.graph.edges.map((edge, index) => {
    const m = edge.metrics || {}, status = edge.accepted ? "accepted" : "rejected";
    return `<tr class="${status}" data-edge-index="${index}" tabindex="0" title="${escapeHtml(edge.reason || "Click to inspect this pair")}">
      <td>I${edge.image_a + 1} ↔ I${edge.image_b + 1}${edge.reason ? `<small>${escapeHtml(edge.reason)}</small>` : ""}</td>
      <td>${status.toUpperCase()}</td>
      <td>${m.inlier_count ?? "—"}</td>
      <td>${m.inlier_ratio == null ? "—" : `${(m.inlier_ratio * 100).toFixed(1)}%`}</td>
      <td>${m.rmse_pixels == null ? "—" : `${fmt(m.rmse_pixels, 3)} px`}</td>
      <td>${m.source_spatial_coverage == null ? "—" : `${(m.source_spatial_coverage * 100).toFixed(0)}%`}</td>
      <td>${edge.ecc_used ? "USED" : "—"}</td>
      <td>${edge.cache_hit ? "HIT" : "MISS"}</td>
    </tr>`;
  }).join("");
  const tbody = $("pairTable").querySelector("tbody");
  tbody.innerHTML = rows || '<tr><td colspan="8">No candidate pairs.</td></tr>';
  tbody.querySelectorAll("[data-edge-index]").forEach(row => {
    const activate = () => inspectPair(data, data.graph.edges[Number(row.dataset.edgeIndex)]);
    row.addEventListener("click", activate);
    row.addEventListener("keydown", event => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); activate(); } });
  });
}

function inspectPair(data, edge) {
  if (!edge) return;
  document.querySelectorAll("#pairTable tbody tr").forEach(row => row.classList.remove("selected"));
  const matching = [...document.querySelectorAll("#pairTable tbody tr")].find(row => row.textContent.includes(`I${edge.image_a + 1} ↔ I${edge.image_b + 1}`));
  if (matching) matching.classList.add("selected");
  $("pairInspector").classList.remove("hidden");
  const m = edge.metrics || {};
  $("inspectorStatus").textContent = `IMAGE ${String(edge.image_a + 1).padStart(2,"0")} ↔ IMAGE ${String(edge.image_b + 1).padStart(2,"0")} · ${(edge.status || (edge.accepted ? "accepted" : "rejected")).toUpperCase()}`;
  $("inspectorMetrics").innerHTML = `
    <div class="inspector-metric-grid">
      <div><span>INLIERS</span><b>${m.inlier_count ?? "—"}</b></div>
      <div><span>INLIER RATIO</span><b>${m.inlier_ratio == null ? "—" : (m.inlier_ratio * 100).toFixed(1) + "%"}</b></div>
      <div><span>RMSE</span><b>${m.rmse_pixels == null ? "—" : fmt(m.rmse_pixels,3) + " px"}</b></div>
      <div><span>COVERAGE</span><b>${m.source_spatial_coverage == null ? "—" : (m.source_spatial_coverage * 100).toFixed(1) + "%"}</b></div>
    </div>
    <p class="inspector-reason">${escapeHtml(edge.reason || "Pair passed the configured acceptance criteria.")}</p>
    <p class="inspector-reason">Candidate score: ${edge.screening_score == null ? "—" : Number(edge.screening_score).toFixed(3)} · ${edge.cache_hit ? "reused cached evidence" : "computed this run"}</p>`;
  renderSpatialGrid(m.source_spatial_grid || m.reference_spatial_grid);
  const pair = data.pairs?.[`${edge.image_a}-${edge.image_b}`] || data.pairs?.[`${edge.image_b}-${edge.image_a}`];
  if (pair?.match_visualization) {
    $("edgeDetails").innerHTML = `<b>PAIR INSPECTOR ACTIVE</b><br>${escapeHtml(pair.match_visualization)}<br><a href="${pair.match_visualization}" target="_blank">OPEN CORRESPONDENCE IMAGE ↗</a>`;
  }
  $("pairInspector").scrollIntoView({ behavior:"smooth", block:"nearest" });
}

function renderSpatialGrid(grid) {
  const target = $("spatialGrid");
  if (!grid) { target.innerHTML = '<span class="chart-empty">No spatial grid available.</span>'; return; }
  const counts = grid.counts || [];
  const max = Math.max(1, ...counts.flat());
  target.innerHTML = counts.flat().map((value, index) => {
    const intensity = value / max;
    return `<div class="spatial-cell" title="Cell ${index + 1}: ${value} inliers" style="--cell-alpha:${0.15 + intensity * 0.85}"><b>${value}</b></div>`;
  }).join("");
  $("spatialGridNote").textContent = `${grid.populated_cells}/${grid.total_cells} cells populated · ${grid.total_inliers} inliers`;
}

function renderDiagnostics(data) {
  const t = data.summary.stage_timings_seconds || {};
  $("stageTimings").innerHTML = Object.entries(t).map(([key,value]) => `<div class="diag-row"><span>${key.replaceAll("_"," ")}</span><b>${fmt(value,3)} s</b></div>`).join("") || "N/A";
  const c = data.summary.cache || {};
  $("cacheDiagnostics").innerHTML = [
    ["feature hits", c.feature_hits ?? 0], ["feature misses", c.feature_misses ?? 0],
    ["pair hits", c.pair_hits ?? 0], ["pair misses", c.pair_misses ?? 0],
    ["processed pairs", data.summary.processed_pairs ?? 0], ["reused pairs", data.summary.reused_pairs ?? 0]
  ].map(([k,v]) => `<div class="diag-row"><span>${k}</span><b>${v}</b></div>`).join("");
}

function renderSensorSummary(data) {
  const groups = {};
  (data.images || []).forEach(item => { const key = item.sensor || "Unknown / Not provided"; groups[key] = (groups[key] || 0) + 1; });
  $("sensorSummary").innerHTML = Object.entries(groups).map(([sensor,count]) => `<div class="sensor-chip"><b>${count}</b><span>${escapeHtml(sensor)}</span></div>`).join("") || "N/A";
}

function updateMapScale() { $("mosaicStage").style.setProperty("--map-scale", multiState.scale); }

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, char => ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;" }[char]));
}

$("robustnessBtn").addEventListener("click", async () => {
  if (!state.source) { toast("Select a source image first."); return; }
  const panel = $("robustnessPanel");
  panel.classList.remove("hidden");
  $("robustnessStatus").textContent = "RUNNING CONTROLLED STRESS TEST…";
  $("robustnessTable").innerHTML = '<div class="loader-line"></div>';
  const fd = new FormData();
  fd.append("image", state.source);
  fd.append("detector", $("detector").value);
  fd.append("ratio", $("ratio").value);
  fd.append("ransac_threshold", $("ransac").value);
  fd.append("illumination_normalization", $("illumination").checked);
  fd.append("spatial_distribution", $("distribution").checked);
  fd.append("ecc_refinement", $("ecc").checked);
  fd.append("max_features", "8000");
  const button = $("robustnessBtn");
  button.disabled = true;
  try {
    const data = await safeFetchJSON("/api/evaluation/synthetic-robustness", { method:"POST", body:fd });
    $("robustnessPassed").textContent = `${data.passed_cases}/${data.total_cases}`;
    $("robustnessTotal").textContent = data.total_cases;
    $("robustnessTime").textContent = `${fmt(data.elapsed_seconds,2)} s`;
    $("robustnessStatus").textContent = "COMPLETE · SYNTHETIC ONLY";
    $("robustnessTable").innerHTML = `<table><thead><tr><th>SCENARIO</th><th>STATUS</th><th>INLIERS</th><th>RATIO</th><th>RMSE</th><th>COVERAGE</th></tr></thead><tbody>${
      data.results.map(r => `<tr><td>${escapeHtml(r.scenario)}</td><td class="${r.status.toLowerCase()}">${r.status}</td><td>${r.inlier_count}</td><td>${(r.inlier_ratio*100).toFixed(1)}%</td><td>${r.rmse_pixels == null ? "—" : fmt(r.rmse_pixels,3)+" px"}</td><td>${(r.spatial_coverage*100).toFixed(0)}%</td></tr>`).join("")
    }</tbody></table>`;
    toast(`Synthetic robustness lab complete: ${data.passed_cases}/${data.total_cases} cases passed.`);
  } catch (error) {
    $("robustnessStatus").textContent = "ERROR";
    $("robustnessTable").textContent = error.message;
    toast(error.message);
  } finally {
    button.disabled = false;
  }
});

$("zoomIn").addEventListener("click", () => { multiState.scale = Math.min(3, multiState.scale + .2); updateMapScale(); });
$("zoomOut").addEventListener("click", () => { multiState.scale = Math.max(.5, multiState.scale - .2); updateMapScale(); });
$("zoomReset").addEventListener("click", () => { multiState.scale = 1; updateMapScale(); });
$("footprintsToggle").addEventListener("change", event => $("footprintsOverlay").classList.toggle("hidden", !event.target.checked));
$("matchPointsToggle").addEventListener("change", event => $("matchPointsOverlay").classList.toggle("hidden", !event.target.checked));


// -------------------- Registered Mosaic 3D Explorer --------------------
// This renderer deliberately uses the actual generated mosaic texture.  Without
// valid lunar georeferencing it presents a relative terrain patch rather than
// inventing a latitude/longitude placement.
const terrainState = { gl: null, program: null, texture: null, image: null, data: null, yaw: 0.35, pitch: -0.35, distance: 3.0, dragging: false, lastX: 0, lastY: 0, auto: false, exaggeration: 0.22 };

function mat4Perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
  return new Float32Array([f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,(2*far*near)*nf,0]);
}
function mat4Mul(a,b) {
  const o = new Float32Array(16);
  for (let c=0;c<4;c++) for (let r=0;r<4;r++) o[c*4+r] =
    a[r]*b[c*4] + a[4+r]*b[c*4+1] + a[8+r]*b[c*4+2] + a[12+r]*b[c*4+3];
  return o;
}
function mat4Translate(z) {
  const m = new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,z,1]); return m;
}
function mat4RotateX(a) {
  const c=Math.cos(a),s=Math.sin(a); return new Float32Array([1,0,0,0,0,c,s,0,0,-s,c,0,0,0,0,1]);
}
function mat4RotateY(a) {
  const c=Math.cos(a),s=Math.sin(a); return new Float32Array([c,0,-s,0,0,1,0,0,s,0,c,0,0,0,0,1]);
}
function shader(gl,type,source) { const s=gl.createShader(type); gl.shaderSource(s,source); gl.compileShader(s); if(!gl.getShaderParameter(s,gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; }

function initTerrainRenderer() {
  const canvas=$("terrainCanvas");
  const gl=canvas.getContext("webgl2",{antialias:true,alpha:false});
  if(!gl){ $("explorerStatus").textContent="WEBGL2 UNAVAILABLE"; return false; }
  const vs=`#version 300 es
  precision highp float;
  layout(location=0) in vec2 aPos;
  layout(location=1) in vec2 aUv;
  uniform mat4 uMvp;
  uniform sampler2D uTex;
  uniform float uExaggeration;
  uniform bool uDisplace;
  out vec2 vUv;
  void main(){
    float h=texture(uTex,aUv).r;
    float z=(uDisplace ? (h-0.5)*uExaggeration : 0.0);
    vUv=aUv;
    gl_Position=uMvp*vec4(aPos.x,z,aPos.y,1.0);
  }`;
  const fs=`#version 300 es
  precision highp float;
  in vec2 vUv;
  uniform sampler2D uTex;
  uniform bool uGrid;
  out vec4 outColor;
  void main(){
    vec4 c=texture(uTex,vUv);
    if(uGrid){
      vec2 g=abs(fract(vUv*16.0)-0.5);
      float line=1.0-smoothstep(0.47,0.5,max(g.x,g.y));
      c.rgb=mix(c.rgb,vec3(0.72,0.76,0.80),line*0.22);
    }
    outColor=vec4(c.rgb,1.0);
  }`;
  const program=gl.createProgram(); gl.attachShader(program,shader(gl,gl.VERTEX_SHADER,vs)); gl.attachShader(program,shader(gl,gl.FRAGMENT_SHADER,fs)); gl.linkProgram(program);
  if(!gl.getProgramParameter(program,gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  const N=96, verts=[], inds=[];
  for(let y=0;y<=N;y++) for(let x=0;x<=N;x++){ const u=x/N,v=y/N; verts.push((u-.5)*2.7,(v-.5)*2.7,u,v); }
  for(let y=0;y<N;y++) for(let x=0;x<N;x++){ const i=y*(N+1)+x; inds.push(i,i+1,i+N+1,i+1,i+N+2,i+N+1); }
  const vao=gl.createVertexArray(), vb=gl.createBuffer(), ib=gl.createBuffer();
  gl.bindVertexArray(vao); gl.bindBuffer(gl.ARRAY_BUFFER,vb); gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(verts),gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0,2,gl.FLOAT,false,16,0);
  gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1,2,gl.FLOAT,false,16,8);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,ib); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,new Uint32Array(inds),gl.STATIC_DRAW);
  terrainState.gl=gl; terrainState.program=program; terrainState.vao=vao; terrainState.indexCount=inds.length;
  gl.enable(gl.DEPTH_TEST); gl.clearColor(0.018,0.02,0.024,1);
  resizeTerrainCanvas();
  return true;
}
function resizeTerrainCanvas(){
  const c=$("terrainCanvas"); if(!c) return; const dpr=Math.min(2,window.devicePixelRatio||1), r=c.getBoundingClientRect();
  c.width=Math.max(1,Math.floor(r.width*dpr)); c.height=Math.max(1,Math.floor(r.height*dpr));
  if(terrainState.gl) terrainState.gl.viewport(0,0,c.width,c.height);
}
function loadTerrainTexture(url){
  if(!terrainState.gl && !initTerrainRenderer()) return;
  const img=new Image(); img.crossOrigin="anonymous";
  img.onload=()=>{ terrainState.image=img; const gl=terrainState.gl; const tex=gl.createTexture(); terrainState.texture=tex;
    gl.bindTexture(gl.TEXTURE_2D,tex); gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,true); gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,img);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR_MIPMAP_LINEAR); gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR); gl.generateMipmap(gl.TEXTURE_2D);
    $("terrainTextureStatus").textContent=`MOSAIC ${img.naturalWidth}×${img.naturalHeight}`; requestAnimationFrame(renderTerrain);
  };
  img.onerror=()=>{ $("terrainTextureStatus").textContent="TEXTURE LOAD FAILED"; };
  img.src=url;
}
function renderTerrain(now=0){
  const gl=terrainState.gl; if(!gl||!terrainState.texture) return;
  if(terrainState.auto) terrainState.yaw += 0.00035*(now-(terrainState.lastFrame||now)); terrainState.lastFrame=now;
  const c=$("terrainCanvas"), aspect=c.width/Math.max(1,c.height);
  const p=mat4Perspective(Math.PI/4,aspect,.1,100), view=mat4Translate(-terrainState.distance);
  const rot=mat4Mul(mat4RotateY(terrainState.yaw),mat4RotateX(terrainState.pitch));
  const mvp=mat4Mul(p,mat4Mul(view,rot));
  gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT); gl.useProgram(terrainState.program); gl.bindVertexArray(terrainState.vao); gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D,terrainState.texture);
  gl.uniformMatrix4fv(gl.getUniformLocation(terrainState.program,"uMvp"),false,mvp); gl.uniform1i(gl.getUniformLocation(terrainState.program,"uTex"),0);
  gl.uniform1f(gl.getUniformLocation(terrainState.program,"uExaggeration"),terrainState.exaggeration);
  gl.uniform1i(gl.getUniformLocation(terrainState.program,"uDisplace"),$("terrainDisplacement").checked?1:0);
  gl.uniform1i(gl.getUniformLocation(terrainState.program,"uGrid"),$("terrainGrid").checked?1:0);
  gl.drawElements(gl.TRIANGLES,terrainState.indexCount,gl.UNSIGNED_INT,0);
  requestAnimationFrame(renderTerrain);
}
function open3DExplorer(){
  if(!multiState.result?.outputs?.mosaic){ toast("A generated mosaic is required before opening the 3D explorer."); return; }
  $("explorer3D").classList.remove("hidden");
  $("explorer3D").scrollIntoView({behavior:"smooth",block:"start"});
  const base=window.location.origin; loadTerrainTexture(base+multiState.result.outputs.mosaic);
}
$("open3DBtn").addEventListener("click",open3DExplorer);
$("terrainExaggeration").addEventListener("input",e=>{terrainState.exaggeration=Number(e.target.value);$("terrainExaggerationValue").textContent=`${e.target.value}×`;});
$("terrainDisplacement").addEventListener("change",()=>requestAnimationFrame(renderTerrain));
$("terrainGrid").addEventListener("change",()=>requestAnimationFrame(renderTerrain));
$("terrainReset").addEventListener("click",()=>{terrainState.yaw=.35;terrainState.pitch=-.35;terrainState.distance=3;});
$("terrainTop").addEventListener("click",()=>{terrainState.pitch=-Math.PI/2+0.03;terrainState.yaw=0;});
$("terrainOrbit").addEventListener("click",()=>{terrainState.auto=!terrainState.auto;$("terrainOrbit").textContent=terrainState.auto?"STOP ORBIT":"AUTO ORBIT";});
$("terrainCanvas").addEventListener("pointerdown",e=>{terrainState.dragging=true;terrainState.lastX=e.clientX;terrainState.lastY=e.clientY;$("terrainCanvas").setPointerCapture(e.pointerId);});
$("terrainCanvas").addEventListener("pointerup",()=>terrainState.dragging=false);
$("terrainCanvas").addEventListener("pointermove",e=>{
  if(terrainState.dragging){const dx=e.clientX-terrainState.lastX,dy=e.clientY-terrainState.lastY;terrainState.yaw+=dx*.008;terrainState.pitch=Math.max(-1.5,Math.min(1.5,terrainState.pitch+dy*.008));terrainState.lastX=e.clientX;terrainState.lastY=e.clientY;}
  const r=e.currentTarget.getBoundingClientRect(),u=Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)),v=Math.max(0,Math.min(1,1-(e.clientY-r.top)/r.height));
  $("terrainX").textContent=`${((u-.5)*2.7).toFixed(3)} rel`; $("terrainY").textContent=`${((v-.5)*2.7).toFixed(3)} rel`; $("terrainZ").textContent=terrainState.exaggeration.toFixed(2)+" rel";
});
$("terrainCanvas").addEventListener("wheel",e=>{e.preventDefault();terrainState.distance=Math.max(1.7,Math.min(6,terrainState.distance+e.deltaY*.002));},{passive:false});
window.addEventListener("resize",resizeTerrainCanvas);

document.addEventListener('DOMContentLoaded', () => Inspector.init());

document.addEventListener('DOMContentLoaded', () => {
  const btn = document.getElementById('loadDiagnosticsBtn');
  if(btn) {
    btn.addEventListener('click', async () => {
      try {
        btn.textContent = 'LOADING...';
        const data = await safeFetchJSON('/api/runtime/status');
        
        document.getElementById('execDevice').textContent = data.execution_device || 'CPU';
        document.getElementById('sysInfo').textContent = (data.libraries && data.libraries.os) ? data.libraries.os : 'Unknown OS';
        
        const caps = data.capabilities || [];
        const capsHtml = caps.map(c => `<li><span>${c.name}</span><span>${c.available ? 'Available' : 'NOT AVAILABLE'}</span></li>`).join('');
        document.getElementById('diagCaps').innerHTML = capsHtml;
        
        if (data.synthetic_consensus) {
          const sc = data.synthetic_consensus;
          if (sc.rmse !== null) document.getElementById('synthConsensusRmse').textContent = sc.rmse.toFixed(3) + ' px';
          if (sc.median !== null) document.getElementById('synthConsensusMedian').textContent = sc.median.toFixed(3) + ' px';
          if (sc.max !== null) document.getElementById('synthConsensusMax').textContent = sc.max.toFixed(3) + ' px';
          document.getElementById('synthTaylorFailures').textContent = `${sc.taylor_failures} / ${sc.total_cases}`;
        }
        
        btn.textContent = 'REFRESH TELEMETRY';
      } catch(e) {
        btn.textContent = 'ERROR LOADING';
      }
    });
    
    // Auto load on start
    btn.click();
  }
});
