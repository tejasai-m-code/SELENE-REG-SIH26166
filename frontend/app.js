const $ = (id) => document.getElementById(id);

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

  input.addEventListener("change", () => {
    if (input.files[0]) {
      state[key] = input.files[0];
      name.textContent = input.files[0].name;
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
  $("registeredImage").src = base + data.outputs.registered_image;
  $("matchesImage").src = base + data.outputs.match_visualization;
  $("registeredLink").href = base + data.outputs.registered_image;
  $("matchesLink").href = base + data.outputs.match_visualization;

  $("homography").textContent = data.homography
    .map(row => row.map(v => Number(v).toFixed(6)).join("   "))
    .join("\n");

  $("pipeline").innerHTML = data.pipeline.map(x => `<li>${x}</li>`).join("");
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
    const response = await fetch("/api/register", { method: "POST", body: fd });
    const data = await response.json();

    if (!response.ok) throw new Error(data.detail || "Registration failed.");

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
