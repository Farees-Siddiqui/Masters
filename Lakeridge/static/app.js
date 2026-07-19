const $ = (id) => document.getElementById(id);

let selectedFile = null;

async function init() {
  try {
    const cfg = await (await fetch("/config")).json();
    $("concepts").value = cfg.default_concepts.join(", ");
    $("deviceChip").textContent = `device: ${cfg.device}`;
    renderChipHints(cfg.default_concepts);
  } catch (e) {
    $("hintLine").textContent = "Could not reach backend.";
  }
}

function renderChipHints(concepts) {
  const wrap = $("chipHints");
  wrap.innerHTML = "";
  concepts.slice(0, 8).forEach((c) => {
    const t = document.createElement("span");
    t.className = "tag";
    t.textContent = "+ " + c;
    t.onclick = () => {
      const cur = $("concepts").value.trim();
      const parts = cur ? cur.split(",").map((s) => s.trim()) : [];
      if (!parts.includes(c)) parts.push(c);
      $("concepts").value = parts.join(", ");
    };
    wrap.appendChild(t);
  });
}

// --- file handling ---
const dz = $("dropzone");
$("fileInput").addEventListener("change", (e) => setFile(e.target.files[0]));
dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("drag"); });
dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
dz.addEventListener("drop", (e) => {
  e.preventDefault();
  dz.classList.remove("drag");
  if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
});

function setFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (ev) => {
    $("dzInner").innerHTML = `<img src="${ev.target.result}" alt="preview" />`;
    dz.classList.add("has-file");
  };
  reader.readAsDataURL(file);
  $("analyzeBtn").disabled = false;
  $("hintLine").textContent = file.name;
}

$("threshold").addEventListener("input", (e) => {
  $("thrVal").textContent = Number(e.target.value).toFixed(2);
});

// --- analyze ---
$("analyzeBtn").addEventListener("click", analyze);

async function analyze() {
  if (!selectedFile) return;
  const btn = $("analyzeBtn");
  btn.disabled = true;
  showSpinner(
    true,
    $("narrative").checked ? "Segmenting + writing incident report…" : "Running SAM 3 segmentation…"
  );

  const fd = new FormData();
  fd.append("image", selectedFile);
  fd.append("concepts", $("concepts").value);
  fd.append("threshold", $("threshold").value);
  fd.append("narrative", $("narrative").checked ? "true" : "false");

  try {
    const res = await fetch("/analyze", { method: "POST", body: fd });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || "Request failed");
    }
    renderResult(await res.json());
  } catch (e) {
    $("spinText").textContent = "Error: " + e.message;
    setTimeout(() => showSpinner(false), 2500);
  } finally {
    btn.disabled = false;
  }
}

function showSpinner(on, text) {
  $("spinner").classList.toggle("hidden", !on);
  if (on) $("spinText").textContent = text || "Running SAM 3 segmentation…";
}

function renderReport(report, accident) {
  const el = $("report");
  if (!report) { el.classList.add("hidden"); return; }
  el.classList.remove("hidden");
  const sev = (report.severity || "moderate").toLowerCase();
  const actions = (report.immediate_actions || []).map((a) => `<li>${escapeHtml(a)}</li>`).join("");
  const evidence = (report.key_evidence || []).map((e) => `<li>${escapeHtml(e)}</li>`).join("");
  const icon = accident && accident.icon ? accident.icon : "🧭";
  const srcNote =
    report.source === "fallback"
      ? `<div class="src fallback">⚠ VLM unavailable — heuristic fallback used${report.error ? ": " + escapeHtml(report.error) : ""}</div>`
      : `<div class="src">Generated on-device by Qwen2.5-VL-7B · fused with SAM 3 detections · no data left the machine</div>`;

  el.innerHTML = `
    <div class="rhead">
      <span style="font-size:22px">${icon}</span>
      <span class="rtype">${escapeHtml(report.accident_type || "Incident")}</span>
      <span class="badge sev-${sev}">${escapeHtml(report.severity || "Moderate")}</span>
      <span class="rconf">assessment confidence <b>${Math.round((report.confidence || 0) * 100)}%</b></span>
    </div>
    <p class="rnarr">${escapeHtml(report.narrative || "")}</p>
    <div class="rcols">
      <div><h4>Immediate actions</h4><ul>${actions || "<li>—</li>"}</ul></div>
      <div><h4>Key visual evidence</h4><ul>${evidence || "<li>—</li>"}</ul></div>
    </div>
    ${srcNote}`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

function renderResult(data) {
  showSpinner(false);
  // overlay
  $("placeholder").classList.add("hidden");
  const img = $("overlay");
  img.src = data.overlay;
  img.classList.remove("hidden");

  // verdict banner (fast heuristic headline)
  const v = $("verdict");
  const a = data.accident;
  v.classList.remove("hidden");
  v.innerHTML = `
    <div class="vic">${a.icon}</div>
    <div>
      <div class="vlabel">${a.label}</div>
      <div class="vrat">${a.rationale}</div>
    </div>
    <div class="vconf">objects<b>${data.detections.length}</b></div>`;

  // rich VLM incident report
  renderReport(data.report, a);

  // summary
  const sum = $("summary");
  sum.innerHTML = "";
  data.summary.forEach((s) => {
    const row = document.createElement("div");
    row.className = "sum-row";
    row.innerHTML = `<span class="sw" style="background:${s.color}"></span>
      <span>${s.label}</span><span class="cnt">×${s.count}</span>`;
    sum.appendChild(row);
  });

  // detection list
  const list = $("detList");
  list.innerHTML = "";
  if (!data.detections.length) {
    list.innerHTML = `<p class="hint">No objects matched. Try lowering the confidence threshold or adding concepts.</p>`;
  }
  data.detections.forEach((d) => {
    const el = document.createElement("div");
    el.className = "det";
    el.innerHTML = `<span class="sw" style="background:${d.color}"></span>
      <span class="lab">${d.label}</span>
      <span class="sc">${Math.round(d.score * 100)}%</span>`;
    list.appendChild(el);
  });

  // meta
  $("meta").innerHTML =
    `${data.detections.length} instances · ${data.summary.length} classes<br />` +
    `concepts: ${data.concepts_used.length} · ${data.elapsed_sec}s on ${data.device}`;
}

init();
