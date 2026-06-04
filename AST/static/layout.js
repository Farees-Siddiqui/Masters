const fileInput = document.getElementById("file");
const granSelect = document.getElementById("granularity");
const goBtn = document.getElementById("go");
const statusEl = document.getElementById("status");

const pager = document.getElementById("pager");
const prevBtn = document.getElementById("prev");
const nextBtn = document.getElementById("next");
const pageIndicator = document.getElementById("page-indicator");
const boxCountEl = document.getElementById("box-count");

const emptyEl = document.getElementById("empty");
const canvasEl = document.getElementById("canvas");
const pageImg = document.getElementById("page-img");
const overlay = document.getElementById("overlay");

const boxInfoEl = document.getElementById("box-info");
const legendEl = document.getElementById("legend");
const legendHint = document.getElementById("legend-hint");

const SVG_NS = "http://www.w3.org/2000/svg";

// Distinct palette (mirrors layout/draw.py in spirit; consistency is per-session).
const PALETTE = [
  "#e6194b", "#3cb44b", "#0082c8", "#f58231", "#911eb4", "#009e73",
  "#f032e6", "#aa6e28", "#008080", "#800000", "#4682b4", "#bc8000",
  "#800080", "#000080", "#d2691e", "#556b2f",
];
const UNKNOWN_COLOR = "#6e6e6e";

let doc = null;          // server response
let pageIdx = 0;         // current page index
let selectedBox = null;  // index of selected box on current page
const hiddenLabels = new Set();

/* ---------- color mapping ---------- */
function crc32(str) {
  let c, crc = 0 ^ -1;
  for (let i = 0; i < str.length; i++) {
    c = (crc ^ str.charCodeAt(i)) & 0xff;
    for (let k = 0; k < 8; k++) c = c & 1 ? (c >>> 1) ^ 0xedb88320 : c >>> 1;
    crc = (crc >>> 8) ^ c;
  }
  return (crc ^ -1) >>> 0;
}
function colorFor(label) {
  if (!label) return UNKNOWN_COLOR;
  return PALETTE[crc32(label) % PALETTE.length];
}

/* ---------- analyze ---------- */
goBtn.addEventListener("click", async () => {
  const f = fileInput.files?.[0];
  if (!f) {
    statusEl.textContent = "Pick a PDF first.";
    return;
  }
  const granularity = granSelect.value;
  goBtn.disabled = true;
  statusEl.textContent = `Running ${granularity} layout analysis… (first run downloads models)`;
  resetView();

  const form = new FormData();
  form.append("file", f);
  form.append("granularity", granularity);

  try {
    const res = await fetch("/api/layout", { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    doc = await res.json();
    statusEl.textContent = `${doc.filename} — ${doc.page_count} page(s), granularity: ${doc.granularity}`;
    pageIdx = 0;
    hiddenLabels.clear();
    renderPage();
  } catch (e) {
    statusEl.textContent = `Error: ${e.message}`;
  } finally {
    goBtn.disabled = false;
  }
});

function resetView() {
  doc = null;
  selectedBox = null;
  canvasEl.hidden = true;
  emptyEl.hidden = false;
  pager.hidden = true;
  overlay.innerHTML = "";
  boxInfoEl.innerHTML = '<p class="muted">Click a bounding box to inspect it.</p>';
  legendEl.innerHTML = "";
  legendHint.textContent = "";
}

/* ---------- page navigation ---------- */
prevBtn.addEventListener("click", () => { if (pageIdx > 0) { pageIdx--; renderPage(); } });
nextBtn.addEventListener("click", () => {
  if (doc && pageIdx < doc.pages.length - 1) { pageIdx++; renderPage(); }
});

function renderPage() {
  if (!doc) return;
  const page = doc.pages[pageIdx];
  selectedBox = null;

  emptyEl.hidden = true;
  canvasEl.hidden = false;
  pager.hidden = false;

  pageIndicator.textContent = `Page ${page.page} / ${doc.page_count}`;
  prevBtn.disabled = pageIdx === 0;
  nextBtn.disabled = pageIdx === doc.pages.length - 1;
  const withText = page.boxes.filter((b) => b.text).length;
  boxCountEl.textContent = `${page.boxes.length} ${doc.granularity} box(es) · ${withText} with text`;

  pageImg.src = page.image_url;
  overlay.setAttribute("viewBox", `0 0 ${page.width} ${page.height}`);
  drawOverlay(page);
  buildLegend(page);
  boxInfoEl.innerHTML = '<p class="muted">Click a bounding box to inspect it.</p>';
}

function drawOverlay(page) {
  overlay.innerHTML = "";
  page.boxes.forEach((b, i) => {
    if (hiddenLabels.has(b.label || "")) return;
    const [x1, y1, x2, y2] = b.bbox;
    const rect = document.createElementNS(SVG_NS, "rect");
    rect.setAttribute("x", x1);
    rect.setAttribute("y", y1);
    rect.setAttribute("width", Math.max(0, x2 - x1));
    rect.setAttribute("height", Math.max(0, y2 - y1));
    rect.setAttribute("class", "lbox");
    rect.setAttribute("stroke", colorFor(b.label));
    rect.dataset.idx = i;
    rect.addEventListener("click", (e) => {
      e.stopPropagation();
      selectBox(i);
    });
    overlay.appendChild(rect);
  });
  highlightSelected();
}

function selectBox(i) {
  selectedBox = i;
  highlightSelected();
  showBoxInfo(doc.pages[pageIdx].boxes[i]);
}

function highlightSelected() {
  overlay.querySelectorAll(".lbox.selected").forEach((r) => r.classList.remove("selected"));
  if (selectedBox != null) {
    const r = overlay.querySelector(`.lbox[data-idx="${selectedBox}"]`);
    if (r) {
      r.classList.add("selected");
      // raise to top so its outline isn't covered by neighbours
      overlay.appendChild(r);
    }
  }
}

/* ---------- box inspector ---------- */
function showBoxInfo(b) {
  const dl = document.createElement("dl");
  const color = colorFor(b.label);
  appendKV(dl, "label", `<span class="pill" style="background:${color}">${escapeHtml(b.label ?? "(none)")}</span>`, true);
  appendKV(dl, "cls_id", b.cls_id ?? "—");
  appendKV(dl, "bbox", `[${b.bbox.map((v) => Math.round(v)).join(", ")}]`);
  appendKV(dl, "size", `${Math.round(b.bbox[2] - b.bbox[0])} × ${Math.round(b.bbox[3] - b.bbox[1])} px`);
  appendKV(dl, "det score", fmtScore(b.score));
  appendKV(dl, "text score", fmtScore(b.text_score));

  boxInfoEl.innerHTML = "";
  boxInfoEl.appendChild(dl);

  const h = document.createElement("dt");
  h.textContent = "text";
  h.className = "block-title";
  const body = document.createElement("div");
  body.className = "kv-block";
  body.textContent = b.text || "(no text)";
  boxInfoEl.appendChild(h);
  boxInfoEl.appendChild(body);
}

function fmtScore(s) {
  return s == null ? "—" : Number(s).toFixed(4);
}

/* ---------- legend (click to toggle a label's boxes) ---------- */
function buildLegend(page) {
  const labels = [...new Set(page.boxes.map((b) => b.label || ""))].sort();
  legendHint.textContent = "(click to toggle)";
  legendEl.innerHTML = "";
  labels.forEach((label) => {
    const item = document.createElement("button");
    item.className = "legend-item" + (hiddenLabels.has(label) ? " off" : "");
    const sw = document.createElement("span");
    sw.className = "swatch";
    sw.style.background = colorFor(label || null);
    const txt = document.createElement("span");
    txt.textContent = label || "(none)";
    const n = page.boxes.filter((b) => (b.label || "") === label).length;
    const cnt = document.createElement("span");
    cnt.className = "count";
    cnt.textContent = n;
    item.append(sw, txt, cnt);
    item.addEventListener("click", () => {
      if (hiddenLabels.has(label)) hiddenLabels.delete(label);
      else hiddenLabels.add(label);
      item.classList.toggle("off");
      drawOverlay(page);
    });
    legendEl.appendChild(item);
  });
}

/* ---------- helpers ---------- */
function appendKV(dl, k, v, isHtml = false) {
  const dt = document.createElement("dt");
  dt.textContent = k;
  const dd = document.createElement("dd");
  if (isHtml) dd.innerHTML = v;
  else dd.textContent = v;
  dl.appendChild(dt);
  dl.appendChild(dd);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

// Keyboard paging.
window.addEventListener("keydown", (e) => {
  if (!doc) return;
  if (e.key === "ArrowLeft") prevBtn.click();
  if (e.key === "ArrowRight") nextBtn.click();
});
