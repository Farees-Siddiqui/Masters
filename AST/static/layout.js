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
const readingOrderToggle = document.getElementById("reading-order");

const SVG_NS = "http://www.w3.org/2000/svg";

// Distinct palette (mirrors layout/draw.py in spirit; consistency is per-session).
const PALETTE = [
  "#e6194b", "#3cb44b", "#0082c8", "#f58231", "#911eb4", "#009e73",
  "#f032e6", "#aa6e28", "#008080", "#800000", "#4682b4", "#bc8000",
  "#800080", "#000080", "#d2691e", "#556b2f",
];
const UNKNOWN_COLOR = "#6e6e6e";

let doc = null;          // { doc, filename, page_count, pages:[{page,width,height,image_url}] }
let pageIdx = 0;         // current page index
let selectedBox = null;  // index of selected box on current page
const hiddenLabels = new Set();
const boxCache = new Map();  // `${page}|${granularity}` -> boxes[]
let reqToken = 0;        // guards against out-of-order async box fetches

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
  goBtn.disabled = true;
  statusEl.textContent = "Rendering pages…";
  resetView();

  const form = new FormData();
  form.append("file", f);

  try {
    const res = await fetch("/api/layout/start", { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    doc = await res.json();
    statusEl.textContent = `${doc.filename} — ${doc.page_count} page(s)`;
    pageIdx = 0;
    hiddenLabels.clear();
    boxCache.clear();
    renderPage();
  } catch (e) {
    statusEl.textContent = `Error: ${e.message}`;
  } finally {
    goBtn.disabled = false;
  }
});

// Switching granularity re-fetches just the current page (lazy, cached).
granSelect.addEventListener("change", () => {
  if (doc) renderPage();
});

// Toggling reading order just redraws the current overlay (no refetch).
readingOrderToggle.addEventListener("change", () => {
  if (doc) drawOverlay(currentVisibleBoxes());
});

function currentVisibleBoxes() {
  return currentBoxes();
}

function resetView() {
  doc = null;
  selectedBox = null;
  boxCache.clear();
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

async function renderPage() {
  if (!doc) return;
  const page = doc.pages[pageIdx];
  const granularity = granSelect.value;
  selectedBox = null;

  emptyEl.hidden = true;
  canvasEl.hidden = false;
  pager.hidden = false;

  pageIndicator.textContent = `Page ${page.page} / ${doc.page_count}`;
  prevBtn.disabled = pageIdx === 0;
  nextBtn.disabled = pageIdx === doc.pages.length - 1;

  // Show the page image right away; boxes load (or come from cache) after.
  pageImg.src = page.image_url;
  overlay.setAttribute("viewBox", `0 0 ${page.width} ${page.height}`);
  overlay.innerHTML = "";
  boxInfoEl.innerHTML = '<p class="muted">Click a bounding box to inspect it.</p>';

  const key = `${page.page}|${granularity}`;
  const token = ++reqToken;

  let boxes = boxCache.get(key);
  if (!boxes) {
    boxCountEl.textContent = `Analyzing page ${page.page} (${granularity})…`;
    legendEl.innerHTML = "";
    legendHint.textContent = "";
    try {
      const url = `/api/layout/page?doc=${encodeURIComponent(doc.doc)}&page=${page.page}&granularity=${granularity}`;
      const res = await fetch(url);
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      boxes = (await res.json()).boxes;
      boxCache.set(key, boxes);
    } catch (e) {
      if (token === reqToken) boxCountEl.textContent = `Error: ${e.message}`;
      return;
    }
  }

  // A newer navigation/granularity change superseded this fetch — drop it.
  if (token !== reqToken) return;

  const withText = boxes.filter((b) => b.text).length;
  boxCountEl.textContent = `${boxes.length} ${granularity} box(es) · ${withText} with text`;
  drawOverlay(boxes);
  buildLegend(boxes);
}

function drawOverlay(boxes) {
  overlay.innerHTML = "";
  boxes.forEach((b, i) => {
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
  if (readingOrderToggle.checked) drawReadingOrder(boxes);
  highlightSelected();
}

/* ---------- reading-order overlay (flow path + numbered pills) ---------- */
function boxCenter(b) {
  return [(b.bbox[0] + b.bbox[2]) / 2, (b.bbox[1] + b.bbox[3]) / 2];
}

// Hue ramp green -> amber -> red, t in [0,1], so the flow shows progress.
function flowColor(t) {
  const stops = [
    [0x22, 0xc5, 0x5e],
    [0xf5, 0x9e, 0x0b],
    [0xef, 0x44, 0x44],
  ];
  t = Math.max(0, Math.min(1, t));
  const seg = t < 0.5 ? 0 : 1;
  const lt = t < 0.5 ? t / 0.5 : (t - 0.5) / 0.5;
  const a = stops[seg], b = stops[seg + 1];
  const m = a.map((v, k) => Math.round(v + (b[k] - v) * lt));
  return `rgb(${m[0]}, ${m[1]}, ${m[2]})`;
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function drawReadingOrder(boxes) {
  // Visible boxes that carry an order, sorted into the reading sequence.
  const seq = boxes
    .filter((b) => !hiddenLabels.has(b.label || "") && b.order != null)
    .sort((a, b) => a.order - b.order);
  if (seq.length === 0) return;

  // Scale glyphs to the page so they read at any zoom (viewBox is page pixels).
  const vb = overlay.viewBox.baseVal;
  const scale = Math.max(vb.width, vb.height) / 1000;
  const r = Math.max(8, 13 * scale);
  const flowW = Math.max(1.2, 1.8 * scale);
  const ringW = Math.max(1.4, 2 * scale);
  const denom = Math.max(1, seq.length - 1);

  ensureRoDefs();

  // Flow path first (under the pills): one short segment per step, tinted by
  // progress; arrowheads inherit each segment's color via context-stroke.
  for (let i = 1; i < seq.length; i++) {
    const [x1, y1] = boxCenter(seq[i - 1]);
    const [x2, y2] = boxCenter(seq[i]);
    const line = document.createElementNS(SVG_NS, "line");
    line.setAttribute("x1", x1);
    line.setAttribute("y1", y1);
    line.setAttribute("x2", x2);
    line.setAttribute("y2", y2);
    line.setAttribute("class", "ro-flow");
    line.setAttribute("stroke", flowColor((i - 1) / denom));
    line.setAttribute("stroke-width", flowW);
    line.setAttribute("marker-end", "url(#ro-arrowhead)");
    overlay.appendChild(line);
  }

  // Numbered pills tucked into each box's top-left corner.
  seq.forEach((b, idx) => {
    const cx = clamp(b.bbox[0] + r * 1.1, r, vb.width - r);
    const cy = clamp(b.bbox[1] + r * 1.1, r, vb.height - r);
    const g = document.createElementNS(SVG_NS, "g");
    g.setAttribute("class", "ro-badge");
    g.setAttribute("filter", "url(#ro-shadow)");
    const circle = document.createElementNS(SVG_NS, "circle");
    circle.setAttribute("cx", cx);
    circle.setAttribute("cy", cy);
    circle.setAttribute("r", r);
    circle.setAttribute("stroke", flowColor(idx / denom));
    circle.setAttribute("stroke-width", ringW);
    const text = document.createElementNS(SVG_NS, "text");
    text.setAttribute("x", cx);
    text.setAttribute("y", cy);
    text.setAttribute("font-size", r * 1.15);
    text.textContent = idx + 1; // 1-based for humans
    g.appendChild(circle);
    g.appendChild(text);
    overlay.appendChild(g);
  });
}

function ensureRoDefs() {
  if (overlay.querySelector("#ro-arrowhead")) return;
  const defs = document.createElementNS(SVG_NS, "defs");

  // Sleek notched arrowhead; fill = context-stroke so it matches its segment.
  const marker = document.createElementNS(SVG_NS, "marker");
  marker.setAttribute("id", "ro-arrowhead");
  marker.setAttribute("viewBox", "0 0 10 10");
  marker.setAttribute("refX", "8.5");
  marker.setAttribute("refY", "5");
  marker.setAttribute("markerWidth", "6");
  marker.setAttribute("markerHeight", "6");
  marker.setAttribute("orient", "auto-start-reverse");
  marker.setAttribute("markerUnits", "strokeWidth");
  const head = document.createElementNS(SVG_NS, "path");
  head.setAttribute("d", "M 0 0 L 10 5 L 0 10 L 3 5 z");
  head.setAttribute("fill", "context-stroke");
  marker.appendChild(head);
  defs.appendChild(marker);

  // Soft drop shadow shared by all pills.
  const filter = document.createElementNS(SVG_NS, "filter");
  filter.setAttribute("id", "ro-shadow");
  filter.setAttribute("x", "-50%");
  filter.setAttribute("y", "-50%");
  filter.setAttribute("width", "200%");
  filter.setAttribute("height", "200%");
  const ds = document.createElementNS(SVG_NS, "feDropShadow");
  ds.setAttribute("dx", "0");
  ds.setAttribute("dy", "1");
  ds.setAttribute("stdDeviation", "1.3");
  ds.setAttribute("flood-color", "#000");
  ds.setAttribute("flood-opacity", "0.55");
  filter.appendChild(ds);
  defs.appendChild(filter);

  overlay.appendChild(defs);
}

function currentBoxes() {
  if (!doc) return [];
  return boxCache.get(`${doc.pages[pageIdx].page}|${granSelect.value}`) || [];
}

function selectBox(i) {
  selectedBox = i;
  highlightSelected();
  showBoxInfo(currentBoxes()[i]);
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
function buildLegend(boxes) {
  const labels = [...new Set(boxes.map((b) => b.label || ""))].sort();
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
    const n = boxes.filter((b) => (b.label || "") === label).length;
    const cnt = document.createElement("span");
    cnt.className = "count";
    cnt.textContent = n;
    item.append(sw, txt, cnt);
    item.addEventListener("click", () => {
      if (hiddenLabels.has(label)) hiddenLabels.delete(label);
      else hiddenLabels.add(label);
      item.classList.toggle("off");
      drawOverlay(boxes);
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
