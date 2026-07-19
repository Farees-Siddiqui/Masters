/* Corpus explorer: a semantic map of the whole collection + concept tracing.
 * Both run on a server-side corpus index (MiniLM embeddings over every AST
 * segment). Every point / hit links to the aligner focused on its node. */

const SVG_NS = "http://www.w3.org/2000/svg";
const mapSvg = document.getElementById("map");
const legendEl = document.getElementById("legend");
const mapMeta = document.getElementById("map-meta");
const statusEl = document.getElementById("status");
const traceInput = document.getElementById("trace-q");
const traceBtn = document.getElementById("trace-btn");
const traceStatus = document.getElementById("trace-status");
const traceResults = document.getElementById("trace-results");

const PALETTE = [
  "#7aa2ff", "#ffb86b", "#9be7a3", "#ff8fb1", "#c5a3ff",
  "#6be0d9", "#ffd166", "#f78fb3", "#84d2f6", "#b5e48c",
];
const W = 1000, H = 660, pad = 26;

let mapData = null;
let activeCluster = null;  // legend filter

/* ---------- tabs ---------- */
document.querySelector(".corpus-tabs").addEventListener("click", (e) => {
  const b = e.target.closest("[data-tab]");
  if (!b) return;
  document.querySelectorAll(".corpus-tabs .tab").forEach((t) => t.classList.remove("active"));
  b.classList.add("active");
  const tab = b.dataset.tab;
  document.getElementById("panel-map").hidden = tab !== "map";
  document.getElementById("panel-trace").hidden = tab !== "trace";
  if (tab === "trace") traceInput.focus();
});

/* ---------- semantic map ---------- */
loadMap();

async function loadMap() {
  statusEl.textContent = "Projecting the corpus (embeddings + t-SNE)…";
  try {
    const res = await fetch("/api/corpus/map");
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `HTTP ${res.status}`);
    mapData = await res.json();
    statusEl.textContent = "";
    drawMap();
  } catch (e) {
    statusEl.textContent = `Error: ${e.message}`;
  }
}

function colorFor(c) { return PALETTE[c % PALETTE.length]; }

function drawMap() {
  mapSvg.innerHTML = "";
  if (!mapData || mapData.points.length === 0) {
    mapMeta.textContent = "No documents indexed yet — align a few PDFs first.";
    legendEl.innerHTML = "";
    return;
  }
  const xs = (v) => pad + v * (W - 2 * pad);
  const ys = (v) => pad + (1 - v) * (H - 2 * pad);

  for (const p of mapData.points) {
    const c = document.createElementNS(SVG_NS, "circle");
    c.setAttribute("cx", xs(p.x).toFixed(1));
    c.setAttribute("cy", ys(p.y).toFixed(1));
    c.setAttribute("r", 4.5);
    c.setAttribute("class", "dot");
    c.setAttribute("fill", colorFor(p.cluster));
    c.dataset.cluster = p.cluster;
    if (activeCluster != null && p.cluster !== activeCluster) c.classList.add("muted");
    const tip = document.createElementNS(SVG_NS, "title");
    tip.textContent = `${p.title} · ${p.node_id}\n${p.snippet}`;
    c.appendChild(tip);
    c.addEventListener("click", () =>
      window.open(`/align?doc=${encodeURIComponent(p.doc)}&focus=${encodeURIComponent(p.node_id)}`, "_blank"));
    mapSvg.appendChild(c);
  }

  // Legend / theme list (click to isolate a theme).
  legendEl.innerHTML = "";
  for (const cl of mapData.clusters) {
    const item = document.createElement("div");
    item.className = "legend-item" + (activeCluster === cl.cluster ? " on" : "");
    item.innerHTML =
      `<span class="swatch" style="background:${colorFor(cl.cluster)}"></span>` +
      `<span class="lbl"></span><span class="cnt">${cl.size}</span>`;
    item.querySelector(".lbl").textContent = cl.label;
    item.addEventListener("click", () => {
      activeCluster = activeCluster === cl.cluster ? null : cl.cluster;
      drawMap();
    });
    legendEl.appendChild(item);
  }
  mapMeta.textContent = `${mapData.shown} of ${mapData.total} passages · ${mapData.clusters.length} themes`;
}

/* ---------- concept tracing ---------- */
traceBtn.addEventListener("click", runTrace);
traceInput.addEventListener("keydown", (e) => { if (e.key === "Enter") runTrace(); });

async function runTrace() {
  const q = traceInput.value.trim();
  if (!q) return;
  traceStatus.textContent = "Tracing…";
  traceBtn.disabled = true;
  try {
    const res = await fetch(`/api/corpus/trace?q=${encodeURIComponent(q)}`);
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `HTTP ${res.status}`);
    const data = await res.json();
    traceStatus.textContent = data.n_hits
      ? `“${q}” appears in ${data.n_docs} document(s) · ${data.n_hits} passages`
      : `no passages matched “${q}”`;
    renderTrace(data);
  } catch (e) {
    traceStatus.textContent = `Error: ${e.message}`;
  } finally {
    traceBtn.disabled = false;
  }
}

function renderTrace(data) {
  traceResults.innerHTML = "";
  for (const g of data.docs) {
    const card = document.createElement("div");
    card.className = "trace-doc";
    const head = document.createElement("div");
    head.className = "trace-doc-head";
    head.innerHTML = `<span class="doc-title"></span><span class="hit-count">${g.count} passage${g.count > 1 ? "s" : ""} · best ${g.best.toFixed(2)}</span>`;
    head.querySelector(".doc-title").textContent = g.title;
    card.appendChild(head);

    for (const h of g.hits) {
      const row = document.createElement("div");
      row.className = "trace-hit";
      row.innerHTML = `<span class="score">${h.score.toFixed(2)}</span><span class="snip"></span>`;
      row.querySelector(".snip").textContent = h.snippet;
      row.title = "open in aligner — highlight on the page";
      row.addEventListener("click", () =>
        window.open(`/align?doc=${encodeURIComponent(g.doc)}&focus=${encodeURIComponent(h.node_id)}`, "_blank"));
      card.appendChild(row);
    }
    traceResults.appendChild(card);
  }
}
