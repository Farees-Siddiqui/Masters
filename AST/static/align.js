/* AST <-> PDF alignment page.
 *
 * Left pane: the AST as a compact indented, collapsible outline (a node-box
 * graph is unusably wide for a real document). Each row is click-to-align.
 * Right pane: the PDF page viewer (adapted from layout.js). Clicking a node
 * highlights the boxes it aligns to and jumps to the matching page.
 */

const fileInput = document.getElementById("file");
const granSelect = document.getElementById("granularity");
const docgranSelect = document.getElementById("docgran");
const methodSelect = document.getElementById("method");
const confControl = document.getElementById("conf-control");
const confInput = document.getElementById("confidence");
const confValue = document.getElementById("conf-value");
const goBtn = document.getElementById("go");
const statusEl = document.getElementById("status");
const coverageEl = document.getElementById("coverage");
const treeEl = document.getElementById("tree");
const expandAllBtn = document.getElementById("expand-all");
const collapseAllBtn = document.getElementById("collapse-all");

const pager = document.getElementById("pager");
const prevBtn = document.getElementById("prev");
const nextBtn = document.getElementById("next");
const pageIndicator = document.getElementById("page-indicator");
const matchInfo = document.getElementById("match-info");

const scopeSel = document.getElementById("scope");
const queryInput = document.getElementById("query");
const searchBtn = document.getElementById("search-btn");
const askBtn = document.getElementById("ask-btn");
const qaStatus = document.getElementById("qa-status");
const qaPanel = document.getElementById("qa-panel");
const corpusInfo = document.getElementById("corpus-info");

let corpusDocs = [];   // [{doc, title, page_count}] available for cross-doc search
let lastResults = [];  // most recent search results (for re-marking after a doc switch)

const emptyEl = document.getElementById("empty");
const canvasEl = document.getElementById("canvas");
const pageImg = document.getElementById("page-img");
const overlay = document.getElementById("overlay");

const SVG_NS = "http://www.w3.org/2000/svg";

let currentAst = null;
let nodeIndex = new Map();        // id -> node
let doc = null;                   // { doc, pages:[{page,width,height,image_url}] }
let pageIdx = 0;
let alignment = {};               // node_id -> [{page, box_index}]  (active granularity)
let reverse = {};                 // "page:box_index" -> [node_id,...] (owner first)
let boxesByPage = new Map();      // page_no -> boxes[]              (active granularity)
let selectedId = null;            // highlighted AST node (tree row)
let selectedSet = new Set();      // "page:box_index" highlighted in the PDF
const rowById = new Map();        // node_id -> .row element (for select + badges)

// Similarity-only: per-pair scores power the live confidence slider + heatmap.
let activeMethod = "stream";      // method backing the current alignment
let scoresByNode = new Map();     // node_id -> [{page, box_index, score}] (>= floor)
let unionScoreByNode = new Map(); // node_id -> Map("page:box_index" -> best subtree score)
let totalSegments = 0;            // # of text-bearing AST nodes (coverage denominator)

// Word/sentence selection (stream method): per-word boxes + the DOM element a
// connector is anchored to (a token chip when a sub-span is selected, else the row).
let tokensByNode = new Map();     // node_id -> [{i, text, boxes:[{page,box_index}]}]
let selectedAnchorEl = null;

// Compute results cached per "granularity|method", so switching back is instant.
// key -> { alignment, reverse, boxes: Map(page->boxes[]), coverage }
const resultsByGran = new Map();

// The cache/request key combines both controls (a given alignment depends on both).
function computeKey() {
  return `${granSelect.value}|${methodSelect.value}`;
}

/* ---------- align flow ---------- */
goBtn.addEventListener("click", async () => {
  const f = fileInput.files?.[0];
  if (!f) { statusEl.textContent = "Pick a PDF first."; return; }
  goBtn.disabled = true;
  resetView();
  statusEl.textContent = "Uploading, OCR & rendering…";

  const form = new FormData();
  form.append("file", f);
  try {
    const startRes = await fetch("/api/align/start", { method: "POST", body: form });
    if (!startRes.ok) throw new Error((await startRes.json().catch(() => ({}))).detail || `HTTP ${startRes.status}`);
    const start = await startRes.json();
    await applyDoc(start);
    statusEl.textContent = `${start.filename} — ${start.page_count} page(s)`;
    refreshCorpus();  // a freshly-aligned doc joins the searchable corpus
  } catch (e) {
    statusEl.textContent = `Error: ${e.message}`;
  } finally {
    goBtn.disabled = false;
  }
});

// Load a doc payload (from /api/align/start or /api/align/load) into the viewer
// and compute its alignment. Clears the previous doc's state but keeps the
// search/Q&A panel, so cross-document results stay clickable across switches.
async function applyDoc(start) {
  resetDocState();
  currentAst = start.ast;
  nodeIndex = indexNodes(currentAst);
  totalSegments = countSegments(currentAst);
  doc = { doc: start.doc, pages: start.pages };
  pageIdx = 0;
  renderTree(currentAst);
  renderPage();
  await computeAndApply();
}

// Switching granularity or method re-aligns (cached after the first time).
async function onControlChange() {
  if (!doc) return;
  granSelect.disabled = methodSelect.disabled = true;
  try {
    await computeAndApply();
  } catch (e) {
    coverageEl.textContent = `Error: ${e.message}`;
  } finally {
    granSelect.disabled = methodSelect.disabled = false;
  }
}
granSelect.addEventListener("change", onControlChange);
methodSelect.addEventListener("change", onControlChange);

// The Doc-side "Select" control. Word/sentence selection needs the stream
// aligner's per-word boxes, so those modes force method=stream (and prefer word
// PDF boxes for a tight 1:1 mapping). Paragraph mode restores the method choice.
async function onDocgranChange() {
  const dg = docgranSelect.value;
  if (!doc) return;
  if (dg !== "paragraph") {
    methodSelect.disabled = true;
    let recompute = false;
    if (methodSelect.value !== "stream") { methodSelect.value = "stream"; recompute = true; }
    if (granSelect.value === "paragraph") { granSelect.value = "word"; recompute = true; }
    if (recompute) await onControlChange();   // re-aligns and re-selects
  } else {
    methodSelect.disabled = false;
  }
  clearTokenStrip();
  if (selectedId != null) selectNode(selectedId);  // re-render under new doc-gran
}
docgranSelect.addEventListener("change", onDocgranChange);

/* Compute alignment for the current granularity+method (or reuse cache). */
async function computeAndApply() {
  const gran = granSelect.value;
  const method = methodSelect.value;
  const key = computeKey();
  if (!resultsByGran.has(key)) {
    coverageEl.textContent = `Computing ${gran} · ${method} alignment (OCR'ing every page)…`;
    const res = await fetch(
      `/api/align/compute?doc=${encodeURIComponent(doc.doc)}&granularity=${gran}&method=${method}`
    );
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `HTTP ${res.status}`);
    const comp = await res.json();
    const boxes = new Map();
    for (const p of comp.pages) boxes.set(p.page, p.boxes);
    resultsByGran.set(key, {
      alignment: comp.alignment || {},
      reverse: comp.reverse || {},
      boxes,
      coverage: comp.coverage,
      scores: comp.scores || null,   // similarity only; null for stream
      tokens: comp.tokens || null,   // stream only; per-word boxes
    });
  }
  // A late-returning request must not clobber the view if the user changed controls.
  if (key !== computeKey()) return;
  const r = resultsByGran.get(key);
  activeMethod = method;
  boxesByPage = r.boxes;
  tokensByNode = r.tokens ? new Map(Object.entries(r.tokens)) : new Map();

  if (method === "similarity" && r.scores) {
    // The slider drives the threshold; derive alignment client-side from scores.
    confControl.hidden = false;
    scoresByNode = new Map(Object.entries(r.scores));
    recomputeSimilarity(parseFloat(confInput.value));
  } else {
    confControl.hidden = true;
    scoresByNode = new Map();
    unionScoreByNode = new Map();
    alignment = r.alignment;
    reverse = r.reverse;
    coverageEl.textContent =
      `${gran} · ${method} · coverage ${(r.coverage * 100).toFixed(0)}% · click a node to align`;
  }
  annotateMatches();
  if (selectedId != null) selectNode(selectedId);  // re-highlight at new settings
  else renderPage();
}

/* Re-derive alignment/reverse from the raw similarity scores at a given
   threshold — runs entirely client-side, so the confidence slider is instant. */
function recomputeSimilarity(threshold) {
  const directScore = new Map();   // node_id -> Map(key -> score) for direct hits
  for (const [nid, pairs] of scoresByNode) {
    const sm = new Map();
    for (const p of pairs) {
      if (p.score >= threshold) sm.set(`${p.page}:${p.box_index}`, p.score);
    }
    if (sm.size) directScore.set(nid, sm);
  }

  // Subtree union, keeping the best score per box (used to colour the heatmap).
  unionScoreByNode = new Map();
  function build(node) {
    const acc = new Map(directScore.get(node.id) || []);
    for (const c of node.children || []) {
      for (const [k, s] of build(c)) if (s > (acc.get(k) ?? -1)) acc.set(k, s);
    }
    unionScoreByNode.set(node.id, acc);
    return acc;
  }
  build(currentAst);

  alignment = {};
  for (const [nid, m] of unionScoreByNode) {
    if (!m.size) continue;
    alignment[nid] = [...m.keys()].map((k) => {
      const [page, box_index] = k.split(":").map(Number);
      return { page, box_index };
    });
  }

  // Reverse map: box -> owning nodes, highest direct score first.
  const owners = new Map();         // key -> Map(node_id -> score)
  for (const [nid, sm] of directScore) {
    for (const [k, s] of sm) {
      let o = owners.get(k);
      if (!o) owners.set(k, (o = new Map()));
      if (s > (o.get(nid) ?? -1)) o.set(nid, s);
    }
  }
  reverse = {};
  for (const [k, o] of owners) {
    reverse[k] = [...o.entries()].sort((a, b) => b[1] - a[1]).map((e) => e[0]);
  }

  const coverage = totalSegments ? directScore.size / totalSegments : 0;
  coverageEl.textContent =
    `${granSelect.value} · similarity · ≥${threshold.toFixed(2)} · coverage ` +
    `${(coverage * 100).toFixed(0)}% · click a node to align`;
}

/* Live confidence slider (similarity only): re-filter + re-highlight instantly. */
confInput.addEventListener("input", () => {
  const t = parseFloat(confInput.value);
  confValue.textContent = t.toFixed(2);
  if (activeMethod !== "similarity" || scoresByNode.size === 0) return;
  recomputeSimilarity(t);
  annotateMatches();
  if (selectedId != null) selectNode(selectedId);
  else drawOverlay();
});

// Clear everything tied to the currently-loaded document (but not the search/Q&A
// panel, so it survives a cross-document switch).
function resetDocState() {
  currentAst = null;
  doc = null;
  alignment = {};
  reverse = {};
  boxesByPage = new Map();
  resultsByGran.clear();
  selectedId = null;
  selectedSet = new Set();
  rowById.clear();
  scoresByNode = new Map();
  unionScoreByNode = new Map();
  tokensByNode = new Map();
  selectedAnchorEl = null;
  treeEl.innerHTML = "";
  overlay.innerHTML = "";
  if (connectorSvg) connectorSvg.innerHTML = "";
  canvasEl.hidden = true;
  emptyEl.hidden = false;
  pager.hidden = true;
  coverageEl.textContent = "";
  matchInfo.textContent = "";
}

// Full reset for a brand-new upload: doc state + the search/Q&A panel.
function resetView() {
  resetDocState();
  if (qaPanel) { qaPanel.hidden = true; qaPanel.innerHTML = ""; }
  if (qaStatus) qaStatus.textContent = "";
}

function indexNodes(root) {
  const map = new Map();
  const stack = [root];
  while (stack.length) {
    const n = stack.pop();
    map.set(n.id, n);
    for (const c of n.children || []) stack.push(c);
  }
  return map;
}

// Count text-bearing nodes (sections carry text in attribs.title) — the
// coverage denominator, mirroring the backend's _iter_segments.
function countSegments(root) {
  let n = 0;
  (function walk(node) {
    const text = node.type === "section" ? node.attribs?.title : node.text;
    if (text) n++;
    for (const c of node.children || []) walk(c);
  })(root);
  return n;
}

/* ---------- left pane: indented outline ---------- */
function nodeLabel(n) {
  const MAX = 80;
  if (n.type === "section" && n.attribs?.title) return n.attribs.title;
  if (n.text) return n.text.length > MAX ? n.text.slice(0, MAX - 1) + "…" : n.text;
  if (n.type === "list") return n.attribs?.ordered === "true" ? "ordered list" : "unordered list";
  if (n.type === "doc") return `${(n.children || []).length} top-level nodes`;
  return "";
}

function renderTree(root) {
  treeEl.innerHTML = "";
  rowById.clear();
  treeEl.appendChild(renderNode(root));
}

function renderNode(node) {
  const wrap = document.createElement("div");
  wrap.className = "node";
  wrap.dataset.type = node.type;

  const row = document.createElement("div");
  row.className = "row";
  row.dataset.id = node.id;
  rowById.set(node.id, row);

  const hasChildren = node.children && node.children.length > 0;

  const toggle = document.createElement("span");
  toggle.className = "toggle";
  toggle.textContent = hasChildren ? "▾" : " ";
  row.appendChild(toggle);

  const type = document.createElement("span");
  type.className = "type";
  type.textContent = node.type;
  row.appendChild(type);

  const label = nodeLabel(node);
  if (label) {
    const t = document.createElement("span");
    t.className = "text";
    t.textContent = `  ${label}`;
    t.title = label;
    row.appendChild(t);
  }

  // a small badge for how many PDF boxes this node aligns to (filled after compute)
  const badge = document.createElement("span");
  badge.className = "match-badge";
  row.appendChild(badge);

  wrap.appendChild(row);

  let kids = null;
  if (hasChildren) {
    kids = document.createElement("div");
    kids.className = "children";
    for (const c of node.children) kids.appendChild(renderNode(c));
    wrap.appendChild(kids);
  }

  // Click the toggle (or a parent's type word) to collapse; click the row to align.
  toggle.addEventListener("click", (e) => {
    if (!hasChildren) return;
    e.stopPropagation();
    wrap.classList.toggle("collapsed");
    toggle.textContent = wrap.classList.contains("collapsed") ? "▸" : "▾";
  });
  row.addEventListener("click", () => selectNode(node.id));

  return wrap;
}

/* Annotate rows with their aligned-box count once /compute returns. */
function annotateMatches() {
  for (const [id, row] of rowById) {
    const n = alignment[id]?.length || 0;
    const badge = row.querySelector(".match-badge");
    if (!badge) continue;
    if (n > 0) { badge.textContent = n; row.classList.add("has-match"); }
    else { badge.textContent = ""; row.classList.remove("has-match"); }
  }
}

function expandCollapseAll(collapse) {
  treeEl.querySelectorAll(".node").forEach((wrap) => {
    const hasChildren = wrap.querySelector(":scope > .children");
    if (!hasChildren) return;
    const toggle = wrap.querySelector(":scope > .row > .toggle");
    wrap.classList.toggle("collapsed", collapse);
    if (toggle) toggle.textContent = collapse ? "▸" : "▾";
  });
}
expandAllBtn.addEventListener("click", () => expandCollapseAll(false));
collapseAllBtn.addEventListener("click", () => expandCollapseAll(true));

/* ---------- selection: AST <-> PDF (bidirectional) ---------- */

// Highlight a node's row in the tree, expanding collapsed ancestors and
// scrolling it into view. Pass null to clear the tree selection.
function setTreeSelection(id, scroll) {
  treeEl.querySelectorAll(".row.selected").forEach((r) => r.classList.remove("selected"));
  selectedId = id;
  if (!id) return;
  // Expand any collapsed ancestors so the row is visible.
  let cur = nodeIndex.get(id);
  while (cur && cur.parent_id) {
    const prow = rowById.get(cur.parent_id);
    const wrap = prow?.parentElement;
    if (wrap?.classList.contains("collapsed")) {
      wrap.classList.remove("collapsed");
      const t = wrap.querySelector(":scope > .row > .toggle");
      if (t) t.textContent = "▾";
    }
    cur = nodeIndex.get(cur.parent_id);
  }
  const row = rowById.get(id);
  if (row) {
    row.classList.add("selected");
    if (scroll) row.scrollIntoView({ block: "nearest" });
  }
}

// Forward: click an AST node -> highlight all the boxes it maps to.
function selectNode(id) {
  setTreeSelection(id, false);
  selectedAnchorEl = rowById.get(id) || null;   // connector starts at the row
  clearTokenStrip();
  renderTokenStrip(nodeIndex.get(id));          // word/sentence chips (if enabled)
  const matches = alignment[id] || [];
  selectedSet = new Set(matches.map((m) => `${m.page}:${m.box_index}`));

  if (matches.length === 0) {
    matchInfo.textContent = boxesByPage.size ? "no alignment found" : "still computing…";
    drawOverlay();
    return;
  }

  const pages = [...new Set(matches.map((m) => m.page))].sort((a, b) => a - b);
  const targetIdx = doc.pages.findIndex((p) => p.page === pages[0]);
  if (targetIdx >= 0 && targetIdx !== pageIdx) { pageIdx = targetIdx; renderPage(); }
  else drawOverlay();

  const onThis = matches.filter((m) => m.page === doc.pages[pageIdx].page).length;
  matchInfo.textContent =
    `${matches.length} box(es) on page ${pages.join(", ")}` + (pages.length > 1 ? ` · ${onThis} here` : "");
}

// Reverse: click a PDF box -> emphasize just that box and reveal its AST node.
function selectBox(page, idx) {
  const key = `${page}:${idx}`;
  selectedSet = new Set([key]);
  const owners = reverse[key] || [];
  const ownerId = owners[0] || null;
  setTreeSelection(ownerId, true);
  selectedAnchorEl = rowById.get(ownerId) || null;
  clearTokenStrip();
  drawOverlay();

  if (ownerId) {
    const n = nodeIndex.get(ownerId);
    const label = n?.type === "section" ? `section “${n.attribs?.title ?? ""}”` : n?.type;
    matchInfo.textContent = `box → ${ownerId} (${label})`;
  } else {
    matchInfo.textContent = "box → no AST node";
  }
}

/* ---------- word / sentence selection (stream method) ---------- */

function clearTokenStrip() {
  treeEl.querySelectorAll(".token-strip").forEach((el) => el.remove());
}

// Group consecutive word tokens into sentences, breaking after . ! ? (plus any
// trailing quotes/brackets).
function sentenceGroups(tokens) {
  const groups = [];
  let cur = [];
  for (const t of tokens) {
    cur.push(t);
    if (/[.!?]["'”’)\]]*$/.test(t.text)) { groups.push(cur); cur = []; }
  }
  if (cur.length) groups.push(cur);
  return groups;
}

// Render the selected leaf's text as clickable word/sentence chips beneath its
// row. Each chip aligns just that span; chips with no PDF box are shown dimmed.
function renderTokenStrip(node) {
  if (!node || docgranSelect.value === "paragraph") return;
  const toks = tokensByNode.get(node.id);
  if (!toks || toks.length === 0) return;
  const row = rowById.get(node.id);
  if (!row) return;

  const strip = document.createElement("div");
  strip.className = "token-strip";

  const makeChip = (label, indices, hasBoxes, extra) => {
    const chip = document.createElement("span");
    chip.className = "chip" + (extra ? " " + extra : "") + (hasBoxes ? "" : " no-box");
    chip.textContent = label;
    if (hasBoxes) {
      chip.addEventListener("click", (e) => { e.stopPropagation(); selectSpan(node.id, indices, chip); });
    }
    strip.appendChild(chip);
  };

  if (docgranSelect.value === "word") {
    toks.forEach((t) => makeChip(t.text, [t.i], t.boxes.length > 0));
  } else {  // sentence
    for (const g of sentenceGroups(toks)) {
      makeChip(g.map((t) => t.text).join(" "), g.map((t) => t.i), g.some((t) => t.boxes.length > 0), "sentence");
    }
  }
  row.parentElement.insertBefore(strip, row.nextSibling);
}

// Highlight exactly the boxes a word/sentence span maps to, anchoring the
// connector at the clicked chip.
function selectSpan(nodeId, indices, chip) {
  const toks = tokensByNode.get(nodeId) || [];
  const want = new Set(indices);
  const keys = new Set();
  let firstPage = null;
  for (const t of toks) {
    if (!want.has(t.i)) continue;
    for (const b of t.boxes) {
      keys.add(`${b.page}:${b.box_index}`);
      if (firstPage == null) firstPage = b.page;
    }
  }
  selectedId = nodeId;
  selectedSet = keys;
  selectedAnchorEl = chip || rowById.get(nodeId);
  treeEl.querySelectorAll(".chip.active").forEach((c) => c.classList.remove("active"));
  chip?.classList.add("active");

  if (keys.size === 0) { matchInfo.textContent = "no PDF box for this selection"; drawOverlay(); return; }
  const targetIdx = doc.pages.findIndex((p) => p.page === firstPage);
  if (targetIdx >= 0 && targetIdx !== pageIdx) { pageIdx = targetIdx; renderPage(); }
  else drawOverlay();
  matchInfo.textContent = `${keys.size} box(es) on page ${firstPage}`;
}

/* ---------- PDF viewer (adapted from layout.js) ---------- */
prevBtn.addEventListener("click", () => { if (pageIdx > 0) { pageIdx--; renderPage(); } });
nextBtn.addEventListener("click", () => { if (doc && pageIdx < doc.pages.length - 1) { pageIdx++; renderPage(); } });
window.addEventListener("keydown", (e) => {
  if (!doc || e.target.tagName === "INPUT") return;
  if (e.key === "ArrowLeft") prevBtn.click();
  if (e.key === "ArrowRight") nextBtn.click();
});

function renderPage() {
  if (!doc) return;
  const page = doc.pages[pageIdx];
  emptyEl.hidden = true;
  canvasEl.hidden = false;
  pager.hidden = false;
  pageIndicator.textContent = `Page ${page.page} / ${doc.pages.length}`;
  prevBtn.disabled = pageIdx === 0;
  nextBtn.disabled = pageIdx === doc.pages.length - 1;

  pageImg.src = page.image_url;
  overlay.setAttribute("viewBox", `0 0 ${page.width} ${page.height}`);
  drawOverlay();
}

function drawOverlay() {
  overlay.innerHTML = "";
  if (!doc) { scheduleConnectors(); return; }
  const page = doc.pages[pageIdx];
  const boxes = boxesByPage.get(page.page) || [];   // empty until /compute returns
  const haveSelection = selectedId != null && selectedSet.size > 0;
  // In similarity mode, colour matched boxes by their score to the selected node.
  const heatScores =
    activeMethod === "similarity" && selectedId != null
      ? unionScoreByNode.get(selectedId)
      : null;

  boxes.forEach((b, i) => {
    const key = `${page.page}:${i}`;
    const [x1, y1, x2, y2] = b.bbox;
    const rect = document.createElementNS(SVG_NS, "rect");
    rect.setAttribute("x", x1);
    rect.setAttribute("y", y1);
    rect.setAttribute("width", Math.max(0, x2 - x1));
    rect.setAttribute("height", Math.max(0, y2 - y1));
    let cls = "lbox";
    if (selectedSet.has(key)) {
      cls += " matched";
      rect.dataset.key = key;                         // anchor for connector lines
      const score = heatScores?.get(key);
      if (score != null) {                            // heatmap: green=high, red=low
        const c = colorForScore(score);
        rect.style.fill = c.fill;
        rect.style.stroke = c.stroke;
        rect.setAttribute("title", score.toFixed(2));
      }
    } else if (haveSelection) {
      cls += " dim";
    }
    rect.setAttribute("class", cls);
    rect.addEventListener("click", (e) => { e.stopPropagation(); selectBox(page.page, i); });
    overlay.appendChild(rect);
  });
  scheduleConnectors();
}

// Confidence -> colour. Scores live in ~[0.2, 0.9]; map to a red→amber→green hue.
function colorForScore(score) {
  const n = Math.max(0, Math.min(1, (score - 0.2) / 0.65));
  const hue = n * 120;  // 0 = red, 120 = green
  return { fill: `hsla(${hue}, 85%, 50%, 0.28)`, stroke: `hsl(${hue}, 85%, 55%)` };
}

/* ---------- semantic search + grounded Q&A ----------
 * Both turn a query into AST nodes, then reuse selectNode() so the matched node
 * highlights in the tree AND its aligned region lights up in the PDF. */

searchBtn.addEventListener("click", searchDoc);
askBtn.addEventListener("click", askDoc);
queryInput.addEventListener("keydown", (e) => {
  if (e.key !== "Enter") return;
  e.preventDefault();
  (e.ctrlKey || e.metaKey) ? askDoc() : searchDoc();
});

async function searchDoc() {
  const q = queryInput.value.trim();
  if (!q) return;
  const corpus = scopeSel.value === "corpus";
  if (!corpus && !doc) { qaStatus.textContent = "Align a PDF first."; return; }
  qaStatus.textContent = "Searching…";
  searchBtn.disabled = askBtn.disabled = true;
  try {
    const url = corpus
      ? `/api/corpus/search?q=${encodeURIComponent(q)}&top_k=12`
      : `/api/align/search?doc=${encodeURIComponent(doc.doc)}&q=${encodeURIComponent(q)}&top_k=8`;
    const res = await fetch(url);
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `HTTP ${res.status}`);
    const data = await res.json();
    if (corpus) {
      const nDocs = new Set(data.results.map((r) => r.doc)).size;
      qaStatus.textContent = data.results.length ? `${data.results.length} matches across ${nDocs} doc(s)` : "no matches";
    } else {
      qaStatus.textContent = data.results.length ? `${data.results.length} matches` : "no matches";
    }
    renderResults(data.results, corpus);
  } catch (e) {
    qaStatus.textContent = `Error: ${e.message}`;
  } finally {
    searchBtn.disabled = askBtn.disabled = false;
  }
}

// Load the corpus list so cross-document search knows what's available.
async function refreshCorpus() {
  try {
    const res = await fetch("/api/corpus/list");
    if (!res.ok) return;
    corpusDocs = (await res.json()).docs || [];
    corpusInfo.textContent = corpusDocs.length ? `${corpusDocs.length} doc(s) indexed` : "";
  } catch { /* ignore */ }
}

// Open a (possibly different) document for a cross-doc result, then highlight it.
async function openResult(r) {
  if (r.doc && (!doc || r.doc !== doc.doc)) {
    qaStatus.textContent = `Opening ${r.title || r.doc}…`;
    searchBtn.disabled = askBtn.disabled = true;
    try {
      const res = await fetch(`/api/align/load?doc=${encodeURIComponent(r.doc)}`);
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `HTTP ${res.status}`);
      await applyDoc(await res.json());
      statusEl.textContent = `${r.title || r.doc}`;
      qaStatus.textContent = "";
      markResultRows(lastResults.filter((x) => x.doc === doc.doc).map((x) => x.node_id));
    } catch (e) {
      qaStatus.textContent = `Error: ${e.message}`;
      return;
    } finally {
      searchBtn.disabled = askBtn.disabled = false;
    }
  }
  focusResult(r.node_id);
}

async function askDoc() {
  const q = queryInput.value.trim();
  if (!doc) { qaStatus.textContent = "Align a PDF first."; return; }
  if (!q) return;
  qaStatus.textContent = "Thinking…";
  searchBtn.disabled = askBtn.disabled = true;
  try {
    const res = await fetch(`/api/align/ask?doc=${encodeURIComponent(doc.doc)}&q=${encodeURIComponent(q)}`);
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `HTTP ${res.status}`);
    const data = await res.json();
    // Highlighting the exact source *line* needs line-granularity stream boxes;
    // switch to them (cached after the first time) before rendering the answer.
    if (granSelect.value !== "line" || methodSelect.value !== "stream" || docgranSelect.value !== "paragraph") {
      qaStatus.textContent = "Locating source…";
      docgranSelect.value = "paragraph";
      methodSelect.value = "stream";
      methodSelect.disabled = false;
      granSelect.value = "line";
      await onControlChange();
    }
    qaStatus.textContent = "";
    renderAnswer(data.answer || "", data.citations || []);
  } catch (e) {
    qaStatus.textContent = `Error: ${e.message}`;
  } finally {
    searchBtn.disabled = askBtn.disabled = false;
  }
}

// Highlight a result/citation node in the tree + PDF, scroll its row into view.
function focusResult(id) {
  if (!nodeIndex.has(id)) return;
  selectNode(id);
  rowById.get(id)?.scrollIntoView({ block: "center", behavior: "smooth" });
}

// Tag the result/citation rows so they stand out in the tree.
function markResultRows(ids) {
  treeEl.querySelectorAll(".row.result").forEach((r) => r.classList.remove("result"));
  for (const id of ids) rowById.get(id)?.classList.add("result");
}

function clearQuery() {
  qaPanel.hidden = true;
  qaPanel.innerHTML = "";
  qaStatus.textContent = "";
  markResultRows([]);
}

function renderResults(results, corpus = false) {
  lastResults = results;
  qaPanel.hidden = false;
  qaPanel.innerHTML = "";
  if (results.length === 0) { qaPanel.hidden = true; markResultRows([]); return; }

  const head = document.createElement("div");
  head.className = "qa-head";
  head.textContent = corpus
    ? "Matches across the corpus — click to open the document and highlight"
    : "Semantic matches — click to highlight in the PDF";
  qaPanel.appendChild(head);

  const list = document.createElement("div");
  list.className = "result-list" + (corpus ? " corpus" : "");
  results.forEach((r, i) => {
    const item = document.createElement("div");
    item.className = "result-item" + (corpus ? " corpus" : "");
    const badge = corpus ? `<span class="doc-badge" title="${escapeAttr(r.title || r.doc)}"></span>` : "";
    item.innerHTML =
      `<span class="rank">${i + 1}</span>` +
      `<span class="score">${r.score.toFixed(2)}</span>` +
      badge +
      `<span class="snip"></span>`;
    item.querySelector(".snip").textContent = r.snippet;
    if (corpus) item.querySelector(".doc-badge").textContent = r.title || r.doc;
    item.addEventListener("click", () => (corpus ? openResult(r) : focusResult(r.node_id)));
    list.appendChild(item);
  });
  qaPanel.appendChild(list);

  // Only the current document's results have tree rows to mark.
  markResultRows(results.filter((r) => !r.doc || (doc && r.doc === doc.doc)).map((r) => r.node_id));
  if (!corpus) focusResult(results[0].node_id);
}

function escapeAttr(s) {
  return (s || "").replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
}

function renderAnswer(answer, citations) {
  qaPanel.hidden = false;
  qaPanel.innerHTML = "";
  // citations: [{id, quote}] — the quote pins the highlight to the source line.
  const quoteById = new Map(citations.map((c) => [c.id, c.quote || ""]));

  const ans = document.createElement("div");
  ans.className = "qa-answer";
  // Replace inline [nXX] citations with clickable chips (built as DOM, no innerHTML).
  const re = /\[(n\d+)\]/g;
  let last = 0, m;
  while ((m = re.exec(answer)) !== null) {
    if (m.index > last) ans.appendChild(document.createTextNode(answer.slice(last, m.index)));
    const id = m[1];
    const chip = document.createElement("span");
    chip.className = "cite-chip";
    chip.textContent = id;
    chip.title = "show the source line in the PDF";
    if (nodeIndex.has(id)) chip.addEventListener("click", () => focusCitation(id, quoteById.get(id)));
    else chip.classList.add("dead");
    ans.appendChild(chip);
    last = re.lastIndex;
  }
  if (last < answer.length) ans.appendChild(document.createTextNode(answer.slice(last)));
  qaPanel.appendChild(ans);

  if (citations.length) {
    const src = document.createElement("div");
    src.className = "qa-sources";
    const label = document.createElement("span");
    label.className = "qa-src-label";
    label.textContent = "Sources:";
    src.appendChild(label);
    citations.forEach((c) => {
      const chip = document.createElement("span");
      chip.className = "cite-chip";
      chip.textContent = c.id;
      if (c.quote) chip.title = `“${c.quote}”`;
      chip.addEventListener("click", () => focusCitation(c.id, c.quote));
      src.appendChild(chip);
    });
    qaPanel.appendChild(src);
  }

  markResultRows(citations.map((c) => c.id));
  if (citations[0]) focusCitation(citations[0].id, citations[0].quote);
}

// Normalize for fuzzy text matching (mirrors the backend normalizer).
function qaNorm(s) {
  return (s || "").toLowerCase().replace(/[^a-z0-9\s]+/g, " ").replace(/\s+/g, " ").trim();
}

// Highlight the exact source line(s): among the cited node's (line-granularity)
// boxes, pick the one whose OCR text best covers the verbatim quote.
function focusCitation(id, quote) {
  if (!nodeIndex.has(id)) return;
  const cands = alignment[id] || [];
  const qWords = new Set(qaNorm(quote).split(" ").filter(Boolean));
  if (qWords.size === 0 || cands.length === 0) { focusResult(id); return; }

  const scored = cands.map((b) => {
    const bWords = new Set(qaNorm(boxesByPage.get(b.page)?.[b.box_index]?.text || "").split(" ").filter(Boolean));
    let inter = 0;
    for (const w of qWords) if (bWords.has(w)) inter++;
    return { b, s: inter / qWords.size };
  });
  const top = scored.reduce((mx, x) => Math.max(mx, x.s), 0);
  if (top <= 0) { focusResult(id); return; }            // OCR mismatch — fall back
  // Keep the best line(s); the threshold catches a quote that spans two lines.
  const keep = scored.filter((x) => x.s >= Math.max(0.34, top * 0.6)).map((x) => x.b);

  setTreeSelection(id, false);
  selectedAnchorEl = rowById.get(id) || null;
  clearTokenStrip();
  selectedSet = new Set(keep.map((b) => `${b.page}:${b.box_index}`));
  const firstPage = keep[0].page;
  const ti = doc.pages.findIndex((p) => p.page === firstPage);
  if (ti >= 0 && ti !== pageIdx) { pageIdx = ti; renderPage(); }
  else drawOverlay();
  rowById.get(id)?.scrollIntoView({ block: "center", behavior: "smooth" });
  matchInfo.textContent = `source line${keep.length > 1 ? "s" : ""} for ${id}`;
}

/* ---------- connector lines: AST row -> its PDF boxes ----------
 * A fixed, full-window SVG drawn in viewport pixels. We bezier from the right
 * edge of the selected tree row to the left edge of each matched box, clamped to
 * each pane so the curves stay anchored while either side scrolls. */
const connectorSvg = document.createElementNS(SVG_NS, "svg");
connectorSvg.id = "connectors";
document.body.appendChild(connectorSvg);

let connectorRAF = 0;
function scheduleConnectors() {
  if (connectorRAF) return;
  connectorRAF = requestAnimationFrame(() => { connectorRAF = 0; drawConnectors(); });
}

function clampRect(y, rect) {
  return Math.max(rect.top + 2, Math.min(rect.bottom - 2, y));
}

function drawConnectors() {
  connectorSvg.innerHTML = "";
  connectorSvg.setAttribute("width", window.innerWidth);
  connectorSvg.setAttribute("height", window.innerHeight);
  if (!selectedId || selectedSet.size === 0) return;

  const anchor = selectedAnchorEl || rowById.get(selectedId);
  const targets = overlay.querySelectorAll("rect.matched");
  if (!anchor || targets.length === 0) return;

  const treeRect = treeEl.getBoundingClientRect();
  const stageRect = document.getElementById("stage").getBoundingClientRect();
  const rowRect = anchor.getBoundingClientRect();
  // Start at the tree pane's right edge, vertically tracking the row (clamped in view).
  const sx = treeRect.right - 2;
  const sy = clampRect(rowRect.top + rowRect.height / 2, treeRect);

  targets.forEach((t) => {
    const r = t.getBoundingClientRect();
    const ex = r.left + 1;
    const ey = clampRect(r.top + r.height / 2, stageRect);
    const dx = Math.max(40, (ex - sx) * 0.45);
    const d = `M ${sx} ${sy} C ${sx + dx} ${sy}, ${ex - dx} ${ey}, ${ex} ${ey}`;

    const path = document.createElementNS(SVG_NS, "path");
    path.setAttribute("d", d);
    path.setAttribute("class", "connector");
    connectorSvg.appendChild(path);

    const dot = document.createElementNS(SVG_NS, "circle");
    dot.setAttribute("cx", ex);
    dot.setAttribute("cy", ey);
    dot.setAttribute("r", 3);
    dot.setAttribute("class", "connector-dot");
    connectorSvg.appendChild(dot);
  });
}

// Keep the curves glued to both panes as the user scrolls or resizes.
treeEl.addEventListener("scroll", scheduleConnectors, { passive: true });
document.getElementById("stage").addEventListener("scroll", scheduleConnectors, { passive: true });
window.addEventListener("scroll", scheduleConnectors, { passive: true });
window.addEventListener("resize", scheduleConnectors, { passive: true });

/* ---------- corpus bootstrap ---------- */
scopeSel.addEventListener("change", () => { if (scopeSel.value === "corpus") refreshCorpus(); });
refreshCorpus();  // populate the "N docs indexed" hint on load

// Deep-link from the diff page: /align?doc=<id>&focus=<node_id> opens an
// already-processed document and highlights the node (provenance for a change).
(async function initFromUrl() {
  const params = new URLSearchParams(location.search);
  const d = params.get("doc");
  const f = params.get("focus");
  if (!d) return;
  statusEl.textContent = `Loading ${d}…`;
  try {
    const res = await fetch(`/api/align/load?doc=${encodeURIComponent(d)}`);
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `HTTP ${res.status}`);
    const start = await res.json();
    await applyDoc(start);
    statusEl.textContent = `${start.filename}`;
    if (f) focusResult(f);
  } catch (e) {
    statusEl.textContent = `Error: ${e.message}`;
  }
})();
