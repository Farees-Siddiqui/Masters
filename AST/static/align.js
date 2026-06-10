/* AST <-> PDF alignment page.
 *
 * Left pane: the AST as a compact indented, collapsible outline (a node-box
 * graph is unusably wide for a real document). Each row is click-to-align.
 * Right pane: the PDF page viewer (adapted from layout.js). Clicking a node
 * highlights the boxes it aligns to and jumps to the matching page.
 */

const fileInput = document.getElementById("file");
const granSelect = document.getElementById("granularity");
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

// Per-granularity compute results, so switching back is instant.
// gran -> { alignment, reverse, boxes: Map(page->boxes[]), coverage }
const resultsByGran = new Map();

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

    currentAst = start.ast;
    nodeIndex = indexNodes(currentAst);
    doc = { doc: start.doc, pages: start.pages };
    pageIdx = 0;
    renderTree(currentAst);
    renderPage();
    statusEl.textContent = `${start.filename} — ${start.page_count} page(s)`;

    await computeAndApply(granSelect.value);
  } catch (e) {
    statusEl.textContent = `Error: ${e.message}`;
  } finally {
    goBtn.disabled = false;
  }
});

// Switching granularity re-aligns at that level (cached after the first time).
granSelect.addEventListener("change", async () => {
  if (!doc) return;
  granSelect.disabled = true;
  try {
    await computeAndApply(granSelect.value);
  } catch (e) {
    coverageEl.textContent = `Error: ${e.message}`;
  } finally {
    granSelect.disabled = false;
  }
});

/* Compute alignment for a granularity (or reuse cache), then make it active. */
async function computeAndApply(gran) {
  if (!resultsByGran.has(gran)) {
    coverageEl.textContent = `Computing ${gran} alignment (OCR'ing every page)…`;
    const res = await fetch(
      `/api/align/compute?doc=${encodeURIComponent(doc.doc)}&granularity=${gran}`
    );
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `HTTP ${res.status}`);
    const comp = await res.json();
    const boxes = new Map();
    for (const p of comp.pages) boxes.set(p.page, p.boxes);
    resultsByGran.set(gran, {
      alignment: comp.alignment || {},
      reverse: comp.reverse || {},
      boxes,
      coverage: comp.coverage,
    });
  }
  const r = resultsByGran.get(gran);
  alignment = r.alignment;
  reverse = r.reverse;
  boxesByPage = r.boxes;
  coverageEl.textContent = `${gran} · coverage ${(r.coverage * 100).toFixed(0)}% · click a node to align`;
  annotateMatches();
  if (selectedId != null) selectNode(selectedId);  // re-highlight at new granularity
  else renderPage();
}

function resetView() {
  currentAst = null;
  doc = null;
  alignment = {};
  reverse = {};
  boxesByPage = new Map();
  resultsByGran.clear();
  selectedId = null;
  selectedSet = new Set();
  rowById.clear();
  treeEl.innerHTML = "";
  overlay.innerHTML = "";
  canvasEl.hidden = true;
  emptyEl.hidden = false;
  pager.hidden = true;
  coverageEl.textContent = "";
  matchInfo.textContent = "";
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
  drawOverlay();

  if (ownerId) {
    const n = nodeIndex.get(ownerId);
    const label = n?.type === "section" ? `section “${n.attribs?.title ?? ""}”` : n?.type;
    matchInfo.textContent = `box → ${ownerId} (${label})`;
  } else {
    matchInfo.textContent = "box → no AST node";
  }
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
  if (!doc) return;
  const page = doc.pages[pageIdx];
  const boxes = boxesByPage.get(page.page) || [];   // empty until /compute returns
  const haveSelection = selectedId != null && selectedSet.size > 0;

  boxes.forEach((b, i) => {
    const [x1, y1, x2, y2] = b.bbox;
    const rect = document.createElementNS(SVG_NS, "rect");
    rect.setAttribute("x", x1);
    rect.setAttribute("y", y1);
    rect.setAttribute("width", Math.max(0, x2 - x1));
    rect.setAttribute("height", Math.max(0, y2 - y1));
    let cls = "lbox";
    if (selectedSet.has(`${page.page}:${i}`)) cls += " matched";
    else if (haveSelection) cls += " dim";
    rect.setAttribute("class", cls);
    rect.addEventListener("click", (e) => { e.stopPropagation(); selectBox(page.page, i); });
    overlay.appendChild(rect);
  });
}
