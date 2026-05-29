const fileInput = document.getElementById("file");
const goBtn = document.getElementById("go");
const statusEl = document.getElementById("status");
const treeEl = document.getElementById("tree");
const jsonEl = document.getElementById("json");
const graphEl = document.getElementById("graph");
const inspectorEl = document.getElementById("inspector");
const inspectorBody = document.getElementById("inspector-body");
const inspectorClose = document.getElementById("inspector-close");

const SVG_NS = "http://www.w3.org/2000/svg";

let currentAst = null;
let nodeIndex = new Map(); // id -> node
let selectedId = null;

/* ---------- tab switching ---------- */
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
  });
});

/* ---------- upload ---------- */
goBtn.addEventListener("click", async () => {
  const f = fileInput.files?.[0];
  if (!f) {
    statusEl.textContent = "Pick a PDF first.";
    return;
  }
  goBtn.disabled = true;
  statusEl.textContent = "Uploading & running OCR…";
  treeEl.innerHTML = "";
  jsonEl.textContent = "";
  graphEl.innerHTML = "";
  inspectorEl.classList.add("hidden");

  const form = new FormData();
  form.append("file", f);

  try {
    const res = await fetch("/api/upload", { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    statusEl.textContent = `${data.filename} — ${data.page_count} page(s)`;
    currentAst = data.ast;
    nodeIndex = indexNodes(currentAst);
    jsonEl.textContent = JSON.stringify(currentAst, null, 2);
    treeEl.appendChild(renderOutline(currentAst));
    renderGraph(currentAst);
  } catch (e) {
    statusEl.textContent = `Error: ${e.message}`;
  } finally {
    goBtn.disabled = false;
  }
});

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

/* ---------- left pane: outline ---------- */
function renderOutline(node) {
  const wrap = document.createElement("div");
  wrap.className = "node";
  wrap.dataset.type = node.type;

  const row = document.createElement("div");
  row.className = "row";

  const toggle = document.createElement("span");
  toggle.className = "toggle";
  const hasChildren = node.children && node.children.length > 0;
  toggle.textContent = hasChildren ? "▾" : " ";
  row.appendChild(toggle);

  const type = document.createElement("span");
  type.className = "type";
  type.textContent = node.type;
  row.appendChild(type);

  if (node.attribs && Object.keys(node.attribs).length) {
    const a = document.createElement("span");
    a.className = "attribs";
    const pairs = Object.entries(node.attribs).map(([k, v]) => ` ${k}="${v}"`).join("");
    a.textContent = pairs;
    row.appendChild(a);
  }

  if (node.text) {
    const t = document.createElement("span");
    t.className = "text";
    const preview = node.text.length > 120 ? node.text.slice(0, 117) + "…" : node.text;
    t.textContent = `  — ${preview}`;
    t.title = node.text;
    row.appendChild(t);
  }

  wrap.appendChild(row);

  if (hasChildren) {
    const kids = document.createElement("div");
    kids.className = "children";
    for (const c of node.children) kids.appendChild(renderOutline(c));
    wrap.appendChild(kids);

    row.addEventListener("click", (e) => {
      e.stopPropagation();
      wrap.classList.toggle("collapsed");
      toggle.textContent = wrap.classList.contains("collapsed") ? "▸" : "▾";
    });
  }

  return wrap;
}

/* ---------- right pane: SVG tree ---------- */
const NODE_W = 180;
const NODE_H = 64;
const H_GAP = 28;
const V_GAP = 70;

function layout(node, leftX, depth) {
  const w = subtreeWidth(node);
  node._y = depth * (NODE_H + V_GAP) + NODE_H / 2;
  if (!node.children || node.children.length === 0) {
    node._x = leftX + w / 2;
  } else {
    let cx = leftX;
    for (const c of node.children) {
      layout(c, cx, depth + 1);
      cx += subtreeWidth(c) + H_GAP;
    }
    const first = node.children[0];
    const last = node.children[node.children.length - 1];
    node._x = (first._x + last._x) / 2;
  }
  return w;
}

function subtreeWidth(node) {
  if (!node.children || node.children.length === 0) return NODE_W;
  let total = 0;
  for (const c of node.children) total += subtreeWidth(c);
  total += H_GAP * (node.children.length - 1);
  return Math.max(total, NODE_W);
}

function collectNodes(root) {
  const out = [];
  const stack = [root];
  while (stack.length) {
    const n = stack.pop();
    out.push(n);
    for (const c of n.children || []) stack.push(c);
  }
  return out;
}

function renderGraph(root) {
  graphEl.innerHTML = "";
  layout(root, 0, 0);

  const nodes = collectNodes(root);
  const minX = Math.min(...nodes.map((n) => n._x)) - NODE_W / 2;
  const maxX = Math.max(...nodes.map((n) => n._x)) + NODE_W / 2;
  const maxY = Math.max(...nodes.map((n) => n._y)) + NODE_H / 2;
  const fullW = maxX - minX + 40;
  const fullH = maxY + 40;
  // shift everything so x >= 20
  const shift = 20 - minX;

  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${fullW} ${fullH}`);
  svg.setAttribute("preserveAspectRatio", "xMinYMin meet");
  // Render at NATURAL pixel size so nodes are readable; the inner <g>
  // transform handles pan/zoom, and the container clips with overflow:hidden.
  svg.setAttribute("width", String(fullW));
  svg.setAttribute("height", String(fullH));

  const inner = document.createElementNS(SVG_NS, "g");
  inner.setAttribute("class", "viewport");
  svg.appendChild(inner);

  // edges first (under nodes)
  const edgesG = document.createElementNS(SVG_NS, "g");
  edgesG.setAttribute("class", "edges");
  inner.appendChild(edgesG);

  for (const n of nodes) {
    for (const c of n.children || []) {
      const x1 = n._x + shift;
      const y1 = n._y + NODE_H / 2;
      const x2 = c._x + shift;
      const y2 = c._y - NODE_H / 2;
      const my = (y1 + y2) / 2;
      const path = document.createElementNS(SVG_NS, "path");
      path.setAttribute("class", "edge");
      path.setAttribute("d", `M ${x1} ${y1} C ${x1} ${my}, ${x2} ${my}, ${x2} ${y2}`);
      edgesG.appendChild(path);
    }
  }

  // nodes
  const nodesG = document.createElementNS(SVG_NS, "g");
  nodesG.setAttribute("class", "nodes");
  inner.appendChild(nodesG);

  for (const n of nodes) {
    const x = n._x + shift - NODE_W / 2;
    const y = n._y - NODE_H / 2;

    const g = document.createElementNS(SVG_NS, "g");
    g.setAttribute("class", "gnode");
    g.setAttribute("data-id", n.id);
    g.setAttribute("data-type", n.type);
    g.setAttribute("transform", `translate(${x}, ${y})`);

    const rect = document.createElementNS(SVG_NS, "rect");
    rect.setAttribute("width", NODE_W);
    rect.setAttribute("height", NODE_H);
    g.appendChild(rect);

    const t1 = document.createElementNS(SVG_NS, "text");
    t1.setAttribute("class", "type-label");
    t1.setAttribute("x", NODE_W / 2);
    t1.setAttribute("y", 26);
    t1.setAttribute("text-anchor", "middle");
    t1.textContent = n.type;
    g.appendChild(t1);

    const t2 = document.createElementNS(SVG_NS, "text");
    t2.setAttribute("class", "meta");
    t2.setAttribute("x", NODE_W / 2);
    t2.setAttribute("y", 48);
    t2.setAttribute("text-anchor", "middle");
    t2.textContent = metaLine(n);
    g.appendChild(t2);

    g.addEventListener("click", (e) => {
      e.stopPropagation();
      selectNode(n.id);
    });

    nodesG.appendChild(g);
  }

  graphEl.appendChild(svg);
  enablePanZoom(svg, inner, fullW, fullH);
}

function metaLine(n) {
  const MAX = 22;
  const trunc = (s) => (s.length > MAX ? s.slice(0, MAX - 1) + "…" : s);
  if (n.type === "section" && n.attribs?.title) return trunc(n.attribs.title);
  if (n.text) return trunc(n.text);
  if (n.type === "list" && n.attribs?.ordered) {
    return n.attribs.ordered === "true" ? "ordered" : "unordered";
  }
  if (n.type === "doc") return `${(n.children || []).length} children`;
  return n.id;
}

/* ---------- pan & zoom ---------- */
function enablePanZoom(svg, inner, fullW, fullH) {
  let scale = 1;
  let tx = 0;
  let ty = 0;
  let dragging = false;
  let startX = 0, startY = 0;

  function apply() {
    inner.setAttribute("transform", `translate(${tx} ${ty}) scale(${scale})`);
  }

  // Initial fit: shrink-to-fit if the tree is wider/taller than the viewport,
  // but never scale UP (small trees show at 1:1 so nodes stay readable).
  function fit() {
    const cw = graphEl.clientWidth || fullW;
    const ch = graphEl.clientHeight || fullH;
    const fitScale = Math.min(cw / fullW, ch / fullH, 1);
    scale = Math.max(0.3, fitScale);
    // Center horizontally, top-align vertically with a small margin.
    tx = (cw - fullW * scale) / 2;
    ty = 16;
    apply();
  }
  fit();

  graphEl.addEventListener("mousedown", (e) => {
    if (e.target.closest(".gnode")) return; // let click handler run
    dragging = true;
    graphEl.classList.add("dragging");
    startX = e.clientX - tx;
    startY = e.clientY - ty;
  });
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    tx = e.clientX - startX;
    ty = e.clientY - startY;
    apply();
  });
  window.addEventListener("mouseup", () => {
    dragging = false;
    graphEl.classList.remove("dragging");
  });

  graphEl.addEventListener("wheel", (e) => {
    e.preventDefault();
    const delta = -e.deltaY * 0.0015;
    const newScale = Math.min(4, Math.max(0.2, scale * (1 + delta)));
    // zoom around cursor
    const rect = graphEl.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    tx = cx - (cx - tx) * (newScale / scale);
    ty = cy - (cy - ty) * (newScale / scale);
    scale = newScale;
    apply();
  }, { passive: false });

  document.getElementById("zoom-in").onclick = () => { scale = Math.min(4, scale * 1.2); apply(); };
  document.getElementById("zoom-out").onclick = () => { scale = Math.max(0.2, scale / 1.2); apply(); };
  document.getElementById("zoom-reset").onclick = () => { fit(); };
}

/* ---------- inspector ---------- */
function selectNode(id) {
  selectedId = id;
  document.querySelectorAll(".gnode.selected").forEach((g) => g.classList.remove("selected"));
  const g = document.querySelector(`.gnode[data-id="${id}"]`);
  if (g) g.classList.add("selected");

  const n = nodeIndex.get(id);
  if (!n) return;
  showInspector(n);
}

inspectorClose.addEventListener("click", () => {
  inspectorEl.classList.add("hidden");
  document.querySelectorAll(".gnode.selected").forEach((g) => g.classList.remove("selected"));
  selectedId = null;
});

function showInspector(n) {
  const parent = n.parent_id ? nodeIndex.get(n.parent_id) : null;
  const siblings = parent ? parent.children : null;
  const siblingIdx = siblings ? siblings.findIndex((c) => c.id === n.id) : -1;

  const pathParts = [];
  let cur = n;
  while (cur) {
    const label = cur.type === "section" && cur.attribs?.title
      ? `${cur.type}[${cur.attribs.title}]`
      : cur.type;
    pathParts.unshift(label);
    cur = cur.parent_id ? nodeIndex.get(cur.parent_id) : null;
  }

  inspectorBody.innerHTML = "";
  const dl = document.createElement("dl");

  appendKV(dl, "id", n.id);
  appendKV(dl, "type", `<span class="pill">${escapeHtml(n.type)}</span>`, true);
  appendKV(dl, "parent_id", n.parent_id ?? "(root)");
  appendKV(dl, "children", `${(n.children || []).length}`);
  if (siblingIdx >= 0) appendKV(dl, "sibling idx", `${siblingIdx} of ${siblings.length}`);
  appendKV(dl, "path", pathParts.join("  ›  "));

  inspectorBody.appendChild(dl);

  if (n.attribs && Object.keys(n.attribs).length) {
    addBlock("attribs", JSON.stringify(n.attribs, null, 2));
  } else {
    addBlock("attribs", "{}");
  }

  if (n.text != null) {
    addBlock("text", n.text);
  }

  // raw JSON (without recursing into children, to keep it focused)
  const flat = { ...n };
  flat.children = (n.children || []).map((c) => ({ id: c.id, type: c.type }));
  addBlock("raw (children flattened)", JSON.stringify(flat, null, 2));

  inspectorEl.classList.remove("hidden");
}

function appendKV(dl, k, v, isHtml = false) {
  const dt = document.createElement("dt");
  dt.textContent = k;
  const dd = document.createElement("dd");
  if (isHtml) dd.innerHTML = v;
  else dd.textContent = v;
  dl.appendChild(dt);
  dl.appendChild(dd);
}

function addBlock(title, content) {
  const h = document.createElement("dt");
  h.textContent = title;
  h.style.marginTop = "10px";
  const pre = document.createElement("div");
  pre.className = "kv-block";
  pre.textContent = content;
  inspectorBody.appendChild(h);
  inspectorBody.appendChild(pre);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}
