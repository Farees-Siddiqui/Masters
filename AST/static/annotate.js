/* Alignment annotation tool.
 *
 * Three queues (excluded / nulls / audit), one item at a time. Every verdict
 * POSTs to /api/annotate/save immediately — there is no local unsaved state
 * beyond the box selection being composed for an "excluded" item.
 */

"use strict";

const $ = (id) => document.getElementById(id);

const state = {
  doc: null, // /api/annotate/doc payload
  queue: "excluded", // active queue name
  idx: 0, // active item index within queue
  pageIdx: 0, // visible page index
  selectedBoxes: new Set(), // excluded-mode box selection
  annotations: {}, // item id -> record
};

// ---------------------------------------------------------------------------
// Bootstrapping
// ---------------------------------------------------------------------------

async function init() {
  const res = await fetch("/api/annotate/docs");
  const data = await res.json();
  const sel = $("doc");
  for (const d of data.docs) {
    const opt = document.createElement("option");
    opt.value = d.doc;
    opt.textContent = d.n_annotated ? `${d.doc} (${d.n_annotated} done)` : d.doc;
    sel.appendChild(opt);
  }
  sel.addEventListener("change", () => sel.value && loadDoc(sel.value));
  $("tabs").addEventListener("click", (e) => {
    const btn = e.target.closest(".tab");
    if (btn) setQueue(btn.dataset.queue);
  });
  $("prev-page").addEventListener("click", () => showPage(state.pageIdx - 1));
  $("next-page").addEventListener("click", () => showPage(state.pageIdx + 1));
  $("page-img").addEventListener("load", renderOverlay);
  window.addEventListener("resize", renderOverlay);
  document.addEventListener("keydown", onKey);
}

async function loadDoc(name) {
  setStatus(`deriving gold for ${name}… (first load reads the PDF text layer)`);
  const res = await fetch(`/api/annotate/doc?doc=${encodeURIComponent(name)}`);
  if (!res.ok) {
    setStatus(`failed: ${(await res.json()).detail || res.status}`);
    return;
  }
  state.doc = await res.json();
  state.annotations = state.doc.annotations || {};
  for (const q of ["excluded", "nulls", "audit"]) {
    $(`count-${q}`).textContent = state.doc.queues[q].length;
  }
  setStatus("");
  setQueue(state.queue);
}

function setQueue(name) {
  state.queue = name;
  document.querySelectorAll(".tab").forEach((t) =>
    t.classList.toggle("active", t.dataset.queue === name)
  );
  if (!state.doc) return;
  // Land on the first undecided item.
  const items = state.doc.queues[name];
  const firstOpen = items.findIndex((it) => !state.annotations[it.id]);
  setItem(firstOpen === -1 ? 0 : firstOpen);
}

// ---------------------------------------------------------------------------
// Item navigation + rendering
// ---------------------------------------------------------------------------

function items() {
  return state.doc ? state.doc.queues[state.queue] : [];
}
function current() {
  return items()[state.idx] || null;
}

function setItem(i) {
  const n = items().length;
  if (!n) {
    renderList();
    renderDetail();
    renderOverlay();
    updateProgress();
    return;
  }
  state.idx = Math.max(0, Math.min(n - 1, i));
  state.selectedBoxes = new Set(
    (state.annotations[current().id] || {}).boxes || []
  );
  jumpToItemPage();
  renderList();
  renderDetail();
  renderOverlay();
  updateProgress();
}

function itemPages(it) {
  // Pages an item touches, for auto-jump and the page hint.
  const pages = new Set();
  if (!it) return [];
  if (it.page) pages.add(it.page);
  for (const s of it.suggestions || []) pages.add(parseInt(s.box.split(":")[0], 10));
  for (const b of state.selectedBoxes) pages.add(parseInt(b.split(":")[0], 10));
  for (const k of it.token_keys || []) {
    const tok = state.doc.tokens[k];
    if (tok) pages.add(tok[0]);
  }
  return [...pages].sort((a, b) => a - b);
}

function jumpToItemPage() {
  const pgs = itemPages(current());
  if (!pgs.length) return;
  const idx = state.doc.pages.findIndex((p) => p.page === pgs[0]);
  if (idx !== -1) showPage(idx, false);
}

function showPage(i, rerender = true) {
  if (!state.doc) return;
  state.pageIdx = Math.max(0, Math.min(state.doc.pages.length - 1, i));
  const p = state.doc.pages[state.pageIdx];
  $("page-label").textContent = `page ${p.page} / ${state.doc.pages.length}`;
  const img = $("page-img");
  if (img.dataset.url !== p.image_url) {
    img.dataset.url = p.image_url;
    img.src = p.image_url; // overlay re-renders on load
  } else if (rerender) {
    renderOverlay();
  }
  const pgs = itemPages(current());
  $("page-hint").textContent = pgs.length
    ? `item touches page${pgs.length > 1 ? "s" : ""} ${pgs.join(", ")}`
    : "";
}

function renderList() {
  const el = $("item-list");
  el.innerHTML = "";
  items().forEach((it, i) => {
    const row = document.createElement("div");
    const rec = state.annotations[it.id];
    row.className =
      "item-row" + (i === state.idx ? " active" : "") + (rec ? " done" : "");
    const badge = badgeFor(it);
    row.innerHTML =
      (badge ? `<span class="badge">${badge}</span>` : "") +
      `<span class="snippet"></span>` +
      (rec ? `<span class="verdict ${verdictClass(rec.verdict)}">${rec.verdict}</span>` : "");
    row.querySelector(".snippet").textContent = snippetFor(it);
    row.addEventListener("click", () => setItem(i));
    el.appendChild(row);
  });
  const active = el.querySelector(".item-row.active");
  if (active) active.scrollIntoView({ block: "nearest" });
}

function badgeFor(it) {
  if (state.queue === "excluded") return it.kind === "too_short" ? "short" : "no-hit";
  if (state.queue === "nulls") return `${it.label} p${it.page}`;
  if (state.queue === "audit") return it.exact ? "exact" : `~${it.ratio}`;
  return "";
}
function snippetFor(it) {
  return (it.text || "").slice(0, 80) || "(empty)";
}
function verdictClass(v) {
  if (v === "accept" || v === "furniture" || v === "boxes" || v === "node") return "ok";
  if (v === "reject" || v === "absent") return "bad";
  return "neutral";
}

function updateProgress() {
  const all = items();
  const done = all.filter((it) => state.annotations[it.id]).length;
  $("progress").textContent = all.length ? `${done} / ${all.length} decided` : "";
}

// ---------------------------------------------------------------------------
// Page overlay
// ---------------------------------------------------------------------------

function renderOverlay() {
  const overlay = $("overlay");
  overlay.innerHTML = "";
  if (!state.doc) return;
  const page = state.doc.pages[state.pageIdx];
  const img = $("page-img");
  if (!img.clientWidth) return;
  const scale = img.clientWidth / page.width;
  const it = current();

  const suggested = new Set((it?.suggestions || []).map((s) => s.box));
  const focusBox = state.queue === "nulls" && it ? it.box : null;

  page.boxes.forEach((b, i) => {
    const key = `${page.page}:${i}`;
    const div = document.createElement("div");
    div.className = "obox";
    if (suggested.has(key)) div.classList.add("suggested");
    if (state.selectedBoxes.has(key)) div.classList.add("selected");
    if (key === focusBox) div.classList.add("focus");
    const [x0, y0, x1, y1] = b.bbox;
    Object.assign(div.style, {
      left: `${x0 * scale}px`,
      top: `${y0 * scale}px`,
      width: `${(x1 - x0) * scale}px`,
      height: `${(y1 - y0) * scale}px`,
    });
    div.title = `${key} ${b.label}\n${(b.text || "").slice(0, 120)}`;
    div.addEventListener("click", () => onBoxClick(key));
    overlay.appendChild(div);
  });

  // Token highlights for the current item (audit + nulls queues). Each token
  // carries [page, [rect, ...]] — one rect per line fragment, so a word
  // hyphenated across a line break draws as two word-sized highlights.
  for (const k of it?.token_keys || []) {
    const tok = state.doc.tokens[k];
    if (!tok || tok[0] !== page.page) continue;
    for (const [x0, y0, x1, y1] of tok[1]) {
      const div = document.createElement("div");
      div.className = "otok";
      Object.assign(div.style, {
        left: `${x0 * scale}px`,
        top: `${y0 * scale}px`,
        width: `${(x1 - x0) * scale}px`,
        height: `${(y1 - y0) * scale}px`,
      });
      overlay.appendChild(div);
    }
  }
}

function onBoxClick(key) {
  if (state.queue !== "excluded" || !current()) return;
  if (state.selectedBoxes.has(key)) state.selectedBoxes.delete(key);
  else state.selectedBoxes.add(key);
  renderOverlay();
  renderDetail();
}

// ---------------------------------------------------------------------------
// Detail pane
// ---------------------------------------------------------------------------

function renderDetail() {
  const el = $("detail");
  const it = current();
  if (!it) {
    el.innerHTML = `<p class="muted">Queue is empty.</p>`;
    return;
  }
  const rec = state.annotations[it.id];
  const saved = rec
    ? `<p class="saved-note">saved: <b>${rec.verdict}</b>` +
      (rec.boxes ? ` → ${rec.boxes.join(", ")}` : "") +
      (rec.node_id ? ` → ${rec.node_id}` : "") +
      ` <button data-act="undo" style="margin-left:6px;">undo</button></p>`
    : "";

  if (state.queue === "excluded") {
    const chips = [...state.selectedBoxes]
      .map((b) => `<span class="chip" data-box="${b}">${b} ✕</span>`)
      .join(" ");
    const sugg = (it.suggestions || [])
      .map((s) => `<span class="chip" data-sugg="${s.box}">${s.box} (${s.score})</span>`)
      .join(" ");
    el.innerHTML = `
      <h3>${it.node_id} — ${it.kind === "too_short" ? "too short to auto-place" : "no text-layer match"}</h3>
      <div class="node-text"></div>
      <div class="meta">Click the box(es) on the page that contain this node's content.</div>
      <div class="meta">${sugg ? "suggestions: " + sugg : "no box suggestions"}</div>
      <div class="meta">${chips ? "selected: " + chips : "nothing selected"}</div>
      <div class="actions">
        <button data-act="boxes" class="primary" ${state.selectedBoxes.size ? "" : "disabled"}>Save boxes (a)</button>
        <button data-act="absent" class="danger">Not on any page (x)</button>
        <button data-act="skip">Skip (s)</button>
      </div>${saved}`;
  } else if (state.queue === "nulls") {
    const s = it.suggested_node;
    const options = state.doc.nodes
      .map((n) => `<option value="${n.node_id}">${n.node_id}: ${esc(n.text.slice(0, 70))}</option>`)
      .join("");
    el.innerHTML = `
      <h3>box ${it.box} (${it.label}) — ${it.n_null}/${it.n_total} tokens owned by nobody</h3>
      <div class="node-text"></div>
      <div class="meta">Is this page furniture (page number, figure text, header), or does it belong to an AST node the oracle missed?</div>
      ${s ? `<div class="meta">suggested owner: <b>${s.node_id}</b> (overlap ${s.score}) — <button data-act="take-sugg">assign it</button></div>` : ""}
      <div class="meta">or pick a node:
        <select id="node-pick"><option value="">—</option>${options}</select>
      </div>
      <div class="actions">
        <button data-act="furniture" class="primary">Furniture / no owner (a)</button>
        <button data-act="node">Assign selected node</button>
        <button data-act="skip">Skip (s)</button>
      </div>${saved}`;
  } else {
    el.innerHTML = `
      <h3>${it.node_id} — oracle placement ${it.exact ? "exact" : `fuzzy, ratio ${it.ratio}`}</h3>
      <div class="node-text"></div>
      <div class="meta">${it.token_keys.length} tokens highlighted on the page. Does the highlight cover this node's real location?</div>
      <div class="actions">
        <button data-act="accept" class="primary">Correct (a)</button>
        <button data-act="reject" class="danger">Wrong (r)</button>
        <button data-act="skip">Skip (s)</button>
      </div>${saved}`;
  }
  el.querySelector(".node-text").textContent = it.text || "(no text)";

  el.querySelectorAll("[data-act]").forEach((b) =>
    b.addEventListener("click", () => onAction(b.dataset.act))
  );
  el.querySelectorAll("[data-box]").forEach((c) =>
    c.addEventListener("click", () => onBoxClick(c.dataset.box))
  );
  el.querySelectorAll("[data-sugg]").forEach((c) =>
    c.addEventListener("click", () => {
      const key = c.dataset.sugg;
      onBoxClick(key);
      const pg = parseInt(key.split(":")[0], 10);
      const idx = state.doc.pages.findIndex((p) => p.page === pg);
      if (idx !== -1) showPage(idx);
    })
  );
}

function esc(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");
}

// ---------------------------------------------------------------------------
// Verdicts
// ---------------------------------------------------------------------------

async function onAction(act) {
  const it = current();
  if (!it) return;
  if (act === "undo") return save(it, "clear");
  if (act === "take-sugg") return save(it, "node", { node_id: it.suggested_node.node_id });
  if (act === "node") {
    const pick = $("node-pick");
    if (!pick || !pick.value) return setStatus("pick a node first");
    return save(it, "node", { node_id: pick.value });
  }
  if (act === "boxes") {
    if (!state.selectedBoxes.size) return;
    return save(it, "boxes", { boxes: [...state.selectedBoxes] });
  }
  return save(it, act); // absent / furniture / accept / reject / skip
}

async function save(it, verdict, extra = {}) {
  const res = await fetch("/api/annotate/save", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ doc: state.doc.doc, id: it.id, verdict, ...extra }),
  });
  if (!res.ok) {
    setStatus(`save failed: ${(await res.json()).detail || res.status}`);
    return;
  }
  if (verdict === "clear") delete state.annotations[it.id];
  else state.annotations[it.id] = { verdict, ...extra };
  setStatus("");
  if (verdict === "clear") {
    setItem(state.idx); // stay put after undo
  } else {
    advance();
  }
}

function advance() {
  // Next undecided item after the current one, else next item, else stay.
  const all = items();
  for (let d = 1; d <= all.length; d++) {
    const j = (state.idx + d) % all.length;
    if (!state.annotations[all[j].id]) return setItem(j);
  }
  setItem(state.idx);
}

// ---------------------------------------------------------------------------
// Keyboard
// ---------------------------------------------------------------------------

function onKey(e) {
  if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
  const map = {
    j: () => setItem(state.idx + 1),
    k: () => setItem(state.idx - 1),
    s: () => onAction("skip"),
    u: () => onAction("undo"),
    ArrowLeft: () => showPage(state.pageIdx - 1),
    ArrowRight: () => showPage(state.pageIdx + 1),
  };
  if (state.queue === "excluded") {
    map.a = () => onAction("boxes");
    map.x = () => onAction("absent");
  } else if (state.queue === "nulls") {
    map.a = () => onAction("furniture");
  } else {
    map.a = () => onAction("accept");
    map.r = () => onAction("reject");
  }
  const fn = map[e.key];
  if (fn) {
    e.preventDefault();
    fn();
  }
}

function setStatus(msg) {
  $("status").textContent = msg;
}

init();
