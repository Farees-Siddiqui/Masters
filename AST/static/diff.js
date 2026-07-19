/* Structural document diff.
 *
 * Two processed documents are aligned segment-to-segment on the server; this
 * page renders the result as a unified, colour-coded diff (added / removed /
 * modified / unchanged). Clicking a change opens the alignment viewer focused on
 * that node, so every diff is traceable to the exact region of the page. */

const docASel = document.getElementById("doc-a");
const docBSel = document.getElementById("doc-b");
const compareBtn = document.getElementById("compare");
const showUnchanged = document.getElementById("show-unchanged");
const statusEl = document.getElementById("status");
const summaryEl = document.getElementById("summary");
const listEl = document.getElementById("diff-list");
const emptyEl = document.getElementById("empty");

let lastItems = [];

init();

async function init() {
  try {
    const res = await fetch("/api/corpus/list");
    const docs = (await res.json()).docs || [];
    if (docs.length < 2) {
      statusEl.textContent = `Need at least 2 processed documents (have ${docs.length}). Align more on the Alignment tab.`;
    }
    for (const sel of [docASel, docBSel]) {
      sel.innerHTML = "";
      for (const d of docs) {
        const opt = document.createElement("option");
        opt.value = d.doc;
        opt.textContent = `${d.title} (${d.page_count}pp)`;
        sel.appendChild(opt);
      }
    }
    if (docs.length >= 2) docBSel.selectedIndex = 1;  // default to two different docs
  } catch (e) {
    statusEl.textContent = `Error loading corpus: ${e.message}`;
  }
}

compareBtn.addEventListener("click", runDiff);
showUnchanged.addEventListener("change", () => renderDiff(lastItems));

async function runDiff() {
  const a = docASel.value, b = docBSel.value;
  if (!a || !b) return;
  if (a === b) { statusEl.textContent = "Pick two different documents."; return; }
  statusEl.textContent = "Aligning structures…";
  compareBtn.disabled = true;
  try {
    const res = await fetch(`/api/diff?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`);
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `HTTP ${res.status}`);
    const data = await res.json();
    statusEl.textContent = "";
    renderSummary(data.counts);
    lastItems = data.items || [];
    renderDiff(lastItems);
  } catch (e) {
    statusEl.textContent = `Error: ${e.message}`;
  } finally {
    compareBtn.disabled = false;
  }
}

function renderSummary(counts) {
  summaryEl.hidden = false;
  summaryEl.innerHTML = "";
  const chips = [
    ["added", counts.added, "+"],
    ["removed", counts.removed, "−"],
    ["modified", counts.modified, "~"],
    ["unchanged", counts.unchanged, "="],
  ];
  for (const [type, n, sym] of chips) {
    const chip = document.createElement("span");
    chip.className = `sum-chip ${type}`;
    chip.innerHTML = `<span class="sym">${sym}</span> ${n} ${type}`;
    summaryEl.appendChild(chip);
  }
}

function renderDiff(items) {
  listEl.innerHTML = "";
  if (!items.length) { listEl.appendChild(emptyEl); emptyEl.hidden = false; return; }
  emptyEl.hidden = true;

  const a = docASel.value, b = docBSel.value;
  let shown = 0;
  for (const it of items) {
    if (it.type === "unchanged" && !showUnchanged.checked) continue;
    shown++;
    const row = document.createElement("div");
    row.className = `diff-item ${it.type}`;

    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = it.type;
    row.appendChild(tag);

    const body = document.createElement("div");
    body.className = "body";
    if (it.type === "modified") {
      const oldL = document.createElement("div");
      oldL.className = "line old";
      oldL.textContent = it.a_text;
      const newL = document.createElement("div");
      newL.className = "line new";
      newL.textContent = it.b_text;
      body.appendChild(oldL);
      body.appendChild(newL);
    } else {
      const line = document.createElement("div");
      line.className = "line";
      line.textContent = it.text;
      body.appendChild(line);
    }
    row.appendChild(body);

    // Click -> open the aligner on the owning document, focused on the node.
    const onB = it.type === "added" || it.type === "modified" || it.type === "unchanged";
    const docId = onB ? b : a;
    const nodeId = onB ? it.b_id : it.a_id;
    row.title = "open in aligner — highlight on the page";
    row.addEventListener("click", () => {
      window.open(`/align?doc=${encodeURIComponent(docId)}&focus=${encodeURIComponent(nodeId)}`, "_blank");
    });

    listEl.appendChild(row);
  }
  if (shown === 0) {
    const note = document.createElement("div");
    note.className = "empty";
    note.textContent = "No changes to show (toggle “show unchanged” to see matched segments).";
    listEl.appendChild(note);
  }
}
