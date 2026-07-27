/* Alignment benchmark dashboard — two modes, answering different questions.
 *
 * `real` (default): token-level scores on a real document against gold derived
 * from the PDF's own text layer (`python -m benchmark real`). This is the
 * cross-engine task the deployed system faces, and the only mode that can see
 * the NULL class — regions no AST node owns. Its headline is not F1: prose is
 * essentially solved, while both aligners invent a source for most page
 * furniture, which is the error that matters for provenance.
 *
 * `v2`: node-level localization (`python -m benchmark_v2_0 --all`). Gold maps
 * each AST node to the set of layout boxes it owns (or ABSENT), in three tiers
 * of trust — human > audited > unverified oracle — and a node scores when the
 * pixel-area IoU of predicted vs. gold box unions clears the threshold. Unlike
 * the token view this asks "did we find the right boxes", not "the right
 * characters", and its false-claim rate is per furniture *box*, not per token.
 *
 * `synthetic`: the OCR-noise sweep (`python -m benchmark synthetic`) as an SVG
 * line chart. Gold is built from the AST's own text, so it measures robustness
 * to character noise under a known box→node owner — a controlled stress test,
 * not a claim about real documents. Each method owns a regime: positional
 * `stream` dominates with reading order intact and collapses when shuffled,
 * where semantic `similarity` takes over.
 *
 * No chart library — the frontend is dependency-free, so axes/lines/points are
 * drawn by hand.
 */

const SVG_NS = "http://www.w3.org/2000/svg";
const chart = document.getElementById("chart");
const legendEl = document.getElementById("legend");
const titleEl = document.getElementById("chart-title");
const takeawayEl = document.getElementById("takeaway");
const tableWrap = document.getElementById("table-wrap");
const statusEl = document.getElementById("status");

const METHODS = [
  { key: "stream", label: "stream (positional)", color: "#7aa2ff" },
  { key: "similarity", label: "similarity (semantic)", color: "#ffb86b" },
];
const METRIC_LABEL = { f1: "F1", precision: "precision", recall: "recall", owner_acc: "owner accuracy" };

// Plot geometry (in the 760×400 viewBox).
const W = 760, H = 400, padL = 54, padR = 18, padT = 18, padB = 46;
const plotW = W - padL - padR, plotH = H - padT - padB;

let regimes = [];           // [{key, label, rows}]
let real = null;            // the real-document run, or null if never generated
let v2 = null;              // the node-localization run, or null if never generated
let regimeKey = "intact";
let metric = "f1";
let mode = "real";

// Per-regime narrative — accurate to the shipped numbers.
const TAKEAWAY = {
  intact:
    "With reading order intact, the positional stream aligner stays near-perfect " +
    "(F1 ≥ 0.86 through 40% character noise). Semantic similarity is ~3× faster " +
    "but decays past ~10% noise as embeddings drift.",
  shuffled:
    "Shuffle the box order (multi-column or disrupted reading order) and stream " +
    "collapses to F1 ≈ 0.08 — it aligns by position. Semantic similarity is " +
    "order-invariant and holds F1 ≈ 0.92 to 10% noise. Each method owns a regime.",
};

init();

async function init() {
  statusEl.textContent = "Loading results…";
  try {
    const res = await fetch("/api/benchmark");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    regimes = data.regimes || [];
    real = data.real || null;
    v2 = data.v2 || null;
    statusEl.textContent = "";
    // ?mode=synthetic|real|v2 deep-links a view, so a specific result is shareable.
    const wanted = new URLSearchParams(location.search).get("mode");
    if (wanted === "synthetic" || wanted === "real" || wanted === "v2") mode = wanted;
    if (mode === "real" && !real) mode = v2 ? "v2" : "synthetic";
    if (mode === "v2" && !v2) mode = real ? "real" : "synthetic";
    syncMode();
    render();
  } catch (e) {
    statusEl.textContent = `Error: ${e.message}`;
  }
}

/* ---------- controls ---------- */
document.getElementById("mode-seg").addEventListener("click", (e) => {
  const b = e.target.closest("[data-mode]");
  if (!b) return;
  mode = b.dataset.mode;
  setActive("mode-seg", b);
  syncMode();
  render();
});

const SUBTITLE = {
  real: "Real documents vs. PDF text-layer gold &mdash; <em>prose is solved; page furniture is not.</em>",
  v2: "Node-level localization vs. tiered gold &mdash; <em>did the aligner find the right boxes?</em>",
  synthetic: "Synthetic OCR-noise sweep, gold by construction &mdash; <em>each method has a regime.</em>",
};

function syncMode() {
  const isSynthetic = mode === "synthetic";
  document.getElementById("real-view").hidden = mode !== "real";
  document.getElementById("v2-view").hidden = mode !== "v2";
  document.getElementById("synthetic-view").hidden = !isSynthetic;
  document.getElementById("regime-seg").hidden = !isSynthetic;
  document.getElementById("metric-seg").hidden = !isSynthetic;
  document.getElementById("page-sub").innerHTML = SUBTITLE[mode];
  // Reflect the active mode button even when set programmatically.
  document.querySelectorAll("#mode-seg .seg-btn").forEach((x) =>
    x.classList.toggle("active", x.dataset.mode === mode)
  );
}

document.getElementById("regime-seg").addEventListener("click", (e) => {
  const b = e.target.closest("[data-regime]");
  if (!b) return;
  regimeKey = b.dataset.regime;
  setActive("regime-seg", b);
  render();
});
document.getElementById("metric-seg").addEventListener("click", (e) => {
  const b = e.target.closest("[data-metric]");
  if (!b) return;
  metric = b.dataset.metric;
  setActive("metric-seg", b);
  render();
});
function setActive(segId, btn) {
  document.getElementById(segId).querySelectorAll(".seg-btn").forEach((x) => x.classList.remove("active"));
  btn.classList.add("active");
}

/* ---------- render ---------- */
function curRegime() {
  return regimes.find((r) => r.key === regimeKey) || { rows: [] };
}

function render() {
  if (mode === "real") return renderReal();
  if (mode === "v2") return renderV2();
  const rows = curRegime().rows;
  titleEl.textContent = `${METRIC_LABEL[metric]} vs. OCR noise — ${curRegime().label || ""}`;
  takeawayEl.textContent = TAKEAWAY[regimeKey] || "";
  drawLegend();
  drawChart(rows);
  drawTable(rows);
}

/* ---------- real-document view ---------- */

// A metric is `null` when it is undefined, not zero — a stratum of pure page
// furniture has no gold owner to recall. Rendering those as 0.000 would read as
// total failure instead of "not applicable".
const fmt = (v, digits = 3) => (v === null || v === undefined ? "—" : v.toFixed(digits));
const pct = (v) => (v === null || v === undefined ? "—" : `${(v * 100).toFixed(1)}%`);

function renderReal() {
  if (!real) {
    document.getElementById("real-meta").textContent =
      "No real-document results yet — run `python -m benchmark real`.";
    return;
  }
  const o = real.oracle;
  const nDocs = real.n_docs ?? 1;
  const scope =
    nDocs > 1
      ? `<strong>${nDocs} documents</strong> · corpus <code>${real.corpus || "?"}</code>`
      : `<code>${real.pdf}</code> · <strong>1 document</strong>`;
  document.getElementById("real-meta").innerHTML =
    `${scope} · ${real.n_pages} pages · ${real.n_boxes} boxes · ` +
    `${real.n_tokens.toLocaleString()} text-layer tokens · granularity <code>${real.granularity}</code>`;

  drawKpis();
  drawRealOverall();
  drawPerDoc();
  // Split strata by MAJORITY: pooled over enough documents a figure region picks
  // up a stray owned token, and "has any owner" would file it under prose and
  // print a meaningless F1 of 0.000 for what is really page furniture.
  const isFurniture = (s) => s.n_gold_null > s.n_gold_owned;
  drawStrata("real-prose", (s) => !isFurniture(s) && s.n_gold_owned > 0, "f1");
  drawStrata("real-furniture", isFurniture, "null_accuracy");
  drawOracle(o);

  // Narrative derived from the shipped numbers, not hardcoded.
  const fa = (r) => r.overall.false_attribution_rate ?? 1;
  const best = real.results.reduce((a, b) => (a.overall.f1 >= b.overall.f1 ? a : b));
  const worstAttrib = real.results.reduce((a, b) => (fa(a) >= fa(b) ? a : b));
  const bestAttrib = real.results.reduce((a, b) => (fa(a) <= fa(b) ? a : b));
  const scopeTxt =
    nDocs > 1
      ? `Across <strong>${nDocs} documents</strong> (${real.n_pages} pages)`
      : `On a single document (${real.n_pages} pages)`;
  const caveat =
    nDocs > 1
      ? ` All ${nDocs} are scientific papers, so this is one domain — it says nothing yet about ` +
        `financial, legal or form-heavy layouts.`
      : ` This is one document; treat it as indicative, not a corpus result.`;
  // The two rankings disagree, and that IS the finding — so report both leaders
  // rather than letting the F1 winner imply it is the best method overall.
  const split =
    best.method === bestAttrib.method
      ? `<code>${best.method}</code> leads on both.`
      : `<code>${best.method}</code> leads on F1 but <code>${bestAttrib.method}</code> ` +
        `is far better at knowing when to stay quiet — the two rankings disagree, and for ` +
        `provenance the second one is the one that matters.`;
  document.getElementById("real-takeaway").innerHTML =
    `${scopeTxt}, every method reconstructs <strong>prose</strong> well ` +
    `(best F1 ${fmt(best.overall.f1)}, <code>${best.method}</code>) — so F1 barely separates them. ` +
    `Where they split is <strong>abstention</strong>: on regions no node owns, ` +
    `<code>${worstAttrib.method}</code> invents a source for ` +
    `<strong>${pct(worstAttrib.overall.false_attribution_rate)}</strong> of tokens, while ` +
    `<code>${bestAttrib.method}</code> is down to ` +
    `<strong>${pct(bestAttrib.overall.false_attribution_rate)}</strong>. ${split}` +
    caveat;
}

function drawKpis() {
  const row = document.getElementById("kpi-row");
  row.innerHTML = "";
  for (const r of real.results) {
    const ov = r.overall;
    const card = document.createElement("div");
    card.className = "kpi";
    card.innerHTML =
      `<span class="kpi-method m-${r.method}">${r.method}</span>` +
      `<div class="kpi-main"><span class="kpi-val">${fmt(ov.f1)}</span><span class="kpi-lab">prose F1</span></div>` +
      `<div class="kpi-sub"><span class="bad">${pct(ov.false_attribution_rate)}</span> false attribution</div>` +
      `<div class="kpi-sub">${r.secs.toFixed(1)}s</div>`;
    row.appendChild(card);
  }
}

function drawRealOverall() {
  const multi = (real.n_docs ?? 1) > 1;
  const cols = [
    ["precision", "P"],
    ["recall", "R"],
    ["f1", multi ? "F1 (µ)" : "F1"],
    ...(multi ? [["macro_f1", "F1 (M)"]] : []),
    ["accuracy", "acc"],
    ["accuracy_up_to_ambiguity", "acc ~amb"],
    ["null_accuracy", "abstain acc"],
    ["false_attribution_rate", "false attrib"],
  ];
  let html = "<table class='bench-table'><thead><tr><th>method</th>";
  for (const [, lab] of cols) html += `<th>${lab}</th>`;
  html += "<th>excluded</th><th>secs</th></tr></thead><tbody>";
  for (const r of real.results) {
    const ov = r.overall;
    html += `<tr><td class="m-${r.method}">${r.method}</td>`;
    for (const [key] of cols) {
      // macro_f1 lives on the result, every other metric on `overall`.
      const v = key === "macro_f1" ? r.macro_f1 : ov[key];
      const cls = key === "false_attribution_rate" ? "bad" : "";
      html += `<td class="${cls}">${fmt(v)}</td>`;
    }
    html += `<td>${ov.n_excluded}</td><td>${r.secs.toFixed(1)}s</td></tr>`;
  }
  html += "</tbody></table>";
  document.getElementById("real-overall").innerHTML = html;
}

function drawPerDoc() {
  const wrap = document.getElementById("real-perdoc");
  const docs = real.docs || [];
  if (docs.length < 2) {
    wrap.innerHTML = "<p class='hint'>Single document — no spread to show.</p>";
    return;
  }
  const methods = real.results.map((r) => r.method);
  let html = "<table class='bench-table'><thead><tr><th>document</th><th>pages</th>" +
    "<th>tokens</th><th>oracle placed</th>";
  for (const m of methods) html += `<th>${m} F1</th>`;
  html += "</tr></thead><tbody>";
  for (const d of docs) {
    html += `<tr><td><code>${d.doc}</code></td><td>${d.n_pages}</td>` +
      `<td>${d.n_tokens.toLocaleString()}</td><td>${pct(d.oracle.placement_rate)}</td>`;
    for (const m of methods) {
      const r = d.results.find((x) => x.method === m);
      const v = r ? r.overall.f1 : null;
      let cls = "";
      if (v !== null && v !== undefined) cls = v >= 0.9 ? "good" : v >= 0.5 ? "mid" : "bad";
      html += `<td class="${cls}">${fmt(v)}</td>`;
    }
    html += "</tr>";
  }
  html += "</tbody></table>";
  wrap.innerHTML = html;
}

function drawStrata(elId, keep, metricKey) {
  const first = real.results[0].by_label;
  const labels = Object.keys(first).filter((l) => keep(first[l])).sort();
  if (!labels.length) {
    document.getElementById(elId).innerHTML = "<p class='hint'>none</p>";
    return;
  }
  let html = "<table class='bench-table'><thead><tr><th>region</th><th>tokens</th>";
  for (const r of real.results) html += `<th>${r.method}</th>`;
  html += "</tr></thead><tbody>";
  for (const lab of labels) {
    const s0 = first[lab];
    const n = s0.n_tokens - s0.n_excluded;
    html += `<tr><td><code>${lab}</code></td><td>${n}</td>`;
    for (const r of real.results) {
      const s = r.by_label[lab];
      const v = s ? s[metricKey] : null;
      // Colour by outcome so a 0.05 abstention score can't be skimmed past.
      let cls = "";
      if (v !== null && v !== undefined) cls = v >= 0.9 ? "good" : v >= 0.5 ? "mid" : "bad";
      html += `<td class="${cls}">${fmt(v)}</td>`;
    }
    html += "</tr>";
  }
  html += "</tbody></table>";
  document.getElementById(elId).innerHTML = html;
}

function drawOracle(o) {
  const rows = [
    ["AST nodes placed", `${o.n_placed} / ${o.n_nodes}`, pct(o.placement_rate)],
    ["Token coverage", `${o.n_gold_tokens} / ${o.n_tokens}`, pct(o.token_coverage)],
    ["NULL-class tokens", o.n_null_tokens, "no node owns these"],
    ["Unplaced nodes", o.n_unplaced, "excluded from gold, not guessed"],
    ["Placement conflicts", o.n_conflicts, "overlapping spans"],
  ];
  let html = "<table class='bench-table'><tbody>";
  for (const [k, v, note] of rows) {
    html += `<tr><td>${k}</td><td><strong>${v}</strong></td><td class="hint">${note}</td></tr>`;
  }
  html += "</tbody></table>";
  if (o.unplaced_samples && o.unplaced_samples.length) {
    html +=
      "<p class='hint' style='margin-top:8px'>Unplaced (sample) — LaTeX, image refs and bare numbers have no literal glyph run:</p><ul class='unplaced'>" +
      o.unplaced_samples.map((s) => `<li><code>${escapeHtml(s)}</code></li>`).join("") +
      "</ul>";
  }
  document.getElementById("real-oracle").innerHTML = html;
}

/* ---------- node-localization (v2) view ---------- */

// Colour thresholds: localization accuracy is good when high, false claim rate
// is good when low — using one scale for both would paint failures green.
const clsHi = (v) => (v === null || v === undefined ? "" : v >= 0.9 ? "good" : v >= 0.5 ? "mid" : "bad");
const clsLo = (v) => (v === null || v === undefined ? "" : v <= 0.1 ? "good" : v <= 0.5 ? "mid" : "bad");

function v2Methods() {
  // Keep the canonical METHODS order, then anything new the backend added.
  const known = METHODS.map((m) => m.key).filter((k) => v2.aggregate[k]);
  const extra = Object.keys(v2.aggregate).filter((k) => !known.includes(k)).sort();
  return [...known, ...extra];
}

function renderV2() {
  if (!v2) {
    document.getElementById("v2-meta").textContent =
      "No node-localization results yet — run `python -m benchmark_v2_0 --all --out results_v2_all.json`.";
    return;
  }
  const methods = v2Methods();
  const docs = v2.docs || [];
  const tiers = { human: 0, audited: 0, oracle: 0 };
  let nNodes = 0, nUngraded = 0, nFurniture = 0;
  for (const d of docs) {
    nNodes += d.gold.n_gold_nodes;
    nUngraded += d.gold.n_ungraded;
    nFurniture += d.gold.n_furniture;
    for (const t of Object.keys(tiers)) tiers[t] += d.gold.by_tier[t] || 0;
  }
  const verified = nNodes ? (tiers.human + tiers.audited) / nNodes : 0;

  document.getElementById("v2-meta").innerHTML =
    `<strong>${docs.length} documents</strong> · ${nNodes.toLocaleString()} gold nodes · ` +
    `IoU &ge; ${v2.iou_threshold} · granularity <code>${v2.granularity}</code> · ` +
    `tiers ${tiers.human} human / ${tiers.audited} audited / ${tiers.oracle} oracle ` +
    `(<strong>${pct(verified)}</strong> verified) · ` +
    `${nFurniture} furniture boxes · ${nUngraded.toLocaleString()} nodes ungraded` +
    (v2.human_only ? " · <strong>human-only gold</strong>" : "");

  drawV2Kpis(methods);
  drawV2Aggregate(methods);
  drawV2PerDoc(methods);
  drawV2ByKind(methods);
  drawV2Tiers(methods);

  // Narrative derived from the shipped numbers, not hardcoded.
  const acc = (m) => v2.aggregate[m].localization_accuracy;
  const claim = (m) => v2.aggregate[m].false_claim_rate;
  const best = methods.reduce((a, b) => (acc(a) >= acc(b) ? a : b));
  const bestClaim = methods.reduce((a, b) => (claim(a) <= claim(b) ? a : b));
  const split =
    best === bestClaim
      ? `<code>${best}</code> also has the lowest false-claim rate (${pct(claim(bestClaim))}).`
      : `but <code>${bestClaim}</code> claims less furniture ` +
        `(${pct(claim(bestClaim))} vs. ${pct(claim(best))}) — the two rankings disagree.`;
  document.getElementById("v2-takeaway").innerHTML =
    `Across <strong>${docs.length} documents</strong>, <code>${best}</code> localizes ` +
    `<strong>${pct(acc(best))}</strong> of gold nodes to the right boxes, ${split} ` +
    `Only <strong>${pct(verified)}</strong> of the gold is human-verified so far — the rest ` +
    `is the unverified oracle tier, so treat small per-kind gaps as leads, not conclusions.`;
}

function drawV2Kpis(methods) {
  const row = document.getElementById("v2-kpi-row");
  row.innerHTML = "";
  for (const m of methods) {
    const a = v2.aggregate[m];
    const card = document.createElement("div");
    card.className = "kpi";
    card.innerHTML =
      `<span class="kpi-method m-${m}">${m}</span>` +
      `<div class="kpi-main"><span class="kpi-val">${fmt(a.localization_accuracy)}</span>` +
      `<span class="kpi-lab">localization acc.</span></div>` +
      `<div class="kpi-sub">mean IoU ${fmt(a.mean_iou)}</div>` +
      `<div class="kpi-sub"><span class="bad">${pct(a.false_claim_rate)}</span> false claims</div>`;
    row.appendChild(card);
  }
}

function drawV2Aggregate(methods) {
  let html = "<table class='bench-table'><thead><tr><th>method</th><th>nodes</th>" +
    "<th>loc. acc</th><th>mean IoU</th><th>exact</th><th>missed</th>" +
    "<th>absent acc</th><th>false claim</th></tr></thead><tbody>";
  for (const m of methods) {
    const a = v2.aggregate[m];
    html += `<tr><td class="m-${m}">${m}</td><td>${a.n_nodes.toLocaleString()}</td>` +
      `<td class="${clsHi(a.localization_accuracy)}">${fmt(a.localization_accuracy)}</td>` +
      `<td>${fmt(a.mean_iou)}</td><td>${a.n_exact.toLocaleString()}</td><td>${a.n_missed}</td>` +
      `<td>${fmt(a.absent_accuracy)}</td>` +
      `<td class="${clsLo(a.false_claim_rate)}">${fmt(a.false_claim_rate)}</td></tr>`;
  }
  html += "</tbody></table>";
  document.getElementById("v2-aggregate").innerHTML = html;
}

function drawV2PerDoc(methods) {
  let html = "<table class='bench-table'><thead><tr><th>document</th><th>gold nodes</th>" +
    "<th>verified</th><th>furniture</th>";
  for (const m of methods) html += `<th>${m} loc. acc</th>`;
  for (const m of methods) html += `<th>${m} false claim</th>`;
  html += "</tr></thead><tbody>";
  for (const d of v2.docs) {
    const g = d.gold;
    const ver = g.n_gold_nodes
      ? ((g.by_tier.human || 0) + (g.by_tier.audited || 0)) / g.n_gold_nodes
      : 0;
    html += `<tr><td><code>${d.doc}</code></td><td>${g.n_gold_nodes}</td>` +
      `<td>${pct(ver)}</td><td>${g.n_furniture}</td>`;
    for (const m of methods) {
      const v = d.results[m] ? d.results[m].localization_accuracy : null;
      html += `<td class="${clsHi(v)}">${fmt(v)}</td>`;
    }
    for (const m of methods) {
      const v = d.results[m] ? d.results[m].false_claim_rate : null;
      html += `<td class="${clsLo(v)}">${fmt(v)}</td>`;
    }
    html += "</tr>";
  }
  html += "</tbody></table>";
  document.getElementById("v2-perdoc").innerHTML = html;
}

function drawV2ByKind(methods) {
  // Union of kinds across methods, biggest population first — prose leads, the
  // contested small-n kinds (table, formula) sink to the bottom where their
  // counts are visible next to the score.
  const kinds = [...new Set(methods.flatMap((m) => Object.keys(v2.aggregate[m].by_kind)))]
    .sort((a, b) => (v2.aggregate[methods[0]].by_kind[b]?.n || 0) - (v2.aggregate[methods[0]].by_kind[a]?.n || 0));
  let html = "<table class='bench-table'><thead><tr><th>kind</th><th>n</th>";
  for (const m of methods) html += `<th>${m}</th>`;
  html += "</tr></thead><tbody>";
  for (const k of kinds) {
    const n = methods.map((m) => v2.aggregate[m].by_kind[k]?.n).find((x) => x !== undefined) ?? 0;
    html += `<tr><td><code>${k}</code></td><td>${n}</td>`;
    for (const m of methods) {
      const s = v2.aggregate[m].by_kind[k];
      const v = s ? s.localization_accuracy : null;
      html += `<td class="${clsHi(v)}">${fmt(v)}</td>`;
    }
    html += "</tr>";
  }
  html += "</tbody></table>";
  document.getElementById("v2-bykind").innerHTML = html;
}

function drawV2Tiers(methods) {
  const order = ["human", "audited", "oracle"];
  const present = order.filter((t) => methods.some((m) => v2.aggregate[m].by_tier[t]));
  let html = "<table class='bench-table'><thead><tr><th>tier</th><th>n</th>";
  for (const m of methods) html += `<th>${m}</th>`;
  html += "</tr></thead><tbody>";
  for (const t of present) {
    const n = methods.map((m) => v2.aggregate[m].by_tier[t]?.n).find((x) => x !== undefined) ?? 0;
    html += `<tr><td><code>${t}</code></td><td>${n.toLocaleString()}</td>`;
    for (const m of methods) {
      const s = v2.aggregate[m].by_tier[t];
      const v = s ? s.localization_accuracy : null;
      html += `<td class="${clsHi(v)}">${fmt(v)}</td>`;
    }
    html += "</tr>";
  }
  html += "</tbody></table>";
  document.getElementById("v2-tiers").innerHTML = html;
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

function seriesFor(rows, methodKey) {
  return rows
    .filter((r) => r.method === methodKey)
    .map((r) => ({ noise: r.noise, value: r[metric] }))
    .sort((a, b) => a.noise - b.noise);
}

function drawChart(rows) {
  chart.innerHTML = "";
  const noises = [...new Set(rows.map((r) => r.noise))].sort((a, b) => a - b);
  const maxNoise = noises.length ? Math.max(...noises) : 1;
  const xFor = (n) => padL + (maxNoise ? n / maxNoise : 0) * plotW;
  const yFor = (v) => padT + (1 - v) * plotH;

  // Horizontal gridlines + y labels (0 .. 1).
  for (let g = 0; g <= 4; g++) {
    const v = g / 4;
    const y = yFor(v);
    line(padL, y, W - padR, y, "grid");
    text(padL - 8, y + 4, v.toFixed(2), "axis-label y");
  }
  // X axis ticks at each noise level.
  for (const n of noises) {
    const x = xFor(n);
    line(x, padT, x, padT + plotH, "grid faint");
    text(x, H - padB + 18, `${Math.round(n * 100)}%`, "axis-label x");
  }
  text(padL + plotW / 2, H - 6, "character noise rate", "axis-title");
  text(16, padT + plotH / 2, METRIC_LABEL[metric], "axis-title rot", -90);

  // One line + points per method.
  for (const m of METHODS) {
    const pts = seriesFor(rows, m.key);
    if (pts.length === 0) continue;
    const d = pts.map((p, i) => `${i ? "L" : "M"} ${xFor(p.noise).toFixed(1)} ${yFor(p.value).toFixed(1)}`).join(" ");
    const path = el("path", { d, class: "series", stroke: m.color });
    chart.appendChild(path);
    for (const p of pts) {
      const c = el("circle", { cx: xFor(p.noise), cy: yFor(p.value), r: 4, class: "pt", fill: m.color });
      const tip = el("title", {});
      tip.textContent = `${m.key} @ ${Math.round(p.noise * 100)}% noise: ${p.value.toFixed(3)}`;
      c.appendChild(tip);
      chart.appendChild(c);
    }
  }
}

function drawLegend() {
  legendEl.innerHTML = "";
  for (const m of METHODS) {
    const item = document.createElement("span");
    item.className = "legend-item";
    item.innerHTML = `<span class="swatch" style="background:${m.color}"></span>${m.label}`;
    legendEl.appendChild(item);
  }
}

function drawTable(rows) {
  const cols = ["precision", "recall", "f1", "owner_acc", "secs"];
  const head = ["method", "noise", "P", "R", "F1", "own.acc", "secs"];
  let html = "<table class='bench-table'><thead><tr>";
  head.forEach((h, i) => {
    const keyForCol = ["", "", "precision", "recall", "f1", "owner_acc", ""][i];
    html += `<th class="${keyForCol === metric ? "hl" : ""}">${h}</th>`;
  });
  html += "</tr></thead><tbody>";
  const sorted = [...rows].sort((a, b) => a.noise - b.noise || a.method.localeCompare(b.method));
  for (const r of sorted) {
    html += `<tr><td class="m-${r.method}">${r.method}</td><td>${Math.round(r.noise * 100)}%</td>`;
    for (const c of cols) {
      const v = c === "secs" ? `${r[c].toFixed(1)}s` : r[c].toFixed(3);
      html += `<td class="${c === metric ? "hl" : ""}">${v}</td>`;
    }
    html += "</tr>";
  }
  html += "</tbody></table>";
  tableWrap.innerHTML = html;
}

/* ---------- tiny SVG helpers ---------- */
function el(tag, attrs) {
  const n = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  return n;
}
function line(x1, y1, x2, y2, cls) {
  chart.appendChild(el("line", { x1, y1, x2, y2, class: cls }));
}
function text(x, y, str, cls, rotate) {
  const t = el("text", { x, y, class: cls });
  if (rotate) t.setAttribute("transform", `rotate(${rotate} ${x} ${y})`);
  t.textContent = str;
  chart.appendChild(t);
}
