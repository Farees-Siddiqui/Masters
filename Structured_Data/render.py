"""Render extracted records as a self-contained, clustered HTML view.

    python render.py records.json --out view.html

Groups records by key, shows each record's value with the verbatim evidence
quote it was grounded in, and flags anything the grounding check couldn't
verify. Output is a single static HTML file (data inlined) suitable for
publishing as an Artifact.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

TEMPLATE = """\
<style>
:root {
  --bg:#f6f8f9; --surface:#ffffff; --surface-2:#f0f3f5;
  --ink:#182029; --muted:#5c6a76; --faint:#8a97a1;
  --border:#e3e8ec; --accent:#0f766e; --accent-soft:#0f766e1a;
  --grounded:#15803d; --grounded-soft:#15803d1a;
  --ungrounded:#b45309; --ungrounded-soft:#b453091f;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg:#0d1117; --surface:#161c23; --surface-2:#1b232c;
    --ink:#e6edf3; --muted:#98a4b0; --faint:#6b7783;
    --border:#28313b; --accent:#2dd4bf; --accent-soft:#2dd4bf1f;
    --grounded:#4ade80; --grounded-soft:#4ade8018;
    --ungrounded:#fbbf24; --ungrounded-soft:#fbbf2418;
  }
}
:root[data-theme="light"] {
  --bg:#f6f8f9; --surface:#ffffff; --surface-2:#f0f3f5;
  --ink:#182029; --muted:#5c6a76; --faint:#8a97a1;
  --border:#e3e8ec; --accent:#0f766e; --accent-soft:#0f766e1a;
  --grounded:#15803d; --grounded-soft:#15803d1a;
  --ungrounded:#b45309; --ungrounded-soft:#b453091f;
}
:root[data-theme="dark"] {
  --bg:#0d1117; --surface:#161c23; --surface-2:#1b232c;
  --ink:#e6edf3; --muted:#98a4b0; --faint:#6b7783;
  --border:#28313b; --accent:#2dd4bf; --accent-soft:#2dd4bf1f;
  --grounded:#4ade80; --grounded-soft:#4ade8018;
  --ungrounded:#fbbf24; --ungrounded-soft:#fbbf2418;
}

* { box-sizing:border-box; }
body {
  margin:0; background:var(--bg); color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  line-height:1.5; -webkit-font-smoothing:antialiased;
}
.mono { font-family:ui-monospace,"Cascadia Code","SF Mono",Menlo,Consolas,monospace; }
.wrap { max-width:940px; margin:0 auto; padding:0 20px 80px; }

header.top {
  position:sticky; top:0; z-index:10;
  background:color-mix(in srgb, var(--bg) 88%, transparent);
  backdrop-filter:blur(8px); border-bottom:1px solid var(--border);
}
.top-inner { max-width:940px; margin:0 auto; padding:18px 20px 16px; }
.eyebrow {
  font-size:12px; letter-spacing:.14em; text-transform:uppercase;
  color:var(--accent); font-weight:600; margin:0 0 4px;
}
h1 { font-size:22px; line-height:1.2; margin:0; letter-spacing:-.01em; text-wrap:balance; }
.source-line { margin:6px 0 0; color:var(--muted); font-size:13px; }
.source-line .mono { color:var(--ink); }

.stats { display:flex; gap:10px; margin-top:14px; flex-wrap:wrap; }
.stat {
  flex:1 1 120px; background:var(--surface); border:1px solid var(--border);
  border-radius:10px; padding:10px 14px;
}
.stat .n { font-size:24px; font-weight:650; letter-spacing:-.02em; font-variant-numeric:tabular-nums; }
.stat .l { font-size:11px; text-transform:uppercase; letter-spacing:.1em; color:var(--muted); margin-top:2px; }

.controls { display:flex; gap:10px; align-items:center; margin:20px 0 6px; flex-wrap:wrap; }
#search {
  flex:1 1 240px; min-width:180px; padding:9px 12px; font-size:14px;
  background:var(--surface); color:var(--ink);
  border:1px solid var(--border); border-radius:9px;
}
#search:focus { outline:2px solid var(--accent); outline-offset:1px; border-color:transparent; }
.toggle { display:flex; align-items:center; gap:7px; font-size:13px; color:var(--muted); cursor:pointer; user-select:none; }
.toggle input { accent-color:var(--accent); width:15px; height:15px; }
.showing { font-size:13px; color:var(--faint); margin:2px 0 14px; }

.chips { display:flex; flex-wrap:wrap; gap:7px; margin-bottom:22px; }
.chip {
  font-size:12px; padding:5px 11px; border-radius:20px; cursor:pointer;
  background:var(--surface); border:1px solid var(--border); color:var(--muted);
  display:inline-flex; align-items:center; gap:6px; transition:.12s;
}
.chip:hover { border-color:var(--accent); color:var(--ink); }
.chip.active { background:var(--accent); border-color:var(--accent); color:#fff; }
:root[data-theme="dark"] .chip.active { color:#0d1117; }
@media (prefers-color-scheme: dark) { .chip.active { color:#0d1117; } }
.chip .c { font-variant-numeric:tabular-nums; opacity:.7; }
.chip .k { font-family:ui-monospace,"Cascadia Code",Menlo,Consolas,monospace; }

.cluster { margin-bottom:26px; scroll-margin-top:150px; }
.cluster-head { display:flex; align-items:baseline; gap:10px; margin-bottom:10px; }
.cluster-head .key {
  font-size:15px; font-weight:600; color:var(--ink);
  background:var(--accent-soft); padding:2px 9px; border-radius:7px;
}
.cluster-head .count { font-size:12px; color:var(--muted); font-variant-numeric:tabular-nums; }
.ratio { flex:1; height:4px; background:var(--surface-2); border-radius:3px; overflow:hidden; max-width:160px; }
.ratio > span { display:block; height:100%; background:var(--grounded); }

.rec {
  background:var(--surface); border:1px solid var(--border); border-radius:11px;
  padding:12px 14px; margin-bottom:8px;
  display:grid; grid-template-columns:1fr auto; gap:6px 12px; align-items:start;
}
.rec .value { font-size:15px; font-weight:550; grid-column:1; }
.rec .badges { grid-column:2; grid-row:1/3; display:flex; flex-direction:column; align-items:flex-end; gap:6px; }
.evidence {
  grid-column:1; font-size:12.5px; color:var(--muted);
  border-left:2px solid var(--border); padding-left:10px;
}
.evidence.ok { border-left-color:var(--grounded); }
.evidence.bad { border-left-color:var(--ungrounded); }
.evidence .lbl { text-transform:uppercase; letter-spacing:.1em; font-size:9.5px; color:var(--faint); display:block; margin-bottom:1px; }

.pill {
  font-size:10.5px; font-weight:600; padding:3px 8px; border-radius:20px;
  text-transform:uppercase; letter-spacing:.04em; white-space:nowrap;
}
.pill.ok  { background:var(--grounded-soft); color:var(--grounded); }
.pill.bad { background:var(--ungrounded-soft); color:var(--ungrounded); }
.conf { font-size:10px; color:var(--faint); text-transform:uppercase; letter-spacing:.06em; }
.conf.low { color:var(--ungrounded); }

.empty { color:var(--faint); text-align:center; padding:60px 0; }
footer { color:var(--faint); font-size:12px; margin-top:36px; padding-top:16px; border-top:1px solid var(--border); }
</style>

<header class="top">
  <div class="top-inner">
    <p class="eyebrow">Structured Data · stage 1 extraction</p>
    <h1>__TITLE__</h1>
    <p class="source-line">from <span class="mono">__SOURCE__</span> · __BACKEND__</p>
    <div class="stats">
      <div class="stat"><div class="n" id="s-total">0</div><div class="l">records</div></div>
      <div class="stat"><div class="n" id="s-keys">0</div><div class="l">distinct keys</div></div>
      <div class="stat"><div class="n" id="s-grounded">0%</div><div class="l">grounded</div></div>
    </div>
  </div>
</header>

<div class="wrap">
  <div class="controls">
    <input id="search" type="text" placeholder="Search value, evidence, or key..." autocomplete="off">
    <label class="toggle"><input type="checkbox" id="grounded-only"> ungrounded only</label>
  </div>
  <p class="showing" id="showing"></p>
  <div class="chips" id="chips"></div>
  <div id="clusters"></div>
  <footer>
    A <span class="mono">record</span> is an inferred key paired with a value and the verbatim
    <span class="mono">evidence</span> it was lifted from. Green = the quote was found in the source;
    amber = the model's quote couldn't be matched verbatim (usually a repaired line-break artifact).
  </footer>
</div>

<script>
const DATA = __DATA__;
const RANK = { high:3, medium:2, low:1 };

const clustersEl = document.getElementById('clusters');
const chipsEl = document.getElementById('chips');
const searchEl = document.getElementById('search');
const groundedOnlyEl = document.getElementById('grounded-only');
const showingEl = document.getElementById('showing');
let activeKey = null;

function esc(s){ return String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }

// Global stats (unfiltered)
document.getElementById('s-total').textContent = DATA.length;
document.getElementById('s-keys').textContent = new Set(DATA.map(r=>r.key)).size;
const gPct = DATA.length ? Math.round(100*DATA.filter(r=>r.grounded).length/DATA.length) : 0;
document.getElementById('s-grounded').textContent = gPct + '%';

// Key order: by count desc
const keyCounts = {};
DATA.forEach(r => keyCounts[r.key] = (keyCounts[r.key]||0)+1);
const keyOrder = Object.keys(keyCounts).sort((a,b)=> keyCounts[b]-keyCounts[a] || a.localeCompare(b));

function renderChips(){
  const chips = [`<span class="chip ${activeKey===null?'active':''}" data-k="__all__"><span class="k">all</span> <span class="c">${DATA.length}</span></span>`];
  keyOrder.forEach(k => {
    chips.push(`<span class="chip ${activeKey===k?'active':''}" data-k="${esc(k)}"><span class="k">${esc(k)}</span> <span class="c">${keyCounts[k]}</span></span>`);
  });
  chipsEl.innerHTML = chips.join('');
  chipsEl.querySelectorAll('.chip').forEach(c => c.onclick = () => {
    const k = c.dataset.k;
    activeKey = (k==='__all__' || k===activeKey) ? null : k;
    renderChips(); render();
  });
}

function render(){
  const q = searchEl.value.trim().toLowerCase();
  const groundedOnly = groundedOnlyEl.checked;
  let shown = 0;
  const html = [];

  keyOrder.forEach(key => {
    if (activeKey && key !== activeKey) return;
    let recs = DATA.filter(r => r.key === key);
    if (groundedOnly) recs = recs.filter(r => !r.grounded);
    if (q) recs = recs.filter(r =>
      r.value.toLowerCase().includes(q) || r.evidence.toLowerCase().includes(q) || r.key.toLowerCase().includes(q));
    if (!recs.length) return;
    recs.sort((a,b)=> (b.grounded-a.grounded) || (RANK[b.confidence]-RANK[a.confidence]));
    shown += recs.length;

    const gc = recs.filter(r=>r.grounded).length;
    const pct = Math.round(100*gc/recs.length);
    const rows = recs.map(r => {
      const gcls = r.grounded ? 'ok' : 'bad';
      const pill = r.grounded ? '<span class="pill ok">grounded</span>' : '<span class="pill bad">unverified</span>';
      const conf = r.confidence !== 'high' ? `<span class="conf ${r.confidence==='low'?'low':''}">${esc(r.confidence)}</span>` : '';
      return `<div class="rec">
        <div class="value mono">${esc(r.value)}</div>
        <div class="badges">${pill}${conf}</div>
        <div class="evidence ${gcls}"><span class="lbl">evidence</span>&ldquo;${esc(r.evidence)}&rdquo;</div>
      </div>`;
    }).join('');

    html.push(`<section class="cluster" id="k-${esc(key)}">
      <div class="cluster-head">
        <span class="key mono">${esc(key)}</span>
        <span class="count">${recs.length} record${recs.length>1?'s':''}</span>
        <span class="ratio"><span style="width:${pct}%"></span></span>
      </div>${rows}</section>`);
  });

  clustersEl.innerHTML = html.length ? html.join('') : '<p class="empty">No records match.</p>';
  showingEl.textContent = `showing ${shown} of ${DATA.length} records` +
    (activeKey ? ` · key: ${activeKey}` : '') + (q ? ` · "${q}"` : '');
}

searchEl.addEventListener('input', render);
groundedOnlyEl.addEventListener('change', render);
renderChips();
render();
</script>
"""


def build(records_path: Path, out_path: Path, backend_label: str) -> None:
    data = json.loads(records_path.read_text(encoding="utf-8"))
    records = data["records"]
    source = Path(data.get("source", records_path.stem)).name
    title = source.rsplit(".", 1)[0].replace("_", " ").title() + " — extracted records"

    payload = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")
    html = (
        TEMPLATE.replace("__DATA__", payload)
        .replace("__TITLE__", title)
        .replace("__SOURCE__", source)
        .replace("__BACKEND__", backend_label)
    )
    out_path.write_text(html, encoding="utf-8")
    kc = Counter(r["key"] for r in records)
    print(f"Wrote {out_path} — {len(records)} records, {len(kc)} keys")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("records", type=Path, nargs="?", default=Path("records.json"))
    ap.add_argument("--out", type=Path, default=Path("view.html"))
    ap.add_argument("--backend", default="local ollama · llama3.3:70b")
    args = ap.parse_args()
    build(args.records, args.out, args.backend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
