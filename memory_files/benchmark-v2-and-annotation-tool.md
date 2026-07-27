---
name: benchmark-v2-and-annotation-tool
description: benchmark_v2_0 (node localization vs tiered human gold) + /annotate tool; v1 contamination + fragment bugs fixed 2026-07-19
metadata: 
  node_type: memory
  type: project
  originSessionId: f40e5348-cfc8-49c8-ab9a-a7ebc092c29b
---

Built 2026-07-19 after concluding v1's token benchmark is rigorous but illegible
and its auto-derived gold needs human verification. Full detail in repo:
`AST/SESSION_2026-07-19.md`.

**`/annotate` tool** (`app/annotate.py` + `static/annotate.*`): three queues per
doc — *excluded* (unplaced nodes → click boxes/absent; this creates gold that
cannot be auto-derived), *nulls* (furniture vs missed owner), *audit* (oracle
placements, fuzzy-first, accept/reject). Verdicts → `layout_output/<doc>/annotations.json`
— **gitignored dir, must `git add -f` the annotations or human gold is lost**.
Backend caches derivation in `_CACHE`; restart server after editing (no --reload,
see [[dev-server-reload-gotcha]]).

**`benchmark_v2_0/`** — headline benchmark now; v1 = diagnostic tier + proposal
generator. Gold `node -> {boxes} | ABSENT`, tiers human > audited > oracle
(majority token ownership), pixel-area IoU ≥ 0.5. CLI:
`python -m benchmark_v2_0 --all [--human-only] [--out f]`.
Baseline (3% verified, `results_v2_all.json`): stream loc.acc .934 / false-claim
.531; similarity .889 / .694. **Stream loses tables .375 vs .875** (complementary
failure modes, n=8). adam = worst doc (thin, partly-wrong gold — see the
span-swallowing bug in [[alignment-benchmark]]).

**v1 fixes landed** (both by agents, verified):
1. Contamination: gold-NULL token whose predicted node is *unplaced* → excluded
   (`n_excluded_unverifiable`), not false_attrib. 75% of stream's FA was artifact;
   real FA is stream .300 / similarity .440 (`results_real_v2.json`). Any FA
   number recorded before this date (incl. CP-SAT 2x2 in [[alignment-benchmark]])
   is inflated — re-run before quoting.
2. `Token.fragments`: pdfium U+FFFE hyphen-broken words split into per-line
   rects; `boxes_to_tokens` votes by fragment centers; annotate payload is
   `key -> [page, [bbox, ...]]` (list of rects, not one).

**Why (user's arc):** they asked repeatedly "does the benchmark say if an aligned
NODE is good" — token F1 never answered it. v2's unit = the node a user clicks;
its two metrics read as sentences. Keep explanations at that altitude.

**Next when resuming:** user annotates resnet (~39 excluded + 41 nulls + rest of
audit; 36 audit accepts done) → rerun v2 resnet → compare tiers. Then 9 docs.
`results_probe.json` is stale (drift, F1 .971 stored vs .998 current) — don't
quote. Docs: artifact 10661eb4-* + `AST/BENCHMARKS.html` (printable twin —
regenerate both together if numbers change).
