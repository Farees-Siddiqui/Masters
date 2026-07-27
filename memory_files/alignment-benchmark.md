---
name: alignment-benchmark
description: "benchmark/ CLI — two modes (synthetic sweep + real text-layer oracle); headline finding = prose solved, abstention broken"
metadata: 
  node_type: memory
  type: project
  originSessionId: fcf19e01-68f4-420c-bac8-c804b3662fe3
---

`benchmark/` evaluates the aligners (see [[project-overview]]). Two modes answering different questions:

- **`python -m benchmark synthetic`** — gold by construction from the AST's own text. Controlled stress, fast, deterministic. Measures robustness to char noise only; **cannot** represent cross-engine disagreement or the NULL class.
- **`python -m benchmark real --doc resnet`** — real PP-OCRv5 boxes + real Mistral AST, gold derived from the PDF text layer (`benchmark/oracle.py`). Writes `results_real.json` to repo root; the `/benchmark` page shows it (mode switch, `?mode=real|synthetic` deep-links).

**Corpus (2026-07): 10 pinned arXiv papers, 136 pages, 81,142 tokens** — `benchmark/corpus.py`, pinned by `id+version` so runs are byte-reproducible. `python -m benchmark fetch` builds (5m32s wall, **3.1s/page measured** on the 4070 Ti — the old "~24s/page" note was CPU-era and wrong). All born-digital (verified). **One domain: all scientific papers.**

**Headline finding (10 docs):** both aligners reconstruct **prose** near-perfectly but **invent a source for 63–65% of tokens no node owns**. Abstention, not matching, is the open problem — the metric that matters for the Lakeridge auditability pitch ([[lakeridge-sam3-demo]]).
**⚠ 2026-07-19 UPDATE: that 63–65% was largely oracle contamination** — tokens of *unplaced* nodes scored as false attributions. Fixed; real FA = stream .300 / similarity .440 (`results_real_v2.json`). All FA numbers in this memory (incl. the CP-SAT table below) are pre-fix and inflated; the *relative* CP-SAT ranking likely holds but re-run before quoting. v1 is now the diagnostic tier under [[benchmark-v2-and-annotation-tool]].

**Multi-doc overturned the single-doc conclusion.** On resnet alone similarity looked equal to stream (.971 vs .973). Across 10 docs: **stream F1 .946 (micro) / .943 (macro), similarity .898 / .895** — resnet was flattering similarity. Never trust a one-document result here.

**CP-SAT experiment (2026-07, `alignment/cpsat_aligner.py`, prof's suggestion).** Alignment as constraint solving: bool `x[node,box]`, hard constraints = one owner per box + monotone reading order (non-crossing, encoded with O(m) position vars — NOT pairwise clauses, which would be O(n²m²)) + length coherence (assigned text ≤ 2× node text). Objective `sum((score - threshold) * x)` → abstention is free (sub-threshold pairs carry negative weight). Scorer pluggable → a clean **2x2 of representation × aggregation**:

| | argmax | CP-SAT |
|---|---|---|
| embedding | similarity .898 / FA .654 | cpsat .927 / FA .359 |
| char n-gram TF-IDF | lexical .926 / FA .190 | cpsat-lex .916 / **FA .168** |

(F1 micro / false-attribution). `stream` still wins F1 at **.946** but FA .634 and is **400x slower** (710s vs lexical 1.7s).

**What this proved:**
- **Neither beat stream on F1** — but F1 was the wrong target. False attribution fell **.634 → .168 (4x)**, which is the auditability metric.
- Representation and constraints are **substitutes, not complements** — each alone gains ~+.03 over similarity; both together is *worse* (.916, recall drops to .867). They fix the same spurious matches, so stacking over-constrains.
- **Lexical transformed abstention**: `chart` .40→.99, `image` .04→.99, `algorithm` .00→1.00. Chart OCR text shares no n-grams with any node → scores below threshold → abstains. MiniLM gives everything cosine >0.5 because embeddings are topically smeared. **This, not the threshold, was the false-attribution cause.**
- **I was WRONG that lexical would fix `reference_content`**: lexical .529 ≈ similarity .527. Bibliography entries share n-grams too (initials, years, "In Proc."). Only *constraints* helped (cpsat .783); `stream` nails it (.996) via order. Unexplained: cpsat-lex stays at .528.

**Per-stratum reversals worth keeping:**
- `reference_content` (8,468 tok): stream **.996** vs similarity **.527** — bibliographies are near-identical to each other, so embeddings can't discriminate. Strongest evidence that this task is lexical, not semantic.
- `table` (454 tok): stream **.251** vs similarity **.954** — reversed.
- `algorithm` (620 tok): abstention 0.000 both — pseudocode blocks are pure false attribution.
- `adam` placed only 54% vs `bert` 90%: placement rate tracks maths density (LaTeX has no glyph run). Gold is thinner on maths-heavy docs, so their F1 is less trustworthy.
- stream is **13x slower** (979s vs 74s over the corpus) — O(n²) difflib.

**Design decisions that matter:**
- Score at **token level**, not box level — box-level conflates aligner quality with box granularity (same aligner: F1 .99 at region vs .40 at line granularity, purely because boxes got smaller).
- **NULL is a first-class label.** Gold-NULL falls out of the oracle for free (page numbers, figure interiors).
- `table`/`display_formula`/`formula_number` are **excluded** from scoring (`ORACLE_BLIND_LABELS`): Mistral emits markdown/LaTeX with no literal glyph run, so gold-NULL there is an oracle limit, not truth — scoring it would penalise a *correct* aligner.
- Undefined metrics return `None`, not 0.0 (a furniture-only stratum has no recall to measure).

**Oracle honesty:** over the corpus places 1461/1928 nodes (75.8%), token coverage 85.4%. Unplaced = LaTeX, `![img](...)` refs, bare section numbers — **excluded from gold, never guessed**, which makes the benchmark *easier* than reality. Reported via `OracleReport`, shown on the page.

**KNOWN ORACLE BUG (diagnosed 2026-07, not yet fixed).** `place_nodes` finds ONE contiguous span in the text stream and `gold_token_owner` hands *every* token in that span to the node. LaTeX floats out of flow, so the PDF content stream interleaves a table's glyphs into the middle of a paragraph's run — and the paragraph's span **swallows them**. Measured by checking whether each token's text actually occurs in its gold node's text:

| doc | inconsistent gold | conflicts |
|---|---|---|
| resnet | 0.8% | 11 |
| bert | 2.5% | 199 |
| adam | **10.0%** | 428 |

Worst strata: adam `display_formula` 60%, `formula_number` 100%, `algorithm` 26%, `text` 10%; bert `table` 75%. **Consequences:** (1) the `table` stratum is unreliable — discount it; (2) **adam's low F1 (.763) is partly an oracle artifact**, its gold is 10% wrong; (3) the 959 conflicts are this same root cause. Body text on clean docs is fine, so headline numbers survive.

**Fix (designed, not built):** only assign a token to a node if the token's text actually appears in that node's text; tokens failing the check are **excluded** (unknown), not relabelled NULL — we know the span-assignment is untrustworthy, not that the token is furniture. Report the exclusion count.

**Gotchas burned once (don't rediscover):**
- pdfium encodes line-break hyphens as **U+FFFE inside the token** (`ex￾tremely`), not ASCII `-`. Strip it in `_norm`; real hyphens in `low-level` must survive.
- The oracle was nondeterministic because `_locate` ranked anchors over `set(needle.split())` — string hash randomisation varies per process. Always `sorted()` + a lexical tiebreak.
- Coordinate transform: PDF points bottom-left → pixels top-left, `scale = dpi/72` (200dpi → 612x792pt = 1700x2200px). Validated against a known layout box; verify before trusting any gold.

**Next:** conformal calibration over the shipped `scores` (threshold sweep → AP), DocLayNet for cross-domain + free edge-case strata via its 11 human labels. See [[alignment-benchmark-datasets]].
