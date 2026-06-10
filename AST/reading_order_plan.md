# Plan: Reading order via XY-Cut++ (block ordering over layout regions)

## Context

We built both sides of the professor’s model — the Doc/AST (`app/`, Mistral OCR → `build_ast`) and the PDF/layout boxes (`layout/`, PP-DocLayoutV3 + PP-OCRv5) — plus a v1 text-stream aligner (`alignment/`). Throughout, **reading order was deliberately deferred**: `alignment/aligner.py` orders PDF boxes by a naive top-to-bottom, left-to-right sort (`key=(bbox[1], bbox[0])`), and `align_plan.md` explicitly flags this as fragile on `resnet.pdf`‘s two columns (our box order ≠ Mistral’s reading order). This plan implements the formal `ReadingOrder` from the whiteboard using **XY-Cut++** (arXiv:2504.10258).

Decisions locked in:

- **Geometry + shallow semantics only.** XY-Cut++ needs nothing beyond a bbox and a PP-DocLayout label per block — both already on every `Region`. No neural model, no text embeddings, no new dependency. (The paper’s “cross-modal” naming oversells it; the only non-geometric signal is the detector label we already have.)
- **Order blocks, not lines/words.** Reading order is computed per page over the **paragraph-level regions** (PP-DocLayoutV3 output). Line/word boxes inherit their containing region’s order and sort locally (y, then x) within it — `_containing_region` already provides this mapping.
- **Per page; global order = page index, then within-page order.** Matches the existing per-page structure.
- **Computed at region-parse time, cached on disk** like every other artifact (it’s pure geometry — microseconds per page, ~500 FPS in the paper). No new OCR cost.
- **Implement all four stages, then ablate.** For academic PDFs (mostly 1–2 column) pre-mask + MGS carry most of the value, but figure/caption/spanning-title remapping (CMM) is exactly what the naive sort gets wrong, so it earns its keep.

## Approach

### 1. Reading-order module (`layout/reading_order.py`, new)

Pure functions, **no paddle import**. Input: a list of `(bbox, label)` per page plus page `(width, height)`. Output: an ordering — a list of region indices (or an `order_index` per region). Mirrors the paper’s four stages (§4):

- **Stage A — Pre-Mask (§4.1).** Identify high-flexibility elements (title / figure / table) by their shallow label and move them to a mask set, so they don’t corrupt the body-text backbone. This is the paper’s fix for the “L-shape” failure. Masked elements are remapped later in Stage D.
- **Stage B — MGS Phase 1: cross-layout detection (§4.2).** Adaptive threshold `T_l = β·median({l_i})`, `β = 1.3`, where `l_i` is block width (horizontal docs). A block is cross-layout if `l_i > T_l` **and** it has horizontal-projection overlap with ≥2 other blocks (Eq. 2). Mask cross-layout elements (e.g. full-width spanning titles/banners) for separate handling.
- **Stage C — MGS Phases 2–3: pre-segment + density-driven recursive cut (§4.2).**
  - *Pre-segment:* isolated central elements (close to page center, no adjacent text — Eq. 3) cut the page into non-overlapping sub-regions `R`.
  - *Recursive cut:* within each `R`, choose the split axis adaptively by regional density `τ_d` (Eq. 4): `τ_d > θ_v (0.9)` → horizontal split first (XY-Cut); else vertical split first (YX-Cut) (Eq. 5). The projection cut operates on **box coordinates, not pixels** — project bboxes onto the axis, split at a coordinate no box straddles (a gap line). Recurse until each sub-region holds one non-masked block. (Conceptually the same valley-finding as `ocr.py`’s `_word_column_spans`, lifted from ink columns to box coordinates.)
- **Stage D — CMM: remap masked elements (§4.3, Algorithm 1).** Restore masked elements in label-priority order `L_order: cross-layout ≻ title ≻ vision ≻ other`, each placed at the anchor minimizing a four-term geometric distance `D = Σ w_k·φ_k`:
  - `φ1` intersection constraint (direction + projection IoU, `τ_overlap = 0.3`),
  - `φ2` boundary proximity (center distances, axis-aligned preferred),
  - `φ3` vertical continuity,
  - `φ4` horizontal ordering (left boundary).
  - Scale-aware dynamic weights `w_k = [max(h,w)², max(h,w), 1, max(h,w)⁻¹]` (Eq. 13) plus the semantic-specific edge-weight table (Eq. 14). Final sort: label priority of `B_p` (desc) → anchor index (asc) → `y1` (asc) → `x1` (asc).
- **Orchestrator:** `compute_reading_order(regions, width, height) -> list[int]` runs A→D.
- **Config constants at top of module** (paper-tuned defaults, exposed for corpus tuning, same pattern as `LINE_UNCLIP` / `WORD_GAP_FRAC`): `BETA = 1.3`, `DENSITY_THRESHOLD = 0.9`, `OVERLAP_THRESHOLD = 0.3`, `CENTER_RATIO = 0.2`, the `L_order` priority list, and the edge-weight table. Plus a **label → category map** (`title` / `vision` / `other`) for *our* PP-DocLayoutV3 label set — cross-layout is computed (Stage B), not labeled.

### 2. Detector integration (`layout/detector.py`, edit)

- Add `order: Optional[int]` to `Region` and `Box` (and to `Box.to_dict()`).
- After `_parse_regions(...)`, call `compute_reading_order(regions, image.width, image.height)` and stamp each region with its index. Do this in **both** code paths: `process_pdf` (batch CLI) and `process_page` (lazy web path), right after regions are parsed and before boxes are built.
- Propagate: `_aggregate_paragraphs` boxes take their region’s `order` directly; `_assign_labels` (line/word) inherit the containing region’s `order`, then a stable secondary sort by `(y1, x1)` gives local order within the region.
- `order` then flows into `layout.json`, `{gran}.json`, and `document.json` for free (cached → instant on re-view).

### 3. Alignment fix (`alignment/aligner.py`, edit)

- Replace the PDF-stream sort `sorted(..., key=lambda b: (b["bbox"][1], b["bbox"][0]))` with sort by `b.get("order")` (fallback to `(y1, x1)` when `order` is absent, for backward compatibility with un-ordered caches). The char stream is then built in true reading order → fixes the two-column fragility `align_plan.md` called out.
- Optional: add `order` to `PdfLoc` in `alignment/types.py` so the alignment result carries it.

### 4. Visualization (`layout/draw.py` + `static/layout.js`, optional)

- `draw.py`: render the `order` index on each box and/or draw arrows between consecutive boxes (the primary verification aid).
- `layout.js`: a “show reading order” toggle on the layout tab (numbers/arrows overlay). Not required for the core fix.

## Files

- New: `layout/reading_order.py`.
- Edit: `layout/detector.py` (`order` field on `Region`/`Box`; call orchestrator in both paths; propagate to line/word), `alignment/aligner.py` (order-based PDF sort), `layout/draw.py` (optional order overlay), `alignment/types.py` (optional `order` on `PdfLoc`), `static/layout.js` (optional toggle).
- Reuse unchanged: `app/ocr.py`, `app/ast_builder.py`, `layout/ocr.py`, `layout/cli.py`.
- Docs: append a `ReadingOrder` / updated `Region`+`Box` section to `DATASTRUCTS.md` (the `order` field, its meaning, and the per-page/within-region invariants).

## Pre-work (before coding)

Dump a `layout.json` for a representative doc and confirm the **exact** label strings PP-DocLayoutV3 emits (e.g. `text`, `title`, `figure`, `table`, `figure_caption`, …). The whole pre-mask stage keys off the label → category map, so it must match our detector’s actual vocabulary, not the paper’s assumed labels.

## Known cost

Near zero. Reading order is deterministic geometry over already-detected regions (no OCR, no model call) — computed once at parse time and baked into the cached JSON. The paper reports the ordering module at ~500 FPS on CPU. Hyperparameters (`β`, `θ_v`, etc.) are tuned in the paper via grid search on a Chinese/newspaper-heavy corpus; our academic 1–2-column docs lean on the regular-subset behavior, so defaults should hold but stay exposed for tuning.

## Verification

1. **Free regression signal:** `Aligner.coverage()` already exists. Correct reading order should *raise* coverage on multi-column pages (more doc chars find a PDF match). Measure coverage before/after on `resnet.pdf` — cheapest possible test, no annotation.
1. **Incremental ablation (mirrors paper Tables 2–3):** implement and measure cumulatively — baseline XY-Cut → +Pre-Mask → +MGS → +CMM — to confirm each stage helps on our docs and to localize regressions.
1. **Ground-truth metrics:** hand-annotate block order on ~3–5 pages (include one multi-column page) in a small JSON; adopt DocBench-100’s schema (`page_id, page_size, bbox, label` + order index) so it’s reusable. Report **Kendall’s Tau** and **block-level BLEU-4** vs GT.
1. **Visual:** numbered/arrow overlay on the layout tab; eyeball `resnet.pdf` two-column pages and a figure-heavy page (caption remapping).
1. **Regression:** confirm single-column pages are unchanged (order should equal the old top-to-bottom result), and that `/` (AST viewer) and `/layout` still work.

## Deferred / out of scope

Sub-page (nested) structures and learned split policies — the paper’s own stated limitations (§6). Not needed for our corpus; revisit only if multi-column-with-nested-panels documents appear.