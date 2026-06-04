# Plan: Alignment tab (AST graph ↔ PDF boxes, node → highlighted matches)

## Context

We've built the two sides of the professor's model — the Doc/AST (`app/`, Mistral OCR → `build_ast`) and the PDF/layout boxes (`layout/`, PP-DocLayoutV3 + PP-OCRv5) — plus a v1 text-stream aligner (`alignment/`). The goal now is to *see* the alignment: a new **Alignment** page where the AST graph sits on one side and the PDF (with bounding boxes) on the other, and **clicking an AST node highlights the PDF boxes it aligns to**.

Decisions locked in:
- Use the **fuzzy two-source aligner** (Mistral AST vs Paddle boxes) — alignment as a real matching problem, not by-construction.
- Highlight at **paragraph-region** granularity for now, but keep granularity a **parameter** so line/word can be offered later.
- Matching is fuzzy (two different OCR engines never agree byte-for-byte): normalize → similarity score → threshold; below threshold = no match (a valid outcome).

## Approach

### 1. Region-level aligner (`alignment/`)
The current `Aligner` is a single global `difflib` char-stream diff — fragile on `resnet.pdf`'s **two columns** (our PDF box order ≠ Mistral's reading order). Add a **region-level matcher** that is order-independent:

- New `alignment/region_aligner.py` (or a method on `Aligner`): `align_nodes_to_regions(ast_root, pages, threshold=0.6) -> {node_id: [{"page", "box_index"}]}` plus a `coverage` figure.
- **Alignable doc segments** — extend the current `_iter_text_nodes`: yield `(node_id, text)` for `paragraph`/`list_item`/`code` via `.text`, **and for `section` nodes use `attribs["title"]`** (headings carry text in `attribs.title`, not `.text` — confirmed in `app/ast_builder.py`).
- **Match**: normalize both sides (lowercase, collapse whitespace); score each doc segment against each page's paragraph-region text with `difflib.SequenceMatcher` (ratio + a longest-match/containment fallback for split paragraphs); assign every region ≥ threshold (a node may match several → the `Set`).
- **Section/internal nodes**: a node's highlight set = its own direct matches ∪ the union of its subtree's matches, so clicking a heading lights up the whole section's regions.
- Keep the existing char-stream `Aligner` (for future substring↔word work); reuse `DocLoc`/`PdfLoc` types. Update `alignment/__init__.py` exports.

### 2. Backend (`app/main.py`) — two-phase so the page paints fast
- `GET /align` → serve `static/align.html`.
- `POST /api/align/start` (multipart `file`): run `run_ocr_on_pdf` + `build_ast` (reuse `app/ocr.py`, `app/ast_builder.py`) → AST; write `ast.json` into the doc dir; `detector.render_document(...)` → page images. Return `{doc, filename, ast, page_count, pages:[{page,width,height,image_url}]}`. (~Mistral 20-30s + render ~2s.)
- `GET /api/align/compute?doc=<stem>&granularity=paragraph`: for each page call `detector.process_page(doc_dir, page, granularity)` (reuse — **cached on disk**, reuses layout-tab cache); load `ast.json`; run the region aligner. Return `{doc, granularity, coverage, pages:[{page, boxes:[...]}], alignment:{node_id:[{page,box_index}]}}`.
- Both run the detector **synchronously on the main thread** (the established PaddlePaddle constraint — see `_get_layout_detector`). Sanitize `doc` with `Path(doc).name`.

### 3. Frontend (`static/`)
- **`static/graph.js`** (new): the SVG tree renderer + pan/zoom **adapted from `app.js`** (`renderGraph`, `layout`, `subtreeWidth`, `enablePanZoom`, `metaLine`), exposed as `renderTree(container, astRoot, { onSelect })` returning a controller with `selectNode(id)`. **Leave `app.js` untouched** (copy, don't refactor) to avoid risk to the working AST viewer; consolidate later.
- **`static/align.html`** + **`static/align.js`** + **`static/align.css`** (new): header with 3-way nav; controls (file input, Align button, status). Two-pane grid: left = AST graph (`graph.js`); right = PDF viewer reusing `layout.js`'s page-nav + SVG box-overlay pattern (`viewBox` = page dims, `non-scaling-stroke`, paragraph boxes shown neutral).
- **Flow**: Align → `/api/align/start` → draw graph + first page image; status "Computing alignment…" → `/api/align/compute` → store boxes + alignment, draw boxes. Click node → `graph.selectNode`, look up `alignment[node_id]`, navigate the right pane to the first matched page, **highlight** those boxes (dim the rest), show "N matched box(es) on page X"; no match → "no alignment found."
- Add the **Alignment** nav link to `static/index.html` and `static/layout.html`; version-bump asset query strings (`?v=`) to dodge browser caching.

## Files
- New: `alignment/region_aligner.py`, `static/align.html`, `static/align.js`, `static/align.css`, `static/graph.js`.
- Edit: `alignment/__init__.py` (exports, extend segment iteration), `app/main.py` (routes + 2 endpoints + persist `ast.json`), `static/index.html` + `static/layout.html` (nav link).
- Reuse unchanged: `app/ocr.py`, `app/ast_builder.py`, `layout/detector.py` (`render_document`, `process_page`), `alignment/types.py`.

## Known cost
Building alignment must OCR **every page once** (paragraph ≈ ~24s/page; ~5 min uncached for resnet's 12 pages), since fuzzy matching needs each box's text. Mitigated by on-disk caching (instant after first run; reuses the layout tab's cache) and the two-phase load (graph + PDF images appear immediately; highlighting activates when compute finishes). Inherent — no shortcut without reading-order-based by-construction alignment (deferred).

## Verification
1. **Unit**: extend the synthetic aligner test — a section title and a paragraph node each map to the right region; an unrelated/figure region maps to nothing; section node unions its children.
2. **End-to-end** (server running, `API_KEY.txt` present): open `/align`, upload `resnet.pdf`; confirm graph + page images paint quickly, then alignment completes. Click a body `paragraph` node → its region highlights on the right (auto-navigating to the page); click a `section` (heading) node → its title/section regions highlight; click a node whose text is a figure/has no match → "no alignment found." Re-run = fast (cached).
3. **Regression**: confirm `/` (AST viewer) and `/layout` still work after adding nav links / `graph.js`.
