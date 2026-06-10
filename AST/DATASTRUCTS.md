# Data Structures

This file documents every data structure used in the project. Update it
whenever a new structure is introduced or an existing one changes.

---

## `Node` — `app/ast_builder.py`

The core AST node. Mirrors the whiteboard definition:

```
Node = {
  type     ∈ Enum[label],
  parent   : Node,
  children : List[Node],
  attribs  : Map[str, str],
  text     : Maybe[str],
}
```

### Fields

| Field      | Type              | Description                                                                 |
|------------|-------------------|-----------------------------------------------------------------------------|
| `id`       | `str`             | Stable identifier (e.g. `n0`, `n1`, …) assigned in pre-order by `_assign_ids()` after `build_ast()` completes. Used by the frontend to address nodes (selection, parent lookup). |
| `type`     | `str` (enum-like) | One of: `doc`, `section`, `paragraph`, `heading`, `list`, `list_item`, `code`, `text`. |
| `attribs`  | `dict[str, str]`  | Free-form key/value attributes. See per-type table below.                   |
| `children` | `list[Node]`      | Ordered child nodes.                                                        |
| `text`     | `str \| None`     | Leaf text content (paragraphs, list items, code blocks).                    |
| `parent`   | `Node \| None`    | In-memory back-pointer set by `Node.add()`. **Serialized as `parent_id` in `to_dict()`** (the raw reference would create a cycle in JSON; the id-reference preserves the relationship without it). |

### Attribute schema by `type`

| `type`      | `attribs`                                         | `text`              | `children`               |
|-------------|---------------------------------------------------|---------------------|--------------------------|
| `doc`       | `{}`                                              | none                | top-level sections/paras |
| `section`   | `{title: str, level: "1".."6"}`                   | none                | body + nested sections   |
| `paragraph` | `{}`                                              | paragraph text      | none                     |
| `list`      | `{ordered: "true" \| "false"}`                    | none                | `list_item`s             |
| `list_item` | `{}`                                              | item text           | none (flat for v1)       |
| `code`      | `{lang?: str}`                                    | code body           | none                     |

### Methods

- `add(child) -> Node` — append child and set `child.parent = self`.
- `to_dict() -> dict` — serializable form; replaces the `parent` reference with `parent_id` to break the cycle. Used for the `/api/upload` JSON response and frontend rendering.

### Serialized shape (`Node.to_dict()`)

```jsonc
{
  "id":        "n5",          // sequential pre-order id
  "type":      "section",
  "attribs":   {"title": "Abstract", "level": "2"},
  "text":      null,
  "parent_id": "n1",          // null only for the root
  "children":  [ /* recursively the same shape */ ]
}
```

### Invariants

- The root returned by `build_ast()` always has `type == "doc"`, `parent is None`, and `id == "n0"`.
- For any non-root node `n`, `n.parent.children` contains `n`, and `n.to_dict()["parent_id"] == n.parent.id`.
- `id`s are unique within an AST; assignment is pre-order so a parent's id is always lexicographically lower than its descendants'.
- `section.attribs["level"]` strictly increases moving down the tree along a section-only path (a deeper section is nested inside a shallower one).

---

## `OCRResult` — `app/ocr.py`

Result of a Mistral OCR run on a single PDF.

| Field        | Type          | Description                                |
|--------------|---------------|--------------------------------------------|
| `markdown`   | `str`         | All pages concatenated with `\n\n`.        |
| `page_count` | `int`         | Number of pages returned by the OCR call.  |
| `pages`      | `list[str]`   | Per-page markdown, indexable for debugging.|

---

## `/api/upload` response — `app/main.py`

JSON object returned to the frontend.

```jsonc
{
  "filename":   "paper.pdf",       // original upload name
  "page_count": 12,                // from OCRResult
  "markdown":   "...",             // raw concatenated markdown (debug aid)
  "ast":        { /* Node.to_dict() output */ }
}
```

`ast` is a recursive `Node.to_dict()` blob — see the `Node` schema above for
its field shape (every node carries `id` and `parent_id`).

---

## Parser state — `app/ast_builder.build_ast()`

Internal only; not exposed via the API. Documented here for completeness.

- **`stack: list[tuple[int, Node]]`** — section stack used to decide where the next block attaches. Each entry is `(heading_level, section_node)`; the root is seeded as `(0, doc_root)`. On encountering a heading of level `L`, entries with `level >= L` are popped before pushing the new section.

---

## `Region` / `Box` — `layout/detector.py`

A **`Region`** is one PP-DocLayoutV3 detection (a labelled bbox). A **`Box`** is
an emitted layout box at a requested granularity (`paragraph` = region with text
aggregated from its lines; `line` / `word` = PP-OCRv5 boxes inheriting their
containing region's label). Both carry an `order` — the formal `ReadingOrder`.

| Field        | Type            | Description                                                                 |
|--------------|-----------------|-----------------------------------------------------------------------------|
| `label`      | `str \| null`   | Semantic label (the Enum part of `BBox`), e.g. `text`, `doc_title`, `chart`.|
| `cls_id`     | `int \| null`   | PP-DocLayoutV3 class id.                                                     |
| `bbox`       | `[x1,y1,x2,y2]` | Axis-aligned box in page-pixel coords (rendered at the doc DPI).             |
| `score`      | `float \| null` | Detection confidence.                                                       |
| `text`       | `str \| null`   | OCR text (`Maybe[Text]`); `null` for non-text regions.                      |
| `text_score` | `float \| null` | Recognition confidence (mean over member lines for `paragraph`).            |
| `order`      | `int \| null`   | **0-based reading position within the page** (see below).                   |

`order` is computed per page by `layout.reading_order.compute_reading_order`
(XY-Cut++, arXiv:2504.10258) over the **paragraph-level regions**, then stamped
on each `Region` and propagated to `Box`es. Invariants:

- **Per page, contiguous 0…N-1.** `order` is a permutation of `0..N-1` over the
  regions of one page; global order = page index, then within-page `order`.
- **Paragraph boxes** take their region's `order` directly.
- **Line / word boxes** inherit their containing region's order, then the whole
  page is renumbered into a single sequence by `(region order, y1, x1)` — so
  lines within a region read top→bottom and the page reads as one stream.
- It flows unchanged into `layout.json`, `{granularity}.json`, and
  `document.json` (cached on disk → instant on re-view), and drives both the
  reading-order overlay (`static/layout.js`) and the alignment char stream
  (`alignment/aligner.py`).

See `layout/reading_order.py` for the stage/category config (which labels are
pre-masked as vision/marginal, the `ENABLE_CROSS_MASK` flag, and the paper-tuned
constants `BETA` / `DENSITY_THRESHOLD` / `OVERLAP_THRESHOLD`).

---

## Alignment (`Loc[Doc] -> Set[Loc[PDF]]`) — `alignment/`

The whiteboard's `Alignment : Loc[Doc] -> Set[Loc[PDF]]`. Two implementations
share the package:

- **`Aligner`** (`alignment/aligner.py`) — v1 char-stream `difflib` diff;
  resolves a `DocLoc` (node + substring) to a `Set[PdfLoc]`. Substring-precise
  but order-fragile.
- **`align_naive`** (`alignment/naive_aligner.py`) — the v1 **region-level naive
  matcher**. Order-independent: for each text-bearing node it scores every PDF
  box by normalized **containment** (score 1.0) or **token overlap**
  `|A∩B| / min(|A|,|B|)`, keeping boxes `>= threshold` (default `0.5`). Good for
  paragraph regions, but over-matches at line/word (a common word is "contained"
  in every node). A node's set is its own matches **∪ its whole subtree's**.
- **`align_stream`** (`alignment/stream_aligner.py`) — the **positional
  char-stream aligner** that powers the Alignment page, at **any granularity**
  (paragraph/line/word). Flattens the AST and the reading-ordered PDF boxes into
  two normalized char streams, aligns them with `difflib.SequenceMatcher`, then
  maps each node's char span to the boxes its characters landed on. Because
  matching is *by position*, the second "the" aligns to the second "the" — so a
  node maps to exactly its run of word/line boxes, not every occurrence. Returns
  both directions: `alignment` (node → boxes, subtree-unioned) and `reverse`
  (`"page:box_index"` → `[node_id, ...]`, primary/most-chars owner first) for
  box → AST highlighting.

### `align_naive(ast_root, pages, threshold=0.5) -> dict`

`pages` is `[{"page": int, "boxes": [box_dict, ...]}, ...]` (paragraph
granularity; each box needs `text`). Returns:

```jsonc
{
  "alignment": {                 // only nodes with >=1 match appear
    "n9": [{"page": 1, "box_index": 8}, ...]   // box_index -> page's boxes[]
  },
  "coverage":  1.0,              // fraction of text segments with a direct match
  "threshold": 0.5
}
```

`_iter_segments` yields `(node_id, text)` for every node carrying alignable
text: `section` uses `attribs["title"]`; everything else uses `.text`.

---

## `/api/align/start` + `/api/align/compute` responses — `app/main.py`

Two-phase (mirrors the layout tab) so the page paints before the slow OCR.

**`POST /api/align/start`** (multipart `file`) — Mistral OCR + `build_ast` +
render page images; persists `ast.json` in the doc dir.

```jsonc
{
  "doc": "resnet", "filename": "resnet.pdf", "page_count": 12,
  "ast": { /* Node.to_dict() */ },
  "pages": [{"page": 1, "width": 1654, "height": 2339, "image_url": "/layout-data/resnet/page1/page.png"}, ...]
}
```

**`GET /api/align/compute?doc=<stem>&granularity=paragraph`** — `process_page`
per page (cached on disk; reuses the layout tab's cache) + `align_stream` at the
requested granularity (`paragraph` | `line` | `word`). The frontend caches the
result per granularity, so switching back is instant.

```jsonc
{
  "doc": "resnet", "granularity": "paragraph", "coverage": 1.0,
  "alignment": { "n9": [{"page": 1, "box_index": 8}, ...] },  // node -> boxes
  "reverse":   { "1:8": ["n9", ...] },                        // box  -> nodes
  "pages": [{"page": 1, "width": 1654, "height": 2339, "boxes": [ /* Box.to_dict() */ ]}, ...]
}
```

The frontend (`static/align.js`) indexes `pages[].boxes` by `box_index`.
Clicking an AST node highlights its `alignment` boxes (forward); clicking a PDF
box uses `reverse` to emphasize that box and scroll to its owning node
(backward). Both work at paragraph / line / word granularity.

---

## Adding a new structure

1. Add the dataclass / TypedDict / schema in its module.
2. Append a section to this file with: fields table, any per-type variants, invariants, and (if part of the HTTP surface) a JSON example.
3. If it replaces an existing structure, update the relevant section in-place rather than appending — keep this file the single source of truth.
