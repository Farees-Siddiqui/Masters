---
name: project-overview
description: "AST project — aligns a PDF's logical AST (Mistral OCR) with its physical layout boxes (PaddlePaddle); FastAPI + 3 browser tabs"
metadata: 
  node_type: memory
  type: project
  originSessionId: fbb2a4fa-a5e3-4f1a-a503-0c9f7a785305
---

`Documents/Masters/AST` implements a whiteboard formal model: `Alignment : Loc[Doc] -> Set[Loc[PDF]]`. It builds two independent representations of the same PDF and matches them so clicking an AST node highlights its PDF boxes (and vice-versa).

**Three pipelines:**
- `app/` — PDF → Mistral OCR → markdown → nested-section AST (`Node` tree in `ast_builder.py`). FastAPI app in `app/main.py`.
- `layout/` — PDF → PP-DocLayoutV3 detection + PP-OCRv5 text → labelled boxes with reading order. See [[paddle-gpu-env]] and [[reading-order-xycut]].
- `alignment/` — matches AST nodes ↔ PDF boxes. Three impls in evolution order: `Aligner` (v1 char-stream difflib, order-fragile), `align_naive` (region-level fuzzy, over-matches at word level), `align_stream` (CURRENT — positional char-stream, any granularity, returns both directions). 

**Frontend** (served by `app/main.py`): `/` AST tree (`app.js`), `/layout` box overlay (`layout.js`), `/align` two-pane viewer (`align.js` + `graph.js`).

**Key constraints:** PaddlePaddle detector must run synchronously on main thread; alignment OCRs every page once (~5 min uncached for resnet.pdf's 12 pages), mitigated by on-disk caching. Mistral key from `MISTRAL_API_KEY` env or `API_KEY.txt` (gitignored).

`DATASTRUCTS.md` is the single source of truth for all data structures (`Node`, `Region`/`Box`, alignment responses). `align_plan.md` / `reading_order_plan.md` are the design plans.
