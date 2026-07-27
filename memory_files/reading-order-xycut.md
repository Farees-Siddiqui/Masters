---
name: reading-order-xycut
description: How block reading order is computed in the AST/layout project (XY-Cut++ + a key deviation)
metadata: 
  node_type: memory
  type: project
  originSessionId: 317792c1-ffe7-4d8b-863f-33cc59879d12
---

Block reading order in the AST project is `layout/reading_order.py`, an
implementation of **XY-Cut++** (arXiv:2504.10258, the `XYCut++.pdf` in the repo;
plan in `reading_order_plan.md`). Pure geometry over PP-DocLayoutV3 regions, no
paddle import. `compute_reading_order(boxes, labels, w, h) -> per-region rank`,
stamped as `order` on `Region`/`Box` in `layout/detector.py` (both `process_pdf`
and `process_page`), flowing into the JSON caches, the `static/layout.js`
overlay (numbers + arrows, "Reading order" toggle), and `alignment/aligner.py`.

**Key deviation from the paper (tuned for academic 1-2 column PDFs, not the
paper's newspapers):**
- The paper pre-masks titles; we DO NOT — academic titles flow correctly with
  the body via the XY-cut (top title → first band). Masking them pushed the
  title behind the authors on ResNet p1.
- We pre-mask only VISION (figures/tables/charts/formulas) and MARGINAL
  furniture (aside_text/header/footer/page-number).
- `ENABLE_CROSS_MASK = False` by default (Stage B mis-fires on full-width titles
  in academic docs; it's for newspaper spanners).
- Recursive cut splits at the **single widest gap** on the preferred axis (not
  all gaps) — otherwise a full-width title's horizontal cut slices the body
  row-major instead of column-major.

Verified on `resnet.pdf`: p1 reads title→authors→affiliations→left col→right
col; p2 clean. See [[paddle-gpu-env]].
