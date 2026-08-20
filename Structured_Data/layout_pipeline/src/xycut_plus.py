"""XY-Cut++ reading-order recovery (arXiv:2504.10258).

Pure geometry over labelled boxes -- no paddle, no torch, no neural model, so
this module imports in milliseconds and is unit-testable without a GPU. The
four phases of the paper:

* **Phase 1 -- Pre-Mask (§4.1).** Pull position-flexible elements (titles,
  headers/footers, figures, tables, formulas) out of the backbone so they cannot
  corrupt the recursive cut. Restored in Phase 4.
* **Phase 2 -- Density analysis (§4.2, Eq. 4-5).** Regional density ratio
  ``tau_d`` = cross-layout area / single-layout area chooses the split axis:
  above ``theta_v`` cut rows first (XY-Cut), below it cut columns first
  (YX-Cut).
* **Phase 3 -- Recursive segmentation (§4.2).** Project the unmasked boxes onto
  the chosen axis, split at the widest straddle-free gap, recurse.
* **Phase 4 -- Hierarchical re-insertion (§4.3, Alg. 1, Eq. 8-14).** Re-anchor
  the masked elements onto the ordered backbone by a scale-weighted four-term
  geometric distance, in label-priority order.

Ported from ``AST/layout/reading_order.py``, which was written against the paper
and validated on academic PDFs. The behavioural differences are deliberate and
configurable rather than hardcoded -- see :class:`XYCutConfig`.

Entry point: :func:`compute_reading_order` (ranks) or :func:`order_indices`
(indices in reading order).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Optional, Sequence

BBox = Sequence[float]  # [x1, y1, x2, y2]

# PP-DocLayoutV3 vocabulary, grouped into the paper's categories.
DEFAULT_TITLE_LABELS = frozenset(
    {"doc_title", "paragraph_title", "title", "sub_title", "chapter_title",
     "figure_title", "table_title", "chart_title"}
)
DEFAULT_VISION_LABELS = frozenset(
    {"image", "figure", "table", "chart", "display_formula", "formula", "seal"}
)
DEFAULT_MARGINAL_LABELS = frozenset(
    {"aside_text", "header", "footer", "header_image", "footer_image", "number",
     "page_number", "footnote", "page_footnote"}
)

# Label priority for the final sort (Eq. 7). The paper's ordering, as restated
# in the pipeline spec: Cross-Layout > Title > Text > Vision, with page
# furniture last.
PRIORITY = {"cross": 4, "title": 3, "text": 2, "vision": 1, "marginal": 0}


@dataclass(frozen=True)
class XYCutConfig:
    """Hyper-parameters. Defaults are the paper's."""

    #: Eq. 1 cross-layout width threshold: T_l = beta * median(width).
    beta: float = 1.3
    #: Eq. 5 density threshold theta_v. tau_d above this -> XY-Cut (rows first).
    theta_v: float = 0.9
    #: Minimum whitespace (px) that counts as a splittable gap in Phase 3.
    min_gap_px: float = 1.0
    #: Eq. 9 minimum projection-IoU for an anchor to count as aligned.
    overlap_threshold: float = 0.3

    #: When density does not force rows-first, pick the axis with the wider
    #: whitespace gap (classic XY-Cut) instead of always cutting columns first.
    #:
    #: Default OFF. It fixes side-by-side rows on single-column pages (the
    #: Transformer author grid is otherwise read down the columns: Vaswani ->
    #: Jones -> Shazeer), but that is worth only 0.006 mean CER on this corpus,
    #: and it breaks a more common case: Phase 1 masking leaves a *phantom* gap
    #: where a figure actually sits, and that phantom gap outvotes the real
    #: column gutter, splitting a two-column body into rows.
    axis_by_widest_gap: bool = False

    #: Phase 1: mask titles out of the backbone.
    #:
    #: Default OFF, against the paper, on measured evidence. Masking titles and
    #: letting Phase 4 re-anchor them costs accuracy on 8 of the 10 arXiv page-1
    #: layouts in ``arxiv_papers/``: mean CER 0.152 masked vs 0.115 unmasked,
    #: and it never helped a single page. In academic PDFs the recursive cut
    #: already peels a full-width title correctly as the top horizontal band,
    #: whereas re-anchoring can place it behind the author block.
    #: ``--paper_defaults`` restores the paper's behaviour.
    mask_titles: bool = False
    #: Phase 1: mask figures/tables/formulas.
    mask_vision: bool = True
    #: Phase 1: mask running heads, page numbers, rotated margin stamps.
    mask_marginal: bool = True
    #: Stage B (Eq. 1-2): detect and separately mask full-width spanners.
    #:
    #: Default OFF, against the paper, on measured evidence. On page 1 of the
    #: BERT paper, enabling it pulls the centred author line, affiliation and
    #: email out as "cross-layout" alongside the title, and Phase 4 re-anchors
    #: all four *after* the entire left column -- the document title lands at
    #: reading position 9 of 13. With it off the same page yields
    #: title -> authors -> affiliation -> email -> Abstract.
    #:
    #: Eq. 1-2 target newspaper-style banners that genuinely sit between
    #: columns; a centred academic title is full-width but is already peeled
    #: correctly by the recursive cut as the top horizontal band. Turn it on
    #: (``--cross_mask``) for newspaper and magazine layouts.
    enable_cross_mask: bool = False

    title_labels: frozenset = DEFAULT_TITLE_LABELS
    vision_labels: frozenset = DEFAULT_VISION_LABELS
    marginal_labels: frozenset = DEFAULT_MARGINAL_LABELS

    def category(self, label: Optional[str]) -> str:
        """Map a detector label onto a paper category."""
        if label in self.title_labels:
            return "title"
        if label in self.vision_labels:
            return "vision"
        if label in self.marginal_labels:
            return "marginal"
        return "text"


DEFAULT_CONFIG = XYCutConfig()


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _w(b: BBox) -> float:
    return b[2] - b[0]


def _h(b: BBox) -> float:
    return b[3] - b[1]


def _cx(b: BBox) -> float:
    return (b[0] + b[2]) / 2.0


def _cy(b: BBox) -> float:
    return (b[1] + b[3]) / 2.0


def _direction(b: BBox) -> str:
    """'h' if the block is wider than tall, else 'v'."""
    return "h" if _w(b) >= _h(b) else "v"


def _proj_overlap(a: BBox, b: BBox, axis: int) -> float:
    lo, hi = (0, 2) if axis == 0 else (1, 3)
    return max(0.0, min(a[hi], b[hi]) - max(a[lo], b[lo]))


def _proj_iou(a: BBox, b: BBox, axis: int) -> float:
    lo, hi = (0, 2) if axis == 0 else (1, 3)
    inter = max(0.0, min(a[hi], b[hi]) - max(a[lo], b[lo]))
    union = max(a[hi], b[hi]) - min(a[lo], b[lo])
    return inter / union if union > 0 else 0.0


# --------------------------------------------------------------------------- #
# Phase 1b / Stage B -- cross-layout detection (Eq. 1-2)
# --------------------------------------------------------------------------- #
def detect_cross_layout(
    boxes: Sequence[BBox], idxs: Sequence[int], cfg: XYCutConfig = DEFAULT_CONFIG
) -> set:
    """Indices among ``idxs`` that span the layout (full-width blocks).

    A block is cross-layout when it is wider than ``beta * median width`` *and*
    its horizontal projection overlaps at least two other blocks (Eq. 1-2).
    These are exactly the elements that bridge a multi-column gutter and would
    otherwise force the recursive cut into an "L-shaped" region.
    """
    idxs = list(idxs)
    if len(idxs) < 3:
        return set()
    threshold = cfg.beta * median(_w(boxes[i]) for i in idxs)
    cross = set()
    for i in idxs:
        if _w(boxes[i]) <= threshold:
            continue
        overlaps = sum(
            1 for j in idxs if j != i and _proj_overlap(boxes[i], boxes[j], 0) > 0
        )
        if overlaps >= 2:
            cross.add(i)
    return cross


# --------------------------------------------------------------------------- #
# Phase 2 -- regional density (Eq. 4)
# --------------------------------------------------------------------------- #
def regional_density(
    boxes: Sequence[BBox], idxs: Sequence[int], cross: set
) -> float:
    """tau_d: cross-layout area over single-layout area inside a region."""
    cross_area = sum(_w(boxes[i]) * _h(boxes[i]) for i in idxs if i in cross)
    single_area = sum(_w(boxes[i]) * _h(boxes[i]) for i in idxs if i not in cross)
    if single_area <= 0:
        return float("inf")
    return cross_area / single_area


# --------------------------------------------------------------------------- #
# Phase 3 -- recursive segmentation
# --------------------------------------------------------------------------- #
def _best_gap(boxes: Sequence[BBox], idxs: Sequence[int], axis: int,
              cfg: XYCutConfig) -> tuple:
    """``(gap_width, split_index)`` for the widest straddle-free gap on ``axis``.

    ``split_index`` is ``None`` when nothing wider than ``min_gap_px`` exists.
    """
    lo, hi = (0, 2) if axis == 0 else (1, 3)
    order = sorted(idxs, key=lambda i: boxes[i][lo])
    best_gap = cfg.min_gap_px
    best_pos: Optional[int] = None
    cur_end = boxes[order[0]][hi]
    for k in range(1, len(order)):
        start = boxes[order[k]][lo]
        if start - cur_end > best_gap:
            best_gap, best_pos = start - cur_end, k
        cur_end = max(cur_end, boxes[order[k]][hi])
    return (best_gap if best_pos is not None else 0.0), best_pos


def _widest_gap_split(
    boxes: Sequence[BBox], idxs: Sequence[int], axis: int, cfg: XYCutConfig
) -> Optional[tuple]:
    """Split ``idxs`` at the single widest straddle-free gap along ``axis``.

    Returns ``(before, after)`` in reading order along that axis, or ``None`` if
    no gap wider than ``min_gap_px`` exists (e.g. a full-width block bridges
    every candidate line).

    Splitting at *one* gap and recursing -- rather than at all gaps at once --
    keeps the preferred axis in control: a full-width heading is peeled by a
    single horizontal cut, and the body beneath then re-tries the vertical
    column cut instead of being sliced row-major by the same pass.
    """
    lo = 0 if axis == 0 else 1
    _, best_pos = _best_gap(boxes, idxs, axis, cfg)
    if best_pos is None:
        return None
    order = sorted(idxs, key=lambda i: boxes[i][lo])
    return order[:best_pos], order[best_pos:]


def recursive_cut(
    boxes: Sequence[BBox],
    idxs: Sequence[int],
    cross: set,
    cfg: XYCutConfig = DEFAULT_CONFIG,
) -> list:
    """Recursively XY/YX-cut ``idxs`` into reading order (Phases 2-3)."""
    idxs = list(idxs)
    if len(idxs) <= 1:
        return idxs
    tau = regional_density(boxes, idxs, cross)
    # Eq. 5: dense in cross-layout content -> XY-Cut (rows first, axis=y);
    # otherwise YX-Cut (columns first, axis=x).
    if tau > cfg.theta_v:
        primary = 1
    elif cfg.axis_by_widest_gap:
        # Once Phase 1 has masked the cross-layout elements, tau is ~0 almost
        # everywhere, so Eq. 5 alone degenerates to "always columns first".
        # That is right for a two-column body and wrong for a single-column
        # page holding a side-by-side row (the Transformer paper's author grid
        # gets read down the columns: Vaswani -> Jones -> Shazeer). Falling back
        # to classic XY-Cut -- take whichever axis actually has the wider
        # whitespace -- resolves both.
        gap_x, _ = _best_gap(boxes, idxs, 0, cfg)
        gap_y, _ = _best_gap(boxes, idxs, 1, cfg)
        primary = 1 if gap_y > gap_x else 0
    else:
        primary = 0
    for axis in (primary, 1 - primary):
        split = _widest_gap_split(boxes, idxs, axis, cfg)
        if split is not None:
            before, after = split
            return (recursive_cut(boxes, before, cross, cfg)
                    + recursive_cut(boxes, after, cross, cfg))
    # Indivisible (mutually overlapping boxes): fall back to (y1, x1).
    return sorted(idxs, key=lambda i: (boxes[i][1], boxes[i][0]))


# --------------------------------------------------------------------------- #
# Phase 4 -- hierarchical re-insertion (Alg. 1, Eq. 8-14)
# --------------------------------------------------------------------------- #
def _edge_weights(cat: str, orient: str) -> list:
    """Semantic-specific edge-weight vector w_edge (Eq. 14)."""
    if cat == "title":
        return [1.0, 0.1, 0.1, 1.0] if orient == "h" else [0.2, 0.1, 1.0, 1.0]
    if cat == "cross":
        return [1.0, 1.0, 0.1, 1.0]
    return [1.0, 1.0, 1.0, 0.1]  # vision / text


def _distance(bp: BBox, bo: BBox, cat: str, page_max: float,
              cfg: XYCutConfig) -> float:
    """Joint geometric distance D(B_p, B_o) from a pending box to an anchor (Eq. 8).

    Scale weights (Eq. 13) stagger the four constraints by magnitude so they act
    lexicographically: intersection dominates, then proximity, then vertical
    continuity, then horizontal ordering.
    """
    # Page furniture has no reliable layout alignment; the axis-aligned
    # min(dx, dy) term would snap a centred page number onto a far-away title.
    # Plain Euclidean centre distance attaches it to its genuinely nearest
    # block, and being lowest priority it sorts immediately after.
    if cat == "marginal":
        return (_cx(bp) - _cx(bo)) ** 2 + (_cy(bp) - _cy(bo)) ** 2

    # phi1 -- intersection constraint (Eq. 9).
    axis = 0 if _direction(bp) == "h" else 1
    same_dir = _direction(bp) == _direction(bo)
    phi1 = 0.0 if (same_dir and _proj_iou(bp, bo, axis) >= cfg.overlap_threshold) else 1.0

    # phi2 -- boundary proximity (Eq. 10).
    dx, dy = abs(_cx(bp) - _cx(bo)), abs(_cy(bp) - _cy(bo))
    aligned = _proj_overlap(bp, bo, 0) > 0 or _proj_overlap(bp, bo, 1) > 0
    phi2 = min(dx, dy) if aligned else dx + dy

    # phi3 -- vertical continuity (Eq. 11).
    phi3 = -bo[3] if (cat == "cross" and bp[1] > bo[3]) else bo[1]

    # phi4 -- horizontal ordering (Eq. 12).
    phi4 = bo[0]

    scale = [page_max * page_max, page_max, 1.0, 1.0 / page_max]
    edge = _edge_weights(cat, _direction(bp))
    phi = [phi1, phi2, phi3, phi4]
    return sum(scale[k] * edge[k] * phi[k] for k in range(4))


def _cross_modal_match(
    boxes: Sequence[BBox],
    anchors: list,
    masked: dict,
    page_max: float,
    cfg: XYCutConfig,
) -> dict:
    """Give every masked block the anchor rank of its best-matching backbone slot.

    Higher-priority elements are restored first and then themselves become
    candidate anchors for lower-priority ones (Alg. 1: the candidate set grows).
    """
    anchor_rank = {idx: r for r, idx in enumerate(anchors)}
    candidates = list(anchors)
    assigned = {}

    for cat in ("cross", "title", "vision", "marginal"):
        for bp in masked.get(cat, []):
            if not candidates:
                break
            best, best_d = None, float("inf")
            for bo in candidates:
                d = _distance(boxes[bp], boxes[bo], cat, page_max, cfg)
                if d < best_d:
                    best_d, best = d, bo
            if best is not None:
                rank = anchor_rank[best]
                anchor_rank[bp] = rank
                assigned[bp] = rank
                candidates.append(bp)
    return assigned


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def order_indices(
    boxes: Sequence[BBox],
    labels: Optional[Sequence[Optional[str]]] = None,
    width: float = 1.0,
    height: float = 1.0,
    config: XYCutConfig = DEFAULT_CONFIG,
) -> list:
    """Box indices in reading order. Runs Phases 1-4."""
    n = len(boxes)
    if n == 0:
        return []
    if n == 1:
        return [0]
    if labels is None:
        labels = [None] * n

    cfg = config
    page_max = max(float(width), float(height), 1.0)
    cats = [cfg.category(labels[i] if i < len(labels) else None) for i in range(n)]

    # ---- Phase 1: pre-mask ----
    masked = {"cross": [], "title": [], "vision": [], "marginal": []}
    backbone = []
    for i in range(n):
        c = cats[i]
        if c == "title" and cfg.mask_titles:
            masked["title"].append(i)
        elif c == "vision" and cfg.mask_vision:
            masked["vision"].append(i)
        elif c == "marginal" and cfg.mask_marginal:
            masked["marginal"].append(i)
        else:
            backbone.append(i)

    # ---- Phase 1b: cross-layout spanners out of the backbone ----
    cross = detect_cross_layout(boxes, backbone, cfg) if cfg.enable_cross_mask else set()
    masked["cross"] = sorted(cross, key=lambda i: (boxes[i][1], boxes[i][0]))
    core = [i for i in backbone if i not in cross]

    for key in ("title", "vision", "marginal"):
        masked[key].sort(key=lambda i: (boxes[i][1], boxes[i][0]))

    # ---- Phases 2-3: recursive cut over the single-layout backbone ----
    if not core:
        # Nothing to anchor against (a figure-only page, or everything masked).
        # Order geometrically and stop; Phase 4 needs a backbone to exist.
        return sorted(range(n), key=lambda i: (boxes[i][1], boxes[i][0]))
    anchors = recursive_cut(boxes, core, set(), cfg)

    # ---- Phase 4: re-insert masked elements ----
    anchor_rank = {idx: r for r, idx in enumerate(anchors)}
    anchor_rank.update(_cross_modal_match(boxes, anchors, masked, page_max, cfg))

    # Safety net: anything unmatched goes after the backbone, geometrically.
    leftover = [i for i in range(n) if i not in anchor_rank]
    for i in sorted(leftover, key=lambda i: (boxes[i][1], boxes[i][0])):
        anchor_rank[i] = len(anchors)

    def cat_of(i: int) -> str:
        return "cross" if i in cross else cats[i]

    # Final sort (Fig. 7): anchor index asc, label priority desc, y1 asc, x1 asc.
    return sorted(
        range(n),
        key=lambda i: (anchor_rank[i], -PRIORITY[cat_of(i)], boxes[i][1], boxes[i][0]),
    )


def compute_reading_order(
    boxes: Sequence[BBox],
    labels: Optional[Sequence[Optional[str]]] = None,
    width: float = 1.0,
    height: float = 1.0,
    config: XYCutConfig = DEFAULT_CONFIG,
) -> list:
    """Per-box reading rank: ``result[i]`` is where box ``i`` falls in the order."""
    ordering = order_indices(boxes, labels, width, height, config)
    ranks = [0] * len(boxes)
    for rank, idx in enumerate(ordering):
        ranks[idx] = rank
    return ranks
