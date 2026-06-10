"""Reading order over layout regions via **XY-Cut++** (arXiv:2504.10258).

Pure geometry + shallow PP-DocLayoutV3 labels — no paddle import, no neural
model, microseconds per page. Given the page's regions (each a bbox + label)
this computes a global block reading order, mirroring the paper's four stages:

  * **A — Pre-Mask (§4.1).** Move highly-dynamic elements (titles, figures,
    tables, formulas) to a mask set so they don't corrupt the body-text
    backbone. Remapped later in Stage D.
  * **B — Cross-layout detection (§4.2, Eq. 1-2).** Flag full-width spanning
    blocks (width > β·median width *and* horizontal-projection overlap with ≥2
    others) and mask them for separate, highest-priority remapping.
  * **C — Density-driven recursive cut (§4.2, Eq. 4-5).** Recursively project
    the remaining (single-layout) boxes onto an axis chosen by regional density
    and split at a coordinate no box straddles (a gap line). YX-Cut (columns
    first) for sparse regions, XY-Cut (rows first) for cross-layout-dense ones.
  * **D — Cross-Modal Matching (§4.3, Alg. 1, Eq. 8-14).** Restore masked
    elements in label-priority order, each anchored to the ordered block that
    minimises a scale-weighted four-term geometric distance. Final order sorts
    by (anchor index, label priority desc, y1, x1) — Index first, per Fig. 7's
    caption (Alg. 1 line 22 lists the keys in the opposite, mislabelled order).

Entry point: :func:`compute_reading_order`.
"""

from __future__ import annotations

from statistics import median
from typing import Optional, Sequence

# --------------------------------------------------------------------------- #
# Paper-tuned config (exposed for corpus tuning, same pattern as ocr.py's
# LINE_UNCLIP / WORD_GAP_FRAC).
# --------------------------------------------------------------------------- #
BETA = 1.3  # cross-layout width threshold scale: T_l = BETA * median(widths) (Eq. 1)
DENSITY_THRESHOLD = 0.9  # theta_v: tau_d above this -> XY-Cut (rows first) (Eq. 5)
OVERLAP_THRESHOLD = 0.3  # tau_overlap: min projection-IoU for an aligned anchor (Eq. 9)
GAP_EPS = 1.0  # px slack so touching boxes aren't split on a phantom gap

# Cross-layout masking (Stage B, Eq. 1-2) targets full-width spanners that sit
# *between* columns and break vertical cuts — common in newspapers, rare in
# academic PDFs, where it also mis-fires on the full-width document title (which
# the recursive cut already peels correctly as the top horizontal band). Off by
# default for our 1-2 column corpus; flip on for newspaper-style layouts.
ENABLE_CROSS_MASK = False

# Label -> category for OUR PP-DocLayoutV3 vocabulary (the paper assumes its own
# label set; this must match what the detector actually emits).
#
# Deviation from the paper's Pre-Mask (§4.1): the paper masks titles too (tuned
# on newspaper layouts). In academic PDFs the document/section titles flow
# correctly with the body via the XY-cut (top title -> first band; section title
# -> band above its paragraph), and masking them pushes the title behind the
# authors. So we keep titles in the backbone and pre-mask only VISION (the truly
# position-flexible figures/tables) and MARGINAL page furniture (rotated arXiv
# stamps, running heads, page numbers) that otherwise corrupt the column cuts.
VISION_LABELS = frozenset(
    {"image", "figure", "table", "chart", "display_formula", "formula", "seal"}
)
MARGINAL_LABELS = frozenset(
    {"aside_text", "header", "footer", "header_image", "footer_image", "number"}
)
# Anything else -> "other" backbone (text, abstract, footnote, formula_number,
# and all *_title labels — kept in the backbone, see above).

# CMM label priority (L_order, Eq. 7), extended for our categories:
# cross-layout > title > vision > other > marginal (furniture restored last).
_PRIORITY = {"cross": 3, "title": 2, "vision": 1, "other": 0, "marginal": -1}


# --------------------------------------------------------------------------- #
# Geometry helpers (bbox = [x1, y1, x2, y2])
# --------------------------------------------------------------------------- #
def _category(label: Optional[str]) -> str:
    if label in VISION_LABELS:
        return "vision"
    if label in MARGINAL_LABELS:
        return "marginal"
    return "other"


def _w(b: Sequence[float]) -> float:
    return b[2] - b[0]


def _h(b: Sequence[float]) -> float:
    return b[3] - b[1]


def _cx(b: Sequence[float]) -> float:
    return (b[0] + b[2]) / 2.0


def _cy(b: Sequence[float]) -> float:
    return (b[1] + b[3]) / 2.0


def _direction(b: Sequence[float]) -> str:
    """Layout orientation of a block: 'h' if wider than tall, else 'v'."""
    return "h" if _w(b) >= _h(b) else "v"


def _proj_overlap(a: Sequence[float], b: Sequence[float], axis: int) -> float:
    """Length of overlap of a,b projected onto axis (0=x, 1=y)."""
    lo, hi = (0, 2) if axis == 0 else (1, 3)
    return max(0.0, min(a[hi], b[hi]) - max(a[lo], b[lo]))


def _proj_iou(a: Sequence[float], b: Sequence[float], axis: int) -> float:
    lo, hi = (0, 2) if axis == 0 else (1, 3)
    inter = max(0.0, min(a[hi], b[hi]) - max(a[lo], b[lo]))
    union = max(a[hi], b[hi]) - min(a[lo], b[lo])
    return inter / union if union > 0 else 0.0


# --------------------------------------------------------------------------- #
# Stage B — cross-layout detection (Eq. 1-2)
# --------------------------------------------------------------------------- #
def _detect_cross_layout(boxes: list, idxs: list[int]) -> set[int]:
    """Indices of full-width spanning blocks among ``idxs``."""
    if len(idxs) < 3:
        return set()
    threshold = BETA * median(_w(boxes[i]) for i in idxs)
    cross: set[int] = set()
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
# Stage C — density-driven recursive projection cut (Eq. 4-5)
# --------------------------------------------------------------------------- #
def _widest_gap_split(
    boxes: list, idxs: list[int], axis: int
) -> Optional[tuple[list[int], list[int]]]:
    """Split ``idxs`` at the single widest straddle-free gap along ``axis``.

    Returns the (before, after) groups in reading order, or ``None`` if no box
    is fully clear of the others along this axis (e.g. a full-width block bridges
    every gap). Splitting at one gap and recursing — rather than at all gaps at
    once — keeps the preferred axis in control: a full-width title is peeled by a
    single horizontal cut, and the body underneath then re-tries the vertical
    (column) cut instead of being sliced row-major by the same pass.
    """
    lo, hi = (0, 2) if axis == 0 else (1, 3)
    order = sorted(idxs, key=lambda i: boxes[i][lo])
    best_gap = GAP_EPS
    best_pos: Optional[int] = None
    cur_end = boxes[order[0]][hi]
    for k in range(1, len(order)):
        start = boxes[order[k]][lo]
        if start - cur_end > best_gap:
            best_gap, best_pos = start - cur_end, k
        cur_end = max(cur_end, boxes[order[k]][hi])
    if best_pos is None:
        return None
    return order[:best_pos], order[best_pos:]


def _regional_density(boxes: list, idxs: list[int], cross: set[int]) -> float:
    """tau_d: cross-layout area / single-layout area within a region (Eq. 4)."""
    cross_area = sum(_w(boxes[i]) * _h(boxes[i]) for i in idxs if i in cross)
    single_area = sum(_w(boxes[i]) * _h(boxes[i]) for i in idxs if i not in cross)
    if single_area <= 0:
        return float("inf")
    return cross_area / single_area


def _recursive_cut(boxes: list, idxs: list[int], cross: set[int]) -> list[int]:
    """Recursively XY/YX-cut ``idxs`` into reading order."""
    if len(idxs) <= 1:
        return list(idxs)
    tau = _regional_density(boxes, idxs, cross)
    # tau_d > theta_v -> XY-Cut (rows first, axis=y); else YX-Cut (columns
    # first, axis=x). With cross-layout already masked, tau is ~0 here, so body
    # text correctly splits into columns before rows. (Eq. 5)
    primary = 1 if tau > DENSITY_THRESHOLD else 0
    for axis in (primary, 1 - primary):
        split = _widest_gap_split(boxes, idxs, axis)
        if split is not None:
            before, after = split
            return _recursive_cut(boxes, before, cross) + _recursive_cut(boxes, after, cross)
    # Indivisible (mutually overlapping boxes): fall back to (y1, x1).
    return sorted(idxs, key=lambda i: (boxes[i][1], boxes[i][0]))


# --------------------------------------------------------------------------- #
# Stage D — cross-modal matching: remap masked elements (Alg. 1, Eq. 8-14)
# --------------------------------------------------------------------------- #
def _edge_weights(cat: str, orient: str) -> list[float]:
    """Semantic-specific edge-weight vector w_edge (Eq. 14)."""
    if cat == "title":
        return [1.0, 0.1, 0.1, 1.0] if orient == "h" else [0.2, 0.1, 1.0, 1.0]
    if cat == "cross":
        return [1.0, 1.0, 0.1, 1.0]
    return [1.0, 1.0, 1.0, 0.1]  # vision / other


def _distance(bp: Sequence[float], bo: Sequence[float], cat: str, page_max: float) -> float:
    """Joint geometric distance D(B_p, B_o) of a pending box to an anchor (Eq. 8).

    Scale weights (Eq. 13) stagger the four constraints by magnitude so they act
    lexicographically: intersection (phi1) dominates, then proximity (phi2),
    then vertical continuity (phi3), then horizontal ordering (phi4).
    """
    # Marginal furniture (page numbers, running heads, rotated stamps) has no
    # reliable layout alignment; the paper's axis-aligned min(dx,dy) term would
    # snap a centered page number onto the far-away title. Use plain Euclidean
    # center distance so it attaches to its genuinely nearest block (and, being
    # lowest priority, sorts right after it).
    if cat == "marginal":
        return (_cx(bp) - _cx(bo)) ** 2 + (_cy(bp) - _cy(bo)) ** 2

    # phi1 — intersection constraint (Eq. 9): 1 (penalised) unless same
    # orientation and projection-IoU >= tau_overlap.
    axis = 0 if _direction(bp) == "h" else 1
    if _direction(bp) != _direction(bo) or _proj_iou(bp, bo, axis) < OVERLAP_THRESHOLD:
        phi1 = 1.0
    else:
        phi1 = 0.0

    # phi2 — boundary proximity (Eq. 10): center distance, axis-aligned preferred.
    dx, dy = abs(_cx(bp) - _cx(bo)), abs(_cy(bp) - _cy(bo))
    aligned = _proj_overlap(bp, bo, 0) > 0 or _proj_overlap(bp, bo, 1) > 0
    phi2 = min(dx, dy) if aligned else dx + dy

    # phi3 — vertical continuity (Eq. 11).
    phi3 = -bo[3] if (cat == "cross" and bp[1] > bo[3]) else bo[1]

    # phi4 — horizontal ordering (Eq. 12): anchor left boundary.
    phi4 = bo[0]

    scale = [page_max * page_max, page_max, 1.0, 1.0 / page_max]
    edge = _edge_weights(cat, _direction(bp))
    phi = [phi1, phi2, phi3, phi4]
    return sum(scale[k] * edge[k] * phi[k] for k in range(4))


def _cross_modal_match(
    boxes: list,
    anchors: list[int],
    masked: dict[str, list[int]],
    page_max: float,
) -> dict[int, int]:
    """Assign every masked block an anchor rank (its slot in the ordered
    backbone). Returns ``{block_index -> anchor_rank}`` for masked blocks.

    Higher-priority masked elements are restored first and become anchors for
    lower-priority ones (Alg. 1: T grows as matches are found).
    """
    anchor_rank: dict[int, int] = {idx: r for r, idx in enumerate(anchors)}
    # T: candidate anchors searched for a match — starts as the backbone, grows.
    candidates: list[int] = list(anchors)
    assigned: dict[int, int] = {}

    for cat in ("cross", "vision", "marginal"):
        for bp in masked.get(cat, []):
            if not candidates:
                break
            best, best_d = None, float("inf")
            for bo in candidates:
                d = _distance(boxes[bp], boxes[bo], cat, page_max)
                if d < best_d:
                    best_d, best = d, bo
            if best is not None:
                rank = anchor_rank[best]
                anchor_rank[bp] = rank
                assigned[bp] = rank
                candidates.append(bp)  # may now anchor lower-priority elements
    return assigned


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def compute_reading_order(
    boxes: list[Sequence[float]],
    labels: Sequence[Optional[str]],
    width: float,
    height: float,
) -> list[int]:
    """Reading order over page regions. Runs Stages A->D.

    Args:
        boxes: per-region bbox ``[x1, y1, x2, y2]`` in page-pixel coords.
        labels: per-region PP-DocLayoutV3 label (parallel to ``boxes``).
        width, height: page dimensions in pixels.

    Returns:
        ``order`` parallel to ``boxes``: ``order[i]`` is the 0-based reading
        position of region ``i``.
    """
    n = len(boxes)
    if n == 0:
        return []
    if n == 1:
        return [0]

    page_max = max(float(width), float(height), 1.0)
    cats = [_category(labels[i] if i < len(labels) else None) for i in range(n)]

    # Stage A — pre-mask vision + marginal furniture; body text and titles are
    # the backbone (see VISION/MARGINAL note above).
    backbone = [i for i in range(n) if cats[i] == "other"]
    masked: dict[str, list[int]] = {
        "cross": [],
        "vision": [i for i in range(n) if cats[i] == "vision"],
        "marginal": [i for i in range(n) if cats[i] == "marginal"],
    }

    # Stage B — optionally pull full-width spanning blocks out of the backbone
    # for highest-priority remapping (off by default; see ENABLE_CROSS_MASK).
    cross = _detect_cross_layout(boxes, backbone) if ENABLE_CROSS_MASK else set()
    masked["cross"] = sorted(cross, key=lambda i: (boxes[i][1], boxes[i][0]))
    core = [i for i in backbone if i not in cross]

    # Stage C — recursive cut orders the single-layout backbone.
    if core:
        anchors = _recursive_cut(boxes, core, set())
    else:
        # No body text (e.g. a figure-only page): order everything geometrically
        # and skip CMM — there are no anchors to remap against.
        return _ranks_from_order(
            sorted(range(n), key=lambda i: (boxes[i][1], boxes[i][0])), n
        )

    # Stage D — remap masked elements onto anchor slots.
    masked_rank = _cross_modal_match(boxes, anchors, masked, page_max)

    anchor_rank: dict[int, int] = {idx: r for r, idx in enumerate(anchors)}
    anchor_rank.update(masked_rank)

    # Any element that never got an anchor (shouldn't happen, but be safe):
    # append after the backbone, geometrically ordered.
    leftover = [i for i in range(n) if i not in anchor_rank]
    big = len(anchors)
    for i in sorted(leftover, key=lambda i: (boxes[i][1], boxes[i][0])):
        anchor_rank[i] = big

    # Final sort: anchor index asc -> label priority desc -> y1 asc -> x1 asc
    # (Fig. 7 caption: "Index, Label Priority, Y1, X1").
    ordering = sorted(
        range(n),
        key=lambda i: (
            anchor_rank[i],
            -_PRIORITY[_cat_with_cross(i, cats, cross)],
            boxes[i][1],
            boxes[i][0],
        ),
    )
    return _ranks_from_order(ordering, n)


def _cat_with_cross(i: int, cats: list[str], cross: set[int]) -> str:
    return "cross" if i in cross else cats[i]


def _ranks_from_order(ordering: list[int], n: int) -> list[int]:
    """Invert an ordering (list of indices in reading order) to per-region rank."""
    ranks = [0] * n
    for rank, idx in enumerate(ordering):
        ranks[idx] = rank
    return ranks
