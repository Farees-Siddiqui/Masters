"""Scoring: predicted box sets vs gold box sets, per node.

A node is **localized** when the pixel area of its predicted boxes overlaps
the area of its gold boxes at IoU >= 0.5. Area (not box identity) is the
comparison so that granularity quirks don't decide the score: if gold says one
box and the aligner claims the same region as two half-boxes, the areas agree
and the node counts as found. Both sides draw from the same box inventory, so
box padding cancels out of the ratio.

Predictions come from the aligner's ``reverse`` map (box -> ranked owners):
the primary owner claims the box, mirroring how the UI colours a region.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .gold import ABSENT, GoldDoc

# Rasterization scale for area union/intersection. 1/4 of a 200-dpi pixel is
# ~0.5mm on paper — far below any decision boundary this metric faces.
_GRID = 4

IOU_THRESHOLD = 0.5


@dataclass
class ScoreCard:
    """Node-level results for one (doc, method) pair."""

    n_nodes: int = 0  # gradeable nodes (gold boxes or ABSENT)
    n_localized: int = 0  # IoU >= threshold
    n_exact: int = 0  # predicted box set == gold box set
    n_missed: int = 0  # gold boxes, predicted nothing
    n_absent: int = 0  # gold ABSENT
    n_absent_correct: int = 0  # ... and aligner predicted nothing
    iou_sum: float = 0.0
    n_furniture: int = 0
    n_false_claims: int = 0  # furniture boxes claimed by some node
    by_kind: dict[str, dict] = field(default_factory=dict)
    by_tier: dict[str, dict] = field(default_factory=dict)

    @property
    def localization_accuracy(self) -> float | None:
        n = self.n_nodes - self.n_absent
        return self.n_localized / n if n else None

    @property
    def mean_iou(self) -> float | None:
        n = self.n_nodes - self.n_absent
        return self.iou_sum / n if n else None

    @property
    def absent_accuracy(self) -> float | None:
        return self.n_absent_correct / self.n_absent if self.n_absent else None

    @property
    def false_claim_rate(self) -> float | None:
        return self.n_false_claims / self.n_furniture if self.n_furniture else None

    def to_dict(self) -> dict:
        def r(v: float | None) -> float | None:
            return round(v, 4) if v is not None else None

        return {
            "n_nodes": self.n_nodes,
            "localization_accuracy": r(self.localization_accuracy),
            "mean_iou": r(self.mean_iou),
            "n_exact": self.n_exact,
            "n_missed": self.n_missed,
            "n_absent": self.n_absent,
            "absent_accuracy": r(self.absent_accuracy),
            "n_furniture": self.n_furniture,
            "false_claim_rate": r(self.false_claim_rate),
            "by_kind": self.by_kind,
            "by_tier": self.by_tier,
        }


def _region_masks(
    box_keys: set[str], pages_by_no: dict[int, dict]
) -> dict[int, np.ndarray]:
    """page -> boolean occupancy mask (at 1/_GRID scale) for a set of boxes."""
    masks: dict[int, np.ndarray] = {}
    for key in box_keys:
        page_no, idx = key.split(":")
        page = pages_by_no.get(int(page_no))
        if page is None:
            continue
        i = int(idx)
        if i >= len(page["boxes"]):
            continue
        if int(page_no) not in masks:
            h = int(page["height"]) // _GRID + 1
            w = int(page["width"]) // _GRID + 1
            masks[int(page_no)] = np.zeros((h, w), dtype=bool)
        x0, y0, x1, y1 = page["boxes"][i]["bbox"]
        m = masks[int(page_no)]
        m[int(y0) // _GRID : int(y1) // _GRID + 1, int(x0) // _GRID : int(x1) // _GRID + 1] = True
    return masks


def _iou(a: dict[int, np.ndarray], b: dict[int, np.ndarray]) -> float:
    inter = union = 0
    for page in set(a) | set(b):
        ma, mb = a.get(page), b.get(page)
        if ma is None:
            union += int(mb.sum())
        elif mb is None:
            union += int(ma.sum())
        else:
            inter += int((ma & mb).sum())
            union += int((ma | mb).sum())
    return inter / union if union else 0.0


def _bump(bucket: dict, key: str, localized: bool, iou: float) -> None:
    b = bucket.setdefault(key, {"n": 0, "localized": 0, "iou_sum": 0.0})
    b["n"] += 1
    b["localized"] += int(localized)
    b["iou_sum"] += iou


def _finish_buckets(bucket: dict) -> dict:
    return {
        k: {
            "n": v["n"],
            "localization_accuracy": round(v["localized"] / v["n"], 4),
            "mean_iou": round(v["iou_sum"] / v["n"], 4),
        }
        for k, v in sorted(bucket.items())
    }


def score_doc(gold: GoldDoc, reverse: dict[str, list[str]]) -> ScoreCard:
    """Score one aligner's ``reverse`` map against a document's gold."""
    card = ScoreCard()
    pages_by_no = {p["page"]: p for p in gold.pages}

    # Primary owner per box, then invert to node -> predicted box set.
    pred: dict[str, set[str]] = {}
    claimed_boxes: set[str] = set()
    for box_key, owners in reverse.items():
        if owners:
            pred.setdefault(owners[0], set()).add(box_key)
            claimed_boxes.add(box_key)

    kind_bucket: dict[str, dict] = {}
    tier_bucket: dict[str, dict] = {}

    for nid, gold_val in gold.nodes.items():
        card.n_nodes += 1
        pred_boxes = pred.get(nid, set())

        if gold_val == ABSENT:
            card.n_absent += 1
            if not pred_boxes:
                card.n_absent_correct += 1
            continue

        assert isinstance(gold_val, set)
        if not pred_boxes:
            card.n_missed += 1
            iou, localized = 0.0, False
        elif pred_boxes == gold_val:
            iou, localized = 1.0, True
            card.n_exact += 1
        else:
            iou = _iou(
                _region_masks(pred_boxes, pages_by_no),
                _region_masks(gold_val, pages_by_no),
            )
            localized = iou >= IOU_THRESHOLD
        card.n_localized += int(localized)
        card.iou_sum += iou
        _bump(kind_bucket, gold.kind.get(nid, "prose"), localized, iou)
        _bump(tier_bucket, gold.tier.get(nid, "oracle"), localized, iou)

    card.by_kind = _finish_buckets(kind_bucket)
    card.by_tier = _finish_buckets(tier_bucket)

    card.n_furniture = len(gold.furniture)
    card.n_false_claims = sum(1 for b in gold.furniture if b in claimed_boxes)
    return card


def merge_cards(cards: list[ScoreCard]) -> ScoreCard:
    """Pool per-doc cards into a corpus card (nodes weighted equally)."""
    out = ScoreCard()
    kind_bucket: dict[str, dict] = {}
    tier_bucket: dict[str, dict] = {}
    for c in cards:
        out.n_nodes += c.n_nodes
        out.n_localized += c.n_localized
        out.n_exact += c.n_exact
        out.n_missed += c.n_missed
        out.n_absent += c.n_absent
        out.n_absent_correct += c.n_absent_correct
        out.iou_sum += c.iou_sum
        out.n_furniture += c.n_furniture
        out.n_false_claims += c.n_false_claims
        for bucket, src in ((kind_bucket, c.by_kind), (tier_bucket, c.by_tier)):
            for k, v in src.items():
                b = bucket.setdefault(k, {"n": 0, "localized": 0, "iou_sum": 0.0})
                b["n"] += v["n"]
                b["localized"] += round(v["localization_accuracy"] * v["n"])
                b["iou_sum"] += v["mean_iou"] * v["n"]
    out.by_kind = _finish_buckets(kind_bucket)
    out.by_tier = _finish_buckets(tier_bucket)
    return out
