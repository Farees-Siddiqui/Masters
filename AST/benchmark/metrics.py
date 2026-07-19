"""Scoring core shared by every benchmark mode.

A benchmark gives us, per PDF box, the *true* AST node that owns it
(``gold_owner: {box_key -> node_id}``). An aligner gives us the reverse map it
predicted (``reverse: {box_key -> [node_id, ...]}``, primary owner first — the
same shape :func:`alignment.align_stream` / :func:`alignment.align_similarity`
return). ``box_key`` is ``"page:box_index"`` on both sides.

From those two we report:

* **precision / recall / F1** over ``(node_id, box_key)`` pairs, taking each
  box's *primary* predicted owner — a precision-honest view (a box maps to one
  node).
* **owner accuracy** — fraction of gold boxes whose predicted primary owner is
  the true one (recall-honest: an unpredicted box counts against you).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Scores:
    precision: float
    recall: float
    f1: float
    owner_accuracy: float
    n_gold: int
    n_pred: int


def prf1(gold: set, pred: set) -> tuple[float, float, float]:
    """Precision/recall/F1 of a predicted pair set against gold."""
    if not gold and not pred:
        return 1.0, 1.0, 1.0
    tp = len(gold & pred)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def score_alignment(gold_owner: dict[str, str], reverse: dict[str, list[str]]) -> Scores:
    """Score a predicted ``reverse`` map against the gold ``box -> owner`` map."""
    gold_pairs = {(nid, bk) for bk, nid in gold_owner.items()}
    pred_pairs = {(nids[0], bk) for bk, nids in reverse.items() if nids}
    precision, recall, f1 = prf1(gold_pairs, pred_pairs)

    correct = sum(
        1
        for bk, nid in gold_owner.items()
        if reverse.get(bk) and reverse[bk][0] == nid
    )
    owner_acc = correct / len(gold_owner) if gold_owner else 0.0
    return Scores(precision, recall, f1, owner_acc, len(gold_pairs), len(pred_pairs))
