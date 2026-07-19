"""Synthetic OCR-noise benchmark — ground truth by construction, no download.

We treat a real AST's text segments as the "documents." Each segment is
optionally split into line-length chunks (simulating the layout engine breaking
a paragraph into several boxes), and each chunk's text is corrupted with
character-level OCR noise at a chosen rate. Because we *generated* every box from
a known node, the true ``box -> node`` mapping is known exactly.

Sweeping the noise rate and scoring each aligner gives the headline curve: the
semantic ``similarity`` aligner should stay accurate as noise rises, while the
exact-character ``stream`` aligner falls off.
"""

from __future__ import annotations

import random

from alignment.naive_aligner import _iter_segments

_ALPHA = "abcdefghijklmnopqrstuvwxyz"


def perturb(text: str, rate: float, rng: random.Random) -> str:
    """Corrupt ``text`` with per-character delete/substitute/duplicate at ``rate``."""
    if rate <= 0:
        return text
    out: list[str] = []
    for ch in text:
        if rng.random() < rate:
            kind = rng.random()
            if kind < 0.34:
                continue  # deletion
            elif kind < 0.67:
                out.append(rng.choice(_ALPHA) if ch.isalpha() else ch)  # substitution
            else:
                out.append(ch)
                out.append(ch)  # duplication
        else:
            out.append(ch)
    return "".join(out)


def chunk(text: str, width: int) -> list[str]:
    """Split ``text`` into ~``width``-char pieces on word boundaries (line boxes)."""
    if width <= 0:
        return [text]
    pieces: list[str] = []
    cur = ""
    for w in text.split():
        if cur and len(cur) + 1 + len(w) > width:
            pieces.append(cur)
            cur = w
        else:
            cur = w if not cur else f"{cur} {w}"
    if cur:
        pieces.append(cur)
    return pieces or [text]


def build_pages(
    ast_root: dict,
    noise: float,
    rng: random.Random,
    split_width: int = 0,
    shuffle: bool = False,
) -> tuple[list[dict], dict[str, str]]:
    """Build synthetic ``pages`` + the gold ``box_key -> node_id`` map.

    Returns ``(pages, gold_owner)`` where ``pages`` matches the aligner input
    shape (``[{"page", "width", "height", "boxes": [{text, bbox, order}]}]``) and
    ``gold_owner`` maps ``"1:<box_index>"`` to the node that text came from.

    With ``shuffle=True`` the box order is randomly permuted and each box's
    ``order`` is set to its new position — a worst-case stand-in for the PDF
    reading order disagreeing with the AST order (multi-column / detection
    reordering). Positional (``stream``) alignment relies on that order; semantic
    matching does not, so this is where the two diverge.
    """
    boxes: list[dict] = []
    gold_owner: dict[str, str] = {}
    y = 0
    for node_id, text in _iter_segments(ast_root):
        text = (text or "").strip()
        if not text:
            continue
        for piece in chunk(text, split_width):
            idx = len(boxes)
            boxes.append(
                {"text": perturb(piece, noise, rng), "bbox": [0, y, 500, y + 20], "order": idx}
            )
            gold_owner[f"1:{idx}"] = node_id
            y += 25

    if shuffle:
        perm = list(range(len(boxes)))
        rng.shuffle(perm)
        reordered: list[dict] = []
        new_gold: dict[str, str] = {}
        for new_idx, old_idx in enumerate(perm):
            b = boxes[old_idx]
            b["order"] = new_idx  # detected reading order, now scrambled vs the AST
            reordered.append(b)
            new_gold[f"1:{new_idx}"] = gold_owner[f"1:{old_idx}"]
        boxes, gold_owner = reordered, new_gold

    pages = [{"page": 1, "width": 500, "height": max(y, 1), "boxes": boxes}]
    return pages, gold_owner
