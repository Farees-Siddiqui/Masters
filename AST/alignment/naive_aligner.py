"""Naive node -> box alignment (v1: simple text matching).

The whiteboard model is ``Alignment : Loc[Doc] -> Set[Loc[PDF]]``. This is the
deliberately-simple first cut of the *region*-level version of that map: for
each AST node that carries text, find the PDF layout boxes whose OCR text
matches, by plain normalized string matching:

  * **containment** — the shorter normalized string appears inside the longer
    one (handles a Mistral paragraph that the layout engine split, or vice
    versa) -> score 1.0;
  * else **token overlap** — ``|A ∩ B| / min(|A|, |B|)`` over word sets, so two
    texts that mostly share words still match despite OCR noise (the two engines
    never agree byte-for-byte).

A box is a match when the score clears ``threshold``. A node's highlight set is
its own direct matches **unioned with its whole subtree's** — so clicking a
section heading lights up every region under it.

This is intentionally crude (no reading order, no fuzzy char diff). It shares
its return shape with the planned region aligner, so a smarter matcher can drop
in behind ``align_naive`` without the backend/frontend changing. The char-stream
:class:`alignment.Aligner` remains for substring<->word work.
"""

from __future__ import annotations

import re
from typing import Iterator

_NON_ALNUM = re.compile(r"[^a-z0-9\s]+")
_WS = re.compile(r"\s+")


def _normalize(s: str) -> str:
    """Lowercase, drop punctuation/markdown, collapse whitespace."""
    s = _NON_ALNUM.sub(" ", s.lower())
    return _WS.sub(" ", s).strip()


def _iter_segments(node: dict) -> Iterator[tuple[str, str]]:
    """Yield ``(node_id, text)`` for every node carrying alignable text.

    Headings (``section``) carry their text in ``attribs["title"]``, not
    ``.text``; everything else (paragraph/list_item/code) uses ``.text``.
    """
    if node.get("type") == "section":
        title = (node.get("attribs") or {}).get("title")
        if title:
            yield node["id"], title
    else:
        text = node.get("text")
        if text:
            yield node["id"], text
    for child in node.get("children") or []:
        yield from _iter_segments(child)


def _score(a: str, b: str) -> float:
    """Naive similarity of two already-normalized strings, in ``[0, 1]``."""
    if not a or not b:
        return 0.0
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if short in long:
        return 1.0
    at, bt = set(a.split()), set(b.split())
    if not at or not bt:
        return 0.0
    return len(at & bt) / min(len(at), len(bt))


def align_naive(ast_root: dict, pages: list[dict], threshold: float = 0.5) -> dict:
    """Map AST nodes to the PDF boxes that render them.

    ``pages`` is ``[{"page": int, "boxes": [box_dict, ...]}, ...]`` where each
    box dict has at least ``text`` (paragraph granularity is expected). Returns::

        {
          "alignment": { node_id: [{"page": int, "box_index": int}, ...] },
          "coverage":  float,   # fraction of text segments with >=1 direct match
          "threshold": float,
        }

    ``box_index`` indexes into that page's ``boxes`` list (the same order the
    ``/api/align/compute`` response ships), so the frontend can highlight by
    index. A node's entry is the union of its own and its descendants' matches.
    """
    # Precompute normalized box texts once: [(page_no, [(box_index, norm), ...])]
    page_boxes: list[tuple[int, list[tuple[int, str]]]] = []
    for page in pages:
        norms = [
            (i, _normalize(b["text"]))
            for i, b in enumerate(page.get("boxes", []))
            if b.get("text")
        ]
        page_boxes.append((page["page"], norms))

    direct: dict[str, set[tuple[int, int]]] = {}
    segments = list(_iter_segments(ast_root))
    matched = 0
    for node_id, text in segments:
        norm = _normalize(text)
        hits: set[tuple[int, int]] = set()
        for page_no, norms in page_boxes:
            for box_index, box_norm in norms:
                if _score(norm, box_norm) >= threshold:
                    hits.add((page_no, box_index))
        if hits:
            matched += 1
        direct[node_id] = hits

    # Each node's highlight set = its own matches U its whole subtree's.
    union: dict[str, set[tuple[int, int]]] = {}

    def _build(node: dict) -> set[tuple[int, int]]:
        acc = set(direct.get(node["id"], ()))
        for child in node.get("children") or []:
            acc |= _build(child)
        union[node["id"]] = acc
        return acc

    _build(ast_root)

    alignment = {
        node_id: [{"page": p, "box_index": b} for p, b in sorted(hits)]
        for node_id, hits in union.items()
        if hits
    }
    coverage = matched / len(segments) if segments else 0.0
    return {"alignment": alignment, "coverage": coverage, "threshold": threshold}
