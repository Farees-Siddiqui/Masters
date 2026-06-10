"""Positional char-stream alignment — works at any granularity.

The naive region matcher (:func:`alignment.align_naive`) scores each node
against each box *independently*, which is fine for paragraph regions but falls
apart at line/word granularity: a common word like ``"the"`` is contained in
(hence "matches") every node it appears in, so clicking one paragraph would
light up every ``the`` on the page.

This aligner is **positional** instead. Both sides carry the same OCR'd text, so
we:

1. flatten the AST into a normalized character stream, remembering which node
   each char came from;
2. flatten the PDF boxes (in reading order) into a character stream, remembering
   which ``(page, box_index)`` each char came from;
3. sequence-align the two streams with :class:`difflib.SequenceMatcher`;
4. for each node, collect the boxes its characters aligned to.

Because step 3 matches *subsequences by position*, the second ``"the"`` in the
document aligns to the second ``"the"`` in the PDF — not to all of them. The same
machinery therefore works for ``paragraph``, ``line`` and ``word`` boxes with no
special-casing: at word granularity a node simply maps to the run of word boxes
that spell it out. A node's set is unioned with its whole subtree's, so a section
heading lights up every box under it. Return shape matches ``align_naive``.
"""

from __future__ import annotations

import difflib

from .naive_aligner import _iter_segments, _normalize


def align_stream(ast_root: dict, pages: list[dict], granularity: str = "paragraph") -> dict:
    """Map AST nodes to PDF boxes by positional char-stream alignment.

    ``pages`` is ``[{"page": int, "boxes": [box_dict, ...]}, ...]`` at the chosen
    granularity; each box needs ``text`` (and ``order``/``bbox`` for sequencing).
    Returns ``{"alignment": {node_id: [{"page", "box_index"}]}, "coverage", "granularity"}``.
    ``box_index`` indexes into that page's ``boxes`` list as supplied here.
    """
    # --- Doc stream: normalized chars + per-node spans --------------------- #
    doc_chars: list[str] = []
    node_span: dict[str, tuple[int, int]] = {}  # node_id -> (start, length)
    for node_id, text in _iter_segments(ast_root):
        norm = _normalize(text)
        if not norm:
            continue
        start = len(doc_chars)
        doc_chars.extend(norm)
        node_span[node_id] = (start, len(norm))
        doc_chars.append(" ")  # separator so adjacent nodes don't fuse

    # --- PDF stream: normalized chars + per-char (page, box_index) --------- #
    pdf_chars: list[str] = []
    pdf_box: list[tuple[int, int] | None] = []
    for page in pages:
        page_no = page["page"]
        boxes = page.get("boxes", [])
        # Reading order (XY-Cut++ `order`), falling back to top->bottom/left->right.
        seq = sorted(
            range(len(boxes)),
            key=lambda i: (
                boxes[i]["order"] if boxes[i].get("order") is not None else float("inf"),
                boxes[i]["bbox"][1],
                boxes[i]["bbox"][0],
            ),
        )
        for i in seq:
            norm = _normalize(boxes[i].get("text") or "")
            if not norm:
                continue
            for ch in norm:
                pdf_chars.append(ch)
                pdf_box.append((page_no, i))
            pdf_chars.append(" ")
            pdf_box.append(None)  # separator belongs to no box

    # --- Align the two streams -------------------------------------------- #
    matcher = difflib.SequenceMatcher(None, "".join(doc_chars), "".join(pdf_chars), autojunk=False)
    doc_to_pdf: dict[int, int] = {}
    for a, b, n in matcher.get_matching_blocks():
        for k in range(n):
            doc_to_pdf[a + k] = b + k

    # --- Per-node direct boxes (+ reverse: box -> owning nodes) ------------ #
    direct: dict[str, set[tuple[int, int]]] = {}
    # box -> {node_id: char_count}; the node contributing the most chars to a box
    # is its primary owner (used for box->AST reverse highlighting).
    box_owners: dict[tuple[int, int], dict[str, int]] = {}
    matched = 0
    for node_id, (start, length) in node_span.items():
        hits: set[tuple[int, int]] = set()
        for di in range(start, start + length):
            pj = doc_to_pdf.get(di)
            if pj is not None and pdf_box[pj] is not None:
                box = pdf_box[pj]
                hits.add(box)
                owners = box_owners.setdefault(box, {})
                owners[node_id] = owners.get(node_id, 0) + 1
        if hits:
            matched += 1
        direct[node_id] = hits

    # Reverse map, primary (most chars) owner first.
    reverse = {
        f"{page}:{idx}": [nid for nid, _ in sorted(owners.items(), key=lambda kv: -kv[1])]
        for (page, idx), owners in box_owners.items()
    }

    # --- Subtree union ---------------------------------------------------- #
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
    coverage = matched / len(node_span) if node_span else 0.0
    return {
        "alignment": alignment,
        "reverse": reverse,
        "coverage": coverage,
        "granularity": granularity,
    }
