"""Structural diff of two documents via AST alignment.

Most diff tools need the source text of both documents. This one works on two
*rendered* PDFs/scans: each is OCR'd into an AST, and we align the two ASTs
**segment-to-segment** by embedding similarity (the same model the similarity
aligner uses). From the matching we classify every segment as:

* **unchanged** — matched, normalized text identical;
* **modified**  — matched, but the text differs (an edit);
* **added**     — present in B with no match in A;
* **removed**   — present in A with no match in B.

The result is a unified, document-ordered diff (B's order, with removed segments
interleaved at their original position) that the UI renders like a structural
``git diff`` and can trace back to the exact region of either page.
"""

from __future__ import annotations

from .naive_aligner import _iter_segments, _normalize
from .similarity_aligner import _clean, _get_model, _DEFAULT_MODEL


def _segments(ast: dict) -> list[tuple[str, str]]:
    out = []
    for nid, text in _iter_segments(ast):
        t = (text or "").strip()
        if t:
            out.append((nid, t))
    return out


def diff_documents(
    ast_a: dict,
    ast_b: dict,
    threshold: float = 0.6,
    model_name: str = _DEFAULT_MODEL,
) -> dict:
    """Return a structural diff of document A vs document B.

    ``{"counts": {...}, "items": [ ... ]}`` where each item is one of::

        {"type": "unchanged", "a_id", "b_id", "text"}
        {"type": "modified",  "a_id", "b_id", "a_text", "b_text", "score"}
        {"type": "added",     "b_id", "text"}
        {"type": "removed",   "a_id", "text"}
    """
    segs_a = _segments(ast_a)
    segs_b = _segments(ast_b)
    n_a, n_b = len(segs_a), len(segs_b)

    # match_a[i] = (j, score); match_b[j] = (i, score)
    match_a: dict[int, tuple[int, float]] = {}
    match_b: dict[int, tuple[int, float]] = {}

    if n_a and n_b:
        model = _get_model(model_name)
        emb_a = model.encode([_clean(t) for _, t in segs_a], convert_to_numpy=True, normalize_embeddings=True, batch_size=64)
        emb_b = model.encode([_clean(t) for _, t in segs_b], convert_to_numpy=True, normalize_embeddings=True, batch_size=64)
        sims = emb_a @ emb_b.T  # (n_a, n_b)

        # Greedy one-to-one assignment by descending similarity (above threshold).
        cand = [
            (float(sims[i, j]), i, j)
            for i in range(n_a)
            for j in range(n_b)
            if sims[i, j] >= threshold
        ]
        cand.sort(reverse=True)
        used_a: set[int] = set()
        used_b: set[int] = set()
        for score, i, j in cand:
            if i in used_a or j in used_b:
                continue
            used_a.add(i)
            used_b.add(j)
            match_a[i] = (j, score)
            match_b[j] = (i, score)

    # Where to interleave each removed (unmatched A) segment: right after the B
    # counterpart of the nearest preceding matched A segment (-1 = before all B).
    removed_after: dict[int, list[int]] = {}
    last_matched_b = -1
    for i in range(n_a):
        if i in match_a:
            last_matched_b = match_a[i][0]
        else:
            removed_after.setdefault(last_matched_b, []).append(i)

    counts = {"added": 0, "removed": 0, "modified": 0, "unchanged": 0}
    items: list[dict] = []

    def _emit_removed(anchor_b: int) -> None:
        for i in removed_after.get(anchor_b, ()):
            counts["removed"] += 1
            items.append({"type": "removed", "a_id": segs_a[i][0], "text": segs_a[i][1]})

    _emit_removed(-1)  # removed segments that precede the first kept B segment
    for j in range(n_b):
        if j in match_b:
            i, score = match_b[j]
            a_id, a_text = segs_a[i]
            b_id, b_text = segs_b[j]
            if _normalize(a_text) == _normalize(b_text):
                counts["unchanged"] += 1
                items.append({"type": "unchanged", "a_id": a_id, "b_id": b_id, "text": b_text})
            else:
                counts["modified"] += 1
                items.append(
                    {"type": "modified", "a_id": a_id, "b_id": b_id, "a_text": a_text, "b_text": b_text, "score": round(score, 4)}
                )
        else:
            counts["added"] += 1
            items.append({"type": "added", "b_id": segs_b[j][0], "text": segs_b[j][1]})
        _emit_removed(j)

    return {"counts": counts, "items": items}
