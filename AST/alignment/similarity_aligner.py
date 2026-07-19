"""Semantic-similarity node -> box alignment.

The stream aligner (:func:`alignment.align_stream`) matches *characters by
position* with :class:`difflib.SequenceMatcher`: it only works when the two OCR
engines (Mistral on the Doc side, PP-OCRv5 on the PDF side) emit near-identical
text. OCR noise, hyphenation, reflowed whitespace or any paraphrase break it.

This aligner scores by **meaning** instead of by characters. Both the AST text
segments and the PDF box texts are embedded with a sentence-transformer
(``all-MiniLM-L6-v2`` by default), and each ``(node, box)`` pair is scored by the
**cosine similarity** of their embeddings. Every box scoring ``>= threshold``
is a match. Because the score is semantic, a node still matches the box that
renders it even when the characters disagree.

Return shape matches :func:`alignment.align_stream` exactly (``alignment`` /
``reverse`` / ``coverage`` / ``granularity``), so it drops in behind
``/api/align/compute`` with no frontend change. A node's highlight set is its own
direct matches unioned with its whole subtree's.

Best suited to ``paragraph`` granularity: sentence embeddings are meaningful for
phrases/sentences, far less so for a single ``word`` box (where the positional
stream aligner is the better tool). The two are switchable per request.
"""

from __future__ import annotations

import re
import threading

from .naive_aligner import _iter_segments

_WS = re.compile(r"\s+")
_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Floor for the scores returned to the client. The frontend's confidence slider
# re-filters matches live from these raw scores, so we ship every node->box pair
# scoring at least this much (independent of the request ``threshold``) to give
# the slider headroom below the default cut.
_SCORE_FLOOR = 0.2

# Loading a transformer is expensive; keep one per model name for the process.
_models: dict[str, object] = {}
_models_lock = threading.Lock()


def _clean(s: str) -> str:
    """Collapse whitespace; keep words/punctuation for the embedding model."""
    return _WS.sub(" ", (s or "").strip())


def _device() -> str:
    """Prefer CUDA (the RTX 4070 Ti) when a CUDA torch build is present."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _get_model(model_name: str):
    """Lazily load + cache the sentence-transformer (thread-safe)."""
    model = _models.get(model_name)
    if model is not None:
        return model
    with _models_lock:
        model = _models.get(model_name)
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name, device=_device())
            _models[model_name] = model
    return model


def align_similarity(
    ast_root: dict,
    pages: list[dict],
    granularity: str = "paragraph",
    threshold: float = 0.5,
    model_name: str = _DEFAULT_MODEL,
) -> dict:
    """Map AST nodes to PDF boxes by embedding cosine similarity.

    ``pages`` is ``[{"page": int, "boxes": [box_dict, ...]}, ...]`` at the chosen
    granularity; each box needs ``text``. Returns
    ``{"alignment": {node_id: [{"page", "box_index"}]}, "reverse", "coverage",
    "granularity", "threshold"}`` (same shape as :func:`align_stream`).
    ``box_index`` indexes into that page's ``boxes`` list as supplied here.
    """
    # --- Gather the two text sides ---------------------------------------- #
    segments = [(nid, _clean(text)) for nid, text in _iter_segments(ast_root)]
    segments = [(nid, t) for nid, t in segments if t]

    box_refs: list[tuple[int, int]] = []  # (page_no, box_index)
    box_texts: list[str] = []
    for page in pages:
        page_no = page["page"]
        for i, b in enumerate(page.get("boxes", [])):
            t = _clean(b.get("text") or "")
            if t:
                box_refs.append((page_no, i))
                box_texts.append(t)

    direct: dict[str, set[tuple[int, int]]] = {}
    # node_id -> [(page, box_index, score)] for every pair scoring >= _SCORE_FLOOR,
    # shipped to the client so the confidence slider can re-filter without a recompute.
    node_scores: dict[str, list[tuple[int, int, float]]] = {}
    # box -> {node_id: best_score}; highest-scoring node is the box's primary owner.
    box_owners: dict[tuple[int, int], dict[str, float]] = {}

    if segments and box_texts:
        model = _get_model(model_name)
        node_emb = model.encode(
            [t for _, t in segments],
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=64,
        )
        box_emb = model.encode(
            box_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=64,
        )
        # Normalized embeddings -> cosine similarity is the dot product.
        sims = node_emb @ box_emb.T  # (n_nodes, n_boxes)

        for ni, (node_id, _) in enumerate(segments):
            row = sims[ni]
            hits: set[tuple[int, int]] = set()
            scored: list[tuple[int, int, float]] = []
            for bj in range(len(box_refs)):
                score = float(row[bj])
                if score < _SCORE_FLOOR:
                    continue
                page_no, box_index = box_refs[bj]
                scored.append((page_no, box_index, round(score, 4)))
                if score >= threshold:
                    ref = box_refs[bj]
                    hits.add(ref)
                    owners = box_owners.setdefault(ref, {})
                    if score > owners.get(node_id, 0.0):
                        owners[node_id] = score
            direct[node_id] = hits
            if scored:
                node_scores[node_id] = scored

    matched = sum(1 for _, h in direct.items() if h)

    # Reverse map: box -> node ids, highest similarity first.
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
    coverage = matched / len(segments) if segments else 0.0
    scores = {
        nid: [{"page": p, "box_index": b, "score": s} for p, b, s in pairs]
        for nid, pairs in node_scores.items()
    }
    return {
        "alignment": alignment,
        "reverse": reverse,
        "coverage": coverage,
        "granularity": granularity,
        "threshold": threshold,
        "scores": scores,
        "score_floor": _SCORE_FLOOR,
    }
