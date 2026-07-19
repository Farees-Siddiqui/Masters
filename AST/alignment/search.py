"""Semantic search over AST nodes.

Embeds every text-bearing AST node and the query with the same sentence
transformer the similarity aligner uses, then ranks nodes by cosine similarity.
The caller maps the returned ``node_id``s back to PDF boxes through the existing
alignment, so a search lights up both the AST tree and the original scan.
"""

from __future__ import annotations

from .naive_aligner import _iter_segments
from .similarity_aligner import _clean, _get_model, _DEFAULT_MODEL


def search_nodes(
    ast_root: dict,
    query: str,
    top_k: int = 8,
    min_score: float = 0.2,
    model_name: str = _DEFAULT_MODEL,
) -> list[dict]:
    """Return the ``top_k`` AST nodes most similar to ``query``.

    Each result is ``{"node_id", "score", "snippet"}``, sorted by descending
    cosine similarity and filtered to ``>= min_score``.
    """
    query = (query or "").strip()
    segments = [(nid, _clean(text)) for nid, text in _iter_segments(ast_root)]
    segments = [(nid, t) for nid, t in segments if t]
    if not query or not segments:
        return []

    model = _get_model(model_name)
    seg_emb = model.encode(
        [t for _, t in segments],
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=64,
    )
    q_emb = model.encode([_clean(query)], convert_to_numpy=True, normalize_embeddings=True)[0]
    sims = seg_emb @ q_emb  # normalized -> cosine

    ranked = sorted(
        (
            {"node_id": segments[i][0], "score": round(float(sims[i]), 4), "snippet": segments[i][1][:180]}
            for i in range(len(segments))
        ),
        key=lambda r: -r["score"],
    )
    return [r for r in ranked[:top_k] if r["score"] >= min_score]
