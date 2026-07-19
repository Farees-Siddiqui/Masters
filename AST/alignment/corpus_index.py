"""Corpus-scale analysis over many documents' AST segments.

Builds (and caches) one embedding index across every processed document, then
serves two views on top of it:

* :func:`corpus_map` — a 2D semantic map of the collection (t-SNE projection,
  KMeans themes with auto-labels) so a large archive can be seen at a glance;
* :func:`corpus_trace` — every passage across the corpus matching a concept,
  grouped by document, for cross-document literature review.

Both reuse the same MiniLM embeddings as the similarity aligner, and every point
/ hit carries its ``doc`` + ``node_id`` so the UI can jump to the exact region.
"""

from __future__ import annotations

import re

import numpy as np

from .naive_aligner import _iter_segments
from .similarity_aligner import _clean, _get_model, _DEFAULT_MODEL

# Cache the whole-corpus index keyed by a signature (doc names + mtimes). The
# embedding pass is the expensive part, so we only redo it when the corpus changes.
_cache: dict[str, dict] = {}

_WORD = re.compile(r"[a-z][a-z0-9\-]{2,}")
_STOP = set(
    "the a an and or of to in for on with as by is are was were be been being this that these those "
    "we our it its they their he she his her you your from at into than then but not can may also such "
    "which who whom whose what when where how why all any each more most other some only own same so "
    "using used use based via given between within over under above below during while figure table".split()
)


def _index(docs: list[dict], signature: str) -> dict:
    """Embed every text segment across ``docs`` (cached by ``signature``).

    ``docs`` is ``[{"doc", "title", "ast"}]``. Returns ``{"embs", "metas"}`` where
    ``metas[i]`` is ``{"doc", "title", "node_id", "text"}``.
    """
    hit = _cache.get(signature)
    if hit is not None:
        return hit

    metas: list[dict] = []
    texts: list[str] = []
    for d in docs:
        for nid, raw in _iter_segments(d["ast"]):
            t = _clean(raw)
            if t:
                metas.append({"doc": d["doc"], "title": d["title"], "node_id": nid, "text": t})
                texts.append(t)

    if texts:
        embs = _get_model(_DEFAULT_MODEL).encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=64
        )
    else:
        embs = np.zeros((0, 384), dtype=np.float32)

    _cache.clear()  # only one corpus in play; keep memory bounded
    _cache[signature] = {"embs": embs, "metas": metas}
    return _cache[signature]


def _cluster_labels(metas: list[dict], labels: np.ndarray, k: int) -> list[dict]:
    """Label each cluster by its most distinctive words (cluster TF vs corpus TF)."""
    overall: dict[str, int] = {}
    per: list[dict[str, int]] = [dict() for _ in range(k)]
    for m, c in zip(metas, labels):
        for w in _WORD.findall(m["text"].lower()):
            if w in _STOP:
                continue
            overall[w] = overall.get(w, 0) + 1
            per[c][w] = per[c].get(w, 0) + 1

    out = []
    for c in range(k):
        scored = sorted(
            per[c].items(),
            key=lambda kv: -(kv[1] / (overall.get(kv[0], 1) ** 0.5)),
        )
        terms = [w for w, _ in scored[:3]]
        out.append({"cluster": c, "label": ", ".join(terms) if terms else f"cluster {c}", "size": int((labels == c).sum())})
    return out


def corpus_map(docs: list[dict], signature: str, max_points: int = 1500, n_clusters: int = 8) -> dict:
    """2D semantic map of the corpus: ``{points, clusters, total, shown}``."""
    idx = _index(docs, signature)
    embs, metas = idx["embs"], idx["metas"]
    n = len(metas)
    if n == 0:
        return {"points": [], "clusters": [], "total": 0, "shown": 0}

    if n > max_points:
        sel = np.random.RandomState(0).choice(n, max_points, replace=False)
    else:
        sel = np.arange(n)
    E = embs[sel]
    M = [metas[i] for i in sel]
    m = len(M)

    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE

    k = max(2, min(n_clusters, m))
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(E) if m >= k else np.zeros(m, dtype=int)

    perplexity = max(5, min(30, m // 4)) if m > 10 else max(2, m - 1)
    xy = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=0).fit_transform(E)
    span = np.ptp(xy, axis=0)
    xy = (xy - xy.min(axis=0)) / np.where(span == 0, 1, span)

    points = [
        {
            "doc": M[i]["doc"],
            "title": M[i]["title"],
            "node_id": M[i]["node_id"],
            "snippet": M[i]["text"][:120],
            "x": round(float(xy[i, 0]), 4),
            "y": round(float(xy[i, 1]), 4),
            "cluster": int(labels[i]),
        }
        for i in range(m)
    ]
    return {"points": points, "clusters": _cluster_labels(M, labels, k), "total": n, "shown": m}


def corpus_trace(
    docs: list[dict],
    signature: str,
    query: str,
    min_score: float = 0.35,
    per_doc: int = 8,
    max_total: int = 80,
) -> dict:
    """Every passage matching ``query`` across the corpus, grouped by document."""
    query = (query or "").strip()
    idx = _index(docs, signature)
    embs, metas = idx["embs"], idx["metas"]
    if not query or len(metas) == 0:
        return {"query": query, "docs": [], "n_docs": 0, "n_hits": 0}

    q = _get_model(_DEFAULT_MODEL).encode([_clean(query)], convert_to_numpy=True, normalize_embeddings=True)[0]
    sims = embs @ q
    order = np.argsort(-sims)

    grouped: dict[str, list[dict]] = {}
    total = 0
    for i in order:
        s = float(sims[i])
        if s < min_score or total >= max_total:
            break
        m = metas[i]
        bucket = grouped.setdefault(m["doc"], [])
        if len(bucket) < per_doc:
            bucket.append({"node_id": m["node_id"], "score": round(s, 4), "snippet": m["text"][:180]})
            total += 1

    title_of = {d["doc"]: d["title"] for d in docs}
    out = [
        {"doc": doc, "title": title_of.get(doc, doc), "count": len(hits), "best": hits[0]["score"], "hits": hits}
        for doc, hits in grouped.items()
    ]
    out.sort(key=lambda g: -g["best"])
    return {"query": query, "docs": out, "n_docs": len(out), "n_hits": total}
