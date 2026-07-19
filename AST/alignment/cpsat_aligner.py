"""Alignment as constraint solving (CP-SAT) rather than independent prediction.

`align_stream` and `align_similarity` both **predict**: they score each
``(node, box)`` pair in isolation and take the argmax. Nothing checks that the
resulting choices are collectively coherent, and the benchmark shows the cost —
on ``reference_content`` the semantic aligner scores F1 0.53, because nothing
stops fifty bibliography boxes from all electing the same node. That is not a
representation failure; it is a *missing constraint*.

This aligner **declares and solves**. A boolean ``x[node, box]`` says "this node
owns this box"; the evidence enters as weights; and what makes an alignment
structurally valid enters as hard constraints:

* **one owner per box** — a box renders text from one node;
* **monotone reading order** — owners are non-decreasing along the page's
  reading order, so the alignment cannot cross itself. This is order as
  *combinatorial structure*, not as position: unlike ``align_stream`` it does
  not assume the two sides are character-aligned, only that they agree on
  sequence. Optional (``monotone=False``) so its contribution is measurable;
* **length coherence** — the text a node is assigned cannot greatly exceed the
  text that node actually has. This is what makes the bibliography collapse
  *inexpressible* rather than merely unlikely.

The objective is ``sum((score - threshold) * x)``. Centring on the threshold
gives abstention for free: a pair scoring below it carries negative weight, so
the solver leaves the box unowned rather than inventing an owner. Nothing forces
coverage.

Return shape matches the other aligners exactly, so it drops into
``/api/align/compute`` and the benchmark unchanged.

Scorers are pluggable (``scorer="embed"|"lexical"``) to keep two hypotheses
separable: whether the fix is a better representation, or the constraints.
"""

from __future__ import annotations

import re

import numpy as np

from .naive_aligner import _iter_segments

_WS = re.compile(r"\s+")

# Pairs scoring below this never become variables — pruning keeps the model
# small. Well under any sane threshold so it cannot silently censor candidates.
_CAND_FLOOR = 0.2

# Keep at most this many candidate nodes per box. Sentence-embedding cosine is
# high between *arbitrary* prose, so the floor alone leaves nearly every pair
# alive and the model balloons; top-K bounds it at K * n_boxes. It is a real
# approximation — a true owner ranked below K is unrecoverable — so K is set
# well past where correct answers plausibly sit, and `n_vars` is reported.
_TOP_K = 12

# Objective coefficients must be integers for CP-SAT.
_SCALE = 10_000


def _clean(s: str) -> str:
    return _WS.sub(" ", (s or "").strip())


def _score_embed(node_texts: list[str], box_texts: list[str], model_name: str):
    """Cosine similarity of sentence embeddings — the same signal `similarity` uses.

    Deliberately identical to :func:`alignment.align_similarity`'s scoring so a
    comparison between the two isolates the *constraints* as the only difference.
    """
    from .similarity_aligner import _get_model

    model = _get_model(model_name)
    ne = model.encode(node_texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=64)
    be = model.encode(box_texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=64)
    return ne @ be.T


def _score_lexical(node_texts: list[str], box_texts: list[str], _model_name: str):
    """Cosine over character n-gram TF-IDF.

    Matching a transcription to the same text rendered by another engine is
    *near-duplicate detection*, which is a lexical problem: the discriminating
    signal between two bibliography entries is surname/year/page digits, exactly
    the surface detail a sentence embedding compresses away. Character n-grams
    keep it and stay robust to OCR noise (a wrong glyph spoils a few grams, not
    the vector).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    X = vec.fit_transform(node_texts + box_texts)  # l2-normalised => dot == cosine
    n = len(node_texts)
    return (X[:n] @ X[n:].T).toarray()


_SCORERS = {"embed": _score_embed, "lexical": _score_lexical}


def align_cpsat(
    ast_root: dict,
    pages: list[dict],
    granularity: str = "paragraph",
    threshold: float = 0.5,
    scorer: str = "embed",
    monotone: bool = True,
    max_len_ratio: float = 2.0,
    top_k: int = _TOP_K,
    time_limit: float = 60.0,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> dict:
    """Solve for the best *coherent* node->box alignment.

    ``max_len_ratio`` bounds how much text a node may be assigned relative to the
    text it has; ``monotone`` toggles the non-crossing constraint. Returns the
    usual ``alignment`` / ``reverse`` / ``coverage`` / ``scores`` dict, plus a
    ``solver`` block reporting status and how many variables the model held —
    the solver's own account of what it did.
    """
    from ortools.sat.python import cp_model

    if scorer not in _SCORERS:
        raise ValueError(f"scorer must be one of {sorted(_SCORERS)}, got {scorer!r}")

    # --- the two sides ---------------------------------------------------- #
    segments = [(nid, _clean(t)) for nid, t in _iter_segments(ast_root)]
    segments = [(nid, t) for nid, t in segments if t]

    box_refs: list[tuple[int, int]] = []  # (page, box_index) — the output key
    box_texts: list[str] = []
    box_order: list[tuple[int, int]] = []  # (page, reading order) — the sequence
    for page in pages:
        pno = page["page"]
        for i, b in enumerate(page.get("boxes", [])):
            t = _clean(b.get("text") or "")
            if not t:
                continue
            box_refs.append((pno, i))
            box_texts.append(t)
            box_order.append((pno, b.get("order", i)))

    empty = {
        "alignment": {},
        "reverse": {},
        "coverage": 0.0,
        "granularity": granularity,
        "threshold": threshold,
        "scores": {},
        "score_floor": _CAND_FLOOR,
        "solver": {"status": "EMPTY", "n_vars": 0, "n_boxes": len(box_texts)},
    }
    if not segments or not box_texts:
        return empty

    sims = _SCORERS[scorer]([t for _, t in segments], box_texts, model_name)

    # --- model ------------------------------------------------------------ #
    model = cp_model.CpModel()
    x: dict[tuple[int, int], object] = {}
    sims = np.asarray(sims)
    k = min(top_k, len(segments))
    for j in range(len(box_texts)):
        col = sims[:, j]
        # argpartition: the K best candidates for this box, cheaply.
        cand = np.argpartition(-col, k - 1)[:k] if k < len(segments) else range(len(segments))
        for i in cand:
            if float(col[i]) >= _CAND_FLOOR:
                x[(int(i), j)] = model.NewBoolVar(f"x_{i}_{j}")

    if not x:
        return empty

    # A box renders one node's text.
    for j in range(len(box_texts)):
        vs = [x[(i, j)] for i in range(len(segments)) if (i, j) in x]
        if vs:
            model.AddAtMostOne(vs)

    # Length coherence: a node cannot be assigned much more text than it has.
    # This is the constraint that makes "50 reference boxes -> 1 node" invalid.
    node_len = [max(len(t), 1) for _, t in segments]
    box_len = [max(len(t), 1) for t in box_texts]
    for i in range(len(segments)):
        vs = [(j, x[(i, j)]) for j in range(len(box_texts)) if (i, j) in x]
        if not vs:
            continue
        cap = int(max_len_ratio * node_len[i])
        model.Add(sum(box_len[j] * v for j, v in vs) <= cap)

    # Non-crossing: owners are non-decreasing along reading order. Encoded with
    # one position var per box (O(m) constraints) rather than forbidding every
    # crossing pair (which would be O(n^2 m^2) clauses and never build).
    if monotone:
        seq = sorted(range(len(box_texts)), key=lambda j: box_order[j])
        pos = [model.NewIntVar(0, len(segments) - 1, f"p_{k}") for k in range(len(seq))]
        for k, j in enumerate(seq):
            for i in range(len(segments)):
                if (i, j) in x:
                    model.Add(pos[k] == i).OnlyEnforceIf(x[(i, j)])
        for k in range(len(seq) - 1):
            model.Add(pos[k] <= pos[k + 1])
        # An unowned box's position is unconstrained by any x, so the solver is
        # free to slot it between its neighbours; it stays transparent here.

    # Objective centred on the threshold: sub-threshold pairs carry negative
    # weight, so leaving a box unowned beats inventing an owner for it.
    model.Maximize(
        sum(
            int(round((float(sims[i][j]) - threshold) * _SCALE)) * v
            for (i, j), v in x.items()
        )
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)
    status_name = solver.StatusName(status)

    direct: dict[str, set[tuple[int, int]]] = {}
    box_owners: dict[tuple[int, int], list[tuple[str, float]]] = {}
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for (i, j), v in x.items():
            if solver.Value(v):
                nid = segments[i][0]
                ref = box_refs[j]
                direct.setdefault(nid, set()).add(ref)
                box_owners.setdefault(ref, []).append((nid, round(float(sims[i][j]), 4)))

    # --- same output shape as the other aligners --------------------------- #
    reverse = {
        f"{p}:{idx}": [nid for nid, _ in sorted(owners, key=lambda kv: -kv[1])]
        for (p, idx), owners in box_owners.items()
    }

    union: dict[str, set[tuple[int, int]]] = {}

    def _build(node: dict) -> set[tuple[int, int]]:
        acc = set(direct.get(node["id"], ()))
        for child in node.get("children") or []:
            acc |= _build(child)
        union[node["id"]] = acc
        return acc

    _build(ast_root)

    alignment = {
        nid: [{"page": p, "box_index": b} for p, b in sorted(hits)]
        for nid, hits in union.items()
        if hits
    }
    matched = sum(1 for h in direct.values() if h)
    coverage = matched / len(segments) if segments else 0.0

    # Ship every surviving candidate's score so the UI slider still works.
    scores: dict[str, list[dict]] = {}
    for (i, j) in x:
        s = float(sims[i][j])
        scores.setdefault(segments[i][0], []).append(
            {"page": box_refs[j][0], "box_index": box_refs[j][1], "score": round(s, 4)}
        )

    return {
        "alignment": alignment,
        "reverse": reverse,
        "coverage": coverage,
        "granularity": granularity,
        "threshold": threshold,
        "scores": scores,
        "score_floor": _CAND_FLOOR,
        "solver": {
            "status": status_name,
            "n_vars": len(x),
            "n_boxes": len(box_texts),
            "n_nodes": len(segments),
            "objective": solver.ObjectiveValue() if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
            "wall_time": round(solver.WallTime(), 2),
            "monotone": monotone,
            "scorer": scorer,
        },
    }
