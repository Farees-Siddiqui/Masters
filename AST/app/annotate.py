"""Annotation tool backend: human gold for the alignment benchmark.

Three queues per document, each a "machine proposes, human confirms/corrects"
flow (see the benchmark critique that motivated this):

- **excluded** — AST nodes the text-layer oracle could not place (LaTeX
  formulas, image refs, too-short markers). The annotator points at the box(es)
  that contain the node, or marks it absent. This is gold that cannot be
  derived automatically at all.
- **nulls** — layout boxes whose tokens the oracle marked as owned-by-nobody.
  The annotator confirms "furniture" (page number, figure innards) or assigns
  the owning node. Decontaminates the false-attribution metric.
- **audit** — a check on the oracle itself: placed nodes, fuzzy placements
  first, accept/reject at a glance. Yields a measured oracle error rate.

Verdicts land in ``layout_output/<doc>/annotations.json`` keyed by a stable
item id, so scoring can later overlay human labels on top of oracle gold
(human wins where both exist).

This module intentionally re-implements the small placement loop instead of
importing :func:`benchmark.oracle.place_nodes`: it needs per-placement ratio /
exactness, and the oracle module is being extended separately. Only stable
primitives (tokenization, normalization, ``_locate``) are shared.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from alignment.naive_aligner import _iter_segments
from benchmark.oracle import (
    _MIN_PLACE_CHARS,
    _locate,
    _norm,
    boxes_to_tokens,
    build_stream,
    extract_tokens,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
LAYOUT_OUTPUT_DIR = REPO_ROOT / "layout_output"
CORPUS_DIR = REPO_ROOT / "corpus"

router = APIRouter()

# doc -> derived working set. Derivation reads the PDF text layer (~seconds);
# the cache makes queue navigation instant. Keyed by doc name only: corpus
# PDFs and cached layouts are immutable during an annotation session.
_CACHE: dict[str, dict] = {}

_WORD = re.compile(r"[a-z0-9]{3,}")


# --------------------------------------------------------------------------- #
# Derivation
# --------------------------------------------------------------------------- #


def _doc_paths(doc: str) -> tuple[Path, Path]:
    """(doc_dir, pdf_path) for an annotatable doc, or 404."""
    doc_dir = LAYOUT_OUTPUT_DIR / Path(doc).name  # sanitize traversal
    pdf = CORPUS_DIR / f"{doc_dir.name}.pdf"
    if not (doc_dir / "ast.json").is_file() or not (doc_dir / "manifest.json").is_file():
        raise HTTPException(status_code=404, detail=f"Unknown document: {doc}")
    if not pdf.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"No corpus PDF for {doc!r} — annotation needs the born-digital PDF.",
        )
    return doc_dir, pdf


def _load_pages(doc_dir: Path, granularity: str = "paragraph") -> list[dict]:
    manifest = json.loads((doc_dir / "manifest.json").read_text(encoding="utf-8"))
    stem = doc_dir.name
    pages: list[dict] = []
    for entry in manifest["pages"]:
        path = doc_dir / entry["dir"] / f"{granularity}.json"
        if not path.is_file():
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        pages.append(
            {
                "page": entry["page"],
                "width": d["width"],
                "height": d["height"],
                "image_url": f"/layout-data/{stem}/{entry['image']}",
                "boxes": [
                    {
                        "bbox": b["bbox"],
                        "label": b.get("label") or "unlabeled",
                        "text": b.get("text") or "",
                    }
                    for b in d.get("boxes", [])
                ],
            }
        )
    return pages


def _word_set(s: str) -> set[str]:
    return set(_WORD.findall(_norm(s)))


def _suggest_boxes(node_text: str, pages: list[dict], top: int = 3) -> list[dict]:
    """Rank boxes by word overlap with a node's text — candidates for excluded nodes.

    Deliberately crude (bag of words, len>=3): it only has to put the right box
    in the top three for a human to click, not decide anything.
    """
    words = _word_set(node_text)
    if not words:
        return []
    scored: list[tuple[float, str]] = []
    for page in pages:
        for i, b in enumerate(page["boxes"]):
            bw = _word_set(b["text"])
            if not bw:
                continue
            inter = len(words & bw)
            if inter == 0:
                continue
            scored.append((inter / len(words | bw), f"{page['page']}:{i}"))
    scored.sort(key=lambda t: -t[0])
    return [{"box": key, "score": round(sc, 3)} for sc, key in scored[:top]]


def _derive(doc: str) -> dict:
    """Build the full annotation working set for one document (cached)."""
    if doc in _CACHE:
        return _CACHE[doc]

    doc_dir, pdf = _doc_paths(doc)
    ast = json.loads((doc_dir / "ast.json").read_text(encoding="utf-8"))
    pages = _load_pages(doc_dir)

    tokens = extract_tokens(pdf, dpi=200)
    stream, tokens = build_stream(tokens)

    # --- placement loop (mirrors oracle.place_nodes, but keeps ratio/exact) --
    placements: dict[str, dict] = {}  # node_id -> {start,end,ratio,exact}
    excluded: list[dict] = []  # queue 1 items
    nodes_brief: list[dict] = []  # for the node picker in the UI
    cursor = 0
    for node_id, text in _iter_segments(ast):
        needle = _norm(text)
        if not needle:
            continue
        nodes_brief.append({"node_id": node_id, "text": text[:200]})
        if len(needle) < _MIN_PLACE_CHARS:
            excluded.append({"node_id": node_id, "kind": "too_short", "text": text})
            continue
        found = _locate(stream, needle, cursor)
        if found is None:
            excluded.append({"node_id": node_id, "kind": "unlocatable", "text": text})
            continue
        # oracle._locate returns (start, end, ratio) historically and
        # (start, end, ratio, exact) since placement provenance landed.
        start, end, ratio, *rest = found
        placements[node_id] = {
            "start": start,
            "end": end,
            "ratio": round(ratio, 3),
            "exact": bool(rest[0]) if rest else ratio >= 0.999,
        }
        cursor = end

    # --- token gold (first-come, like the oracle) ---------------------------
    owner: dict[str, str | None] = {t.key: None for t in tokens}
    for node_id, p in placements.items():
        for t in tokens:
            if t.stream_start < p["end"] and t.stream_end > p["start"] and owner[t.key] is None:
                owner[t.key] = node_id

    box_tokens = boxes_to_tokens(pages, tokens)
    tok_by_key = {t.key: t for t in tokens}

    # --- queue 1: excluded nodes, with candidate boxes ----------------------
    q_excluded = [
        {
            "id": f"excluded:{it['node_id']}",
            "node_id": it["node_id"],
            "kind": it["kind"],
            "text": it["text"],
            "suggestions": _suggest_boxes(it["text"], pages),
        }
        for it in excluded
    ]

    # --- queue 2: boxes holding gold-NULL tokens ----------------------------
    q_nulls: list[dict] = []
    for box_key, tkeys in sorted(box_tokens.items(), key=_box_sort_key):
        null_keys = [k for k in tkeys if owner[k] is None]
        if not null_keys:
            continue
        page_no, box_idx = box_key.split(":")
        page = next(p for p in pages if p["page"] == int(page_no))
        box = page["boxes"][int(box_idx)]
        null_text = " ".join(tok_by_key[k].text for k in null_keys)
        q_nulls.append(
            {
                "id": f"null:{box_key}",
                "box": box_key,
                "page": int(page_no),
                "label": box["label"],
                "n_null": len(null_keys),
                "n_total": len(tkeys),
                "token_keys": null_keys,
                "text": null_text[:300],
                "suggested_node": _suggest_node(null_text, nodes_brief, placements),
            }
        )

    # --- queue 3: oracle audit — every placed node, fuzzy first -------------
    span_tokens: dict[str, list[str]] = {nid: [] for nid in placements}
    for t in tokens:
        nid = owner[t.key]
        if nid is not None:
            span_tokens[nid].append(t.key)
    q_audit = [
        {
            "id": f"audit:{nid}",
            "node_id": nid,
            "text": _node_text(nodes_brief, nid),
            "ratio": p["ratio"],
            "exact": p["exact"],
            "token_keys": span_tokens.get(nid, []),
        }
        for nid, p in placements.items()
    ]
    q_audit.sort(key=lambda a: (a["exact"], a["ratio"]))

    data = {
        "doc": doc_dir.name,
        "pages": pages,
        # compact token map for client-side highlighting: key -> [page, bbox]
        "tokens": {
            t.key: [t.page, [round(v, 1) for v in t.bbox]] for t in tokens
        },
        "nodes": nodes_brief,
        "queues": {"excluded": q_excluded, "nulls": q_nulls, "audit": q_audit},
    }
    _CACHE[doc] = data
    return data


def _box_sort_key(item: tuple[str, list]) -> tuple[int, int]:
    page, idx = item[0].split(":")
    return int(page), int(idx)


def _node_text(nodes_brief: list[dict], node_id: str) -> str:
    for n in nodes_brief:
        if n["node_id"] == node_id:
            return n["text"]
    return ""


def _suggest_node(
    text: str, nodes_brief: list[dict], placements: dict[str, dict]
) -> dict | None:
    """Best unplaced-node match for a run of NULL tokens.

    Only *unplaced* nodes are candidates: if a placed node owned this text the
    oracle would already have claimed it, so suggesting placed nodes would just
    re-argue with the oracle — the audit queue exists for that.
    """
    words = _word_set(text)
    if not words:
        return None
    best: tuple[float, str] | None = None
    for n in nodes_brief:
        if n["node_id"] in placements:
            continue
        nw = _word_set(n["text"])
        if not nw:
            continue
        score = len(words & nw) / len(words | nw)
        if score > 0.2 and (best is None or score > best[0]):
            best = (score, n["node_id"])
    if best is None:
        return None
    return {"node_id": best[1], "score": round(best[0], 3)}


# --------------------------------------------------------------------------- #
# Annotation storage
# --------------------------------------------------------------------------- #


def _ann_path(doc_dir: Path) -> Path:
    return doc_dir / "annotations.json"


def _load_annotations(doc_dir: Path) -> dict:
    path = _ann_path(doc_dir)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"version": 1, "items": {}}


class SaveRequest(BaseModel):
    doc: str
    id: str  # queue item id, e.g. "excluded:n42"
    verdict: str  # queue-specific, see VERDICTS; "clear" removes the record
    boxes: list[str] | None = None  # excluded queue: assigned box keys
    node_id: str | None = None  # nulls queue: assigned owner
    note: str | None = None


VERDICTS = {
    "excluded": {"boxes", "absent", "skip"},
    "null": {"furniture", "node", "skip"},
    "audit": {"accept", "reject", "skip"},
}


@router.post("/api/annotate/save")
def annotate_save(req: SaveRequest) -> dict:
    doc_dir, _ = _doc_paths(req.doc)
    kind = req.id.split(":", 1)[0]
    allowed = VERDICTS.get(kind)
    if allowed is None:
        raise HTTPException(status_code=400, detail=f"Bad item id: {req.id}")
    if req.verdict != "clear" and req.verdict not in allowed:
        raise HTTPException(
            status_code=400, detail=f"verdict for {kind!r} must be one of {sorted(allowed)}"
        )
    if req.verdict == "boxes" and not req.boxes:
        raise HTTPException(status_code=400, detail="verdict 'boxes' needs boxes")
    if req.verdict == "node" and not req.node_id:
        raise HTTPException(status_code=400, detail="verdict 'node' needs node_id")

    ann = _load_annotations(doc_dir)
    if req.verdict == "clear":
        ann["items"].pop(req.id, None)
    else:
        record: dict = {"verdict": req.verdict, "ts": int(time.time())}
        if req.boxes:
            record["boxes"] = req.boxes
        if req.node_id:
            record["node_id"] = req.node_id
        if req.note:
            record["note"] = req.note
        ann["items"][req.id] = record
    _ann_path(doc_dir).write_text(json.dumps(ann, indent=1), encoding="utf-8")
    return {"ok": True, "n_items": len(ann["items"])}


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@router.get("/api/annotate/docs")
def annotate_docs() -> dict:
    """Annotatable docs (layout cache + corpus PDF), with progress if derived."""
    docs = []
    for pdf in sorted(CORPUS_DIR.glob("*.pdf")):
        name = pdf.stem
        doc_dir = LAYOUT_OUTPUT_DIR / name
        if not (doc_dir / "ast.json").is_file():
            continue
        entry: dict = {"doc": name}
        ann = _load_annotations(doc_dir)
        entry["n_annotated"] = len(ann["items"])
        if name in _CACHE:
            q = _CACHE[name]["queues"]
            entry["n_items"] = sum(len(v) for v in q.values())
        docs.append(entry)
    return {"docs": docs}


@router.get("/api/annotate/doc")
def annotate_doc(doc: str) -> dict:
    """Full working set for one document: pages, queues, existing annotations."""
    data = _derive(doc)
    doc_dir, _ = _doc_paths(doc)
    return {**data, "annotations": _load_annotations(doc_dir)["items"]}
