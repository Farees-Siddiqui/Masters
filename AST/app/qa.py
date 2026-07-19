"""Grounded question answering over the document AST.

The whole point of the AST<->PDF alignment is *provenance*: an answer the model
gives should be traceable to the exact region of the original scan. So we feed
the model the AST text with each segment labelled by its ``node_id`` and require
it to cite those ids inline (``[n12]``). The frontend turns each citation into a
click that highlights the source region in the PDF via the existing alignment.
"""

from __future__ import annotations

import json
import re

from mistralai import Mistral

from alignment.naive_aligner import _iter_segments

from .ocr import _load_api_key

_CITE_RE = re.compile(r"\[(n\d+)\]")
_MODEL = "mistral-small-latest"
_MAX_CONTEXT_CHARS = 24000  # keep the prompt well within the model's window

_SYSTEM = (
    "You answer questions about a document using ONLY the numbered context "
    "segments provided. Each segment is prefixed with an id like [n12].\n"
    "Respond with a JSON object of the form:\n"
    '{"answer": "<concise answer with inline [nID] citations after each claim>",\n'
    ' "citations": [{"id": "<nID>", "quote": "<the exact substring, copied '
    'verbatim from that segment, that supports the claim>"}]}\n'
    "Rules: cite ids exactly as given; the quote MUST be copied verbatim (a short "
    "phrase or sentence) from the cited segment so it can be located in the page; "
    "keep quotes under ~120 characters. If the answer is not in the context, set "
    "answer accordingly and citations to []."
)


def _build_context(ast_root: dict) -> str:
    parts: list[str] = []
    total = 0
    for nid, text in _iter_segments(ast_root):
        text = (text or "").strip()
        if not text:
            continue
        line = f"[{nid}] {text}"
        if total + len(line) > _MAX_CONTEXT_CHARS:
            break
        parts.append(line)
        total += len(line)
    return "\n".join(parts)


def answer_question(ast_root: dict, question: str) -> dict:
    """Answer ``question`` from the AST; return ``{answer, citations}``.

    ``citations`` is a list of ``{"id": node_id, "quote": <verbatim supporting
    text>}``, distinct by id in first-seen order and restricted to ids that
    actually exist in the AST. The quote lets the frontend highlight the exact
    source *line* (not the whole paragraph) in the PDF.
    """
    question = (question or "").strip()
    if not question:
        return {"answer": "", "citations": []}

    valid_ids = {nid for nid, _ in _iter_segments(ast_root)}
    context = _build_context(ast_root)

    client = Mistral(api_key=_load_api_key())
    resp = client.chat.complete(
        model=_MODEL,
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    raw = resp.choices[0].message.content or ""

    answer, raw_citations = _parse_response(raw)

    # Distinct, valid citations in first-seen order, carrying their quotes.
    seen: dict[str, str] = {}
    for c in raw_citations:
        nid = c.get("id")
        if nid in valid_ids and nid not in seen:
            seen[nid] = (c.get("quote") or "").strip()
    # Catch any inline [nID] the model cited but omitted from the citations list.
    for nid in _CITE_RE.findall(answer):
        if nid in valid_ids and nid not in seen:
            seen[nid] = ""
    citations = [{"id": nid, "quote": q} for nid, q in seen.items()]
    return {"answer": answer, "citations": citations}


def _parse_response(raw: str) -> tuple[str, list[dict]]:
    """Pull (answer, citations) out of the model's JSON reply, defensively."""
    try:
        data = json.loads(raw)
        answer = (data.get("answer") or "").strip()
        cites = data.get("citations") or []
        if isinstance(answer, str) and isinstance(cites, list):
            return answer, [c for c in cites if isinstance(c, dict)]
    except (json.JSONDecodeError, AttributeError):
        pass
    # Fallback: treat the whole reply as the answer, mine inline [nID] citations.
    return raw, [{"id": nid, "quote": ""} for nid in _CITE_RE.findall(raw)]
