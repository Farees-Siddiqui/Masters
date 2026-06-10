"""Text-based alignment: ``Loc[Doc] -> Set[Loc[PDF]]``.

Both sides carry OCR text (AST nodes from Mistral OCR; PDF boxes from PP-OCR),
so we align the two as character streams:

1. Flatten the AST into a doc char stream, recording for each char which
   ``(node_id, local_offset)`` it came from.
2. Flatten the PDF boxes into a char stream, recording which box each char
   came from (word granularity preferred for precision).
3. Sequence-align the two streams with :class:`difflib.SequenceMatcher`
   (lowercased) — the two OCR engines won't agree byte-for-byte, so this is
   approximate matching, not equality.
4. :meth:`Aligner.align` resolves a ``DocLoc`` (node + ``[start, end)``) to its
   doc-stream range, maps it through the alignment to PDF chars, and collects
   the boxes covering them into a ``Set[PdfLoc]``.

This is v1 (text correspondence). The PDF char stream now follows the formal
``ReadingOrder`` (XY-Cut++ ``order`` stamped on each box by ``layout.detector``),
falling back to top-to-bottom, left-to-right for older caches without it.
"""

from __future__ import annotations

import difflib
from typing import Iterator, Optional

from .types import DocLoc, PdfLoc

# Preference order when a layout page exposes multiple granularities.
_GRANULARITY_PREFERENCE = ("word", "line", "paragraph")


def _iter_text_nodes(node: dict) -> Iterator[tuple[str, str]]:
    """Yield ``(node_id, text)`` for every node with text, in document order."""
    text = node.get("text")
    if text:
        yield node["id"], text
    for child in node.get("children", []) or []:
        yield from _iter_text_nodes(child)


def _pick_granularity(page: dict, preferred: Optional[str]) -> Optional[str]:
    boxes = page.get("boxes", {})
    if preferred and boxes.get(preferred):
        return preferred
    for g in _GRANULARITY_PREFERENCE:
        if boxes.get(g):
            return g
    return None


class Aligner:
    """Builds the doc/PDF char streams once, then answers alignment queries."""

    def __init__(self, ast_root: dict, layout_doc: dict, granularity: Optional[str] = None):
        # --- Doc stream ------------------------------------------------------
        # doc_chars[i] is one lowercased char; doc_prov[i] = (node_id, local_off)
        self._doc_chars: list[str] = []
        self._doc_prov: list[tuple[str, int]] = []
        self._node_span: dict[str, tuple[int, int]] = {}  # node_id -> (stream_start, text_len)

        for node_id, text in _iter_text_nodes(ast_root):
            start = len(self._doc_chars)
            for local_off, ch in enumerate(text):
                self._doc_chars.append(ch.lower())
                self._doc_prov.append((node_id, local_off))
            self._node_span[node_id] = (start, len(text))
            # separator so adjacent nodes don't fuse into one match run
            self._doc_chars.append(" ")
            self._doc_prov.append((node_id, len(text)))

        # --- PDF stream ------------------------------------------------------
        # pdf_prov[j] = index into self._pdf_locs
        self._pdf_chars: list[str] = []
        self._pdf_prov: list[int] = []
        self._pdf_locs: list[PdfLoc] = []

        for page in layout_doc.get("pages", []):
            gran = _pick_granularity(page, granularity)
            if gran is None:
                continue
            page_no = page.get("page")
            # Build the PDF char stream in true reading order (XY-Cut++ `order`),
            # falling back to top->bottom, left->right for caches written before
            # reading order existed. This fixes the two-column fragility where
            # our naive box order diverged from Mistral's reading order.
            boxes = sorted(
                (b for b in page["boxes"][gran] if b.get("text")),
                key=lambda b: (
                    b["order"] if b.get("order") is not None else float("inf"),
                    b["bbox"][1],
                    b["bbox"][0],
                ),
            )
            for box in boxes:
                loc_idx = len(self._pdf_locs)
                self._pdf_locs.append(
                    PdfLoc(
                        page=page_no,
                        bbox=tuple(box["bbox"]),
                        label=box.get("label"),
                        text=box.get("text"),
                    )
                )
                for ch in box["text"]:
                    self._pdf_chars.append(ch.lower())
                    self._pdf_prov.append(loc_idx)
                self._pdf_chars.append(" ")
                self._pdf_prov.append(loc_idx)

        # --- Alignment -------------------------------------------------------
        # doc_to_pdf[i] = aligned pdf-stream index, for matched doc chars only.
        self._doc_to_pdf: dict[int, int] = {}
        matcher = difflib.SequenceMatcher(
            None, "".join(self._doc_chars), "".join(self._pdf_chars), autojunk=False
        )
        for i, j, n in matcher.get_matching_blocks():
            for k in range(n):
                self._doc_to_pdf[i + k] = j + k

    # ------------------------------------------------------------------ #
    def align(self, loc: DocLoc) -> set[PdfLoc]:
        """Map a doc location to the set of PDF boxes that render it."""
        span = self._node_span.get(loc.node_id)
        if span is None:
            return set()
        node_start, text_len = span
        a = node_start + max(0, loc.start)
        b = node_start + min(text_len, loc.end)

        box_ids: set[int] = set()
        for di in range(a, b):
            pj = self._doc_to_pdf.get(di)
            if pj is not None:
                box_ids.add(self._pdf_prov[pj])
        return {self._pdf_locs[i] for i in box_ids}

    def coverage(self) -> float:
        """Fraction of doc chars that aligned to a PDF char (a quality signal)."""
        if not self._doc_chars:
            return 0.0
        return len(self._doc_to_pdf) / len(self._doc_chars)
