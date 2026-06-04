"""The two coordinate systems being aligned (from the professor's model).

    Alignment : Loc[Doc] -> Set[Loc[PDF]]

``Loc[Doc]`` — a location in the Document (AST): a Node **and** a substring of
that node's text (the two together form one coordinate). ``Loc[PDF]`` — a
location in the PDF: a bounding box on a page.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DocLoc:
    """A location in the Doc: a node together with a substring of its text.

    A coordinate is always anchored to a specific ``node_id`` AND a ``[start,
    end)`` character span within that node's ``text`` (``end`` exclusive). The
    "whole node" case is just the span covering all of ``node.text`` — use
    :meth:`whole_node`.
    """

    node_id: str
    start: int
    end: int

    @classmethod
    def whole_node(cls, node_id: str, text_len: int) -> "DocLoc":
        return cls(node_id=node_id, start=0, end=text_len)

    def to_dict(self) -> dict:
        return {"node_id": self.node_id, "start": self.start, "end": self.end}


@dataclass(frozen=True)
class PdfLoc:
    """A location in the PDF: one bounding box on a page.

    Frozen + hashable so alignment results can be collected into a ``set``.
    """

    page: int  # 1-indexed
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) in page-pixel coords
    label: Optional[str] = None
    text: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "page": self.page,
            "bbox": list(self.bbox),
            "label": self.label,
            "text": self.text,
        }
