"""Document <-> PDF alignment (the whiteboard's ``Alignment : Loc[Doc] -> Set[Loc[PDF]]``).

The Doc side (AST nodes) and the PDF side (layout boxes) both carry OCR text;
:class:`Aligner` aligns the two as character streams and maps a :class:`DocLoc`
(node + substring) to the set of :class:`PdfLoc` boxes that render it.
"""

from .aligner import Aligner
from .types import DocLoc, PdfLoc

__all__ = ["Aligner", "DocLoc", "PdfLoc"]
