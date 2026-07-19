"""Document <-> PDF alignment (the whiteboard's ``Alignment : Loc[Doc] -> Set[Loc[PDF]]``).

The Doc side (AST nodes) and the PDF side (layout boxes) both carry OCR text;
:class:`Aligner` aligns the two as character streams and maps a :class:`DocLoc`
(node + substring) to the set of :class:`PdfLoc` boxes that render it.
"""

from .aligner import Aligner
from .cpsat_aligner import align_cpsat
from .naive_aligner import align_naive
from .similarity_aligner import align_similarity
from .stream_aligner import align_stream
from .types import DocLoc, PdfLoc

__all__ = [
    "Aligner",
    "DocLoc",
    "PdfLoc",
    "align_cpsat",
    "align_naive",
    "align_similarity",
    "align_stream",
]
