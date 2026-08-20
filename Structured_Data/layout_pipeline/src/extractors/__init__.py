"""Per-type extractors for the Dual Extraction Engine.

Each module here turns one kind of cropped region into its proper textual
representation: formulas into LaTeX, and (later) tables into a grid.
"""

from .formula_extractor import FormulaExtractor
from .table_extractor import TableExtractor

__all__ = ["FormulaExtractor", "TableExtractor"]
