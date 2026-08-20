"""Cropped table image -> Markdown or HTML.

Same shape as the formula extractor: a swappable local backend behind a method
that never raises. The default is **PP-Structure's table pipeline**
(``TableRecognitionPipelineV2``: table classification, SLANet/SLANeXt structure
recognition, cell detection and OCR), which runs in ``env_paddle`` with weights
already cached.

Output format is chosen by the table, not by configuration. A grid of plain
cells becomes a Markdown pipe table, which is far more readable downstream. A
table with merged cells cannot be expressed in Markdown at all -- Markdown has
no ``colspan`` -- so those stay as HTML rather than being silently flattened
into a grid that misstates the data. The DenseNet architecture table is exactly
this case: its header spans two columns per model.
"""

from __future__ import annotations

import logging
import os
import re
from html.parser import HTMLParser
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger(__name__)

DEFAULT_DEVICE = "gpu:0"
CELL_TAGS = {"td", "th"}


class _TableParser(HTMLParser):
    """Pull a table's cells out of PP-Structure's HTML.

    Deliberately stdlib: the extractor stays importable (and testable) without
    beautifulsoup, and the markup here is machine-generated and simple.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows: List[List[dict]] = []
        self.header_rows = 0
        self._row: Optional[List[dict]] = None
        self._cell: Optional[dict] = None
        self._in_thead = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "thead":
            self._in_thead = True
        elif tag == "tr":
            self._row = []
        elif tag in CELL_TAGS:
            self._cell = {
                "text": "",
                "colspan": _int(a.get("colspan"), 1),
                "rowspan": _int(a.get("rowspan"), 1),
                "header": tag == "th" or self._in_thead,
            }

    def handle_endtag(self, tag):
        if tag == "thead":
            self._in_thead = False
        elif tag in CELL_TAGS and self._cell is not None:
            self._cell["text"] = " ".join(self._cell["text"].split())
            if self._row is not None:
                self._row.append(self._cell)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
                if all(c["header"] for c in self._row):
                    if len(self.rows) == self.header_rows + 1:
                        self.header_rows += 1
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell["text"] += data


def _int(value, default=1):
    try:
        n = int(str(value).strip())
        return n if n > 0 else default
    except (TypeError, ValueError):
        return default


def has_merged_cells(rows: List[List[dict]]) -> bool:
    return any(c["colspan"] > 1 or c["rowspan"] > 1 for row in rows for c in row)


def body_fill_ratio(rows: List[List[dict]]) -> float:
    """Fraction of non-header cells carrying text.

    The header alone is not evidence of a parsed table -- structure recognition
    frequently gets the column names and nothing else. Measuring the body is
    what distinguishes a real extraction from an empty grid.
    """
    body = [c for row in rows[1:] for c in row] if len(rows) > 1 else \
        [c for row in rows for c in row]
    if not body:
        return 0.0
    return sum(1 for c in body if c["text"].strip()) / len(body)


def clean_html(html: str) -> str:
    """Reduce PP-Structure's output to the bare ``<table>`` element."""
    s = (html or "").strip()
    match = re.search(r"<table\b.*?</table>", s, re.DOTALL | re.IGNORECASE)
    return match.group(0) if match else s


def rows_to_markdown(rows: List[List[dict]], header_rows: int = 0) -> str:
    """Render a span-free grid as a Markdown pipe table.

    The first row becomes the header: Markdown requires one, and a table whose
    detector reported no ``<thead>`` almost always still leads with its column
    names.
    """
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    grid = [[_escape_cell(c["text"]) for c in row] + [""] * (width - len(row))
            for row in rows]

    head, body = (grid[0], grid[1:]) if len(grid) > 1 else (grid[0], [])
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * width) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def _escape_cell(text: str) -> str:
    # A literal pipe would break the row into extra columns.
    return (text or "").replace("|", "\\|").strip()


class PaddleTableBackend:
    """PP-Structure table pipeline. Local, GPU, weights cached."""

    def __init__(self, device: str = DEFAULT_DEVICE,
                 use_layout_detection: bool = False):
        self.device = device
        # The input is already a table crop -- the router's PP-DocLayout stage
        # decided that. Letting the pipeline run layout detection *again* on the
        # crop is redundant and actively harmful: on a tight crop it often fails
        # to find a table region and returns nothing at all. Every borderless
        # table in the StudentRecord corpus (6 of 6) came back empty with this
        # on, and all 6 extract cleanly with it off.
        self.use_layout_detection = use_layout_detection
        self._pipeline = None
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    @property
    def name(self) -> str:
        return "paddle:TableRecognitionPipelineV2"

    def _load(self):
        if self._pipeline is None:
            from paddleocr import TableRecognitionPipelineV2

            self._pipeline = TableRecognitionPipelineV2(
                device=self.device,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
            )
        return self._pipeline

    def __call__(self, image) -> str:
        import numpy as np

        pipeline = self._load()
        arr = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
        res = pipeline.predict(arr,
                               use_layout_detection=self.use_layout_detection)
        r = res[0] if isinstance(res, list) else res
        payload = r.json.get("res", r.json) if hasattr(r, "json") else dict(r)
        tables = payload.get("table_res_list") or []
        if not tables:
            return ""
        # The crop is one table; if the pipeline finds several, the largest
        # markup is the real one and the rest are fragments.
        return max((t.get("pred_html") or "" for t in tables), key=len)


class TableExtractor:
    """Turn a cropped table image into Markdown or HTML.

    ``backend`` is any callable ``image -> html str``, which is what lets the
    tests run without model weights.
    """

    MIN_SIDE_PX = 12
    #: Minimum fraction of *body* cells that must carry text.
    #:
    #: Structure recognition can find a grid and still fail to attach any text
    #: to it. On the DenseNet ImageNet-results table the backend returned a
    #: correct 4x3 grid with one filled cell out of nine, from a crop that is
    #: perfectly legible. Without this guard that empty grid is reported
    #: ``completed`` -- a table that lies about being parsed is worse than one
    #: that admits it failed, because the OCR fallback text is then discarded.
    MIN_CELL_FILL = 0.3

    def __init__(self, backend: Optional[Callable] = None,
                 device: str = DEFAULT_DEVICE,
                 prefer_markdown: bool = True,
                 min_cell_fill: Optional[float] = None):
        self.prefer_markdown = prefer_markdown
        self.min_cell_fill = self.MIN_CELL_FILL if min_cell_fill is None \
            else min_cell_fill
        self.last_error: Optional[str] = None
        self._failed = False
        self._backend = backend if backend is not None \
            else PaddleTableBackend(device=device)

    @property
    def backend(self) -> Any:
        return self._backend

    @property
    def name(self) -> str:
        return getattr(self._backend, "name", type(self._backend).__name__)

    def extract_table(self, crop_img) -> Optional[Dict[str, Any]]:
        """``{"content": str, "format": "markdown"|"html"}``, or ``None``.

        Never raises. Every failure -- missing weights, CUDA error, unparseable
        markup, empty result -- returns ``None`` so the router falls back to the
        OCR text and flags it.
        """
        self.last_error = None
        if crop_img is None:
            self.last_error = "no crop image"
            return None
        size = getattr(crop_img, "size", None)
        if size and min(size) < self.MIN_SIDE_PX:
            self.last_error = f"crop too small: {size}"
            return None
        if self._failed:
            self.last_error = "backend previously failed to load"
            return None

        try:
            html = self._backend(crop_img)
        except ImportError as exc:
            self._failed = True
            self.last_error = f"backend unavailable: {exc}"
            log.warning("table backend unavailable: %s", exc)
            return None
        except Exception as exc:  # noqa: BLE001 - fallback is the contract
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.warning("table extraction failed: %s", self.last_error)
            return None

        if not isinstance(html, str) or not html.strip():
            self.last_error = "backend returned no markup"
            return None
        return self.to_content(html)

    def to_content(self, html: str) -> Optional[Dict[str, Any]]:
        """Choose Markdown or HTML for one table's markup."""
        parser = _TableParser()
        try:
            parser.feed(clean_html(html))
            parser.close()
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"unparseable markup: {type(exc).__name__}: {exc}"
            return None

        rows = parser.rows
        if not rows or not any(c["text"] for row in rows for c in row):
            self.last_error = "table has no cell text"
            return None

        fill = body_fill_ratio(rows)
        if fill < self.min_cell_fill:
            self.last_error = (f"only {fill:.0%} of body cells carry text "
                               f"(min {self.min_cell_fill:.0%}) -- structure "
                               f"found but cell text not associated")
            return None

        merged = has_merged_cells(rows)
        common = {
            "rows": len(rows),
            "columns": max(len(r) for r in rows),
            "merged_cells": merged,
            "cell_fill": round(fill, 3),
        }
        if merged or not self.prefer_markdown:
            return {"content": clean_html(html), "format": "html", **common}
        return {"content": rows_to_markdown(rows, parser.header_rows),
                "format": "markdown", **common}
