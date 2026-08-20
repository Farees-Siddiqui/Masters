"""Dual Extraction Engine: semantic blocks, routing, and per-type extraction.

XY-Cut++ answers *what order* the blocks are read in. This module answers *how
each one should be parsed*: prose goes straight through as text, while tables,
formulas and figures each need their own extractor.

    ordered blocks (pipeline.OrderedBlock)  + the page image
        -> DualExtractionRouter.route()
        -> [SemanticBlock, ...]  in the same order

Status by type:

* ``TEXT`` / ``TITLE`` / ``UNKNOWN`` -- extracted (lines joined).
* ``VISION`` -- completed: the region is cropped, published under ``figures/``
  and rendered as a Markdown image tag.
* ``TABLE`` / ``FORMULA`` -- pending, with the crop written to ``crops/`` and
  ``crop_path`` populated so a later step can feed it to a structure or LaTeX
  model.

The block-type mapping here is deliberately *not* the category mapping in
``xycut_plus``. That one exists to decide what to mask during reading-order
recovery, so it lumps ``table``, ``display_formula`` and ``image`` together as
"vision" -- they are all position-flexible. Extraction cares about the
difference: a table needs structure recognition, a formula needs LaTeX, a figure
needs a crop. Same labels, different question.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Optional, Sequence


class BlockType(str, Enum):
    """What kind of extractor a block needs.

    ``str`` mixin so the value serialises straight to JSON.
    """

    TITLE = "TITLE"
    TEXT = "TEXT"
    TABLE = "TABLE"
    FORMULA = "FORMULA"
    VISION = "VISION"
    UNKNOWN = "UNKNOWN"


# PP-DocLayoutV3 label -> BlockType. Labels observed on the arxiv_papers corpus
# are marked; the rest are documented vocabulary kept so a new layout model does
# not silently fall through to UNKNOWN.
DEFAULT_LABEL_MAP = {
    # -- headings (become "# " / "## " downstream) ------------------------- #
    "doc_title": BlockType.TITLE,          # observed
    "paragraph_title": BlockType.TITLE,    # observed
    "title": BlockType.TITLE,
    "sub_title": BlockType.TITLE,
    "chapter_title": BlockType.TITLE,

    # -- prose ------------------------------------------------------------- #
    "text": BlockType.TEXT,                # observed
    "abstract": BlockType.TEXT,            # observed
    "content": BlockType.TEXT,
    "reference": BlockType.TEXT,
    "reference_content": BlockType.TEXT,
    "algorithm": BlockType.TEXT,
    # Captions are prose, not headings: rendering "Figure 3: ..." as a heading
    # would be wrong even though the detector calls it a *_title.
    "figure_title": BlockType.TEXT,        # observed
    "table_title": BlockType.TEXT,
    "chart_title": BlockType.TEXT,
    "image_caption": BlockType.TEXT,
    "table_caption": BlockType.TEXT,
    # Page furniture. Still text; kept routed to text so nothing is dropped,
    # and the original label survives in metadata for filtering.
    "aside_text": BlockType.TEXT,          # observed
    "header": BlockType.TEXT,              # observed
    "footer": BlockType.TEXT,              # observed
    "footnote": BlockType.TEXT,            # observed
    "page_footnote": BlockType.TEXT,
    "number": BlockType.TEXT,              # observed
    "page_number": BlockType.TEXT,
    "formula_number": BlockType.TEXT,

    # -- specialised extractors -------------------------------------------- #
    "table": BlockType.TABLE,
    "formula": BlockType.FORMULA,
    "display_formula": BlockType.FORMULA,  # observed
    "inline_formula": BlockType.FORMULA,
    "image": BlockType.VISION,             # observed
    "figure": BlockType.VISION,
    "chart": BlockType.VISION,             # observed
    "seal": BlockType.VISION,
    "header_image": BlockType.VISION,
    "footer_image": BlockType.VISION,
}

# Labels that are page furniture rather than document body. Not a separate
# BlockType (the spec fixes the six), but worth flagging in metadata.
MARGINAL_LABELS = frozenset(
    {"aside_text", "header", "footer", "footnote", "page_footnote", "number",
     "page_number"}
)


#: Block types that get their pixels sliced out for a downstream model.
CROPPED_TYPES = frozenset({BlockType.TABLE, BlockType.FORMULA, BlockType.VISION})


@dataclass
class RouteContext:
    """Everything a handler needs beyond the block itself.

    Passed as an optional second argument so the step-1 handler contract
    (``handler(block)``) still holds for callers that do not crop.
    """

    block_id: int = 0
    page_number: int = 0
    block_type: BlockType = BlockType.UNKNOWN
    page_image: Any = None
    output_dir: Optional[str] = None
    crop_path: Optional[str] = None
    #: The cropped region itself, kept so a handler can run a model on it
    #: without reading back the PNG it just wrote.
    crop_image: Any = None
    crop_error: Optional[str] = None


@dataclass
class SemanticBlock:
    """One reading-ordered block plus whatever the router made of it."""

    block_id: int
    block_type: BlockType
    bbox: list                                  # [x_min, y_min, x_max, y_max]
    raw_lines: list = field(default_factory=list)
    parsed_content: str = ""
    metadata: dict = field(default_factory=dict)
    #: Where the block's pixels were written, for TABLE / FORMULA / VISION.
    crop_path: Optional[str] = None

    @property
    def status(self) -> Optional[str]:
        """``extracted`` | ``completed`` | ``fallback`` | ``pending``.

        Lives in ``metadata`` so it round-trips through JSON with everything
        else; exposed here because it is the field callers actually branch on.
        """
        return self.metadata.get("status")

    @classmethod
    def from_dict(cls, payload: dict) -> "SemanticBlock":
        """Rebuild from :meth:`to_dict`, so a saved ``*.blocks.json`` can be
        re-processed without re-running inference."""
        raw = payload.get("block_type", BlockType.UNKNOWN)
        try:
            btype = raw if isinstance(raw, BlockType) else BlockType(str(raw))
        except ValueError:
            btype = BlockType.UNKNOWN
        return cls(
            block_id=payload.get("block_id", 0),
            block_type=btype,
            bbox=list(payload.get("bbox") or []),
            raw_lines=list(payload.get("raw_lines") or []),
            parsed_content=payload.get("parsed_content", "") or "",
            metadata=dict(payload.get("metadata") or {}),
            crop_path=payload.get("crop_path"),
        )

    def to_dict(self) -> dict:
        """JSON-ready. ``raw_lines`` uses each line's own ``to_dict`` when it has
        one, so both real ``TextLine`` objects and plain dicts work."""
        return {
            "block_id": self.block_id,
            "block_type": self.block_type.value,
            "bbox": [round(float(v), 1) for v in self.bbox],
            "parsed_content": self.parsed_content,
            "crop_path": self.crop_path,
            "metadata": self.metadata,
            "raw_lines": [
                l.to_dict() if hasattr(l, "to_dict")
                else (asdict(l) if hasattr(l, "__dataclass_fields__") else l)
                for l in self.raw_lines
            ],
        }


# A handler takes the incoming block and returns ``{"content": str,
# "metadata": dict}``. Step 2 replaces the stub bodies without touching the
# router or this contract.
HandlerResult = dict


def _line_text(line: Any) -> str:
    if hasattr(line, "text"):
        return line.text or ""
    if isinstance(line, dict):
        return line.get("text") or ""
    return str(line)


def _line_score(line: Any) -> Optional[float]:
    if hasattr(line, "score"):
        return line.score
    if isinstance(line, dict):
        return line.get("score")
    return None


class DualExtractionRouter:
    """Dispatch reading-ordered blocks to a per-type extraction handler.

    Reading order is the contract: ``route`` returns one :class:`SemanticBlock`
    per input block, in exactly the sequence it received them. It does not
    re-sort -- the input is already ordered by XY-Cut++, and silently reordering
    here would hide an upstream bug.

    ``block_id`` comes from a counter on the instance, so routing several pages
    through one router yields globally unique ids.
    """

    def __init__(self, label_map: Optional[dict] = None, start_id: int = 0,
                 crop_engine: Any = None, output_dir: Optional[str] = None,
                 crop_padding: int = 5, formula_extractor: Any = None,
                 formula_model: Optional[str] = None, device: str = "gpu:0",
                 table_extractor: Any = None):
        self.label_map = dict(DEFAULT_LABEL_MAP if label_map is None else label_map)
        self._next_id = start_id
        self.output_dir = output_dir
        self.crop_padding = crop_padding
        if crop_engine is None:
            from .crop_engine import CropEngine

            crop_engine = CropEngine(padding=crop_padding)
        self.crop_engine = crop_engine

        # ``False`` disables formula extraction outright; ``None`` builds the
        # default lazily, so a document with no formulas never pays to load a
        # model, and unit tests that inject a stub never touch paddle.
        self._formula_extractor = formula_extractor
        self._table_extractor = table_extractor
        self.formula_model = formula_model
        self.device = device

    @property
    def table_extractor(self):
        if self._table_extractor is False:
            return None
        if self._table_extractor is None:
            from .extractors.table_extractor import TableExtractor

            self._table_extractor = TableExtractor(device=self.device)
        return self._table_extractor

    @property
    def formula_extractor(self):
        if self._formula_extractor is False:
            return None
        if self._formula_extractor is None:
            from .extractors.formula_extractor import (DEFAULT_PADDLE_MODEL,
                                                       FormulaExtractor)

            self._formula_extractor = FormulaExtractor(
                model_name=self.formula_model or DEFAULT_PADDLE_MODEL,
                device=self.device)
        return self._formula_extractor

    # -- crop paths --------------------------------------------------------- #
    @staticmethod
    def crop_filename(page_number: int, block_id: int) -> str:
        return f"page_{page_number}_block_{block_id}.png"

    @staticmethod
    def figure_filename(page_number: int, block_id: int) -> str:
        return f"fig_p{page_number}_{block_id}.png"

    @staticmethod
    def figure_rel_path(page_number: int, block_id: int) -> str:
        """Path written into the Markdown, relative to ``output_dir``."""
        return f"figures/{DualExtractionRouter.figure_filename(page_number, block_id)}"

    # -- classification ---------------------------------------------------- #
    def classify(self, label: Optional[str]) -> BlockType:
        """Detector label -> :class:`BlockType`.

        Unknown, empty and ``None`` labels become ``UNKNOWN``, which the
        dispatch table sends to the text handler: an unrecognised block is far
        more likely to be prose than a table, and treating it as text keeps its
        content in the output.
        """
        if not label:
            return BlockType.UNKNOWN
        return self.label_map.get(str(label).strip().lower(), BlockType.UNKNOWN)

    @property
    def dispatch(self) -> dict:
        """BlockType -> handler. TITLE and UNKNOWN share the text handler;
        the distinction survives in ``block_type`` for downstream rendering."""
        return {
            BlockType.TITLE: self._handle_text,
            BlockType.TEXT: self._handle_text,
            BlockType.UNKNOWN: self._handle_text,
            BlockType.TABLE: self._handle_table,
            BlockType.FORMULA: self._handle_formula,
            BlockType.VISION: self._handle_vision,
        }

    def handler_for(self, block_type: BlockType) -> Callable:
        return self.dispatch.get(block_type, self._handle_text)

    # -- handlers ---------------------------------------------------------- #
    def _handle_text(self, block: Any, ctx: Optional[RouteContext] = None) -> HandlerResult:
        """Prose: concatenate the recognised lines in order.

        Real work, not a stub -- the lines were already ordered within the block
        by the pipeline, so joining them is the whole job.
        """
        lines = _get(block, "lines", []) or []
        text = "\n".join(t for t in (_line_text(l) for l in lines) if t)
        return {"content": text, "metadata": {"status": "extracted"}}

    def _handle_table(self, block: Any, ctx: Optional[RouteContext] = None) -> HandlerResult:
        """Tables: run structure recognition over the crop.

        Emits a Markdown pipe table when the grid has no merged cells, HTML when
        it does -- Markdown cannot express ``colspan``, and flattening a spanned
        header into a plain grid would misstate the data. Falls back to the OCR
        text with a visible warning on any failure.
        """
        ctx = ctx or RouteContext()
        lines = _get(block, "lines", []) or []
        ocr_text = "\n".join(t for t in (_line_text(l) for l in lines) if t)

        image = self._crop_for(ctx)
        result, reason = None, None
        if image is None:
            reason = ctx.crop_error or "no page image supplied"
        else:
            extractor = self.table_extractor
            if extractor is None:
                reason = "table extraction disabled"
            else:
                result = extractor.extract_table(image)
                reason = None if result else (extractor.last_error or "unknown")

        if result:
            return {
                "content": result["content"],
                "metadata": {
                    "status": "completed",
                    "table_format": result["format"],
                    "table_rows": result.get("rows"),
                    "table_columns": result.get("columns"),
                    "merged_cells": result.get("merged_cells"),
                    "cell_fill": result.get("cell_fill"),
                    "extractor": getattr(extractor, "name", "unknown"),
                    "ocr_fallback_text": ocr_text,
                },
            }

        warning = "<!-- WARNING: Table extraction failed -->"
        return {
            "content": f"{warning}\n{ocr_text}" if ocr_text else warning,
            "metadata": {
                "status": "fallback",
                "table_format": None,
                "table_error": reason,
                "warning": warning,
                "ocr_fallback_text": ocr_text,
            },
        }

    def _handle_formula(self, block: Any, ctx: Optional[RouteContext] = None) -> HandlerResult:
        """Formulas: run image-to-LaTeX over the crop and wrap it as display math.

        Falls back to the block's OCR text with a visible warning whenever LaTeX
        cannot be produced -- no page image, no extractor, or a model failure.
        The fallback is marked ``fallback`` rather than ``completed`` so a
        formula that silently degraded to OCR noise is never mistaken for a
        parsed one.
        """
        ctx = ctx or RouteContext()
        lines = _get(block, "lines", []) or []
        ocr_text = "\n".join(t for t in (_line_text(l) for l in lines) if t)

        image = self._crop_for(ctx)
        latex, reason = None, None
        if image is None:
            reason = ctx.crop_error or "no page image supplied"
        else:
            extractor = self.formula_extractor
            if extractor is None:
                reason = "formula extraction disabled"
            else:
                latex = extractor.extract_latex(image)
                reason = None if latex else (extractor.last_error or "unknown")

        if latex:
            return {
                "content": f"$$\n{latex}\n$$",
                "metadata": {
                    "status": "completed",
                    "latex": latex,
                    "extractor": getattr(extractor, "name", "unknown"),
                    "ocr_fallback_text": ocr_text,
                },
            }

        warning = "<!-- WARNING: LaTeX extraction failed -->"
        return {
            "content": f"{warning}\n{ocr_text}" if ocr_text else warning,
            "metadata": {
                "status": "fallback",
                "latex": None,
                "latex_error": reason,
                "warning": warning,
                "ocr_fallback_text": ocr_text,
            },
        }

    def _crop_for(self, ctx: RouteContext):
        """The cropped region for ``ctx``: in memory if step 2 kept it, else
        re-read from the PNG it wrote."""
        if ctx.crop_image is not None:
            return ctx.crop_image
        if ctx.crop_path and os.path.isfile(ctx.crop_path):
            try:
                from PIL import Image

                return Image.open(ctx.crop_path).convert("RGB")
            except Exception:  # noqa: BLE001 - treated as "no crop"
                return None
        return None

    def _handle_vision(self, block: Any, ctx: Optional[RouteContext] = None) -> HandlerResult:
        """Figures: publish the crop as a standalone asset and emit Markdown.

        Complete, not a stub -- a figure *is* its pixels, so once the region is
        cropped there is nothing further to extract. Falls back to a pending
        marker when no page image was supplied, rather than emitting a Markdown
        tag that points at a file which was never written.
        """
        ctx = ctx or RouteContext()
        lines = _get(block, "lines", []) or []
        caption = "\n".join(t for t in (_line_text(l) for l in lines) if t)

        if not ctx.crop_path:
            result = self._stub(block, BlockType.VISION, ctx,
                                "figure crop (no page image supplied)")
            return result

        rel = self.figure_rel_path(ctx.page_number, ctx.block_id)
        figure_path = os.path.join(ctx.output_dir or "", rel) if ctx.output_dir else rel
        try:
            os.makedirs(os.path.dirname(os.path.abspath(figure_path)), exist_ok=True)
            shutil.copyfile(ctx.crop_path, figure_path)
            error = None
        except OSError as exc:
            error = f"{type(exc).__name__}: {exc}"

        if error:
            return {
                "content": f"<!-- VISION: crop saved but figure copy failed -->",
                "metadata": {"status": "pending", "figure_error": error,
                             "ocr_fallback_text": caption},
            }

        markdown = f"![Figure]({rel})"
        return {
            "content": f"{markdown}\n\n{caption}" if caption else markdown,
            "metadata": {
                "status": "completed",
                "figure_path": figure_path,
                "figure_rel_path": rel,
                "markdown": markdown,
                "caption_text": caption,
            },
        }

    def _stub(self, block: Any, block_type: BlockType,
              ctx: Optional[RouteContext] = None,
              pending: str = "") -> HandlerResult:
        lines = _get(block, "lines", []) or []
        ocr_text = "\n".join(t for t in (_line_text(l) for l in lines) if t)
        bbox = [round(float(v)) for v in (_get(block, "bbox", []) or [])]
        marker = f"<!-- {block_type.value}: pending {pending} -->"
        return {
            "content": f"{marker}\n{ocr_text}" if ocr_text else marker,
            "metadata": {
                "status": "pending",
                "pending_step": pending,
                "marker": marker,
                "ocr_fallback_text": ocr_text,
                "crop_bbox": bbox,
            },
        }

    # -- entry point -------------------------------------------------------- #
    def _make_crop(self, block: Any, ctx: RouteContext) -> None:
        """Slice the block's pixels to ``crops/`` and record the path on ``ctx``.

        A failure here is per-block: a detector occasionally emits a box off the
        page edge, and losing one crop must not abort the document.
        """
        if ctx.page_image is None or ctx.block_type not in CROPPED_TYPES:
            return
        rel = os.path.join("crops", self.crop_filename(ctx.page_number, ctx.block_id))
        path = os.path.join(ctx.output_dir, rel) if ctx.output_dir else rel
        try:
            crop = self.crop_engine.crop_block(
                ctx.page_image, _get(block, "bbox", []) or [], self.crop_padding)
            ctx.crop_image = crop
            ctx.crop_path = self.crop_engine.save_crop(crop, path)
        except (ValueError, TypeError, OSError) as exc:
            ctx.crop_error = f"{type(exc).__name__}: {exc}"

    def route(self, blocks: Iterable[Any], page_number: int = 0,
              page_image: Any = None, output_dir: Optional[str] = None) -> list:
        """Route ``blocks`` and return :class:`SemanticBlock`s in the same order.

        When ``page_image`` is supplied, TABLE / FORMULA / VISION blocks are
        cropped to ``<output_dir>/crops/`` first, so their handlers receive a
        real file to work from.
        """
        out = []
        output_dir = output_dir if output_dir is not None else self.output_dir
        for position, block in enumerate(blocks):
            label = _get(block, "label", None)
            block_type = self.classify(label)
            handler = self.handler_for(block_type)

            # The block id is fixed before dispatch: crop and figure filenames
            # are built from it, so it cannot be assigned afterwards.
            ctx = RouteContext(
                block_id=self._next_id, page_number=page_number,
                block_type=block_type, page_image=page_image,
                output_dir=output_dir,
            )
            self._make_crop(block, ctx)
            result = handler(block, ctx) or {}

            lines = list(_get(block, "lines", []) or [])
            scores = [s for s in (_line_score(l) for l in lines) if s is not None]
            metadata = {
                "label": label,
                "page": page_number,
                "reading_position": position,
                "handler": handler.__name__,
                "n_lines": len(lines),
                "mean_line_confidence": (
                    round(sum(scores) / len(scores), 4) if scores else None
                ),
                "is_marginal": str(label).strip().lower() in MARGINAL_LABELS
                if label else False,
            }
            # Source order, when the caller passed real OrderedBlocks. Recorded
            # rather than used, so a mismatch with reading_position is visible.
            source_order = _get(block, "order", None)
            if source_order is not None:
                metadata["source_order"] = source_order
            if _get(block, "synthetic", False):
                metadata["synthetic"] = True
            if ctx.crop_error:
                metadata["crop_error"] = ctx.crop_error
            metadata.update(result.get("metadata") or {})

            out.append(SemanticBlock(
                block_id=ctx.block_id,
                block_type=block_type,
                bbox=list(_get(block, "bbox", []) or []),
                raw_lines=lines,
                parsed_content=result.get("content", ""),
                metadata=metadata,
                crop_path=ctx.crop_path,
            ))
            self._next_id += 1
        return out

    def route_pages(self, pages_and_blocks: Iterable[Any],
                    output_dir: Optional[str] = None) -> list:
        """Route a whole document. Ids stay unique across pages.

        Each item may be:

        * ``(page, image)`` -- a page object or dict plus its rendered image,
        * a page carrying its own image on ``.image``, or
        * a bare page, in which case nothing is cropped.

        Returns one list of :class:`SemanticBlock` per page.
        """
        out = []
        for i, item in enumerate(pages_and_blocks):
            image = None
            page = item
            if isinstance(item, tuple) and len(item) == 2:
                page, image = item
            if image is None:
                image = _get(page, "image", None)
            out.append(self.route(
                _get(page, "blocks", []) or [],
                page_number=_get(page, "index", i),
                page_image=image,
                output_dir=output_dir,
            ))
        return out

    def reset(self, start_id: int = 0) -> None:
        self._next_id = start_id


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Attribute or dict lookup, so real objects and plain dicts both route."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)
