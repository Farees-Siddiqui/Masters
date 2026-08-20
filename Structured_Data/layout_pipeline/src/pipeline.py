"""Document extraction pipeline: PaddleOCR -> XY-Cut++ -> ordered blocks.

    render page -> det+rec text lines  -\
                                         >- attach lines to blocks -> XY-Cut++ -> ordered output
                -> layout blocks       -/

Text comes from det+rec (highest coverage), structure from PP-DocLayout. A line
that falls inside no detected block still becomes a block of its own, so layout
recall problems can never silently delete text -- the failure mode measured on
PP-StructureV3 in this repo's engine survey.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .ocr_engine import (DEFAULT_DPI, LayoutBlock, OCREngine, Page, TextLine,
                         load_inputs)
from .xycut_plus import DEFAULT_CONFIG, XYCutConfig, order_indices

# Labels that carry no body text; kept for structure but never merged into prose.
NON_TEXT_LABELS = frozenset(
    {"image", "figure", "chart", "table", "seal", "figure_title"}
)
# Minimum fraction of a line's area that must fall inside a block to attach it.
CONTAINMENT_THRESHOLD = 0.5


@dataclass
class OrderedBlock:
    """A layout block with its lines, in reading order."""

    order: int
    label: str
    bbox: list
    lines: list = field(default_factory=list)
    score: Optional[float] = None
    synthetic: bool = False   # created from unattached lines, not detected

    @property
    def text(self) -> str:
        return "\n".join(l.text for l in self.lines if l.text)

    def to_dict(self) -> dict:
        return {
            "order": self.order,
            "label": self.label,
            "bbox": [round(v, 1) for v in self.bbox],
            "score": round(self.score, 4) if self.score is not None else None,
            "synthetic": self.synthetic,
            "text": self.text,
            "lines": [l.to_dict() for l in self.lines],
        }


@dataclass
class OrderedPage:
    source: str
    index: int
    width: int
    height: int
    blocks: list = field(default_factory=list)
    #: The rendered page, kept so the extraction router can crop regions out of
    #: it. Never serialised -- ``to_dict`` is the on-disk contract.
    image: Optional[object] = None

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "page_index": self.index,
            "width": self.width,
            "height": self.height,
            "n_blocks": len(self.blocks),
            "blocks": [b.to_dict() for b in self.blocks],
        }


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def _area(b: Sequence[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _containment(inner: Sequence[float], outer: Sequence[float]) -> float:
    """Fraction of ``inner``'s area that lies inside ``outer``."""
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    a = _area(inner)
    return (ix * iy / a) if a > 0 else 0.0


def _union(boxes: Sequence[Sequence[float]]) -> list:
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def attach_lines(lines: Sequence[TextLine], blocks: Sequence[LayoutBlock],
                 threshold: float = CONTAINMENT_THRESHOLD) -> tuple:
    """Assign each line to its best-containing block.

    Returns ``(per_block_lines, orphan_lines)``. A line goes to the block that
    contains the largest fraction of it, provided that fraction clears
    ``threshold``; otherwise it is an orphan and the caller keeps it as its own
    block rather than dropping it.
    """
    per_block = [[] for _ in blocks]
    orphans = []
    for line in lines:
        best_i, best_c = None, 0.0
        for i, blk in enumerate(blocks):
            c = _containment(line.bbox, blk.bbox)
            if c > best_c:
                best_i, best_c = i, c
        if best_i is not None and best_c >= threshold:
            per_block[best_i].append(line)
        else:
            orphans.append(line)
    return per_block, orphans


def group_orphans(orphans: Sequence[TextLine], gap_factor: float = 1.6) -> list:
    """Cluster unattached lines into pseudo-blocks by vertical adjacency.

    Without this every orphan line becomes its own block, which fragments the
    reading order. Lines are grouped when they overlap horizontally and sit
    within ``gap_factor`` line-heights of each other.
    """
    if not orphans:
        return []
    items = sorted(orphans, key=lambda l: (l.bbox[1], l.bbox[0]))
    groups = [[items[0]]]
    for line in items[1:]:
        prev = groups[-1][-1]
        h = max(1.0, prev.bbox[3] - prev.bbox[1])
        vgap = line.bbox[1] - prev.bbox[3]
        overlap = min(line.bbox[2], prev.bbox[2]) - max(line.bbox[0], prev.bbox[0])
        if vgap <= gap_factor * h and overlap > 0:
            groups[-1].append(line)
        else:
            groups.append([line])
    return groups


def sort_lines(lines: Sequence[TextLine], config: XYCutConfig = None,
               row_overlap: float = 0.5) -> list:
    """Order lines inside one block: rows top to bottom, then left to right.

    Deliberately *not* the recursive cut. Inside a block the lines are one
    vertical stack, and the cut prefers a column split whenever a vertical gap
    exists -- so a single short line's ragged right edge opens a phantom column
    and the block comes out scrambled. On the ViT page that put the citation
    year "2017" ahead of the sentence it belongs to.

    Grouping by vertical overlap first also handles the legitimate side-by-side
    case (two short lines sharing a baseline) without any column logic.
    """
    if len(lines) <= 1:
        return list(lines)

    items = sorted(lines, key=lambda l: (l.bbox[1], l.bbox[0]))
    rows = [[items[0]]]
    for line in items[1:]:
        row = rows[-1]
        top = min(x.bbox[1] for x in row)
        bottom = max(x.bbox[3] for x in row)
        height = max(1.0, min(bottom - top, line.bbox[3] - line.bbox[1]))
        overlap = min(bottom, line.bbox[3]) - max(top, line.bbox[1])
        if overlap > row_overlap * height:
            row.append(line)
        else:
            rows.append([line])

    ordered = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda l: l.bbox[0]))
    return ordered


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
class DocumentPipeline:
    """End-to-end: input path -> ordered blocks per page."""

    def __init__(self, engine: Optional[OCREngine] = None,
                 config: XYCutConfig = DEFAULT_CONFIG,
                 dpi: int = DEFAULT_DPI, batch_size: int = 4):
        self.engine = engine if engine is not None else OCREngine()
        self.config = config
        self.dpi = dpi
        self.batch_size = batch_size

    def run(self, input_path: str) -> list:
        """Process every page of every input, returning :class:`OrderedPage`s."""
        items = load_inputs(input_path, dpi=self.dpi)
        images = [img for _, _, img in items]
        detections = self.engine.process(images, batch_size=self.batch_size)

        pages = []
        for (source, page_index, img), (lines, blocks) in zip(items, detections):
            w, h = img.size
            page = self.assemble(source, page_index, w, h, lines, blocks)
            page.image = img
            pages.append(page)
        return pages

    def assemble(self, source: str, page_index: int, width: int, height: int,
                 lines: Sequence[TextLine],
                 blocks: Sequence[LayoutBlock]) -> OrderedPage:
        """Attach lines to blocks and order everything. Pure geometry -- no models,
        so this is directly unit-testable."""
        per_block, orphans = attach_lines(lines, blocks)

        candidates = []
        for blk, blines in zip(blocks, per_block):
            if not blines and blk.label not in NON_TEXT_LABELS:
                # An empty text-ish region contributes nothing and would only add
                # a spurious node to the ordering.
                continue
            candidates.append(OrderedBlock(
                order=-1, label=blk.label, bbox=list(blk.bbox),
                lines=sort_lines(blines, self.config), score=blk.score,
            ))

        for group in group_orphans(orphans):
            candidates.append(OrderedBlock(
                order=-1, label="text", bbox=_union([l.bbox for l in group]),
                lines=sort_lines(group, self.config), synthetic=True,
            ))

        if not candidates:
            return OrderedPage(source, page_index, width, height, [])

        ordering = order_indices(
            [c.bbox for c in candidates], [c.label for c in candidates],
            width, height, self.config,
        )
        ordered = []
        for rank, idx in enumerate(ordering):
            blk = candidates[idx]
            blk.order = rank
            ordered.append(blk)
        return OrderedPage(source, page_index, width, height, ordered)


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #
def to_json(pages: Sequence[OrderedPage], config: XYCutConfig) -> str:
    return json.dumps({
        "config": {
            "beta": config.beta,
            "theta_v": config.theta_v,
            "min_gap_px": config.min_gap_px,
            "overlap_threshold": config.overlap_threshold,
            "mask_titles": config.mask_titles,
            "mask_vision": config.mask_vision,
            "mask_marginal": config.mask_marginal,
            "enable_cross_mask": config.enable_cross_mask,
        },
        "n_pages": len(pages),
        "pages": [p.to_dict() for p in pages],
    }, indent=2, ensure_ascii=False)


def to_markdown(pages: Sequence[OrderedPage]) -> str:
    """Stitch pages into Markdown, preserving the recovered reading order."""
    out = []
    for page in pages:
        if len(pages) > 1:
            out.append(f"<!-- {page.source} page {page.index + 1} -->\n")
        for blk in page.blocks:
            label, text = blk.label, blk.text
            if label in ("doc_title",):
                out.append(f"# {text}\n" if text else "")
            elif label in ("paragraph_title", "title", "sub_title", "chapter_title"):
                out.append(f"## {text}\n" if text else "")
            elif label in ("formula", "display_formula"):
                out.append(f"$$\n{text}\n$$\n" if text else "")
            elif label in ("table",):
                out.append("<!-- table -->\n" + (f"{text}\n" if text else ""))
            elif label in ("image", "figure", "chart", "seal"):
                bbox = ", ".join(str(round(v)) for v in blk.bbox)
                out.append(f"<!-- {label} at [{bbox}] -->\n")
                if text:
                    out.append(f"{text}\n")
            elif label in ("aside_text", "header", "footer", "number",
                           "page_number", "footnote", "page_footnote"):
                if text:
                    out.append(f"<!-- {label}: {text} -->\n")
            elif text:
                out.append(f"{text}\n")
    return "\n".join(s for s in out if s)


def write_outputs(pages: Sequence[OrderedPage], output_dir: str,
                  config: XYCutConfig, stem: str = "document",
                  formats: Sequence[str] = ("json", "md")) -> list:
    os.makedirs(output_dir, exist_ok=True)
    written = []
    if "json" in formats:
        p = os.path.join(output_dir, f"{stem}.json")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(to_json(pages, config))
        written.append(p)
    if "md" in formats:
        p = os.path.join(output_dir, f"{stem}.md")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(to_markdown(pages))
        written.append(p)
    return written
