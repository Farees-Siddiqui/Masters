"""Core layout-detection + per-box text pipeline.

Pipeline per document:

    PDF --(pypdfium2)--> page images --(PP-DocLayoutV3)--> region boxes+labels
                                     \\--(PP-OCRv5 det+rec)--> line/word boxes+text

``--granularity`` selects which BBox level(s) to emit; every box carries
``(bbox, label, text)`` — the whiteboard's ``BBox in R^4 x Enum`` plus
``OCR : BBox -> Maybe[Text]``:

  * ``paragraph`` : PP-DocLayoutV3 regions (label per region), text aggregated
                    from the recognized lines that fall inside each region.
  * ``line``      : PP-OCRv5 line boxes; label inherited from containing region.
  * ``word``      : line boxes split at whitespace gaps; label inherited likewise.
  * ``all``       : all three (default).

Output layout, for an input ``resnet.pdf`` and ``-o output``::

    output/resnet/
        document.json            # doc-level summary: per-page boxes for each granularity
        page1/
            page.png             # rendered source image
            layout.png/.json     # raw PP-DocLayoutV3 region detection
            paragraph.png/.json  # boxes drawn + bbox/label/text per granularity
            line.png/.json
            word.png/.json
        page2/ ...
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pypdfium2 as pdfium

from .draw import draw_boxes

DEFAULT_DPI = 200
DEFAULT_MODEL = "PP-DocLayoutV3"
DEFAULT_DEVICE = "cpu"  # Apple Silicon has no CUDA; CPU is the only option.
GRANULARITIES = ("paragraph", "line", "word")
ALL_GRANULARITY = "all"
DEFAULT_GRANULARITY = ALL_GRANULARITY


@dataclass
class Region:
    """A PP-DocLayoutV3 detection: a labelled bounding box."""

    label: Optional[str]
    cls_id: Optional[int]
    score: Optional[float]
    bbox: list[float]  # [x1, y1, x2, y2]


@dataclass
class Box:
    """An emitted layout box at a given granularity."""

    label: Optional[str]  # semantic label (Enum part of BBox)
    cls_id: Optional[int]
    bbox: list[float]  # [x1, y1, x2, y2] in page-pixel coords
    score: Optional[float]  # detection confidence
    text: Optional[str]  # OCR text (Maybe[Text])
    text_score: Optional[float]  # recognition confidence

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "cls_id": self.cls_id,
            "bbox": self.bbox,
            "score": self.score,
            "text": self.text,
            "text_score": self.text_score,
        }


@dataclass
class PageResult:
    page: int  # 1-indexed
    dir: str  # relative folder name, e.g. "page1"
    image: str  # relative path to the rendered page image
    width: int
    height: int
    boxes: dict[str, list[Box]] = field(default_factory=dict)  # granularity -> boxes


@dataclass
class DocumentResult:
    source_pdf: str
    output_dir: str
    page_count: int
    dpi: int
    model: str
    granularities: list[str]
    pages: list[PageResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "source_pdf": self.source_pdf,
            "output_dir": self.output_dir,
            "page_count": self.page_count,
            "dpi": self.dpi,
            "model": self.model,
            "granularities": self.granularities,
            "pages": [
                {
                    "page": p.page,
                    "dir": p.dir,
                    "image": p.image,
                    "width": p.width,
                    "height": p.height,
                    "boxes": {g: [b.to_dict() for b in boxes] for g, boxes in p.boxes.items()},
                }
                for p in self.pages
            ],
        }


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #

def _round_bbox(bbox: list[float]) -> list[float]:
    return [round(float(c), 2) for c in bbox]


def _center(bbox: list[float]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _containing_region(bbox: list[float], regions: list[Region]) -> Optional[int]:
    """Index of the region containing ``bbox``'s center (smallest if nested).

    Falls back to the region with the largest overlap if none strictly contains
    the center. Returns None if nothing overlaps.
    """
    cx, cy = _center(bbox)
    best_idx: Optional[int] = None
    best_area = float("inf")
    for i, r in enumerate(regions):
        rb = r.bbox
        if rb[0] <= cx <= rb[2] and rb[1] <= cy <= rb[3] and _area(rb) < best_area:
            best_area, best_idx = _area(rb), i
    if best_idx is not None:
        return best_idx

    best_overlap = 0.0
    for i, r in enumerate(regions):
        rb = r.bbox
        ox = max(0.0, min(bbox[2], rb[2]) - max(bbox[0], rb[0]))
        oy = max(0.0, min(bbox[3], rb[3]) - max(bbox[1], rb[1]))
        if ox * oy > best_overlap:
            best_overlap, best_idx = ox * oy, i
    return best_idx


# --------------------------------------------------------------------------- #
# Box construction from extracted text
# --------------------------------------------------------------------------- #

def _assign_labels(text_boxes, regions: list[Region]) -> list[Box]:
    """Turn recognized line/word TextBoxes into labelled Boxes."""
    boxes: list[Box] = []
    for tb in text_boxes:
        idx = _containing_region(tb.bbox, regions)
        region = regions[idx] if idx is not None else None
        boxes.append(
            Box(
                label=region.label if region else None,
                cls_id=region.cls_id if region else None,
                bbox=_round_bbox(tb.bbox),
                score=round(tb.det_score, 4) if tb.det_score is not None else None,
                text=tb.text,
                text_score=round(tb.text_score, 4) if tb.text_score is not None else None,
            )
        )
    return boxes


def _aggregate_paragraphs(regions: list[Region], line_boxes) -> list[Box]:
    """Region boxes with text aggregated from the recognized lines inside them."""
    buckets: dict[int, list] = {i: [] for i in range(len(regions))}
    for tb in line_boxes:
        idx = _containing_region(tb.bbox, regions)
        if idx is not None:
            buckets[idx].append(tb)

    boxes: list[Box] = []
    for i, region in enumerate(regions):
        members = sorted(buckets[i], key=lambda t: (t.bbox[1], t.bbox[0]))  # top->bottom
        texts = [m.text for m in members if m.text]
        scores = [m.text_score for m in members if m.text_score is not None]
        boxes.append(
            Box(
                label=region.label,
                cls_id=region.cls_id,
                bbox=region.bbox,
                score=region.score,
                text=" ".join(texts) if texts else None,
                text_score=round(sum(scores) / len(scores), 4) if scores else None,
            )
        )
    return boxes


# --------------------------------------------------------------------------- #
# PDF rendering / layout parsing
# --------------------------------------------------------------------------- #

def _render_pdf_pages(pdf_path: Path, dpi: int):
    """Yield (1-indexed page number, PIL.Image RGB) for each page."""
    scale = dpi / 72.0  # PDF user-space units are 1/72 inch
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        for i in range(len(pdf)):
            page = pdf[i]
            bitmap = page.render(scale=scale)
            image = bitmap.to_pil().convert("RGB")
            yield i + 1, image
    finally:
        pdf.close()


def _parse_regions(layout_json_path: Path) -> list[Region]:
    """Read PP-DocLayoutV3's per-page JSON into Region objects."""
    try:
        data = json.loads(layout_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    raw_boxes = data.get("boxes", []) if isinstance(data, dict) else []
    regions: list[Region] = []
    for b in raw_boxes:
        coord = b.get("coordinate") or b.get("bbox")
        if not coord:
            continue
        regions.append(
            Region(
                label=b.get("label"),
                cls_id=b.get("cls_id"),
                score=round(float(b["score"]), 4) if b.get("score") is not None else None,
                bbox=_round_bbox(coord),
            )
        )
    return regions


def _resolve_granularities(granularity: str) -> list[str]:
    if granularity == ALL_GRANULARITY:
        return list(GRANULARITIES)
    if granularity in GRANULARITIES:
        return [granularity]
    raise ValueError(
        f"granularity must be one of {GRANULARITIES + (ALL_GRANULARITY,)}, got {granularity!r}"
    )


class LayoutDetector:
    """Runs PP-DocLayoutV3 and, for finer granularities, PaddleOCR det+rec.

    Instantiate once (models load + download here) and call
    :meth:`process_pdf` per document. The OCR text models load lazily.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = DEFAULT_DEVICE,
    ) -> None:
        from paddleocr import LayoutDetection  # lazy: avoids paddle import cost

        self.model_name = model_name
        self.device = device
        self._model = LayoutDetection(model_name=model_name, device=device)
        self._text_extractor = None  # created on demand

    def _ocr(self):
        if self._text_extractor is None:
            from .ocr import TextExtractor

            self._text_extractor = TextExtractor(device=self.device)
        return self._text_extractor

    def process_pdf(
        self,
        pdf_path: str | Path,
        output_dir: str | Path,
        dpi: int = DEFAULT_DPI,
        granularity: str = DEFAULT_GRANULARITY,
    ) -> DocumentResult:
        grans = _resolve_granularities(granularity)

        pdf_path = Path(pdf_path).expanduser().resolve()
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc_dir = Path(output_dir).expanduser() / pdf_path.stem
        doc_dir.mkdir(parents=True, exist_ok=True)

        need_line = "line" in grans or "paragraph" in grans
        need_word = "word" in grans

        pages: list[PageResult] = []
        for page_no, image in _render_pdf_pages(pdf_path, dpi):
            page_dir = doc_dir / f"page{page_no}"
            page_dir.mkdir(parents=True, exist_ok=True)

            page_img = page_dir / "page.png"
            image.save(page_img)

            # 1) Layout regions (+ labels) via PP-DocLayoutV3.
            for res in self._model.predict(input=str(page_img), batch_size=1):
                res.save_to_img(save_path=str(page_dir / "layout.png"))
                res.save_to_json(save_path=str(page_dir / "layout.json"))
            regions = _parse_regions(page_dir / "layout.json")

            # 2) Text extraction (only what the requested granularities need).
            line_tbs: list = []
            word_tbs: list = []
            if need_line or need_word:
                page_bgr = np.asarray(image)[:, :, ::-1].copy()  # RGB -> BGR for PaddleOCR
                if need_line and need_word:
                    line_tbs, word_tbs = self._ocr().extract_line_and_word(str(page_img), page_bgr)
                elif need_line:
                    line_tbs = self._ocr().extract(str(page_img), page_bgr, "line")
                elif need_word:
                    word_tbs = self._ocr().extract(str(page_img), page_bgr, "word")

            # 3) Build boxes per granularity, write JSON + annotated image.
            boxes_by_gran: dict[str, list[Box]] = {}
            for g in grans:
                if g == "paragraph":
                    boxes = _aggregate_paragraphs(regions, line_tbs)
                elif g == "line":
                    boxes = _assign_labels(line_tbs, regions)
                else:  # word
                    boxes = _assign_labels(word_tbs, regions)
                boxes_by_gran[g] = boxes
                self._write_page_outputs(page_dir, g, page_no, image, boxes)

            pages.append(
                PageResult(
                    page=page_no,
                    dir=page_dir.name,
                    image=f"{page_dir.name}/page.png",
                    width=image.width,
                    height=image.height,
                    boxes=boxes_by_gran,
                )
            )

        result = DocumentResult(
            source_pdf=pdf_path.name,
            output_dir=str(doc_dir),
            page_count=len(pages),
            dpi=dpi,
            model=self.model_name,
            granularities=grans,
            pages=pages,
        )
        (doc_dir / "document.json").write_text(
            json.dumps(result.to_dict(), indent=2), encoding="utf-8"
        )
        return result

    # ------------------------------------------------------------------ #
    @staticmethod
    def _write_page_outputs(page_dir: Path, gran: str, page_no: int, image, boxes: list[Box]) -> None:
        """Write ``{gran}.json`` (bbox/label/text) and ``{gran}.png`` (drawn boxes)."""
        payload = {
            "page": page_no,
            "granularity": gran,
            "image": "page.png",
            "width": image.width,
            "height": image.height,
            "boxes": [b.to_dict() for b in boxes],
        }
        (page_dir / f"{gran}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

        annotated = draw_boxes(image, boxes, show_inline_labels=(gran == "paragraph"))
        annotated.save(page_dir / f"{gran}.png")


def detect_layout(
    pdf_path: str | Path,
    output_dir: str | Path,
    dpi: int = DEFAULT_DPI,
    model_name: str = DEFAULT_MODEL,
    device: str = DEFAULT_DEVICE,
    granularity: str = DEFAULT_GRANULARITY,
) -> DocumentResult:
    """One-shot convenience wrapper: build a detector and process one PDF."""
    detector = LayoutDetector(model_name=model_name, device=device)
    return detector.process_pdf(pdf_path, output_dir, dpi=dpi, granularity=granularity)
