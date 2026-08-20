"""PaddleOCR wrapper: text lines plus layout blocks, on GPU.

Two detectors, deliberately:

* **PP-OCR det+rec** gives every text line with a quadrilateral and a confidence.
  On the ten-paper survey in this repo it scored the highest text coverage of
  any engine measured (Word F1 0.978) at ~3 s/page.
* **PP-DocLayout** gives labelled regions (doc_title, abstract, table, image,
  formula, aside_text...). XY-Cut++ needs those labels: Phase 1 cannot mask
  "titles and visual elements" and Phase 4 cannot rank by label priority
  without them.

Running only PP-StructureV3 would provide labels in one pass, but it was
measured dropping ~21% of words on average (and ~45% on two pages), so text
comes from det+rec and labels are overlaid on top.

Nothing here imports torch; paddle is imported lazily so that `--help` and the
pure-geometry tests do not pay for it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

DEFAULT_DPI = 200
DEFAULT_LAYOUT_MODEL = "PP-DocLayoutV3"


@dataclass
class TextLine:
    """One recognised line of text."""

    text: str
    bbox: list          # [x1, y1, x2, y2]
    quad: Optional[list] = None   # original 4-point polygon
    score: Optional[float] = None

    def to_dict(self) -> dict:
        return {"text": self.text, "bbox": [round(v, 1) for v in self.bbox],
                "score": round(self.score, 4) if self.score is not None else None}


@dataclass
class LayoutBlock:
    """One detected layout region."""

    label: str
    bbox: list
    score: Optional[float] = None

    def to_dict(self) -> dict:
        return {"label": self.label, "bbox": [round(v, 1) for v in self.bbox],
                "score": round(self.score, 4) if self.score is not None else None}


@dataclass
class Page:
    """A rendered page and everything detected on it."""

    index: int
    width: int
    height: int
    lines: list = field(default_factory=list)
    blocks: list = field(default_factory=list)
    image_path: Optional[str] = None


def quad_to_bbox(quad: Sequence) -> list:
    """Quadrilateral (4 corner points) -> axis-aligned [x_min, y_min, x_max, y_max]."""
    xs = [float(p[0]) for p in quad]
    ys = [float(p[1]) for p in quad]
    return [min(xs), min(ys), max(xs), max(ys)]


def _tolist(v):
    return v.tolist() if hasattr(v, "tolist") else v


def to_ndarray(img):
    """PIL image -> BGR ndarray for paddle.

    paddle's predictors accept only ``str`` paths or ``numpy.ndarray``; handing
    them a PIL image logs "Not supported input data type!" and silently drops
    the page, yielding zero detections rather than an error. Arrays follow the
    ``cv2.imread`` convention, so RGB must be reversed to BGR.
    """
    if hasattr(img, "mode"):  # PIL
        import numpy as np

        arr = np.asarray(img.convert("RGB"))
        return arr[:, :, ::-1].copy()
    return img


def _result_payload(res):
    """Unwrap a paddleocr predict() result into a plain dict."""
    r = res[0] if isinstance(res, list) else res
    if hasattr(r, "json"):
        j = r.json
        return j.get("res", j)
    return dict(r)


# --------------------------------------------------------------------------- #
# PDF / image loading
# --------------------------------------------------------------------------- #
def render_pdf(path: str, dpi: int = DEFAULT_DPI,
               first_page: Optional[int] = None,
               last_page: Optional[int] = None) -> list:
    """Render PDF pages to PIL images.

    Uses pypdfium2 (already a paddlex dependency, no poppler needed) and falls
    back to pdf2image if it is missing.
    """
    try:
        import pypdfium2 as pdfium

        doc = pdfium.PdfDocument(path)
        n = len(doc)
        lo = (first_page - 1) if first_page else 0
        hi = last_page if last_page else n
        scale = dpi / 72.0
        out = []
        for i in range(max(0, lo), min(n, hi)):
            out.append(doc[i].render(scale=scale).to_pil().convert("RGB"))
        return out
    except ImportError:
        from pdf2image import convert_from_path

        return convert_from_path(path, dpi=dpi, first_page=first_page,
                                 last_page=last_page)


def load_inputs(path: str, dpi: int = DEFAULT_DPI) -> list:
    """Load a PDF, a single image, or a directory of either, as PIL images.

    Returns a list of ``(source_name, page_index, PIL.Image)``.
    """
    from PIL import Image

    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    items = []

    def add_file(fp: str):
        low = fp.lower()
        if low.endswith(".pdf"):
            for i, img in enumerate(render_pdf(fp, dpi=dpi)):
                items.append((os.path.basename(fp), i, img))
        elif os.path.splitext(low)[1] in exts:
            items.append((os.path.basename(fp), 0, Image.open(fp).convert("RGB")))

    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            add_file(os.path.join(path, name))
    else:
        add_file(path)
    if not items:
        raise FileNotFoundError(f"no PDF or image inputs found at {path}")
    return items


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class OCREngine:
    """Lazily-constructed PP-OCR + PP-DocLayout pair.

    Model construction is expensive (seconds to a couple of minutes), so both
    predictors are built once on first use and reused for every page.
    """

    def __init__(self, device: str = "gpu:0", layout_model: str = DEFAULT_LAYOUT_MODEL,
                 detect_layout: bool = True, require_gpu: bool = False,
                 layout_threshold: Optional[float] = None):
        self.device = device
        self.layout_model = layout_model
        self.detect_layout = detect_layout
        self.require_gpu = require_gpu
        self.layout_threshold = layout_threshold
        self._ocr = None
        self._layout = None

    # -- construction ------------------------------------------------------ #
    def _check_gpu(self):
        import paddle

        if self.device.startswith("gpu"):
            if not paddle.device.is_compiled_with_cuda():
                raise RuntimeError("paddle build has no CUDA support")
            if paddle.device.cuda.device_count() < 1:
                raise RuntimeError("no CUDA device visible")
        elif self.require_gpu:
            raise RuntimeError(f"GPU required but device is {self.device!r}")

    @property
    def ocr(self):
        if self._ocr is None:
            self._check_gpu()
            from paddleocr import PaddleOCR

            self._ocr = PaddleOCR(
                device=self.device,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        return self._ocr

    @property
    def layout(self):
        if self._layout is None and self.detect_layout:
            self._check_gpu()
            from paddleocr import LayoutDetection

            kwargs = {"model_name": self.layout_model, "device": self.device}
            if self.layout_threshold is not None:
                kwargs["threshold"] = self.layout_threshold
            self._layout = LayoutDetection(**kwargs)
        return self._layout

    # -- inference --------------------------------------------------------- #
    def read_lines(self, images: Sequence) -> list:
        """Run det+rec. Returns one list of :class:`TextLine` per image.

        The predictor accepts a batch, so a multi-page document costs one call
        rather than one call per page.
        """
        results = self.ocr.predict([to_ndarray(i) for i in images])
        if not isinstance(results, list):
            results = [results]
        out = []
        for res in results:
            d = _result_payload(res)
            texts = d.get("rec_texts") or []
            polys = _tolist(d.get("rec_polys") if d.get("rec_polys") is not None
                            else d.get("dt_polys")) or []
            scores = _tolist(d.get("rec_scores")) or []
            lines = []
            for i, t in enumerate(texts):
                quad = _tolist(polys[i]) if i < len(polys) else None
                if quad is None:
                    continue
                lines.append(TextLine(
                    text=t,
                    bbox=quad_to_bbox(quad),
                    quad=[[float(p[0]), float(p[1])] for p in quad],
                    score=float(scores[i]) if i < len(scores) else None,
                ))
            out.append(lines)
        return out

    def read_layout(self, images: Sequence) -> list:
        """Run layout detection. Returns one list of :class:`LayoutBlock` per image."""
        if not self.detect_layout:
            return [[] for _ in images]
        results = self.layout.predict([to_ndarray(i) for i in images])
        if not isinstance(results, list):
            results = [results]
        out = []
        for res in results:
            d = _result_payload(res)
            blocks = []
            for b in d.get("boxes") or []:
                coord = _tolist(b.get("coordinate"))
                if not coord:
                    continue
                blocks.append(LayoutBlock(
                    label=b.get("label") or "text",
                    bbox=[float(c) for c in coord],
                    score=float(b["score"]) if b.get("score") is not None else None,
                ))
            out.append(blocks)
        return out

    def process(self, images: Sequence, batch_size: int = 4) -> list:
        """Run both detectors over ``images`` in batches.

        Returns a list of ``(lines, blocks)`` tuples, one per image. Batching
        bounds peak VRAM: a whole 40-page PDF handed to the predictor at once
        can exhaust the card, and on this box the 16 GiB container RAM cap bites
        first.
        """
        out = []
        for start in range(0, len(images), max(1, batch_size)):
            chunk = list(images[start:start + batch_size])
            lines = self.read_lines(chunk)
            blocks = self.read_layout(chunk)
            for i in range(len(chunk)):
                out.append((lines[i] if i < len(lines) else [],
                            blocks[i] if i < len(blocks) else []))
        return out
