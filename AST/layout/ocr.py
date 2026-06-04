"""Per-box text extraction (the whiteboard's ``OCR : BBox -> Maybe[Text]``).

PaddleOCR's text **detection** model emits LINE-level boxes (in dense, justified
text the DB detector merges words; ``unclip_ratio`` only dilates boxes, it does
not separate words — empirically verified). So:

  * ``line`` : detect lines, recognize each crop.
  * ``word`` : detect lines, then split each line at its internal whitespace
               gaps via a vertical projection profile, and recognize each word
               crop. The boxes are real (pixel-accurate to the gaps), all from
               PaddleOCR det + rec.

PP-DocLayoutV3 (see :mod:`layout.detector`) already provides the coarsest,
paragraph/region level; this module provides the finer two.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

DEFAULT_DET_MODEL = "PP-OCRv5_server_det"
DEFAULT_REC_MODEL = "PP-OCRv5_server_rec"

LINE_UNCLIP = 1.5  # PP-OCRv5 default — line-level boxes

# Word-splitting: a blank-column run wider than this fraction of the line height
# is treated as a space between words.
WORD_GAP_FRAC = 0.28
# Minimum crop dimensions worth sending to the recognizer.
_MIN_CROP_PX = 3


@dataclass
class TextBox:
    bbox: list[float]  # axis-aligned [x1, y1, x2, y2] in page-pixel coords
    text: Optional[str]
    det_score: Optional[float]  # detection confidence
    text_score: Optional[float]  # recognition confidence


def _res_dict(res) -> dict:
    """Best-effort extraction of a PaddleOCR result object's payload dict."""
    j = getattr(res, "json", None)
    if isinstance(j, dict):
        return j.get("res", j)
    try:
        return dict(res)  # some result objects are mapping-like
    except Exception:
        return {}


def _word_column_spans(crop_bgr: np.ndarray, gap_frac: float) -> list[tuple[int, int]]:
    """Find word column spans (x1, x2) within a line crop via its ink profile.

    Binarizes the crop, sums ink per column, and treats blank-column runs wider
    than ``gap_frac * line_height`` as spaces separating words.
    """
    if crop_bgr.size == 0:
        return []
    gray = crop_bgr.mean(axis=2)
    h, w = gray.shape
    # Foreground = darker than a margin below the (light-background) mean.
    thresh = gray.mean() - 0.30 * gray.std()
    ink = (gray < thresh).sum(axis=0)  # ink count per column
    blank = ink == 0

    min_gap = max(3, int(round(gap_frac * h)))

    spans: list[tuple[int, int]] = []
    word_start: Optional[int] = None  # start col of the current word
    word_end = 0  # last ink col + 1 of the current word
    run = 0  # length of the current blank run

    for col in range(w):
        if blank[col]:
            run += 1
            continue
        # An ink column: does the preceding blank run end the previous word?
        if word_start is not None and run >= min_gap:
            spans.append((word_start, word_end))
            word_start = None
        if word_start is None:
            word_start = col
        word_end = col + 1
        run = 0

    if word_start is not None:
        spans.append((word_start, word_end))
    return spans


def _poly_to_bbox(poly) -> list[float]:
    """Convert a 4-point detection polygon to an axis-aligned bbox."""
    pts = np.asarray(poly, dtype=float).reshape(-1, 2)
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    return [float(x1), float(y1), float(x2), float(y2)]


class TextExtractor:
    """Wraps PaddleOCR ``TextDetection`` + ``TextRecognition``.

    Instantiate once (loads + downloads both models) and reuse across pages.
    """

    def __init__(
        self,
        det_model: str = DEFAULT_DET_MODEL,
        rec_model: str = DEFAULT_REC_MODEL,
        device: str = "cpu",
        rec_batch_size: int = 16,
    ) -> None:
        from paddleocr import TextDetection, TextRecognition  # lazy import

        self.device = device
        self.rec_batch_size = rec_batch_size
        self._det = TextDetection(model_name=det_model, device=device)
        self._rec = TextRecognition(model_name=rec_model, device=device)

    def _detect(self, image_path: str, unclip_ratio: float) -> list[tuple[list[float], float]]:
        """Return [(bbox, det_score), ...] for one page image."""
        boxes: list[tuple[list[float], float]] = []
        for res in self._det.predict(input=image_path, unclip_ratio=unclip_ratio, batch_size=1):
            data = _res_dict(res)
            polys = data.get("dt_polys", []) or []
            scores = data.get("dt_scores", []) or []
            for i, poly in enumerate(polys):
                score = float(scores[i]) if i < len(scores) and scores[i] is not None else None
                boxes.append((_poly_to_bbox(poly), score))
        return boxes

    def _recognize(self, crops: list[np.ndarray]) -> list[tuple[Optional[str], Optional[float]]]:
        """Recognize cropped (BGR) word/line images, order preserved.

        Crops are sorted by width before batching so each batch pads to a
        similar width (PP-OCR pads a batch to its widest crop) — this noticeably
        cuts wasted compute on CPU. Results are unsorted back to input order.
        """
        if not crops:
            return []
        order = sorted(range(len(crops)), key=lambda i: crops[i].shape[1])
        sorted_crops = [crops[i] for i in order]

        rec_sorted: list[tuple[Optional[str], Optional[float]]] = []
        for res in self._rec.predict(input=sorted_crops, batch_size=self.rec_batch_size):
            data = _res_dict(res)
            text = data.get("rec_text")
            score = data.get("rec_score")
            rec_sorted.append((text or None, float(score) if score is not None else None))

        out: list[tuple[Optional[str], Optional[float]]] = [(None, None)] * len(crops)
        for pos, i in enumerate(order):
            out[i] = rec_sorted[pos]
        return out

    def _boxes_from_dets(
        self,
        dets: list[tuple[list[float], float]],
        page_bgr: np.ndarray,
    ) -> list[TextBox]:
        """Crop each detection from the page and recognize its text."""
        h, w = page_bgr.shape[:2]
        crops: list[np.ndarray] = []
        keep: list[tuple[list[float], float]] = []
        for bbox, score in dets:
            x1 = max(0, int(round(bbox[0])))
            y1 = max(0, int(round(bbox[1])))
            x2 = min(w, int(round(bbox[2])))
            y2 = min(h, int(round(bbox[3])))
            if x2 - x1 < _MIN_CROP_PX or y2 - y1 < _MIN_CROP_PX:
                continue
            crops.append(page_bgr[y1:y2, x1:x2])
            keep.append((bbox, score))

        recognized = self._recognize(crops)
        return [
            TextBox(bbox=bbox, text=text, det_score=det_score, text_score=text_score)
            for (bbox, det_score), (text, text_score) in zip(keep, recognized)
        ]

    def extract(
        self,
        image_path: str,
        page_bgr: np.ndarray,
        granularity: str,
    ) -> list[TextBox]:
        """Detect + recognize text boxes at ``line`` or ``word`` granularity."""
        if granularity not in ("line", "word"):
            raise ValueError(f"TextExtractor handles 'line'/'word', not {granularity!r}")
        line_dets = self._detect(image_path, unclip_ratio=LINE_UNCLIP)
        dets = line_dets if granularity == "line" else self._split_lines_to_words(line_dets, page_bgr)
        return self._boxes_from_dets(dets, page_bgr)

    def extract_line_and_word(
        self,
        image_path: str,
        page_bgr: np.ndarray,
    ) -> tuple[list[TextBox], list[TextBox]]:
        """Detect lines once, return both line boxes and word boxes (recognized)."""
        line_dets = self._detect(image_path, unclip_ratio=LINE_UNCLIP)
        line_boxes = self._boxes_from_dets(line_dets, page_bgr)
        word_dets = self._split_lines_to_words(line_dets, page_bgr)
        word_boxes = self._boxes_from_dets(word_dets, page_bgr)
        return line_boxes, word_boxes

    def _split_lines_to_words(
        self,
        line_dets: list[tuple[list[float], float]],
        page_bgr: np.ndarray,
    ) -> list[tuple[list[float], float]]:
        """Split each line box into word boxes at internal whitespace gaps."""
        h, w = page_bgr.shape[:2]
        words: list[tuple[list[float], float]] = []
        for bbox, score in line_dets:
            x1 = max(0, int(round(bbox[0])))
            y1 = max(0, int(round(bbox[1])))
            x2 = min(w, int(round(bbox[2])))
            y2 = min(h, int(round(bbox[3])))
            if x2 - x1 < _MIN_CROP_PX or y2 - y1 < _MIN_CROP_PX:
                continue
            crop = page_bgr[y1:y2, x1:x2]
            spans = _word_column_spans(crop, gap_frac=WORD_GAP_FRAC)
            if len(spans) <= 1:
                words.append((bbox, score))  # single word / couldn't split
                continue
            for sx1, sx2 in spans:
                words.append(([float(x1 + sx1), float(y1), float(x1 + sx2), float(y2)], score))
        return words
