"""Render detected boxes onto a page image.

Each box is drawn as a rectangle colored by its semantic label (the Enum part of
``BBox``); a legend maps colors to labels. For the coarse ``paragraph`` level the
label is also drawn inline; for the dense ``line``/``word`` levels the color +
legend carry the label without cluttering the page.
"""

from __future__ import annotations

import zlib
from typing import Iterable, Optional, Protocol

from PIL import Image, ImageDraw, ImageFont

# Distinct, reasonably readable-on-white palette.
_PALETTE = [
    (230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (0, 158, 115), (240, 50, 230), (170, 110, 40),
    (0, 128, 128), (128, 0, 0), (70, 130, 180), (188, 128, 0),
    (128, 0, 128), (0, 0, 128), (210, 105, 30), (85, 107, 47),
]
_UNKNOWN_COLOR = (110, 110, 110)

_FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
)


class _BoxLike(Protocol):
    bbox: list[float]
    label: Optional[str]


def color_for(label: Optional[str]) -> tuple[int, int, int]:
    """Deterministic color for a label (stable across pages/granularities)."""
    if not label:
        return _UNKNOWN_COLOR
    idx = zlib.crc32(label.encode("utf-8")) % len(_PALETTE)
    return _PALETTE[idx]


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_legend(draw: ImageDraw.ImageDraw, labels: list[str], font, img_w: int) -> None:
    if not labels:
        return
    pad, swatch = 8, max(10, font.size if hasattr(font, "size") else 12)
    line_h = swatch + 6
    text_w = max((draw.textlength(lbl, font=font) for lbl in labels), default=0)
    box_w = int(swatch + pad * 3 + text_w)
    box_h = line_h * len(labels) + pad * 2
    x0, y0 = img_w - box_w - 12, 12

    draw.rectangle([x0, y0, x0 + box_w, y0 + box_h], fill=(255, 255, 255), outline=(0, 0, 0))
    y = y0 + pad
    for lbl in labels:
        draw.rectangle([x0 + pad, y, x0 + pad + swatch, y + swatch], fill=color_for(lbl))
        draw.text((x0 + pad * 2 + swatch, y - 1), lbl, fill=(0, 0, 0), font=font)
        y += line_h


def draw_boxes(
    image: Image.Image,
    boxes: Iterable[_BoxLike],
    show_inline_labels: bool = False,
) -> Image.Image:
    """Return a copy of ``image`` with ``boxes`` drawn, colored by label."""
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    font = _load_font(max(12, h // 110))
    line_w = max(2, h // 800)

    present: list[str] = []
    for b in boxes:
        color = color_for(b.label)
        x1, y1, x2, y2 = b.bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)
        if b.label and b.label not in present:
            present.append(b.label)
        if show_inline_labels and b.label:
            tb = draw.textbbox((0, 0), b.label, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            ty = max(0, y1 - th - 3)
            draw.rectangle([x1, ty, x1 + tw + 4, ty + th + 3], fill=color)
            draw.text((x1 + 2, ty + 1), b.label, fill=(255, 255, 255), font=font)

    _draw_legend(draw, sorted(present), font, w)
    return img
