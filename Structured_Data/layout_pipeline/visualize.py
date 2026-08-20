#!/usr/bin/env python3
"""Draw routed blocks over the page image, for eyeballing reading order.

    ocr_venvs/env_paddle/bin/python layout_pipeline/visualize.py \
        --results out/doc.json --image page.png --output overlay.png

Each block gets a box coloured by :class:`BlockType`, a badge with its reading
position, and a polyline threading the blocks in order -- so a wrong reading
order is visible as a line that jumps between columns instead of running down
one and then the other.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src.dual_extractor import BlockType, DualExtractionRouter
else:  # pragma: no cover
    from .src.dual_extractor import BlockType, DualExtractionRouter

FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

COLORS = {
    BlockType.TITLE:   (217, 70, 239),    # magenta
    BlockType.TEXT:    (37, 99, 235),     # blue
    BlockType.TABLE:   (245, 158, 11),    # amber
    BlockType.FORMULA: (16, 185, 129),    # green
    BlockType.VISION:  (239, 68, 68),     # red
    BlockType.UNKNOWN: (107, 114, 128),   # grey
}


def _font(size):
    from PIL import ImageFont

    for path in FONT_PATHS:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_overlay(image, blocks, show_path=True, dim=0.35):
    """Return a new RGB image with ``blocks`` drawn over ``image``."""
    from PIL import Image, ImageDraw

    base = image.convert("RGB")
    # Fade the page so the annotations read clearly against dense body text.
    if dim:
        white = Image.new("RGB", base.size, (255, 255, 255))
        base = Image.blend(base, white, dim)

    w, h = base.size
    scale = max(1.0, w / 1000.0)
    line_w = max(2, int(2.5 * scale))
    badge_r = int(17 * scale)
    font = _font(int(21 * scale))
    small = _font(int(15 * scale))

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)

    # Translucent fills first, so outlines and badges stay on top.
    for b in blocks:
        x1, y1, x2, y2 = [float(v) for v in b.bbox]
        colour = COLORS.get(b.block_type, COLORS[BlockType.UNKNOWN])
        od.rectangle([x1, y1, x2, y2], fill=colour + (38,))
    base = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
    d = ImageDraw.Draw(base)

    for b in blocks:
        x1, y1, x2, y2 = [float(v) for v in b.bbox]
        colour = COLORS.get(b.block_type, COLORS[BlockType.UNKNOWN])
        d.rectangle([x1, y1, x2, y2], outline=colour, width=line_w)
        # A pending stub gets a dashed inner border so it is obvious at a glance.
        if b.metadata.get("status") == "pending":
            for off in range(int(6 * scale), int(7 * scale) + 1):
                d.rectangle([x1 + off, y1 + off, x2 - off, y2 - off],
                            outline=colour, width=1)

    if show_path and len(blocks) > 1:
        centres = [((b.bbox[0] + b.bbox[2]) / 2, (b.bbox[1] + b.bbox[3]) / 2)
                   for b in blocks]
        d.line(centres, fill=(255, 255, 255), width=line_w + 3)
        d.line(centres, fill=(55, 65, 81), width=max(1, line_w - 1))
        for (ax, ay), (bx, by) in zip(centres, centres[1:]):
            # Small arrowhead at the midpoint of each hop.
            mx, my = (ax + bx) / 2, (ay + by) / 2
            dx, dy = bx - ax, by - ay
            n = max(1e-6, (dx * dx + dy * dy) ** 0.5)
            ux, uy = dx / n, dy / n
            s = 7 * scale
            d.polygon([(mx + ux * s, my + uy * s),
                       (mx - ux * s - uy * s * 0.7, my - uy * s + ux * s * 0.7),
                       (mx - ux * s + uy * s * 0.7, my - uy * s - ux * s * 0.7)],
                      fill=(17, 24, 39))

    # Order badges last so nothing covers them.
    for i, b in enumerate(blocks):
        x1, y1 = float(b.bbox[0]), float(b.bbox[1])
        colour = COLORS.get(b.block_type, COLORS[BlockType.UNKNOWN])
        cx, cy = x1 + badge_r, y1 + badge_r
        d.ellipse([cx - badge_r, cy - badge_r, cx + badge_r, cy + badge_r],
                  fill=colour, outline=(255, 255, 255), width=max(1, line_w - 1))
        label = str(i)
        tb = d.textbbox((0, 0), label, font=font)
        d.text((cx - (tb[2] - tb[0]) / 2, cy - (tb[3] - tb[1]) / 2 - tb[1]),
               label, font=font, fill=(255, 255, 255))
        # Only tag the non-prose types. TEXT is the large majority, and labelling
        # every one of them buries the page under overlapping captions when the
        # colour already carries the same information.
        if b.block_type is not BlockType.TEXT:
            d.text((cx + badge_r + 5 * scale, y1 + 3 * scale), b.block_type.value,
                   font=small, fill=colour, stroke_width=max(1, int(2 * scale)),
                   stroke_fill=(255, 255, 255))

    return _add_legend(base, blocks, scale)


def _add_legend(img, blocks, scale):
    from PIL import Image, ImageDraw

    present = []
    for b in blocks:
        if b.block_type not in present:
            present.append(b.block_type)
    pad = int(14 * scale)
    row = int(30 * scale)
    strip_h = pad * 2 + row * len(present)
    out = Image.new("RGB", (img.width, img.height + strip_h), (255, 255, 255))
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    font = _font(int(19 * scale))
    counts = {t: sum(1 for b in blocks if b.block_type == t) for t in present}
    y = img.height + pad
    for t in present:
        colour = COLORS.get(t, COLORS[BlockType.UNKNOWN])
        d.rectangle([pad, y + 4, pad + int(26 * scale), y + row - 6],
                    fill=colour + (0,) if False else colour)
        stub = " (dashed = step-2 stub)" if t in (
            BlockType.TABLE, BlockType.FORMULA, BlockType.VISION) else ""
        d.text((pad + int(36 * scale), y + 2),
               f"{t.value}  x{counts[t]}{stub}", font=font, fill=(17, 24, 39))
        y += row
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="pipeline JSON output")
    ap.add_argument("--image", required=True, help="the rendered page image")
    ap.add_argument("--output", required=True)
    ap.add_argument("--page", type=int, default=0,
                    help="index into results['pages'] when it holds several")
    ap.add_argument("--match", help="pick the page whose source contains this")
    ap.add_argument("--no_path", action="store_true",
                    help="omit the reading-order polyline")
    args = ap.parse_args(argv)

    from PIL import Image

    data = json.loads(open(args.results, encoding="utf-8").read())
    pages = data["pages"]
    page = next((p for p in pages if args.match in p["source"]), None) \
        if args.match else pages[args.page]
    if page is None:
        print(f"no page matching {args.match!r}", file=sys.stderr)
        return 2

    blocks = DualExtractionRouter().route(page["blocks"])
    img = draw_overlay(Image.open(args.image), blocks, show_path=not args.no_path)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    img.save(args.output)
    print(f"{len(blocks)} blocks -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
