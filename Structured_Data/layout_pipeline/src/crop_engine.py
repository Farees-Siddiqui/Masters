"""Crop block regions out of a page image.

Step 2 of the Dual Extraction Engine. Tables, formulas and figures cannot be
parsed from OCR text alone -- each needs its own model fed the actual pixels, so
the router slices the region out and hands downstream steps a file path.

Deliberately free of paddle and of the router, so it is testable on a blank
image and reusable anywhere a bbox needs turning into a PNG.
"""

from __future__ import annotations

import math
import os
from typing import Any, Optional, Sequence

DEFAULT_PADDING = 5


class CropEngine:
    """Turn ``[x_min, y_min, x_max, y_max]`` regions into saved image crops.

    ``padding`` widens every box by a few pixels: layout boxes sit tight against
    the glyphs, and a downstream formula or table model does better with a
    little whitespace than with clipped ascenders.
    """

    def __init__(self, padding: int = DEFAULT_PADDING, image_format: str = "PNG",
                 assume_bgr: bool = False):
        self.padding = padding
        self.image_format = image_format
        # numpy arrays coming from OpenCV / paddle are BGR; arrays produced from
        # PIL are RGB. Guessing wrong silently swaps red and blue in every saved
        # figure, so it is an explicit switch rather than a heuristic.
        self.assume_bgr = assume_bgr

    # -- helpers ----------------------------------------------------------- #
    def _to_pil(self, page_image: Any):
        from PIL import Image

        if hasattr(page_image, "shape"):  # numpy ndarray
            arr = page_image
            if self.assume_bgr and getattr(arr, "ndim", 0) == 3 and arr.shape[2] >= 3:
                arr = arr[:, :, ::-1]
            return Image.fromarray(arr)
        if isinstance(page_image, Image.Image):
            return page_image
        raise TypeError(
            f"page_image must be a PIL image or numpy array, got "
            f"{type(page_image).__name__}")

    def clamp_bbox(self, bbox: Sequence[float], width: int, height: int,
                   padding: Optional[int] = None) -> tuple:
        """Pad, normalise and clamp a bbox to the page. Returns ints.

        Handles the three ways a detector box goes wrong: inverted corners
        (x2 < x1), fractional coordinates, and boxes that run past the page
        edge. The result is always inside ``[0, width] x [0, height]``.
        """
        if bbox is None or len(bbox) < 4:
            raise ValueError(f"bbox must have 4 values, got {bbox!r}")
        pad = self.padding if padding is None else padding

        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
        # Normalise inverted corners before padding, or padding shrinks the box.
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        x1 = int(math.floor(x1 - pad))
        y1 = int(math.floor(y1 - pad))
        x2 = int(math.ceil(x2 + pad))
        y2 = int(math.ceil(y2 + pad))

        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        return x1, y1, x2, y2

    # -- api ---------------------------------------------------------------- #
    def crop_block(self, page_image: Any, bbox: Sequence[float],
                   padding: int = DEFAULT_PADDING):
        """Return the padded, clamped crop of ``bbox`` from ``page_image``.

        Raises ``ValueError`` when the region has no area once clamped -- a box
        entirely off the page. Callers in the router treat that as a per-block
        failure and carry on rather than aborting the document.
        """
        img = self._to_pil(page_image)
        width, height = img.size
        x1, y1, x2, y2 = self.clamp_bbox(bbox, width, height, padding)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"empty crop region for bbox={list(bbox)[:4]} on a "
                f"{width}x{height} page (clamped to {(x1, y1, x2, y2)})")
        return img.crop((x1, y1, x2, y2))

    def save_crop(self, crop_img, output_path: str) -> str:
        """Write ``crop_img`` to ``output_path``, creating parent dirs."""
        parent = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(parent, exist_ok=True)
        img = crop_img
        # PNG cannot hold a float or 16-bit-per-channel mode from some arrays.
        if img.mode not in ("RGB", "RGBA", "L", "P"):
            img = img.convert("RGB")
        img.save(output_path, format=self.image_format)
        return output_path

    def crop_and_save(self, page_image: Any, bbox: Sequence[float],
                      output_path: str, padding: int = DEFAULT_PADDING) -> str:
        """crop_block + save_crop in one call."""
        return self.save_crop(self.crop_block(page_image, bbox, padding),
                              output_path)
