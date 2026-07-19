"""SAM 3 promptable-concept segmentation engine.

Loads facebook/sam3 once, then segments an image against a list of open-vocabulary
text concepts. SAM 3 takes one concept phrase at a time and returns every matching
instance, so we loop over the vocabulary and aggregate the results.
"""
from __future__ import annotations

import colorsys
import io
import time
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

MODEL_ID = "facebook/sam3"

# Default incident-scene vocabulary. Editable from the UI.
DEFAULT_CONCEPTS = [
    "car", "truck", "bus", "motorcycle", "bicycle", "person",
    "fire", "smoke", "debris", "traffic cone", "traffic light",
    "broken glass", "spilled liquid", "ladder", "forklift", "helmet",
]


@dataclass
class Detection:
    label: str
    score: float
    box: list[float]      # xyxy, absolute pixels
    mask: np.ndarray      # bool HxW
    color: tuple[int, int, int]


class Sam3Engine:
    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # -- lazy load ---------------------------------------------------------
    def load(self) -> None:
        if self._model is not None:
            return
        from transformers import Sam3Model, Sam3Processor

        self._model = Sam3Model.from_pretrained(MODEL_ID, device_map="auto")
        self._model.eval()
        self._processor = Sam3Processor.from_pretrained(MODEL_ID)

    @property
    def loaded(self) -> bool:
        return self._model is not None

    # -- inference ---------------------------------------------------------
    @torch.no_grad()
    def _segment_concept(self, image: Image.Image, concept: str, threshold: float) -> list[Detection]:
        proc = self._processor
        inputs = proc(images=image, text=concept, return_tensors="pt").to(self._model.device)
        outputs = self._model(**inputs)
        results = proc.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        dets: list[Detection] = []
        masks = results.get("masks", [])
        boxes = results.get("boxes", [])
        scores = results.get("scores", [])
        for mask, box, score in zip(masks, boxes, scores):
            m = _to_numpy(mask).astype(bool)
            dets.append(
                Detection(
                    label=concept,
                    score=float(_to_numpy(score)),
                    box=[float(x) for x in _to_numpy(box).reshape(-1)[:4]],
                    mask=m,
                    color=(0, 0, 0),  # assigned later, per label
                )
            )
        return dets

    def analyze(
        self, image: Image.Image, concepts: list[str], threshold: float = 0.5
    ) -> tuple[list[Detection], float]:
        self.load()
        image = image.convert("RGB")
        t0 = time.time()
        all_dets: list[Detection] = []
        for concept in concepts:
            concept = concept.strip()
            if not concept:
                continue
            all_dets.extend(self._segment_concept(image, concept, threshold))

        # assign a stable color per unique label
        labels = sorted({d.label for d in all_dets})
        palette = _make_palette(len(labels))
        color_map = {lab: palette[i] for i, lab in enumerate(labels)}
        for d in all_dets:
            d.color = color_map[d.label]

        elapsed = time.time() - t0
        return all_dets, elapsed


# -- rendering -------------------------------------------------------------
def render_overlay(image: Image.Image, dets: list[Detection], alpha: float = 0.45) -> Image.Image:
    base = image.convert("RGB")
    arr = np.asarray(base).astype(np.float32)

    # blend masks
    for d in sorted(dets, key=lambda x: x.score):  # draw low-conf first
        color = np.array(d.color, dtype=np.float32)
        m = d.mask
        if m.shape[:2] != arr.shape[:2]:
            continue
        arr[m] = arr[m] * (1 - alpha) + color * alpha

    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(out)
    font = _load_font(max(14, out.width // 70))

    for d in sorted(dets, key=lambda x: x.score):
        x0, y0, x1, y1 = d.box
        draw.rectangle([x0, y0, x1, y1], outline=d.color, width=3)
        tag = f"{d.label} {d.score:.0%}"
        tw, th = _text_size(draw, tag, font)
        ty = max(0, y0 - th - 4)
        draw.rectangle([x0, ty, x0 + tw + 8, ty + th + 4], fill=d.color)
        draw.text((x0 + 4, ty + 2), tag, fill=_readable_text(d.color), font=font)

    return out


def image_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


# -- helpers ---------------------------------------------------------------
def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _make_palette(n: int) -> list[tuple[int, int, int]]:
    if n == 0:
        return []
    out = []
    for i in range(n):
        h = (i * 0.61803398875) % 1.0  # golden-ratio hue spacing
        r, g, b = colorsys.hsv_to_rgb(h, 0.72, 0.98)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    return out


def _readable_text(bg: tuple[int, int, int]) -> tuple[int, int, int]:
    lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    return (0, 0, 0) if lum > 140 else (255, 255, 255)


def _load_font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_size(draw, text, font) -> tuple[int, int]:
    try:
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return r - l, b - t
    except Exception:
        return draw.textlength(text, font=font), font.size


# module-level singleton
engine = Sam3Engine()
