"""Image-to-LaTeX for cropped formula regions.

A thin abstraction over a swappable local model. The default backend is
**PP-FormulaNet**, chosen because its weights are already cached on this box
(``~/.paddlex/official_models/PP-FormulaNet*``) and it runs in the same
``env_paddle`` as the rest of the pipeline -- no extra venv, no download, no
second CUDA stack. Texify / UniMERNet style transformers checkpoints are
supported through the same interface for anyone without paddle.

The contract that matters to the router: :meth:`FormulaExtractor.extract_latex`
returns a LaTeX string, or ``None``. It never raises. A missing dependency, an
un-downloadable checkpoint, a CUDA error and an unreadable crop all land in the
same place -- ``None`` -- because the caller's response to all of them is the
same: fall back to the OCR text and say so.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)

DEFAULT_PADDLE_MODEL = "PP-FormulaNet_plus-M"
DEFAULT_DEVICE = "gpu:0"

# Wrappers a model may put around the expression. Stripped only when they
# enclose the *whole* string, so an inline $..$ inside a longer expression is
# left alone.
_DELIMITERS = [
    ("$$", "$$"),
    (r"\[", r"\]"),
    (r"\(", r"\)"),
    ("$", "$"),
    (r"\begin{equation*}", r"\end{equation*}"),
    (r"\begin{equation}", r"\end{equation}"),
    (r"\begin{displaymath}", r"\end{displaymath}"),
]
_FENCE_RE = re.compile(r"^```(?:latex|tex|math)?\s*\n?(.*?)\n?```$", re.DOTALL)


def strip_delimiters(latex: str) -> str:
    """Remove math delimiters the model wrapped around the whole expression.

    The router adds its own ``$$`` fences, so a model that already emitted them
    would otherwise produce ``$$ $$ x $$ $$``.
    """
    s = (latex or "").strip()
    fence = _FENCE_RE.match(s)
    if fence:
        s = fence.group(1).strip()
    changed = True
    while changed and s:
        changed = False
        for open_d, close_d in _DELIMITERS:
            if len(s) > len(open_d) + len(close_d) \
                    and s.startswith(open_d) and s.endswith(close_d):
                s = s[len(open_d):-len(close_d)].strip()
                changed = True
                break
    return s


class PaddleFormulaBackend:
    """PP-FormulaNet via paddleocr. Local, GPU, weights already cached."""

    def __init__(self, model_name: str = DEFAULT_PADDLE_MODEL,
                 device: str = DEFAULT_DEVICE):
        self.model_name = model_name
        self.device = device
        self._model = None
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    @property
    def name(self) -> str:
        return f"paddle:{self.model_name}"

    def _load(self):
        if self._model is None:
            from paddleocr import FormulaRecognition

            self._model = FormulaRecognition(model_name=self.model_name,
                                             device=self.device)
        return self._model

    def __call__(self, image) -> str:
        import numpy as np

        model = self._load()
        # paddle predictors take str paths or ndarrays in OpenCV BGR order.
        arr = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
        res = model.predict(arr)
        r = res[0] if isinstance(res, list) else res
        payload = r.json.get("res", r.json) if hasattr(r, "json") else dict(r)
        return payload.get("rec_formula") or ""


class TransformersFormulaBackend:
    """Texify / UniMERNet style vision-encoder-decoder checkpoints.

    Not the default: nothing of this kind is cached on this machine, so
    constructing it would trigger a multi-GB download mid-pipeline.
    """

    def __init__(self, model_id: str, device: str = "cuda:0",
                 max_new_tokens: int = 512):
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None

    @property
    def name(self) -> str:
        return f"transformers:{self.model_id}"

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self.model_id)
            dtype = torch.float16 if str(self.device).startswith("cuda") \
                else torch.float32
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.model_id, dtype=dtype).to(self.device).eval()
        return self._model, self._processor

    def __call__(self, image) -> str:
        import torch

        model, processor = self._load()
        inputs = processor(images=image.convert("RGB"),
                           return_tensors="pt").to(self.device)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        return processor.batch_decode(ids, skip_special_tokens=True)[0]


class FormulaExtractor:
    """Turn a cropped formula image into LaTeX.

    ``backend`` may be any callable ``image -> str``, which is what makes this
    testable without model weights. Pass ``backend_name='transformers'`` with a
    ``model_id`` to use a HF checkpoint instead of paddle.
    """

    #: Below this, a crop is too small to hold a legible expression.
    MIN_SIDE_PX = 6

    def __init__(self, backend: Optional[Callable] = None,
                 backend_name: str = "paddle",
                 model_name: str = DEFAULT_PADDLE_MODEL,
                 model_id: Optional[str] = None,
                 device: str = DEFAULT_DEVICE):
        self.backend_name = backend_name
        self.model_name = model_name
        self.device = device
        self.last_error: Optional[str] = None
        self._failed = False

        if backend is not None:
            self._backend = backend
        elif backend_name == "transformers":
            if not model_id:
                raise ValueError("backend_name='transformers' needs a model_id")
            self._backend = TransformersFormulaBackend(model_id, device=device)
        elif backend_name == "paddle":
            self._backend = PaddleFormulaBackend(model_name, device=device)
        else:
            raise ValueError(f"unknown backend {backend_name!r}")

    @property
    def backend(self) -> Any:
        return self._backend

    @property
    def name(self) -> str:
        return getattr(self._backend, "name", getattr(
            self._backend, "__name__", type(self._backend).__name__))

    def extract_latex(self, crop_img) -> Optional[str]:
        """LaTeX for ``crop_img``, or ``None`` if it could not be produced.

        Never raises: every failure mode is reported as ``None`` with the reason
        left on :attr:`last_error`, because the router's response to all of them
        is identical.
        """
        self.last_error = None
        if crop_img is None:
            self.last_error = "no crop image"
            return None
        size = getattr(crop_img, "size", None)
        if size and min(size) < self.MIN_SIDE_PX:
            self.last_error = f"crop too small: {size}"
            return None
        # One hard failure (missing dependency, absent weights) will repeat for
        # every block, so stop paying for it after the first.
        if self._failed:
            self.last_error = "backend previously failed to load"
            return None

        try:
            raw = self._backend(crop_img)
        except ImportError as exc:
            self._failed = True
            self.last_error = f"backend unavailable: {exc}"
            log.warning("formula backend unavailable: %s", exc)
            return None
        except Exception as exc:  # noqa: BLE001 - fallback is the contract
            self.last_error = f"{type(exc).__name__}: {exc}"
            log.warning("formula extraction failed: %s", self.last_error)
            return None

        if not isinstance(raw, str):
            self.last_error = f"backend returned {type(raw).__name__}, expected str"
            return None
        latex = strip_delimiters(raw)
        if not latex:
            self.last_error = "backend returned an empty expression"
            return None
        # Leftover punctuation is not an expression. "$$$$" unwraps to "$$",
        # which would otherwise be emitted as a formula and render as stray
        # delimiters. Every real expression has an alphanumeric somewhere --
        # even \Theta and \sum_{i=1}.
        if not any(c.isalnum() for c in latex):
            self.last_error = f"no expression content in {latex!r}"
            return None
        return latex
