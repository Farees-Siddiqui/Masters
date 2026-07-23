"""Common interface over several OCR engines so they can be compared 1:1.

Each engine is a function (image_path, out_dir) -> extracted text. Engines are
imported lazily and cached, so a missing dependency only disables that engine.

Unlimited-OCR parameters follow ../docs/unlimited-ocr.html: gundam mode for
single images, exact prompt string, documented repetition guards.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

_UNLIMITED = None
_PADDLE = None
_EASYOCR = None


@dataclass
class OCRResult:
    engine: str
    text: str
    seconds: float
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


# --- Unlimited-OCR (baidu/Unlimited-OCR, 3B VLM, GPU required) ---

def _load_unlimited():
    global _UNLIMITED
    if _UNLIMITED is None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "baidu/Unlimited-OCR", trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            "baidu/Unlimited-OCR",
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
        )
        _UNLIMITED = (tokenizer, model.eval().cuda())
    return _UNLIMITED


def run_unlimited(image_path: Path, out_dir: Path) -> str:
    tokenizer, model = _load_unlimited()
    out_dir.mkdir(parents=True, exist_ok=True)
    result = model.infer(
        tokenizer,
        prompt="<image>document parsing.",  # fixed training prompt — do not edit
        image_file=str(image_path),
        output_path=str(out_dir),
        base_size=1024, image_size=640, crop_mode=True,  # gundam mode
        max_length=32768,
        no_repeat_ngram_size=35, ngram_window=128,
        save_results=True,
    )
    if isinstance(result, str) and result.strip():
        return result
    # infer() may return structured data; save_results always writes the parse
    saved = sorted(
        (p for p in out_dir.rglob("*") if p.suffix in {".md", ".mmd", ".txt"}),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if saved:
        return saved[0].read_text(encoding="utf-8")
    raise RuntimeError(f"Unlimited-OCR produced no readable output in {out_dir}")


# --- PaddleOCR ---

def _load_paddle():
    global _PADDLE
    if _PADDLE is None:
        from paddleocr import PaddleOCR

        # enable_mkldnn=False: oneDNN backend crashes CPU inference on Windows
        _PADDLE = PaddleOCR(lang="en", use_doc_orientation_classify=False,
                            use_doc_unwarping=False, enable_mkldnn=False)
    return _PADDLE


def run_paddle(image_path: Path, out_dir: Path) -> str:
    ocr = _load_paddle()
    pages = ocr.predict(str(image_path))
    lines: list[str] = []
    for page in pages:
        lines.extend(page.get("rec_texts", []))
    return "\n".join(lines)


# --- Tesseract ---

def run_tesseract(image_path: Path, out_dir: Path) -> str:
    import pytesseract
    from PIL import Image

    return pytesseract.image_to_string(Image.open(image_path))


# --- EasyOCR ---

def _load_easyocr():
    global _EASYOCR
    if _EASYOCR is None:
        import easyocr

        _EASYOCR = easyocr.Reader(["en"])
    return _EASYOCR


def run_easyocr(image_path: Path, out_dir: Path) -> str:
    reader = _load_easyocr()
    return "\n".join(reader.readtext(str(image_path), detail=0, paragraph=True))


ENGINES = {
    "unlimited": run_unlimited,
    "paddle": run_paddle,
    "tesseract": run_tesseract,
    "easyocr": run_easyocr,
}


def run_engine(name: str, image_path: Path, out_dir: Path) -> OCRResult:
    """Run one engine, timing it and turning failures into a reportable result."""
    start = time.perf_counter()
    try:
        text = ENGINES[name](image_path, out_dir / name)
        return OCRResult(name, text, time.perf_counter() - start)
    except ImportError as exc:
        return OCRResult(name, "", 0.0, error=f"not installed ({exc.name})")
    except Exception as exc:  # noqa: BLE001 - comparison should survive one engine failing
        return OCRResult(name, "", time.perf_counter() - start, error=str(exc))
