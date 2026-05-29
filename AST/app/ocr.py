"""Mistral OCR wrapper.

Uploads a PDF to Mistral's Files API, then runs OCR and returns the
concatenated markdown across all pages.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from mistralai import Mistral


@dataclass
class OCRResult:
    markdown: str
    page_count: int
    pages: list[str]


def _load_api_key() -> str:
    key = os.environ.get("MISTRAL_API_KEY")
    if key:
        return key.strip()
    # Fallback: API_KEY.txt at repo root (one level above app/)
    repo_root = Path(__file__).resolve().parent.parent
    key_file = repo_root / "API_KEY.txt"
    if key_file.exists():
        # File may have leading line numbers from a paste; take last token.
        raw = key_file.read_text(encoding="utf-8").strip()
        # Strip a leading "1\t" or "1 " if present.
        for sep in ("\t", " "):
            if sep in raw and raw.split(sep, 1)[0].isdigit():
                raw = raw.split(sep, 1)[1]
                break
        return raw.strip()
    raise RuntimeError(
        "No Mistral API key found. Set MISTRAL_API_KEY or create API_KEY.txt at the repo root."
    )


def run_ocr_on_pdf(pdf_bytes: bytes, filename: str = "document.pdf") -> OCRResult:
    """Send a PDF to Mistral OCR and return markdown per page."""
    client = Mistral(api_key=_load_api_key())

    uploaded = client.files.upload(
        file={"file_name": filename, "content": pdf_bytes},
        purpose="ocr",
    )
    signed = client.files.get_signed_url(file_id=uploaded.id)

    resp = client.ocr.process(
        model="mistral-ocr-latest",
        document={"type": "document_url", "document_url": signed.url},
    )

    pages: list[str] = []
    for page in getattr(resp, "pages", []) or []:
        md = getattr(page, "markdown", None) or ""
        pages.append(md)

    return OCRResult(
        markdown="\n\n".join(pages),
        page_count=len(pages),
        pages=pages,
    )
