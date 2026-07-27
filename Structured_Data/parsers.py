"""Pluggable document -> text parsers.

The extractor only needs a string of text. Which parser produces it is a
swappable choice — the same seam as the LLM backends:

- pypdf   : fast, naive text layer. Garbles two-column layout and flattens
            tables. Fine for simple/born-digital PDFs.
- mistral : Mistral OCR (`mistral-ocr-latest`). Returns clean markdown with
            correct reading order and tables rendered as markdown tables.
            Works on scans and born-digital PDFs alike. Paid API.

`.txt`/`.md` files are read directly regardless of parser.

Mistral OCR results are cached next to the source as `<name>.mistral.md` so
re-runs don't re-pay for OCR (delete the cache or pass use_cache=False to
force a fresh call).
"""

from __future__ import annotations

import os
from pathlib import Path

_TEXT_SUFFIXES = {".txt", ".md"}


def _mistral_key() -> str:
    """MISTRAL_API_KEY from env, then a local key file, then the AST project's."""
    key = os.environ.get("MISTRAL_API_KEY")
    if key:
        return key.strip()
    here = Path(__file__).resolve().parent
    candidates = [
        here / "MISTRAL_API_KEY.txt",
        here.parent / "AST" / "API_KEY.txt",  # the AST project's key file
    ]
    for path in candidates:
        if path.exists():
            raw = path.read_text(encoding="utf-8").strip()
            # Tolerate a leading "1\t" / "1 " line-number from a paste.
            for sep in ("\t", " "):
                head = raw.split(sep, 1)
                if len(head) == 2 and head[0].isdigit():
                    raw = head[1]
                    break
            return raw.strip()
    raise RuntimeError(
        "No Mistral API key. Set MISTRAL_API_KEY or create MISTRAL_API_KEY.txt."
    )


def parse_pypdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def parse_mistral(path: Path, use_cache: bool = True) -> str:
    cache = path.with_suffix(".mistral.md")
    if use_cache and cache.exists():
        return cache.read_text(encoding="utf-8")

    try:  # mistralai 1.x
        from mistralai import Mistral
    except ImportError:  # mistralai 2.x restructured the package
        from mistralai.client import Mistral

    client = Mistral(api_key=_mistral_key())
    uploaded = client.files.upload(
        file={"file_name": path.name, "content": path.read_bytes()},
        purpose="ocr",
    )
    signed = client.files.get_signed_url(file_id=uploaded.id)
    resp = client.ocr.process(
        model="mistral-ocr-latest",
        document={"type": "document_url", "document_url": signed.url},
    )
    pages = [getattr(p, "markdown", "") or "" for p in getattr(resp, "pages", []) or []]
    markdown = "\n\n".join(pages)
    cache.write_text(markdown, encoding="utf-8")
    return markdown


def parse(path: Path, parser: str = "pypdf") -> str:
    """Return the document's text using the chosen parser."""
    if path.suffix.lower() in _TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8")
    if parser == "pypdf":
        return parse_pypdf(path)
    if parser == "mistral":
        return parse_mistral(path)
    raise ValueError(f"Unknown parser: {parser!r}")
