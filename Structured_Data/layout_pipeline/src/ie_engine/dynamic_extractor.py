"""Discover a document's own schema, rather than imposing one.

Takes the ordered blocks the layout pipeline produced, renders them back into
text, asks a local Llama to describe the structure it sees as JSON, and turns
whatever comes back into a :class:`DynamicElement` tree.

    [SemanticBlock, ...] -> render -> LLM -> open JSON -> DynamicDocument

No target schema is supplied at any point. The tree accepts any tag and any
attribute key, so a key the model invents for one document does not have to
exist in any other.

Nothing here raises on a bad model response. An unreachable endpoint, empty
output, prose instead of JSON, or JSON of an unexpected shape all produce an
empty :class:`DynamicDocument` with the reason recorded in ``metadata`` — a
document that failed extraction has to be distinguishable from one that
genuinely had no content, and neither should stop a batch.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, List, Optional, Sequence

from .llm_client import (SCHEMA_DISCOVERY_SYSTEM_PROMPT, LocalLLMClient,
                         extract_json)
from .node_schema import (DynamicDocument, DynamicElement, document_from_json,
                          element_from_json)

log = logging.getLogger(__name__)

#: Characters of document text per LLM call. Long documents are split rather
#: than truncated, so a table on the last page is not silently dropped.
DEFAULT_CHUNK_CHARS = 12000

#: Block types worth sending. Figures carry no narrative and their OCR text is
#: axis-label noise, which costs context and yields nothing.
DEFAULT_INCLUDE_TYPES = ("TITLE", "TEXT", "TABLE", "FORMULA", "UNKNOWN")


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _type_name(block: Any) -> str:
    bt = _get(block, "block_type", "")
    return getattr(bt, "value", str(bt)).upper()


def block_text(block: Any) -> str:
    """The text a block contributes to the prompt."""
    content = _get(block, "parsed_content", "") or ""
    if isinstance(content, str) and content.strip():
        return content.strip()
    meta = _get(block, "metadata", {}) or {}
    return (meta.get("ocr_fallback_text") or "").strip()


def render_blocks(blocks: Sequence[Any],
                  include: Sequence[str] = DEFAULT_INCLUDE_TYPES) -> str:
    """Flatten ordered blocks back into plain text for the prompt.

    Headings are marked so the model can see the document's own sectioning,
    which is a strong hint about grouping; everything else goes in verbatim and
    in reading order.
    """
    parts: List[str] = []
    for block in blocks:
        btype = _type_name(block)
        if include and btype not in include:
            continue
        text = block_text(block)
        if not text:
            continue
        parts.append(f"## {text}" if btype == "TITLE" else text)
    return "\n\n".join(parts)


def chunk_text(text: str, limit: int = DEFAULT_CHUNK_CHARS) -> List[str]:
    """Split on paragraph boundaries, never mid-paragraph."""
    if len(text) <= limit:
        return [text] if text.strip() else []
    chunks, current = [], ""
    for para in text.split("\n\n"):
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) > limit and current:
            chunks.append(current)
            current = para
        else:
            current = candidate
    if current.strip():
        chunks.append(current)
    return chunks


class DynamicInformationExtractor:
    """Infer a document's entity tree with a local LLM.

    ``client`` may be any object exposing ``complete_json(system, user)``, which
    is how the tests run without a model. Passing nothing builds a
    :class:`LocalLLMClient` against the default local endpoint.
    """

    def __init__(self, client: Optional[Any] = None,
                 system_prompt: str = SCHEMA_DISCOVERY_SYSTEM_PROMPT,
                 chunk_chars: int = DEFAULT_CHUNK_CHARS,
                 include_types: Sequence[str] = DEFAULT_INCLUDE_TYPES,
                 root_tag: str = "document"):
        self.client = client if client is not None else LocalLLMClient()
        self.system_prompt = system_prompt
        self.chunk_chars = chunk_chars
        self.include_types = tuple(include_types)
        self.root_tag = root_tag

    # -- prompting ---------------------------------------------------------- #
    def build_prompt(self, text: str, source: Optional[str] = None) -> str:
        header = f"Document: {source}\n\n" if source else ""
        return (f"{header}Infer the structure of the following document and "
                f"return it as JSON.\n\n---\n{text}\n---")

    def _call(self, text: str, source: Optional[str]) -> tuple:
        """``(parsed_json_or_None, error_or_None)``."""
        prompt = self.build_prompt(text, source)
        try:
            parsed = self.client.complete_json(self.system_prompt, prompt)
        except Exception as exc:  # noqa: BLE001 - a batch must survive one failure
            log.warning("schema discovery failed: %s", exc)
            return None, f"{type(exc).__name__}: {exc}"
        if parsed is None:
            return None, getattr(self.client, "last_error", None) or \
                "model returned no usable JSON"
        return parsed, None

    # -- entry points -------------------------------------------------------- #
    def extract_from_text(self, text: str,
                          source: Optional[str] = None) -> DynamicDocument:
        """Discover the schema of a plain string."""
        chunks = chunk_text(text, self.chunk_chars)
        if not chunks:
            return DynamicDocument(root=DynamicElement(self.root_tag),
                                   source=source,
                                   metadata={"status": "empty",
                                             "reason": "no text to extract",
                                             "chunks": 0})

        results, errors = [], []
        for chunk in chunks:
            parsed, error = self._call(chunk, source)
            if error:
                errors.append(error)
            elif parsed is not None:
                results.append(parsed)

        if not results:
            return DynamicDocument(
                root=DynamicElement(self.root_tag), source=source,
                metadata={"status": "failed", "chunks": len(chunks),
                          "reason": errors[0] if errors else "no output",
                          "errors": errors})

        if len(results) == 1:
            doc = document_from_json(results[0], source=source,
                                     root_tag=self.root_tag)
        else:
            # Several chunks: keep each discovered tree intact under a shared
            # root rather than merging them, since merging would have to invent
            # a reconciliation rule the model never agreed to.
            root = DynamicElement(self.root_tag)
            for i, payload in enumerate(results):
                part = document_from_json(payload, root_tag=f"part-{i + 1}").root
                root.add(part)
            doc = DynamicDocument(root=root, source=source)

        doc.metadata.update({
            "status": "extracted" if not errors else "partial",
            "chunks": len(chunks),
            "chunks_ok": len(results),
            "elements": doc.elements,
        })
        if errors:
            doc.metadata["errors"] = errors
        return doc

    def extract(self, blocks: Sequence[Any],
                source: Optional[str] = None) -> DynamicDocument:
        """Discover the schema of a block sequence from the layout pipeline."""
        text = render_blocks(blocks, self.include_types)
        doc = self.extract_from_text(text, source=source)
        doc.metadata.setdefault("blocks", len(list(blocks)))
        return doc

    def extract_pages(self, pages: Iterable[Iterable[Any]],
                      source: Optional[str] = None) -> DynamicDocument:
        """Discover the schema across a whole document's pages."""
        return self.extract([b for page in pages for b in page], source=source)

    # -- for callers holding raw model output --------------------------------- #
    @staticmethod
    def parse_response(raw: str, source: Optional[str] = None,
                       root_tag: str = "document") -> DynamicDocument:
        """Turn a raw model response into a tree, tolerating fences and prose."""
        parsed = extract_json(raw) if isinstance(raw, str) else raw
        if parsed is None:
            return DynamicDocument(root=DynamicElement(root_tag), source=source,
                                   metadata={"status": "failed",
                                             "reason": "unparseable response"})
        doc = document_from_json(parsed, source=source, root_tag=root_tag)
        doc.metadata["status"] = "extracted"
        return doc
