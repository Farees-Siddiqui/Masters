"""Project extracted semantic blocks into a structured XML document.

The Markdown output is for reading; this is for machines. Every block becomes an
element that keeps its provenance — the block id, the pixel box it came from and
whether its extraction actually succeeded — so a consumer can trace any fragment
back to the region of the page it was read from, and can tell a parsed table
from one that fell back to OCR text.

    [SemanticBlock, ...]  ->  XMLProjector.project_to_xml()  ->  str

Tag mapping:

===========  ==========================================================
BlockType    element
===========  ==========================================================
TITLE        ``<heading level="1|2|3">``
TEXT         ``<paragraph>``
FORMULA      ``<formula format="latex" status="...">`` holding ``$$…$$``
TABLE        ``<table format="markdown|html">``
VISION       ``<figure src="…" alt="…">``
UNKNOWN      ``<paragraph>`` (matches how the router extracts it)
===========  ==========================================================

Escaping is ElementTree's, not hand-rolled: ``&``, ``<``, ``>`` in text and
additionally ``"`` in attributes. Raw HTML from a table is stored as escaped
text rather than CDATA — ElementTree cannot emit CDATA, and an escaped string
parses back byte-identical, which is what matters for round-tripping.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from typing import Any, Iterable, List, Optional, Sequence

from .dual_extractor import BlockType

#: Detector label -> heading level. Anything else falls back to DEFAULT_LEVEL.
HEADING_LEVELS = {
    "doc_title": 1,
    "title": 1,
    "chapter_title": 2,
    "paragraph_title": 2,
    "sub_title": 3,
}
DEFAULT_LEVEL = 2

# XML 1.0 forbids most control characters outright; they cannot be escaped, only
# removed. OCR of a noisy scan does occasionally emit them, and one such byte
# makes the whole document unparseable.
_ILLEGAL = re.compile(
    "[^\u0009\u000A\u000D\u0020-\uD7FF\uE000-\uFFFD"
    "\U00010000-\U0010FFFF]")


def xml_safe(text: Optional[str]) -> str:
    """Drop characters XML 1.0 cannot represent. Escaping is ElementTree's job."""
    return _ILLEGAL.sub("", text or "")


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _meta(block: Any) -> dict:
    return _get(block, "metadata", {}) or {}


def _block_type(block: Any) -> BlockType:
    bt = _get(block, "block_type", BlockType.UNKNOWN)
    if isinstance(bt, BlockType):
        return bt
    try:
        return BlockType(str(bt))
    except ValueError:
        return BlockType.UNKNOWN


def format_bbox(bbox: Optional[Sequence[float]]) -> str:
    """``x1,y1,x2,y2`` rounded to one decimal."""
    if not bbox or len(bbox) < 4:
        return ""
    return ",".join(f"{float(v):.1f}" for v in bbox[:4])


def infer_title(blocks: Iterable[Any], default: str = "Document") -> str:
    """First document-level heading, for use as the ``title`` attribute."""
    for b in blocks:
        if _block_type(b) is BlockType.TITLE and \
                _meta(b).get("label") in ("doc_title", "title"):
            text = " ".join((_get(b, "parsed_content", "") or "").split())
            if text:
                return text
    return default


class XMLProjector:
    """Serialise ordered :class:`SemanticBlock` objects to XML.

    Accepts real ``SemanticBlock`` instances or the plain dicts produced by
    ``SemanticBlock.to_dict``, so a saved ``*.blocks.json`` can be re-projected
    without re-running the pipeline.
    """

    def __init__(self, include_label: bool = True, indent: str = "  "):
        #: Emit the raw detector label alongside the mapped type. Cheap, and it
        #: is the only way to tell an abstract from a footnote after mapping.
        self.include_label = include_label
        self.indent = indent

    # -- element construction ---------------------------------------------- #
    def _provenance(self, el: ET.Element, block: Any) -> None:
        """Attach ``id`` / ``bbox`` / ``status`` (and ``label``) to ``el``."""
        block_id = _get(block, "block_id", None)
        if block_id is not None:
            el.set("id", str(block_id))
        bbox = format_bbox(_get(block, "bbox", None))
        if bbox:
            el.set("bbox", bbox)
        status = _meta(block).get("status")
        if status:
            el.set("status", str(status))
        if self.include_label:
            label = _meta(block).get("label")
            if label:
                el.set("label", str(label))

    def block_to_element(self, block: Any) -> ET.Element:
        """One block -> one element, provenance attached."""
        btype = _block_type(block)
        meta = _meta(block)
        content = xml_safe(_get(block, "parsed_content", "") or "")

        if btype is BlockType.TITLE:
            el = ET.Element("heading")
            el.set("level", str(HEADING_LEVELS.get(meta.get("label"),
                                                   DEFAULT_LEVEL)))
            el.text = content

        elif btype is BlockType.FORMULA:
            el = ET.Element("formula")
            el.set("format", "latex")
            # The $$…$$ wrapper is kept: it is what the Markdown carries, and
            # dropping it here would make the two outputs disagree.
            el.text = content
            latex = meta.get("latex")
            if latex:
                el.set("latex", xml_safe(latex))

        elif btype is BlockType.TABLE:
            el = ET.Element("table")
            fmt = meta.get("table_format")
            el.set("format", str(fmt) if fmt else "none")
            for key, attr in (("table_rows", "rows"),
                              ("table_columns", "columns")):
                if meta.get(key) is not None:
                    el.set(attr, str(meta[key]))
            if meta.get("merged_cells"):
                el.set("merged_cells", "true")
            el.text = content

        elif btype is BlockType.VISION:
            el = ET.Element("figure")
            src = meta.get("figure_rel_path") or ""
            if src:
                el.set("src", xml_safe(src))
            alt = meta.get("caption_text") or meta.get("ocr_fallback_text") or ""
            el.set("alt", xml_safe(" ".join(alt.split())) or "Figure")
            # A figure carries no prose of its own; the caption lives in alt.
            if not src:
                el.text = content

        else:  # TEXT and UNKNOWN
            el = ET.Element("paragraph")
            if btype is BlockType.UNKNOWN:
                el.set("unmapped", "true")
            el.text = content

        self._provenance(el, block)
        return el

    # -- document ----------------------------------------------------------- #
    def project_to_xml(self, blocks: List[Any],
                       doc_title: str = "Document") -> str:
        """Build the XML document. Block order is preserved exactly.

        Blocks are grouped into ``<page number="N">`` in the order they arrive;
        a page number that reappears after another page opens a second element
        rather than being merged, so the emitted sequence always matches the
        input sequence.
        """
        root = ET.Element("document")
        root.set("title", xml_safe(doc_title))

        page_el: Optional[ET.Element] = None
        current: Any = object()  # sentinel: no page open yet
        n_pages = 0
        for block in blocks:
            page_no = _meta(block).get("page", 0)
            if page_el is None or page_no != current:
                page_el = ET.SubElement(root, "page")
                page_el.set("number", str(page_no))
                current = page_no
                n_pages += 1
            page_el.append(self.block_to_element(block))

        root.set("pages", str(n_pages))
        root.set("blocks", str(len(list(blocks))))

        tree = ET.ElementTree(root)
        ET.indent(tree, space=self.indent)
        return ET.tostring(root, encoding="unicode", xml_declaration=True)

    def project_pages(self, pages: Iterable[Iterable[Any]],
                      doc_title: str = "Document") -> str:
        """Convenience for ``route_pages`` output: a list of per-page lists."""
        return self.project_to_xml([b for page in pages for b in page], doc_title)

    # -- dynamic (discovered-schema) projection ----------------------------- #
    @staticmethod
    def _attr_text(value: Any) -> str:
        """Any attribute value -> a clean string.

        The IE tree accepts whatever JSON the model produced, so a value can be
        a bool, a number, or occasionally a nested structure. XML attributes are
        strings only; ``True`` must become ``"true"`` rather than Python's
        ``"True"``, and a stray container is JSON-encoded rather than rendered
        via ``repr``.
        """
        if isinstance(value, bool):
            return "true" if value else "false"
        if value is None:
            return ""
        if isinstance(value, (str, int, float)):
            return str(value)
        try:
            import json as _json

            return _json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(value)

    def dynamic_element_to_xml(self, element: Any) -> ET.Element:
        """One :class:`DynamicElement` -> one XML element, recursively.

        Tags and attribute keys are re-sanitised here even though the node type
        normalises them on construction: a tree assembled by hand, or rebuilt
        from JSON, can carry raw keys, and an illegal tag name raises deep
        inside ElementTree with no indication of which node caused it.
        """
        from .ie_engine.node_schema import sanitize_tag

        el = ET.Element(sanitize_tag(_get(element, "tag_name", "element")))

        for key, value in (_get(element, "attributes", {}) or {}).items():
            text = xml_safe(self._attr_text(value))
            if text:
                el.set(sanitize_tag(key, default="attr"), text)

        content = _get(element, "text_content", None)
        if content:
            el.text = xml_safe(str(content))

        for child in _get(element, "children", []) or []:
            el.append(self.dynamic_element_to_xml(child))
        return el

    def project_dynamic_xml(self, root_element: Any,
                            doc_title: Optional[str] = None,
                            source: Optional[str] = None) -> str:
        """Serialise a discovered-schema tree to an indented XML string.

        Unlike :meth:`project_to_xml` this imposes no element vocabulary at all
        — every tag came from the model. ``doc_title`` and ``source``, when
        given, are attached to the root so a semantic document can still be
        traced back to the file it came from.
        """
        root = self.dynamic_element_to_xml(root_element)
        if doc_title:
            root.set("title", xml_safe(doc_title))
        if source:
            root.set("source", xml_safe(source))

        tree = ET.ElementTree(root)
        ET.indent(tree, space=self.indent)
        return ET.tostring(root, encoding="unicode", xml_declaration=True)

    def project_dynamic_document(self, document: Any) -> str:
        """Convenience wrapper for a whole :class:`DynamicDocument`."""
        return self.project_dynamic_xml(_get(document, "root", None),
                                        source=_get(document, "source", None))

    def save_xml(self, xml_string: str, output_path: str) -> None:
        """Indent ``xml_string`` and write it, creating parent directories.

        Re-parses rather than trusting the caller's formatting, so a string
        assembled elsewhere still lands pretty-printed and is validated as
        well-formed before anything is written to disk.
        """
        root = ET.fromstring(xml_string)
        tree = ET.ElementTree(root)
        ET.indent(tree, space=self.indent)
        parent = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(parent, exist_ok=True)
        tree.write(output_path, encoding="utf-8", xml_declaration=True)
