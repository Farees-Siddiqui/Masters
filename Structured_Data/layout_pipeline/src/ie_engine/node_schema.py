"""A generic tree for whatever schema a document turns out to have.

Deliberately *not* a fixed model. The IE engine's premise is that the schema is
discovered per document — a report card yields `<student-record>` with
`<evaluation>` children, an email yields `<message>` with `<sender>` — so the
node type has to accept any tag, any attribute key, and any nesting depth
without validation errors on keys it has never seen.

    DynamicElement(tag_name, attributes: dict, text_content: str|None, children: [...])

Plain dataclasses rather than pydantic: a model with declared fields is exactly
the thing this engine must not have.

The one hard constraint is that ``tag_name`` must survive becoming an XML tag
downstream, so names are normalised here rather than at serialisation time —
"Student Record" and "2024 grades" are perfectly good LLM output and neither is
a legal XML name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

#: Characters legal in an XML name, after the first.
_ILLEGAL_TAG_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")
_LEADING = re.compile(r"^[^A-Za-z_]+")

DEFAULT_TAG = "element"
MAX_TAG_LEN = 64


def sanitize_tag(name: Any, default: str = DEFAULT_TAG) -> str:
    """Turn arbitrary text into a legal, readable XML tag name.

    ``"Student Record"`` -> ``"student-record"``, ``"2024 grades"`` ->
    ``"n2024-grades"`` (XML names cannot start with a digit), ``""`` ->
    ``"element"``.
    """
    text = str(name if name is not None else "").strip().lower()
    text = text.replace(" ", "-").replace("/", "-")
    text = _ILLEGAL_TAG_CHARS.sub("-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-.")
    if not text:
        return default
    if _LEADING.match(text):
        # A leading digit is legal *inside* a name, so prefix rather than drop.
        text = "n" + text
    text = text[:MAX_TAG_LEN].rstrip("-.")
    return text or default


def _scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _as_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return "" if value is None else str(value)


@dataclass
class DynamicElement:
    """One discovered node. Any tag, any attributes, any children."""

    tag_name: str = DEFAULT_TAG
    attributes: Dict[str, Any] = field(default_factory=dict)
    text_content: Optional[str] = None
    children: List["DynamicElement"] = field(default_factory=list)

    def __post_init__(self):
        self.tag_name = sanitize_tag(self.tag_name)

    # -- building ---------------------------------------------------------- #
    def add(self, child: "DynamicElement") -> "DynamicElement":
        self.children.append(child)
        return child

    def set(self, key: Any, value: Any) -> None:
        """Attribute keys are normalised the same way tags are, so the pair can
        round-trip through XML."""
        self.attributes[sanitize_tag(key, default="attr")] = _as_text(value)

    # -- reading ------------------------------------------------------------ #
    def walk(self) -> Iterator["DynamicElement"]:
        """Depth-first, self first."""
        yield self
        for child in self.children:
            yield from child.walk()

    def find(self, tag: str) -> Optional["DynamicElement"]:
        target = sanitize_tag(tag)
        return next((e for e in self.walk() if e.tag_name == target), None)

    def find_all(self, tag: str) -> List["DynamicElement"]:
        target = sanitize_tag(tag)
        return [e for e in self.walk() if e.tag_name == target]

    @property
    def depth(self) -> int:
        return 1 + max((c.depth for c in self.children), default=0)

    def count(self) -> int:
        return sum(1 for _ in self.walk())

    def to_dict(self) -> dict:
        out: Dict[str, Any] = {"tag_name": self.tag_name}
        if self.attributes:
            out["attributes"] = dict(self.attributes)
        if self.text_content:
            out["text_content"] = self.text_content
        if self.children:
            out["children"] = [c.to_dict() for c in self.children]
        return out

    @classmethod
    def from_dict(cls, payload: dict) -> "DynamicElement":
        """Rebuild a tree from :meth:`to_dict`, recursing through ``children``.

        This is the round-trip half only: it reads *this class's own*
        serialisation, whose keys are ``tag_name``/``attributes``/``children``.
        It is not the reader for open model output — a discovered schema has
        arbitrary keys and no ``tag_name`` anywhere, and is turned into a tree by
        :func:`element_from_json` instead. Keeping the two apart is deliberate:
        one payload format cannot be both, since a document could legitimately
        contain a key called "tag_name".
        """
        return cls(
            tag_name=payload.get("tag_name", DEFAULT_TAG),
            attributes=dict(payload.get("attributes") or {}),
            text_content=payload.get("text_content"),
            children=[cls.from_dict(c) for c in payload.get("children") or []],
        )


@dataclass
class DynamicDocument:
    """A discovered schema for one document, plus how it was produced."""

    root: DynamicElement = field(
        default_factory=lambda: DynamicElement("document"))
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def elements(self) -> int:
        return self.root.count()

    @property
    def ok(self) -> bool:
        """Whether anything was actually discovered."""
        return bool(self.root.children or self.root.attributes
                    or self.root.text_content)

    def to_dict(self) -> dict:
        return {"source": self.source, "metadata": self.metadata,
                "root": self.root.to_dict()}

    @classmethod
    def from_dict(cls, payload: dict) -> "DynamicDocument":
        return cls(root=DynamicElement.from_dict(payload.get("root") or {}),
                   source=payload.get("source"),
                   metadata=dict(payload.get("metadata") or {}))


# --------------------------------------------------------------------------- #
# Arbitrary JSON -> tree
# --------------------------------------------------------------------------- #
def element_from_json(value: Any, tag_name: str = "root",
                      max_depth: int = 24) -> DynamicElement:
    """Convert any JSON value into a :class:`DynamicElement` tree.

    The mapping, chosen so a discovered schema reads naturally as XML:

    * ``{"grade": "90"}`` — scalar values become **attributes**.
    * ``{"address": {...}}`` — nested objects become **child elements**.
    * ``{"courses": [ ... ]}`` — each list item becomes its own child under the
      list's key, so repeated records stay repeated rather than being flattened
      into one comma-joined attribute.
    * A bare scalar becomes an element's **text_content**.

    ``max_depth`` guards against a self-referential structure; anything deeper
    is rendered as text rather than recursed into.
    """
    el = DynamicElement(tag_name)
    _fill(el, value, max_depth)
    return el


def _fill(el: DynamicElement, value: Any, depth: int) -> None:
    if depth <= 0:
        el.text_content = _as_text(value)[:2000]
        return

    if isinstance(value, dict):
        for key, item in value.items():
            if _scalar(item):
                if item is None or _as_text(item) == "":
                    continue
                el.set(key, item)
            elif isinstance(item, dict):
                el.add(element_from_json(item, key, depth - 1))
            elif isinstance(item, (list, tuple)):
                _fill_list(el, key, list(item), depth)
            else:
                el.set(key, item)

    elif isinstance(value, (list, tuple)):
        _fill_list(el, "item", list(value), depth)

    else:
        text = _as_text(value)
        if text:
            el.text_content = text


def _fill_list(parent: DynamicElement, key: Any, items: list, depth: int) -> None:
    """One child per item, all sharing the list's key as their tag."""
    tag = sanitize_tag(key)
    for item in items:
        if _scalar(item):
            text = _as_text(item)
            if text:
                parent.add(DynamicElement(tag, text_content=text))
        else:
            parent.add(element_from_json(item, tag, depth - 1))


def document_from_json(payload: Any, source: Optional[str] = None,
                       root_tag: str = "document",
                       metadata: Optional[dict] = None) -> DynamicDocument:
    """Wrap :func:`element_from_json` as a whole document.

    A payload with exactly one top-level object key is unwrapped so that
    ``{"student_record": {...}}`` becomes ``<student-record>`` rather than
    ``<document><student-record>``: the model naming the root is a real signal
    about the document type and worth keeping.
    """
    if isinstance(payload, dict) and len(payload) == 1:
        (only_key, only_value), = payload.items()
        if isinstance(only_value, dict):
            root = element_from_json(only_value, only_key)
            return DynamicDocument(root=root, source=source,
                                   metadata=dict(metadata or {}))
    root = element_from_json(payload, root_tag)
    return DynamicDocument(root=root, source=source,
                           metadata=dict(metadata or {}))
