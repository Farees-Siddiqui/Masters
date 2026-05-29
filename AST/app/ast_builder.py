"""AST construction from markdown.

Produces the nested-section representation shown on the whiteboard:
each heading opens a `section` whose body (paragraphs, lists, etc.) and
deeper-level headings become its children.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


NodeType = str  # "doc" | "section" | "paragraph" | "heading" | "list" | "list_item" | "code" | "text"


@dataclass
class Node:
    type: NodeType
    attribs: dict[str, str] = field(default_factory=dict)
    children: list["Node"] = field(default_factory=list)
    text: Optional[str] = None
    parent: Optional["Node"] = field(default=None, repr=False, compare=False)
    id: str = ""  # assigned by _assign_ids() at end of build_ast

    def add(self, child: "Node") -> "Node":
        child.parent = self
        self.children.append(child)
        return child

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "attribs": self.attribs,
            "text": self.text,
            "parent_id": self.parent.id if self.parent is not None else None,
            "children": [c.to_dict() for c in self.children],
        }


def _assign_ids(root: "Node") -> None:
    """Assign sequential ids (n0, n1, ...) in pre-order."""
    counter = 0
    stack: list[Node] = [root]
    while stack:
        node = stack.pop()
        node.id = f"n{counter}"
        counter += 1
        # push children in reverse so leftmost is visited first
        for c in reversed(node.children):
            stack.append(c)


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_ULIST_RE = re.compile(r"^\s*[-*+]\s+(.*)$")
_OLIST_RE = re.compile(r"^\s*\d+\.\s+(.*)$")
_FENCE_RE = re.compile(r"^```(\w*)\s*$")


def build_ast(markdown: str) -> Node:
    """Parse markdown into a nested-section AST.

    Implements the second representation from the whiteboard: headings open
    `section` nodes nested by level; paragraphs and lists become children of
    the most recently opened section.
    """
    root = Node(type="doc")
    # Stack of (level, section_node). level 0 is the root doc.
    stack: list[tuple[int, Node]] = [(0, root)]

    lines = markdown.splitlines()
    i = 0
    n = len(lines)

    def current_parent() -> Node:
        return stack[-1][1]

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # Fenced code block
        fence = _FENCE_RE.match(stripped)
        if fence:
            lang = fence.group(1) or ""
            i += 1
            buf: list[str] = []
            while i < n and not _FENCE_RE.match(lines[i].strip()):
                buf.append(lines[i])
                i += 1
            i += 1  # consume closing fence
            code_node = Node(type="code", attribs={"lang": lang} if lang else {}, text="\n".join(buf))
            current_parent().add(code_node)
            continue

        # Heading -> open / close sections by level
        heading = _HEADING_RE.match(stripped)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2).strip()

            # Pop until top-of-stack level < this heading's level
            while stack and stack[-1][0] >= level:
                stack.pop()
            if not stack:
                stack.append((0, root))

            section = Node(
                type="section",
                attribs={"title": title, "level": str(level)},
            )
            stack[-1][1].add(section)
            stack.append((level, section))
            i += 1
            continue

        # Lists (collect consecutive list items at the same marker family)
        ulist = _ULIST_RE.match(line)
        olist = _OLIST_RE.match(line)
        if ulist or olist:
            ordered = olist is not None
            list_node = Node(type="list", attribs={"ordered": "true" if ordered else "false"})
            while i < n:
                m = _OLIST_RE.match(lines[i]) if ordered else _ULIST_RE.match(lines[i])
                if not m:
                    break
                item = Node(type="list_item", text=m.group(1).strip())
                list_node.add(item)
                i += 1
            current_parent().add(list_node)
            continue

        # Paragraph: gather until blank line / structural line
        buf = [stripped]
        i += 1
        while i < n:
            nxt = lines[i]
            s = nxt.strip()
            if not s:
                break
            if _HEADING_RE.match(s) or _FENCE_RE.match(s) or _ULIST_RE.match(nxt) or _OLIST_RE.match(nxt):
                break
            buf.append(s)
            i += 1
        para = Node(type="paragraph", text=" ".join(buf))
        current_parent().add(para)

    _assign_ids(root)
    return root
