"""Normalise a discovered-schema tree into relational tables.

    DynamicElement / semantic.xml / extractor JSON
        -> RelationalExporter -> {table: rows} -> SQLite | CSV bundle

The IE engine's output is a tree whose shape is different for every document,
which is exactly what makes it awkward to query. This module projects that tree
onto the one shape SQL can work with, without imposing a schema of its own:

* every distinct tag becomes a **table**;
* every element becomes a **row** with an auto-incrementing ``id``;
* a child element's row carries ``parent_id`` referencing its parent's row;
* attributes and text content become **columns** (``text_content`` for the
  latter), unioned across all rows of that tag;
* repeated children — the array case, ``<students>`` twice under one register —
  are simply several rows in the child table, so nothing is comma-joined.

Two design points worth stating, because both are visible in the output:

``parent_table``
    The same tag can appear under different parents (an ``<address>`` under
    ``<student>`` and another under ``<issuer>``), so ``parent_id`` alone is
    ambiguous. Every child row records which table its parent lives in. A real
    ``FOREIGN KEY`` is declared only when a tag turns out to have exactly one
    parent tag in this document; when it has several the column is kept but the
    constraint is not, since SQL has no polymorphic foreign key.

Everything is ``TEXT``
    Values arrive as strings read off a page. Inferring INTEGER would silently
    rewrite ``"007"`` and postal codes, and a grade is no more numeric than a
    student number. Callers who want arithmetic can ``CAST`` at query time.
"""

from __future__ import annotations

import csv
import os
import re
import sqlite3
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from .node_schema import (DynamicDocument, DynamicElement, document_from_json,
                          element_from_json)

#: Columns this module owns on every table, in fixed leading order.
ID_COLUMN = "id"
PARENT_ID_COLUMN = "parent_id"
PARENT_TABLE_COLUMN = "parent_table"
TEXT_COLUMN = "text_content"
BASE_COLUMNS = (ID_COLUMN, PARENT_ID_COLUMN, PARENT_TABLE_COLUMN)
#: An attribute whose name collides with one of ours is prefixed rather than
#: dropped: a document may legitimately have its own "id".
RESERVED = frozenset(BASE_COLUMNS)
ATTR_PREFIX = "attr_"

_NON_IDENT = re.compile(r"[^0-9a-zA-Z_]+")


def sql_ident(name: Any, default: str = "value") -> str:
    """Turn a tag or attribute name into a bare SQL identifier.

    Tags are already XML-safe (``student-record``), but ``-`` and ``.`` force
    quoting in every query written against the result, so they become ``_``:
    ``"student-record"`` -> ``"student_record"``.
    """
    text = _NON_IDENT.sub("_", str(name if name is not None else "").strip().lower())
    text = re.sub(r"_{2,}", "_", text).strip("_")
    if not text:
        return default
    if text[0].isdigit():
        text = "n" + text
    return text


@dataclass
class Table:
    """One normalised table: its columns in order, and its rows."""

    name: str
    #: Attribute columns, in the order the tags first produced them.
    attr_columns: List[str] = field(default_factory=list)
    rows: List[Dict[str, Any]] = field(default_factory=list)
    #: Which tables rows in this one point at. One entry -> a real FOREIGN KEY.
    parent_tables: Set[str] = field(default_factory=set)
    has_text: bool = False

    @property
    def columns(self) -> List[str]:
        """Base columns, then attributes, then text — stable across runs."""
        cols = list(BASE_COLUMNS) + list(self.attr_columns)
        if self.has_text:
            cols.append(TEXT_COLUMN)
        return cols

    def column(self, name: str) -> str:
        """Register (or look up) an attribute column, avoiding our own names."""
        col = sql_ident(name, default="value")
        if col in RESERVED or col == TEXT_COLUMN:
            col = ATTR_PREFIX + col
        if col not in self.attr_columns:
            self.attr_columns.append(col)
        return col

    def matrix(self) -> List[List[Any]]:
        """Rows as a rectangular list-of-lists, missing cells as ``None``."""
        cols = self.columns
        return [[row.get(c) for c in cols] for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)


class RelationalExporter:
    """Flatten a dynamic tree into normalised tables, then write them out.

    ::

        exporter = RelationalExporter(doc.root)
        exporter.to_sqlite("doc01_email.db")
        exporter.to_csv_bundle("doc01_email_tables/")

    The input may be a :class:`DynamicElement`, a :class:`DynamicDocument`, a
    parsed ``semantic.xml`` (``ElementTree`` or ``Element``), a path to one, or
    the raw JSON payload the extractor got back from the model.
    """

    def __init__(self, source: Any, root_tag: str = "document"):
        self.root: DynamicElement = coerce_element(source, root_tag=root_tag)
        self.tables: Dict[str, Table] = {}
        #: tag -> table name, so two tags that sanitise alike stay separate.
        self._table_names: Dict[str, str] = {}
        self._flatten(self.root, parent_table=None, parent_id=None)

    # -- normalisation ------------------------------------------------------ #
    def _table_for(self, tag: str) -> Table:
        name = self._table_names.get(tag)
        if name is None:
            name = sql_ident(tag, default="element")
            # Distinct tags can sanitise to the same identifier ("a-b", "a.b").
            # Suffix rather than merge: they are different things.
            if name in self.tables:
                base, n = name, 2
                while name in self.tables:
                    name = f"{base}_{n}"
                    n += 1
            self._table_names[tag] = name
            self.tables[name] = Table(name=name)
        return self.tables[name]

    def _flatten(self, element: DynamicElement, parent_table: Optional[str],
                 parent_id: Optional[int]) -> int:
        """Insert ``element`` as a row, then recurse into its children.

        Pre-order matters: a parent's table is always created before any table
        that references it, which is what lets SQLite enforce the foreign keys
        at insert time rather than deferring them.
        """
        table = self._table_for(element.tag_name)
        row_id = len(table.rows) + 1
        row: Dict[str, Any] = {
            ID_COLUMN: row_id,
            PARENT_ID_COLUMN: parent_id,
            PARENT_TABLE_COLUMN: parent_table,
        }
        for key, value in (element.attributes or {}).items():
            row[table.column(key)] = _as_cell(value)

        text = (element.text_content or "").strip()
        if text:
            table.has_text = True
            row[TEXT_COLUMN] = text

        if parent_table:
            table.parent_tables.add(parent_table)
        table.rows.append(row)

        for child in element.children:
            self._flatten(child, parent_table=table.name, parent_id=row_id)
        return row_id

    # -- inspection --------------------------------------------------------- #
    @property
    def root_table(self) -> str:
        return self._table_names[self.root.tag_name]

    def table_names(self) -> List[str]:
        """Table names in creation order: parents before their children."""
        return list(self.tables)

    def to_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """The whole normalisation as plain data — ``{table: [row, ...]}``."""
        return {name: [dict(r) for r in t.rows] for name, t in self.tables.items()}

    def summary(self) -> str:
        return ", ".join(f"{n}({len(t)})" for n, t in self.tables.items())

    # -- output ------------------------------------------------------------- #
    def to_sqlite(self, db_path: str) -> str:
        """Create and populate a normalised SQLite database. Returns the path.

        Existing tables of the same names are dropped, so re-running the
        pipeline refreshes a database rather than doubling its rows. Foreign
        keys are enforced during the load: a broken parent link fails here
        rather than surfacing as a silently empty join later.
        """
        parent = os.path.dirname(os.path.abspath(db_path))
        os.makedirs(parent, exist_ok=True)

        conn = sqlite3.connect(db_path)
        try:
            # Drop children before parents, and with enforcement off: a
            # previous run's rows still reference each other, so tearing the
            # old schema down would otherwise trip its own foreign keys.
            conn.execute("PRAGMA foreign_keys = OFF")
            for table in reversed(list(self.tables.values())):
                conn.execute(f'DROP TABLE IF EXISTS "{table.name}"')
            for table in self.tables.values():
                conn.execute(self.create_statement(table))
            conn.commit()

            conn.execute("PRAGMA foreign_keys = ON")
            for table in self.tables.values():
                cols = table.columns
                placeholders = ", ".join("?" for _ in cols)
                quoted = ", ".join(f'"{c}"' for c in cols)
                conn.executemany(
                    f'INSERT INTO "{table.name}" ({quoted}) VALUES ({placeholders})',
                    table.matrix())
            conn.commit()
        finally:
            conn.close()
        return db_path

    def create_statement(self, table: Table) -> str:
        """``CREATE TABLE`` for one table, with a foreign key where it is sound."""
        lines = [f'  "{ID_COLUMN}" INTEGER PRIMARY KEY AUTOINCREMENT',
                 f'  "{PARENT_ID_COLUMN}" INTEGER',
                 f'  "{PARENT_TABLE_COLUMN}" TEXT']
        for col in table.columns[len(BASE_COLUMNS):]:
            lines.append(f'  "{col}" TEXT')
        # Only one candidate parent can be named; see the module docstring.
        if len(table.parent_tables) == 1:
            (parent,) = table.parent_tables
            lines.append(f'  FOREIGN KEY ("{PARENT_ID_COLUMN}") '
                         f'REFERENCES "{parent}" ("{ID_COLUMN}")')
        body = ",\n".join(lines)
        return f'CREATE TABLE "{table.name}" (\n{body}\n)'

    def to_csv_bundle(self, output_dir: str) -> List[str]:
        """Write one ``<table>.csv`` per table. Returns the paths written."""
        os.makedirs(output_dir, exist_ok=True)
        written = []
        for table in self.tables.values():
            path = os.path.join(output_dir, f"{table.name}.csv")
            with open(path, "w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(table.columns)
                for row in table.matrix():
                    writer.writerow(["" if v is None else v for v in row])
            written.append(path)
        return written


# --------------------------------------------------------------------------- #
# Input coercion
# --------------------------------------------------------------------------- #
def _as_cell(value: Any) -> Any:
    if isinstance(value, bool):
        return "true" if value else "false"
    return value if value is None else str(value)


def element_from_etree(node: ET.Element) -> DynamicElement:
    """Convert a parsed XML element into a :class:`DynamicElement`.

    Text is stripped: a pretty-printed ``semantic.xml`` puts indentation
    whitespace in ``.text`` on every element that has children, and that is
    formatting, not content.
    """
    text = (node.text or "").strip()
    return DynamicElement(
        tag_name=node.tag,
        attributes=dict(node.attrib),
        text_content=text or None,
        children=[element_from_etree(c) for c in node],
    )


def coerce_element(source: Any, root_tag: str = "document") -> DynamicElement:
    """Accept any of the engine's output forms and return a tree root."""
    if isinstance(source, DynamicElement):
        return source
    if isinstance(source, DynamicDocument):
        return source.root
    if isinstance(source, ET.ElementTree):
        return element_from_etree(source.getroot())
    if isinstance(source, ET.Element):
        return element_from_etree(source)
    if hasattr(source, "root") and isinstance(getattr(source, "root"),
                                              DynamicElement):
        return source.root  # a DynamicDocument-alike

    if isinstance(source, str):
        stripped = source.lstrip()
        if stripped.startswith("<"):
            return element_from_etree(ET.fromstring(source))
        if os.path.exists(source):
            return element_from_etree(ET.parse(source).getroot())
        raise ValueError(f"not an XML string or an existing path: {source!r}")

    if isinstance(source, dict):
        # Its own serialisation round-trips exactly; anything else is open model
        # output and goes through the JSON -> tree mapping. node_schema draws
        # the same line, for the same reason: one payload cannot be both.
        if "tag_name" in source:
            return DynamicElement.from_dict(source)
        if isinstance(source.get("root"), dict):
            return DynamicDocument.from_dict(source).root
        return document_from_json(source, root_tag=root_tag).root

    if isinstance(source, (list, tuple)):
        return element_from_json(list(source), root_tag)

    raise TypeError(f"cannot export {type(source).__name__} as relational tables")


def export_semantic_xml(xml_path: str, db_path: Optional[str] = None,
                        csv_dir: Optional[str] = None) -> List[str]:
    """Convenience: read a ``semantic.xml`` and write its database / CSVs."""
    exporter = RelationalExporter(xml_path)
    written = []
    if db_path:
        written.append(exporter.to_sqlite(db_path))
    if csv_dir:
        written += exporter.to_csv_bundle(csv_dir)
    return written
