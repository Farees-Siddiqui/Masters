"""Dataclasses for the populated instance graph.

    SchemaGraph -> ParametricInstanceGenerator -> InstanceGraph -> instances.json

Where :mod:`schema_types` declares what the data *may* look like, this declares
what it *is*: one :class:`Record` per synthesised row, carrying its own
identifier, its field values, and the identifiers of the parent rows it points
at.

Keys are deliberately kept apart from ordinary fields:

``id``            the record's own identity, and the value of its entity's
                  primary key attribute.
``attributes``    every non-key field, plus the primary key. A value of ``None``
                  is a *deliberate* null, injected at ``--null-probability``.
``foreign_keys``  only the foreign key columns, mapping column name to the
                  parent record's ``id``.

The split is what makes the ground truth checkable: a later stage renders these
into a document and an extractor has to recover them, so "which of these values
is a join and which is a fact" cannot be a matter of guessing from the name.
:meth:`Record.fields` merges the two back together for renderers that just want
one flat row.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set

#: A foreign key deliberately pointed at a parent that does not exist, injected
#: at ``--orphan-rate``. Values look like ``Customer:__orphan_3__`` so a dangling
#: reference is recognisable on sight in a rendered document.
ORPHAN_MARKER = "__orphan_"


@dataclass
class Record:
    """One concrete data instance of one entity."""

    id: str
    entity_name: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    foreign_keys: Dict[str, str] = field(default_factory=dict)
    #: Foreign key columns on this record that were deliberately orphaned.
    #: Kept per-record rather than tallied on the graph so that a scorer can
    #: tell injected noise from an extraction error on the very row it is
    #: scoring — the whole point of generating the noise in the first place.
    orphaned_keys: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.id = str(self.id)
        self.entity_name = str(self.entity_name)
        self.attributes = dict(self.attributes or {})
        self.foreign_keys = {str(k): v for k, v in
                             (self.foreign_keys or {}).items()}
        self.orphaned_keys = [str(k) for k in (self.orphaned_keys or [])]

    def fields(self) -> Dict[str, Any]:
        """Attributes and foreign keys as one flat row, for rendering."""
        merged: Dict[str, Any] = dict(self.attributes)
        merged.update(self.foreign_keys)
        return merged

    def null_attributes(self) -> List[str]:
        """Names of attributes holding a deliberate null."""
        return [k for k, v in self.attributes.items() if v is None]

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Record":
        return cls(id=payload["id"],
                   entity_name=payload["entity_name"],
                   attributes=payload.get("attributes") or {},
                   foreign_keys=payload.get("foreign_keys") or {},
                   orphaned_keys=payload.get("orphaned_keys") or [])

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "id": self.id,
            "entity_name": self.entity_name,
            "attributes": dict(self.attributes),
            "foreign_keys": dict(self.foreign_keys),
        }
        if self.orphaned_keys:
            out["orphaned_keys"] = list(self.orphaned_keys)
        return out


@dataclass
class InstanceGraph:
    """Every record generated for one schema, in generation order.

    Generation order is parent-before-child, so iterating ``records`` never
    reaches a foreign key whose target has not been seen yet.
    """

    schema_domain: str
    records: List[Record] = field(default_factory=list)
    #: Redundant with ``len(records)`` by design: the spec'd artefact carries an
    #: explicit count, so it is recomputed on every mutation rather than
    #: trusted from input.
    total_records: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.schema_domain = str(self.schema_domain or "").strip() or "unspecified"
        self.records = [r if isinstance(r, Record) else Record.from_dict(r)
                        for r in (self.records or [])]
        self.total_records = len(self.records)

    # -- mutation ----------------------------------------------------------- #
    def add(self, record: Record) -> Record:
        self.records.append(record)
        self.total_records = len(self.records)
        return record

    def extend(self, records: Iterable[Record]) -> None:
        for record in records:
            self.add(record)

    # -- accessors ---------------------------------------------------------- #
    def record(self, record_id: str) -> Optional[Record]:
        for rec in self.records:
            if rec.id == record_id:
                return rec
        return None

    def ids(self) -> Set[str]:
        return {r.id for r in self.records}

    def by_entity(self, entity_name: str) -> List[Record]:
        return [r for r in self.records if r.entity_name == entity_name]

    def counts_by_entity(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for rec in self.records:
            counts[rec.entity_name] = counts.get(rec.entity_name, 0) + 1
        return counts

    def foreign_key_count(self) -> int:
        """Every foreign key value present, orphans and nulls included."""
        return sum(len(r.foreign_keys) for r in self.records)

    def resolved_foreign_key_count(self) -> int:
        """Foreign keys that actually resolve to a record in this graph."""
        known = self.ids()
        return sum(1 for r in self.records for v in r.foreign_keys.values()
                   if v is not None and v in known)

    def orphan_count(self) -> int:
        return sum(len(r.orphaned_keys) for r in self.records)

    def dangling_foreign_keys(self) -> List[str]:
        """``Record.column -> value`` for every key that does not resolve.

        Orphans are expected to appear here — that is what makes them orphans.
        Anything here that is *not* in a record's ``orphaned_keys`` is a bug in
        the generator, and the tests assert exactly that.
        """
        known = self.ids()
        out = []
        for rec in self.records:
            for column, value in rec.foreign_keys.items():
                if value is None or value not in known:
                    out.append(f"{rec.id}.{column} -> {value}")
        return out

    # -- serialisation ------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_domain": self.schema_domain,
            "total_records": self.total_records,
            "metadata": dict(self.metadata),
            "counts_by_entity": self.counts_by_entity(),
            "foreign_keys": {
                "declared": self.foreign_key_count(),
                "resolved": self.resolved_foreign_key_count(),
                "orphaned": self.orphan_count(),
            },
            "records": [r.to_dict() for r in self.records],
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "InstanceGraph":
        graph = cls(schema_domain=payload.get("schema_domain", ""),
                    records=[Record.from_dict(r)
                             for r in payload.get("records") or []])
        graph.metadata = dict(payload.get("metadata") or {})
        graph.warnings = list(payload.get("warnings") or [])
        return graph

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False,
                          default=str)

    def summary(self) -> str:
        """One line per entity, then the join tally. Used for the stderr log."""
        counts = self.counts_by_entity()
        nulls = sum(len(r.null_attributes()) for r in self.records)
        lines = [f"domain={self.schema_domain} records={self.total_records} "
                 f"entities={len(counts)} joins={self.foreign_key_count()} "
                 f"nulls={nulls}"]
        for name, count in counts.items():
            entity_records = self.by_entity(name)
            entity_nulls = sum(len(r.null_attributes()) for r in entity_records)
            entity_fks = sum(len(r.foreign_keys) for r in entity_records)
            lines.append(f"  {name}: {count} record(s), {entity_fks} fk(s), "
                         f"{entity_nulls} null(s)")
        resolved = self.resolved_foreign_key_count()
        orphaned = self.orphan_count()
        lines.append(f"  joins: {resolved} resolved, {orphaned} orphaned "
                     f"of {self.foreign_key_count()} total")
        return "\n".join(lines)


__all__ = ["Record", "InstanceGraph", "ORPHAN_MARKER"]
