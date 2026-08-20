"""Dataclasses for a parametric Entity-Relationship schema.

    LLM JSON -> SchemaGraph(entities=[EntitySchema(attributes=[Attribute])],
                            relationships=[Relationship])

This is the *opposite* of the dynamic tree in ``layout_pipeline``'s IE engine.
There the schema is discovered per document and so cannot be declared; here the
schema is the thing being generated, and every later stage of the data
generation pipeline (record synthesis, document rendering, ground-truth
scoring) reads it back. That contract has to be pinned, so these are declared
dataclasses with a fixed field set.

Plain ``dataclasses`` rather than pydantic, matching the rest of this repo's
stdlib-only extraction path: the validation needed here is structural, and
``from_payload`` does it in one pass while normalising the many shapes a local
model will hand back for the same idea.

Relationships are strictly 1:m and directed child -> parent, written the way
the prompt asks for them::

    Order.customer_id -> Customer.id
    ^^^^^ ^^^^^^^^^^^    ^^^^^^^^ ^^
    child   fk column     parent   pk
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

#: Primitive attribute types the generator will emit. Deliberately semantic
#: rather than SQL-shaped ("currency", "address", "email"): a later stage has to
#: *render* each value into a document, and "currency" tells it to produce
#: "$1,240.50" where "decimal" would only say 1240.5.
PRIMITIVE_TYPES: Tuple[str, ...] = (
    "string", "text", "integer", "decimal", "currency", "percent", "boolean",
    "date", "datetime", "email", "phone", "url", "address", "id", "enum",
)

#: What models actually return when asked for the types above. Anything not
#: listed and not already primitive degrades to "string" with a warning rather
#: than failing the run — an unknown type still renders, a hard error loses the
#: whole schema.
TYPE_ALIASES: Dict[str, str] = {
    "str": "string", "varchar": "string", "char": "string", "name": "string",
    "int": "integer", "int64": "integer", "number": "integer",
    "bigint": "integer", "smallint": "integer", "serial": "integer",
    "float": "decimal", "double": "decimal", "numeric": "decimal",
    "real": "decimal", "money": "currency", "amount": "currency",
    "price": "currency", "usd": "currency", "cad": "currency",
    "bool": "boolean", "flag": "boolean",
    "timestamp": "datetime", "time": "datetime", "datetime2": "datetime",
    "e-mail": "email", "email_address": "email",
    "telephone": "phone", "tel": "phone", "phone_number": "phone",
    "uri": "url", "link": "url", "website": "url",
    "postal_address": "address", "location": "address",
    "uuid": "id", "guid": "id", "identifier": "id", "pk": "id", "fk": "id",
    "foreign_key": "id", "primary_key": "id", "reference": "id",
    "category": "enum", "choice": "enum", "status": "enum",
    "longtext": "text", "description": "text", "notes": "text",
}

DEFAULT_TYPE = "string"
DEFAULT_PRIMARY_KEY = "id"
#: Cardinality is fixed: the pipeline models hierarchical documents, which are
#: 1:m all the way down. m:n would need a join entity, and that is a Stage 2
#: parameter this piece does not take.
CARDINALITY = "1:m"

_NON_IDENT = re.compile(r"[^0-9a-zA-Z]+")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


class SchemaValidationError(ValueError):
    """The payload could not be read as an ER graph at all."""


def snake_case(name: Any, default: str = "field") -> str:
    """``"Order Line"`` / ``"OrderLine"`` -> ``"order_line"``.

    Attribute and entity names become SQL identifiers, XML tags and dict keys
    downstream, so they are normalised once here instead of at each use.
    """
    text = str(name if name is not None else "").strip()
    if not text:
        return default
    text = _CAMEL_BOUNDARY.sub("_", text)
    text = _NON_IDENT.sub("_", text).strip("_").lower()
    text = re.sub(r"_{2,}", "_", text)
    if not text:
        return default
    if text[0].isdigit():  # an identifier cannot start with a digit
        text = "n" + text
    return text


def pascal_case(name: Any, default: str = "Entity") -> str:
    """``"order_line"`` -> ``"OrderLine"``. Entity names are classes/tables."""
    parts = [p for p in snake_case(name, default).split("_") if p]
    if not parts:
        return default
    return "".join(p[:1].upper() + p[1:] for p in parts)


def normalize_type(raw: Any) -> Tuple[str, bool]:
    """Map a model-supplied type onto :data:`PRIMITIVE_TYPES`.

    Returns ``(type, recognised)``. ``recognised`` is False when the value fell
    back to ``string``, which the caller records as a warning.
    """
    text = snake_case(raw, "")
    if not text:
        return DEFAULT_TYPE, False
    if text in PRIMITIVE_TYPES:
        return text, True
    if text in TYPE_ALIASES:
        return TYPE_ALIASES[text], True
    # "decimal(10,2)" / "varchar_255" -> try the leading word alone.
    head = text.split("_")[0]
    if head in PRIMITIVE_TYPES:
        return head, True
    if head in TYPE_ALIASES:
        return TYPE_ALIASES[head], True
    return DEFAULT_TYPE, False


def _as_list(value: Any) -> List[Any]:
    """Accept a list, a single object, or a ``{name: {...}}`` mapping."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        # {"Customer": {...}, "Order": {...}} -> [{"name": "Customer", ...}]
        out = []
        for key, item in value.items():
            if isinstance(item, dict):
                out.append({"name": key, **item})
            else:
                out.append({"name": key, "type": item})
        return out
    return [value]


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in ("true", "yes", "y", "1", "required")


# --------------------------------------------------------------------------- #
# Attribute
# --------------------------------------------------------------------------- #
@dataclass
class Attribute:
    """One column on an entity."""

    name: str
    type: str = DEFAULT_TYPE
    description: str = ""
    required: bool = False
    unique: bool = False
    #: Populated for ``enum`` attributes; empty otherwise.
    values: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.name = snake_case(self.name)
        self.type, _ = normalize_type(self.type)
        self.description = str(self.description or "").strip()
        self.required = bool(self.required)
        self.unique = bool(self.unique)
        self.values = [str(v) for v in (self.values or [])]

    @classmethod
    def from_payload(cls, payload: Any) -> "Attribute":
        """Read the several shapes a model uses for one attribute.

        ``{"name": "total", "type": "currency"}``, ``{"total": "currency"}``
        and the bare string ``"total"`` all arrive here.
        """
        if isinstance(payload, str):
            return cls(name=payload)
        if not isinstance(payload, dict):
            raise SchemaValidationError(f"attribute is not an object: {payload!r}")

        name = payload.get("name") or payload.get("attribute") or \
            payload.get("field") or payload.get("column")
        raw_type = payload.get("type") or payload.get("data_type") or \
            payload.get("dtype")
        if name is None:
            # {"total": "currency"} — a single-pair mapping, name as the key.
            extras = {k: v for k, v in payload.items()
                      if k not in ("type", "data_type", "dtype", "description",
                                   "required", "nullable", "unique", "values",
                                   "enum")}
            if len(extras) != 1:
                raise SchemaValidationError(
                    f"attribute has no usable name: {payload!r}")
            name, inline_type = next(iter(extras.items()))
            raw_type = raw_type or inline_type

        required = payload.get("required")
        if required is None and "nullable" in payload:
            required = not _as_bool(payload.get("nullable"), True)
        return cls(
            name=str(name),
            type=raw_type if raw_type is not None else DEFAULT_TYPE,
            description=payload.get("description") or payload.get("desc") or "",
            required=_as_bool(required),
            unique=_as_bool(payload.get("unique")),
            values=[str(v) for v in (payload.get("values")
                                     or payload.get("enum") or [])],
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"name": self.name, "type": self.type,
                               "required": self.required}
        if self.unique:
            out["unique"] = True
        if self.description:
            out["description"] = self.description
        if self.values:
            out["values"] = list(self.values)
        return out


# --------------------------------------------------------------------------- #
# EntitySchema
# --------------------------------------------------------------------------- #
@dataclass
class EntitySchema:
    """One table / class in the generated schema."""

    name: str
    attributes: List[Attribute] = field(default_factory=list)
    description: str = ""
    primary_key: str = DEFAULT_PRIMARY_KEY

    def __post_init__(self) -> None:
        self.name = pascal_case(self.name)
        self.description = str(self.description or "").strip()
        self.primary_key = snake_case(self.primary_key or DEFAULT_PRIMARY_KEY,
                                      DEFAULT_PRIMARY_KEY)
        self.attributes = [a if isinstance(a, Attribute)
                           else Attribute.from_payload(a)
                           for a in self.attributes]
        self._dedupe()

    def _dedupe(self) -> None:
        """Keep the first of each attribute name; models repeat themselves."""
        seen: Set[str] = set()
        kept: List[Attribute] = []
        for attr in self.attributes:
            if attr.name in seen:
                continue
            seen.add(attr.name)
            kept.append(attr)
        self.attributes = kept

    @classmethod
    def from_payload(cls, payload: Any) -> "EntitySchema":
        if isinstance(payload, str):
            return cls(name=payload)
        if not isinstance(payload, dict):
            raise SchemaValidationError(f"entity is not an object: {payload!r}")
        name = payload.get("name") or payload.get("entity") or \
            payload.get("table") or payload.get("class")
        if not name:
            raise SchemaValidationError(f"entity has no name: {payload!r}")
        raw_attrs = payload.get("attributes")
        if raw_attrs is None:
            raw_attrs = payload.get("fields") or payload.get("columns") or \
                payload.get("properties")
        return cls(
            name=str(name),
            attributes=[Attribute.from_payload(a) for a in _as_list(raw_attrs)],
            description=payload.get("description") or payload.get("desc") or "",
            primary_key=payload.get("primary_key") or payload.get("pk")
            or DEFAULT_PRIMARY_KEY,
        )

    # -- accessors ---------------------------------------------------------- #
    @property
    def attribute_names(self) -> List[str]:
        return [a.name for a in self.attributes]

    def attribute(self, name: str) -> Optional[Attribute]:
        target = snake_case(name)
        for attr in self.attributes:
            if attr.name == target:
                return attr
        return None

    def add_attribute(self, attr: Attribute, index: Optional[int] = None) -> bool:
        """Add ``attr`` unless the name is already taken. True if added."""
        if self.attribute(attr.name) is not None:
            return False
        if index is None:
            self.attributes.append(attr)
        else:
            self.attributes.insert(index, attr)
        return True

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"name": self.name,
                               "primary_key": self.primary_key}
        if self.description:
            out["description"] = self.description
        out["attributes"] = [a.to_dict() for a in self.attributes]
        return out


# --------------------------------------------------------------------------- #
# Relationship
# --------------------------------------------------------------------------- #
@dataclass
class Relationship:
    """A 1:m foreign key: ``child.child_attribute -> parent.parent_attribute``.

    ``parent`` is the *one* side, ``child`` the *many* side, so
    ``Order.customer_id -> Customer.id`` is
    ``Relationship(parent_entity="Customer", child_entity="Order", ...)``.
    """

    parent_entity: str
    child_entity: str
    child_attribute: str = ""
    parent_attribute: str = DEFAULT_PRIMARY_KEY
    name: str = ""
    cardinality: str = CARDINALITY

    def __post_init__(self) -> None:
        self.parent_entity = pascal_case(self.parent_entity)
        self.child_entity = pascal_case(self.child_entity)
        self.parent_attribute = snake_case(
            self.parent_attribute or DEFAULT_PRIMARY_KEY, DEFAULT_PRIMARY_KEY)
        self.child_attribute = snake_case(
            self.child_attribute or f"{self.parent_entity}_{self.parent_attribute}",
            "parent_id")
        self.name = snake_case(self.name, "") or \
            f"{self.child_entity.lower()}_to_{self.parent_entity.lower()}"
        self.cardinality = str(self.cardinality or CARDINALITY)

    # -- parsing ------------------------------------------------------------ #
    @staticmethod
    def _split_ref(ref: Any) -> Tuple[str, str]:
        """``"Order.customer_id"`` -> ``("Order", "customer_id")``."""
        text = str(ref if ref is not None else "").strip()
        if not text:
            return "", ""
        for sep in (".", "::", ":", "->", "/"):
            if sep in text:
                head, _, tail = text.partition(sep)
                return head.strip(), tail.strip()
        return text, ""

    @classmethod
    def from_payload(cls, payload: Any) -> "Relationship":
        """Read either notation the prompt permits.

        String form, exactly as the prompt writes it::

            "Order.customer_id -> Customer.id"

        Object form, which models drift into even when told not to::

            {"child": "Order", "child_attribute": "customer_id",
             "parent": "Customer", "parent_attribute": "id"}
        """
        if isinstance(payload, str):
            left, arrow, right = payload.partition("->")
            if not arrow:
                raise SchemaValidationError(
                    f"relationship is missing '->': {payload!r}")
            child, child_attr = cls._split_ref(left)
            parent, parent_attr = cls._split_ref(right)
            if not child or not parent:
                raise SchemaValidationError(
                    f"relationship names an unknown side: {payload!r}")
            return cls(parent_entity=parent, child_entity=child,
                       child_attribute=child_attr,
                       parent_attribute=parent_attr or DEFAULT_PRIMARY_KEY)

        if not isinstance(payload, dict):
            raise SchemaValidationError(
                f"relationship is not an object: {payload!r}")

        # A single "Order.customer_id -> Customer.id" under any key.
        for key in ("fk", "foreign_key", "relationship", "link", "join", "ref"):
            if isinstance(payload.get(key), str) and "->" in payload[key]:
                base = cls.from_payload(payload[key])
                base.name = snake_case(payload.get("name"), "") or base.name
                return base

        child_ref = payload.get("child") or payload.get("from") or \
            payload.get("child_entity") or payload.get("many") or \
            payload.get("source")
        parent_ref = payload.get("parent") or payload.get("to") or \
            payload.get("parent_entity") or payload.get("one") or \
            payload.get("target") or payload.get("references")
        child, child_attr = cls._split_ref(child_ref)
        parent, parent_attr = cls._split_ref(parent_ref)
        child_attr = payload.get("child_attribute") or \
            payload.get("foreign_key_attribute") or \
            payload.get("fk_column") or child_attr
        parent_attr = payload.get("parent_attribute") or \
            payload.get("referenced_attribute") or \
            payload.get("pk_column") or parent_attr
        if not child or not parent:
            raise SchemaValidationError(
                f"relationship names an unknown side: {payload!r}")
        return cls(parent_entity=parent, child_entity=child,
                   child_attribute=str(child_attr or ""),
                   parent_attribute=str(parent_attr or DEFAULT_PRIMARY_KEY),
                   name=payload.get("name") or "",
                   cardinality=payload.get("cardinality") or CARDINALITY)

    def as_fk(self) -> str:
        """The notation the prompt asks for: ``Order.customer_id -> Customer.id``."""
        return (f"{self.child_entity}.{self.child_attribute} -> "
                f"{self.parent_entity}.{self.parent_attribute}")

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name,
                "parent_entity": self.parent_entity,
                "parent_attribute": self.parent_attribute,
                "child_entity": self.child_entity,
                "child_attribute": self.child_attribute,
                "cardinality": self.cardinality,
                "fk": self.as_fk()}


# --------------------------------------------------------------------------- #
# SchemaGraph
# --------------------------------------------------------------------------- #
@dataclass
class SchemaGraph:
    """The whole Stage 2 artefact: entities plus their 1:m links."""

    domain: str
    entities: List[EntitySchema] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    #: Requested parameters and provenance, echoed into ``schema.json`` so a
    #: corpus can be traced back to the command that produced it.
    metadata: Dict[str, Any] = field(default_factory=dict)
    #: Repairs applied during enforcement (dropped edges, coerced types, ...).
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.domain = str(self.domain or "").strip() or "unspecified"
        self.entities = [e if isinstance(e, EntitySchema)
                         else EntitySchema.from_payload(e)
                         for e in self.entities]
        self.relationships = [r if isinstance(r, Relationship)
                              else Relationship.from_payload(r)
                              for r in self.relationships]

    # -- parsing ------------------------------------------------------------ #
    @classmethod
    def from_payload(cls, payload: Any, domain: str = "") -> "SchemaGraph":
        """Build a graph from raw model JSON.

        Structural problems raise :class:`SchemaValidationError`; individual
        malformed entries are dropped and recorded in ``warnings``, because one
        unreadable relationship should not cost the other nine.
        """
        if not isinstance(payload, dict):
            raise SchemaValidationError(
                f"schema payload is not an object: {type(payload).__name__}")

        # Some models wrap everything one level down under "schema"/"er_graph".
        for wrapper in ("schema", "er_graph", "graph", "result", "output"):
            inner = payload.get(wrapper)
            if isinstance(inner, dict) and ("entities" in inner
                                            or "tables" in inner):
                payload = inner
                break

        raw_entities = payload.get("entities")
        if raw_entities is None:
            raw_entities = payload.get("tables") or payload.get("classes")
        raw_rels = payload.get("relationships")
        if raw_rels is None:
            raw_rels = payload.get("foreign_keys") or payload.get("links") or \
                payload.get("relations") or payload.get("joins")

        warnings: List[str] = []
        entities: List[EntitySchema] = []
        for item in _as_list(raw_entities):
            try:
                entities.append(EntitySchema.from_payload(item))
            except SchemaValidationError as exc:
                warnings.append(f"dropped unreadable entity: {exc}")
        if not entities:
            raise SchemaValidationError(
                "payload contains no readable entities")

        relationships: List[Relationship] = []
        for item in _as_list(raw_rels):
            try:
                relationships.append(Relationship.from_payload(item))
            except SchemaValidationError as exc:
                warnings.append(f"dropped unreadable relationship: {exc}")

        graph = cls(domain=domain or payload.get("domain") or "",
                    entities=entities, relationships=relationships)
        graph.warnings.extend(warnings)
        return graph

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SchemaGraph":
        """Inverse of :meth:`to_dict` — read back a serialised ``schema.json``."""
        graph = cls.from_payload(payload, domain=payload.get("domain", ""))
        graph.metadata = dict(payload.get("metadata") or {})
        graph.warnings = list(payload.get("warnings") or [])
        return graph

    # -- accessors ---------------------------------------------------------- #
    @property
    def entity_names(self) -> List[str]:
        return [e.name for e in self.entities]

    def entity(self, name: str) -> Optional[EntitySchema]:
        target = pascal_case(name)
        for ent in self.entities:
            if ent.name == target:
                return ent
        return None

    def parents_of(self, name: str) -> List[str]:
        target = pascal_case(name)
        return [r.parent_entity for r in self.relationships
                if r.child_entity == target]

    def children_of(self, name: str) -> List[str]:
        target = pascal_case(name)
        return [r.child_entity for r in self.relationships
                if r.parent_entity == target]

    def roots(self) -> List[str]:
        """Entities with no parent — the top level of the document hierarchy."""
        childed = {r.child_entity for r in self.relationships}
        return [e.name for e in self.entities if e.name not in childed]

    def depths(self) -> Dict[str, int]:
        """Level of each entity, roots at 1.

        Computed iteratively (relaxation) rather than recursively so that a
        cycle the caller has not yet removed cannot blow the stack: the loop is
        bounded by the entity count, and anything still unsettled is left at its
        last value.
        """
        levels = {e.name: 1 for e in self.entities}
        edges = [(r.child_entity, r.parent_entity) for r in self.relationships
                 if r.child_entity in levels and r.parent_entity in levels]
        for _ in range(len(levels)):
            changed = False
            for child, parent in edges:
                candidate = levels[parent] + 1
                if candidate > levels[child]:
                    levels[child] = candidate
                    changed = True
            if not changed:
                break
        return levels

    def depth(self) -> int:
        """Deepest level in the graph. 0 for an empty graph."""
        levels = self.depths()
        return max(levels.values()) if levels else 0

    # -- serialisation ------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        levels = self.depths()
        return {
            "domain": self.domain,
            "metadata": dict(self.metadata),
            "depth": self.depth(),
            "roots": self.roots(),
            "entity_depths": levels,
            "entities": [e.to_dict() for e in self.entities],
            "relationships": [r.to_dict() for r in self.relationships],
            "warnings": list(self.warnings),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False,
                          sort_keys=False)

    def summary(self) -> str:
        """One-line-per-entity human summary, used for the stderr log."""
        levels = self.depths()
        lines = [f"domain={self.domain} entities={len(self.entities)} "
                 f"joins={len(self.relationships)} depth={self.depth()}"]
        for ent in self.entities:
            types = ", ".join(f"{a.name}:{a.type}" for a in ent.attributes)
            lines.append(f"  [L{levels.get(ent.name, 1)}] {ent.name}"
                         f"(pk={ent.primary_key}) {types}")
        for rel in self.relationships:
            lines.append(f"  FK {rel.as_fk()}  [{rel.cardinality}]")
        return "\n".join(lines)


__all__ = [
    "Attribute",
    "EntitySchema",
    "Relationship",
    "SchemaGraph",
    "SchemaValidationError",
    "PRIMITIVE_TYPES",
    "TYPE_ALIASES",
    "CARDINALITY",
    "DEFAULT_PRIMARY_KEY",
    "normalize_type",
    "snake_case",
    "pascal_case",
]
