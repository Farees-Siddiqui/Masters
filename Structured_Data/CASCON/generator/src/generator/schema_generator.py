"""Stage 1 & 2: ask local Llama 3 for a parametric ER schema, then enforce it.

    domain, num_entities, max_depth
        -> ParametricSchemaGenerator -> SchemaGraph -> schema.json

Stage 1 is the parameter set (what domain, how wide, how deep); Stage 2 is the
schema graph itself. Both live here because the parameters are only meaningful
as constraints on the graph, and the constraints have to be *enforced* rather
than merely requested: a local model asked for four entities at depth two will
sometimes return five, a foreign key onto a table it never declared, or a chain
three levels deep. Later stages read this file as ground truth, so anything the
model got wrong is repaired here and recorded in ``warnings`` — a schema that
silently violates its own parameters would poison every document generated from
it.

Repairs are always subtractive on the relationship side (drop the offending
edge, keep the entity as a new root) and additive on the attribute side
(synthesise the missing key column). Nothing is invented that the schema does
not already imply.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .llm_bridge import DEFAULT_MODEL, build_client
from .schema_types import (CARDINALITY, Attribute, EntitySchema, Relationship,
                           SchemaGraph, SchemaValidationError, normalize_type,
                           pascal_case, snake_case)

log = logging.getLogger(__name__)

#: The generator's whole premise is in this prompt: the schema is *invented* to
#: order, so the parameters have to survive the trip into it.
#:
#: The example is a beekeeping registry, chosen because it shares no vocabulary
#: with any target domain this pipeline is aimed at (medical, small_business,
#: education). Same discipline as the extraction prompt in
#: ``layout_pipeline/src/ie_engine/llm_client.py``: an example drawn from a
#: target domain stops demonstrating *shape* and starts supplying *content*, and
#: the model then returns the example's entities back with a new label on them.
SCHEMA_GENERATION_SYSTEM_PROMPT = """\
You are a data architect. You design relational schemas to order and return \
them as JSON.

Output rules:
- Return ONE JSON object and nothing else. No prose, no markdown fences.
- The object has exactly two keys: "entities" and "relationships".

"entities" is a list. Each entry is an object:
  "name"         PascalCase, singular, named for one real thing in the domain.
  "description"  one short sentence.
  "primary_key"  the name of this entity's key attribute.
  "attributes"   a list of {"name", "type", "description", "required"}.
                 "name" is snake_case. Give each entity 4 to 8 attributes.

Every "type" must be one of exactly these, and nothing else:
  string, text, integer, decimal, currency, percent, boolean, date, datetime,
  email, phone, url, address, id, enum
Choose the most specific one that fits: a total is currency, not decimal; a \
postal destination is address, not string; a key column is id.
An "enum" attribute must also carry "values": a list of 2 to 6 literal choices.

"relationships" is a list of one-to-many foreign keys, each written as a single \
string in exactly this form:
  "Child.foreign_key_column -> Parent.primary_key"
The child is the MANY side and the parent is the ONE side, so "one parent row \
has many child rows" must read true.

Hard constraints:
- Every entity declares its own key attribute, of type id.
- Every foreign key column named on the left of "->" must also appear in that \
child entity's own "attributes" list, with type id.
- Both entities named in a relationship must exist in "entities".
- One-to-many only. No many-to-many, and no join or bridge entities.
- The hierarchy must be a tree or forest: no entity is its own ancestor, and \
each entity has at most one parent.
- Return exactly the number of entities requested, and do not exceed the \
requested nesting depth.

Shape only, from an unrelated domain -- do not reuse these names or invent a \
schema about bees:
{"entities": [
  {"name": "Apiary", "description": "A registered hive site.",
   "primary_key": "id",
   "attributes": [
     {"name": "id", "type": "id", "description": "Key.", "required": true},
     {"name": "site_name", "type": "string", "description": "Yard name.", "required": true},
     {"name": "location", "type": "address", "description": "Where the yard is.", "required": true},
     {"name": "registered_on", "type": "date", "description": "Registration date.", "required": true},
     {"name": "hive_count", "type": "integer", "description": "Colonies on site.", "required": false}]},
  {"name": "Inspection", "description": "One visit to an apiary.",
   "primary_key": "id",
   "attributes": [
     {"name": "id", "type": "id", "description": "Key.", "required": true},
     {"name": "apiary_id", "type": "id", "description": "Apiary inspected.", "required": true},
     {"name": "visited_on", "type": "date", "description": "Visit date.", "required": true},
     {"name": "verdict", "type": "enum", "description": "Outcome.", "required": true,
      "values": ["clear", "advisory", "quarantine"]},
     {"name": "levy", "type": "currency", "description": "Fee charged.", "required": false}]}],
 "relationships": ["Inspection.apiary_id -> Apiary.id"]}
"""

#: Appended when a first attempt comes back unusable. Naming the failure works
#: better than repeating the whole instruction set.
_RETRY_NUDGE = """\

Your previous response could not be used: {reason}
Return ONLY the JSON object described above, with both "entities" and \
"relationships" keys present.
"""


class SchemaGenerationError(RuntimeError):
    """The model never produced a usable ER graph."""


def build_user_prompt(domain: str, num_entities: int, max_depth: int) -> str:
    """The Stage 1 parameters, stated as a request.

    Depth is spelled out in levels because "depth 2" is otherwise read as
    "two links below the root" about as often as "two levels in total".
    """
    if max_depth <= 1:
        depth_clause = (
            "Nesting depth: 1. Return NO relationships at all -- every entity "
            "must stand alone with no foreign keys between them."
        )
    else:
        depth_clause = (
            f"Nesting depth: at most {max_depth} levels. A top-level entity "
            f"with no parent is level 1; a child of it is level 2. The longest "
            f"parent-to-child chain must therefore not exceed {max_depth} "
            f"entities, so a level-{max_depth} entity must have no children of "
            f"its own."
        )
    return (
        f"Domain: {domain}\n"
        f"Number of entities: exactly {num_entities}.\n"
        f"{depth_clause}\n"
        f"\nDesign the schema a real organisation in this domain would keep: "
        f"the entities its records are actually about, the attributes those "
        f"records actually carry, and the one-to-many links between them. "
        f"Return the JSON object now."
    )


class ParametricSchemaGenerator:
    """Turns Stage 1 parameters into a validated :class:`SchemaGraph`.

    The client is injected so tests never need a server; left out, it defaults
    to the local Llama 3 endpoint at ``127.0.0.1:11434`` via
    :func:`llm_bridge.build_client`.
    """

    def __init__(self, client: Any = None, *,
                 model: str = DEFAULT_MODEL,
                 seed: Optional[int] = None,
                 max_attempts: int = 3) -> None:
        self.client = client if client is not None \
            else build_client(model=model, seed=seed)
        self.seed = seed
        self.max_attempts = max(1, int(max_attempts))
        #: Raw text of the last call, kept for debugging a bad generation.
        self.last_payload: Optional[Any] = None

    # -- public api --------------------------------------------------------- #
    def generate_schema(self, domain: str, num_entities: int,
                        max_depth: int) -> SchemaGraph:
        """Ask for a schema, validate it, and enforce the parameters.

        Raises :class:`SchemaGenerationError` if every attempt came back
        unusable, and :class:`ValueError` for out-of-range parameters — a
        request for zero entities is a caller bug, not a bad generation.
        """
        if num_entities < 1:
            raise ValueError(f"num_entities must be >= 1, got {num_entities}")
        if max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")

        system = SCHEMA_GENERATION_SYSTEM_PROMPT
        user = build_user_prompt(domain, num_entities, max_depth)
        reasons: List[str] = []

        for attempt in range(1, self.max_attempts + 1):
            prompt = user if attempt == 1 else \
                user + _RETRY_NUDGE.format(reason=reasons[-1])
            log.info("requesting schema for %r (%d entities, depth %d), "
                     "attempt %d/%d", domain, num_entities, max_depth,
                     attempt, self.max_attempts)
            payload = self.client.complete_json(system, prompt)
            self.last_payload = payload

            if payload is None:
                reason = getattr(self.client, "last_error", None) \
                    or "no JSON in the response"
                reasons.append(str(reason))
                log.warning("attempt %d produced no JSON: %s", attempt, reason)
                continue

            try:
                graph = SchemaGraph.from_payload(payload, domain=domain)
            except SchemaValidationError as exc:
                reasons.append(str(exc))
                log.warning("attempt %d failed validation: %s", attempt, exc)
                continue

            graph.warnings.extend(self._audit_types(payload))
            self._enforce(graph, num_entities=num_entities,
                          max_depth=max_depth)
            graph.metadata = self._metadata(domain, num_entities, max_depth,
                                            attempt)
            for warning in graph.warnings:
                log.warning("schema repair: %s", warning)
            return graph

        raise SchemaGenerationError(
            f"no usable schema after {self.max_attempts} attempt(s) for domain "
            f"{domain!r}: " + "; ".join(reasons))

    # -- provenance --------------------------------------------------------- #
    def _metadata(self, domain: str, num_entities: int, max_depth: int,
                  attempts: int) -> Dict[str, Any]:
        client = self.client
        return {
            "stage": "1-2:parametric_schema",
            "domain": domain,
            "requested_entities": num_entities,
            "max_depth": max_depth,
            "seed": self.seed,
            "attempts": attempts,
            "model": getattr(client, "model", None),
            "base_url": getattr(client, "base_url", None),
            "backend": getattr(client, "backend", None),
            "temperature": getattr(client, "temperature", None),
            "generated_at": datetime.datetime.now(
                datetime.timezone.utc).replace(microsecond=0).isoformat(),
        }

    # -- validation --------------------------------------------------------- #
    @staticmethod
    def _audit_types(payload: Any) -> List[str]:
        """Report attribute types that had to be coerced.

        Run against the *raw* payload, because ``Attribute.__post_init__`` has
        already normalised by the time a dataclass exists and the original
        spelling is gone.
        """
        notes: List[str] = []
        entities = payload.get("entities") if isinstance(payload, dict) else None
        if isinstance(entities, dict):
            entities = list(entities.values())
        if not isinstance(entities, list):
            return notes
        for ent in entities:
            if not isinstance(ent, dict):
                continue
            name = ent.get("name") or ent.get("table") or "?"
            attrs = ent.get("attributes") or ent.get("fields") or \
                ent.get("columns") or []
            if isinstance(attrs, dict):
                attrs = [{"name": k, "type": v} for k, v in attrs.items()]
            if not isinstance(attrs, list):
                continue
            for attr in attrs:
                if not isinstance(attr, dict):
                    continue
                raw = attr.get("type") or attr.get("data_type") or \
                    attr.get("dtype")
                if raw is None:
                    continue
                coerced, known = normalize_type(raw)
                if not known:
                    notes.append(
                        f"{pascal_case(name)}.{snake_case(attr.get('name'))}: "
                        f"unknown type {str(raw)!r} coerced to {coerced!r}")
        return notes

    def _enforce(self, graph: SchemaGraph, *, num_entities: int,
                 max_depth: int) -> SchemaGraph:
        """Make the graph satisfy the Stage 1 parameters, in place.

        Order matters. Referential integrity first (an edge onto a
        non-existent entity cannot be reasoned about), then entity count, then
        depth — trimming entities removes edges, so checking depth first would
        drop edges that the trim was about to remove anyway.
        """
        self._enforce_entity_names(graph)
        self._enforce_referential_integrity(graph)
        self._enforce_entity_count(graph, num_entities)
        self._enforce_depth(graph, max_depth)
        self._enforce_keys(graph)
        return graph

    @staticmethod
    def _enforce_entity_names(graph: SchemaGraph) -> None:
        """Collapse duplicate entity names, keeping the richer definition."""
        kept: Dict[str, EntitySchema] = {}
        for ent in graph.entities:
            existing = kept.get(ent.name)
            if existing is None:
                kept[ent.name] = ent
                continue
            graph.warnings.append(
                f"entity {ent.name!r} declared twice; merged attributes")
            for attr in ent.attributes:
                existing.add_attribute(attr)
        graph.entities = list(kept.values())

    @staticmethod
    def _enforce_referential_integrity(graph: SchemaGraph) -> None:
        """Drop edges that cannot hold: unknown side, self-loop, duplicate, cycle.

        Also drops a second parent for an already-parented child. The hierarchy
        has to be a forest for a document to be rendered from it: a node with
        two parents would have to appear under both, and a node that is its own
        ancestor could not be rendered at all. Cycles are caught here rather
        than left to the depth pass, which would only notice them when the
        implied depth happened to exceed ``max_depth``.
        """
        known = set(graph.entity_names)
        kept: List[Relationship] = []
        seen: Set[Tuple[str, str, str]] = set()
        parent_of: Dict[str, str] = {}

        def creates_cycle(child: str, parent: str) -> bool:
            """True if ``parent`` already descends from ``child``."""
            walker: Optional[str] = parent
            hops = 0
            while walker is not None and hops <= len(known):
                if walker == child:
                    return True
                walker = parent_of.get(walker)
                hops += 1
            return False

        for rel in graph.relationships:
            if rel.parent_entity not in known:
                graph.warnings.append(
                    f"dropped FK {rel.as_fk()}: parent entity "
                    f"{rel.parent_entity!r} is not in the schema")
                continue
            if rel.child_entity not in known:
                graph.warnings.append(
                    f"dropped FK {rel.as_fk()}: child entity "
                    f"{rel.child_entity!r} is not in the schema")
                continue
            if rel.parent_entity == rel.child_entity:
                graph.warnings.append(
                    f"dropped FK {rel.as_fk()}: an entity cannot be its own "
                    f"parent")
                continue
            key = (rel.child_entity, rel.child_attribute, rel.parent_entity)
            if key in seen:
                graph.warnings.append(f"dropped duplicate FK {rel.as_fk()}")
                continue
            if rel.child_entity in parent_of:
                graph.warnings.append(
                    f"dropped FK {rel.as_fk()}: {rel.child_entity} already has "
                    f"a parent, and the hierarchy must be a forest")
                continue
            if creates_cycle(rel.child_entity, rel.parent_entity):
                graph.warnings.append(
                    f"dropped FK {rel.as_fk()}: {rel.parent_entity} already "
                    f"descends from {rel.child_entity}, and no entity may be "
                    f"its own ancestor")
                continue
            if rel.cardinality != CARDINALITY:
                graph.warnings.append(
                    f"FK {rel.as_fk()}: cardinality {rel.cardinality!r} "
                    f"coerced to {CARDINALITY!r}")
                rel.cardinality = CARDINALITY
            seen.add(key)
            parent_of[rel.child_entity] = rel.parent_entity
            kept.append(rel)

        graph.relationships = kept

    @staticmethod
    def _trim_priority(graph: SchemaGraph) -> List[str]:
        """Entity names in the order they should be *kept* when over budget.

        Connected groups of entities come first, largest first, and within a
        group parents precede their children. Any prefix of this list is
        therefore a valid schema: every kept entity's parent is also kept.

        Preferring linked groups over standalone entities is the whole point.
        The obvious policy — breadth-first from the roots — measured badly
        against a real llama3.3:70b response for ``small_business``: the model
        returned six entities of which four happened to be parentless, so the
        trim kept exactly those four and dropped Order and OrderItem, the only
        two carrying the hierarchy. A ``--max-depth 2`` request came back flat
        with zero joins. Ranking by retained structure keeps the tree and drops
        the standalone entities instead.
        """
        by_name = {e.name: e for e in graph.entities}
        declared = {name: i for i, name in enumerate(by_name)}
        parents: Dict[str, List[str]] = {n: [] for n in by_name}
        children: Dict[str, List[str]] = {n: [] for n in by_name}
        for rel in graph.relationships:
            children[rel.parent_entity].append(rel.child_entity)
            parents[rel.child_entity].append(rel.parent_entity)

        # Weakly connected components: a group is kept or dropped as a unit,
        # because half a hierarchy is worth less than a whole smaller one.
        groups: List[List[str]] = []
        seen: Set[str] = set()
        for name in by_name:
            if name in seen:
                continue
            group, queue = [], [name]
            seen.add(name)
            while queue:
                current = queue.pop(0)
                group.append(current)
                for neighbour in children[current] + parents[current]:
                    if neighbour not in seen:
                        seen.add(neighbour)
                        queue.append(neighbour)
            groups.append(group)

        def group_edges(group: Sequence[str]) -> int:
            members = set(group)
            return sum(1 for rel in graph.relationships
                       if rel.child_entity in members
                       and rel.parent_entity in members)

        groups.sort(key=lambda g: (-group_edges(g), -len(g),
                                   min(declared[n] for n in g)))

        order: List[str] = []
        for group in groups:
            members = set(group)
            roots = [n for n in group if not parents[n]]
            queue = sorted(roots or group, key=lambda n: declared[n])
            visited = set(queue)
            while queue:
                current = queue.pop(0)
                order.append(current)
                for child in sorted(children[current], key=lambda n: declared[n]):
                    if child in members and child not in visited:
                        visited.add(child)
                        queue.append(child)
            # Anything a cycle kept out of the walk still has to be listed.
            order.extend(sorted((n for n in group if n not in visited),
                                key=lambda n: declared[n]))
        return order

    @staticmethod
    def _enforce_entity_count(graph: SchemaGraph, num_entities: int) -> None:
        """Trim to ``num_entities``, keeping as much structure as possible.

        An under-count is only reported. The graph is still internally
        consistent, and re-prompting for one more entity would invalidate
        everything already agreed.
        """
        if len(graph.entities) <= num_entities:
            if len(graph.entities) < num_entities:
                graph.warnings.append(
                    f"model returned {len(graph.entities)} entities, "
                    f"{num_entities} requested")
            return

        order = ParametricSchemaGenerator._trim_priority(graph)
        keep = set(order[:num_entities])
        dropped = order[num_entities:]
        graph.warnings.append(
            f"model returned {len(graph.entities)} entities, {num_entities} "
            f"requested; dropped {', '.join(dropped)}")
        # Declaration order is preserved among the survivors; only membership
        # is decided by priority.
        graph.entities = [e for e in graph.entities if e.name in keep]
        surviving = []
        for rel in graph.relationships:
            if rel.parent_entity in keep and rel.child_entity in keep:
                surviving.append(rel)
            else:
                graph.warnings.append(
                    f"dropped FK {rel.as_fk()}: it references a trimmed entity")
        graph.relationships = surviving

    @staticmethod
    def _enforce_depth(graph: SchemaGraph, max_depth: int) -> None:
        """Keep edges only while the graph stays within ``max_depth`` levels.

        Edges are re-admitted one at a time in the order the model gave them,
        and any edge that would push some entity past level ``max_depth`` is
        dropped — its child simply becomes another root. Greedy in model order
        rather than optimal: the model's own ordering is the only signal
        available about which links matter most, and a maximum-edge subgraph
        under a depth bound is not worth solving for four tables.
        """
        candidates = graph.relationships
        graph.relationships = []
        for rel in candidates:
            graph.relationships.append(rel)
            depth = graph.depth()
            if depth > max_depth:
                graph.relationships.pop()
                graph.warnings.append(
                    f"dropped FK {rel.as_fk()}: it would make the hierarchy "
                    f"{depth} levels deep, over the max_depth of {max_depth}")

    @staticmethod
    def _enforce_keys(graph: SchemaGraph) -> None:
        """Guarantee every declared key column actually exists.

        Three repairs, all additive: a missing primary key is synthesised at
        position 0; a foreign key column named on the left of ``->`` but absent
        from the child's attributes is appended; and a parent attribute that
        does not exist falls back to the parent's primary key, which does.
        """
        for ent in graph.entities:
            if ent.attribute(ent.primary_key) is None:
                ent.add_attribute(
                    Attribute(name=ent.primary_key, type="id",
                              description=f"Primary key for {ent.name}.",
                              required=True, unique=True), index=0)
                graph.warnings.append(
                    f"{ent.name}: added missing primary key "
                    f"{ent.primary_key!r}")
            else:
                key_attr = ent.attribute(ent.primary_key)
                if key_attr is not None:
                    key_attr.type = "id"
                    key_attr.required = True
                    key_attr.unique = True

        for rel in graph.relationships:
            parent = graph.entity(rel.parent_entity)
            child = graph.entity(rel.child_entity)
            if parent is None or child is None:  # pragma: no cover - pre-filtered
                continue
            if parent.attribute(rel.parent_attribute) is None:
                graph.warnings.append(
                    f"FK {rel.as_fk()}: {rel.parent_entity} has no attribute "
                    f"{rel.parent_attribute!r}; retargeted to its primary key "
                    f"{parent.primary_key!r}")
                rel.parent_attribute = parent.primary_key
            if child.attribute(rel.child_attribute) is None:
                child.add_attribute(Attribute(
                    name=rel.child_attribute, type="id",
                    description=f"References {rel.parent_entity}."
                                f"{rel.parent_attribute}.",
                    required=True))
                graph.warnings.append(
                    f"{child.name}: added missing foreign key column "
                    f"{rel.child_attribute!r} for {rel.as_fk()}")
            else:
                fk_attr = child.attribute(rel.child_attribute)
                if fk_attr is not None and fk_attr.type != "id":
                    graph.warnings.append(
                        f"{child.name}.{fk_attr.name}: type {fk_attr.type!r} "
                        f"coerced to 'id' because it is a foreign key")
                    fk_attr.type = "id"


def write_schema(graph: SchemaGraph, path: str) -> str:
    """Serialise ``graph`` to ``path`` as JSON. Returns the path written."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(graph.to_json())
        handle.write("\n")
    return path


__all__ = [
    "ParametricSchemaGenerator",
    "SchemaGenerationError",
    "SCHEMA_GENERATION_SYSTEM_PROMPT",
    "build_user_prompt",
    "write_schema",
]
