"""Stages 3 & 4: populate a schema with concrete records and link them up.

    SchemaGraph -> ParametricInstanceGenerator -> InstanceGraph -> instances.json

The division of labour here is the load-bearing decision. The model supplies
**field values** — the part that has to read as real domain content — and this
module supplies **structure**: identifiers, which child points at which parent,
which optional field is null, which foreign key is deliberately dangling.

Structure is not asked of the model for the same reason Stage 2 enforces rather
than relays: instances.json is ground truth. A join the model invented cannot be
verified, whereas a join this module made is known by construction, and
``--null-probability`` / ``--orphan-rate`` mean nothing unless the rate actually
holds. Everything structural therefore runs off one seeded RNG, so the same
``--seed`` reproduces the same graph down to which row was nulled.

Entities are populated parent-before-child, so a child's foreign key is always
bound to a parent row that already exists.
"""

from __future__ import annotations

import datetime
import logging
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .instance_types import ORPHAN_MARKER, InstanceGraph, Record
from .schema_types import (Attribute, EntitySchema, Relationship, SchemaGraph,
                           snake_case)

log = logging.getLogger(__name__)

#: Values are asked for in the domain's own register, but the *shape* example is
#: kept in an unrelated domain for the reason recorded in
#: ``schema_generator.SCHEMA_GENERATION_SYSTEM_PROMPT``: an example drawn from
#: the target domain stops demonstrating shape and starts supplying content.
INSTANCE_GENERATION_SYSTEM_PROMPT = """\
You invent realistic sample data. You are given one entity from a schema and \
asked for a number of rows of it, and you return them as JSON.

Output rules:
- Return ONE JSON object and nothing else. No prose, no markdown fences.
- The object has exactly one key, "records", whose value is a list of objects.
- Each object in "records" has exactly the attribute names you were given, and \
no others. Every attribute must be present on every record.
- Every value is a JSON string, number or boolean. Never null, and never a \
nested object or list.

Write values that fit the attribute's stated type:
  string    a short phrase             text      one or two plain sentences
  integer   a whole number             decimal   a number with decimals
  currency  an amount with its symbol, e.g. "$1,240.50"
  percent   e.g. "12%"                 boolean   true or false
  date      YYYY-MM-DD                 datetime  YYYY-MM-DD HH:MM
  email     a plausible address        phone     a plausible number
  url       a plausible web address    address   a full postal address
  enum      exactly one of the listed choices, copied verbatim

Make the rows differ from each other, and make them read like records a real \
organisation in the stated domain would hold: names, places and amounts that \
belong together on the same row. Do not number them ("Item 1", "Item 2") and \
do not repeat one row with small edits.

Shape only, for an entity from an unrelated domain -- two rows of \
Apiary(site_name: string, location: address, hive_count: integer):
{"records": [
  {"site_name": "Ridgeway Yard", "location": "118 Culvert Lane, Hexham",
   "hive_count": 14},
  {"site_name": "Old Orchard", "location": "7 Mill Bank, Corbridge",
   "hive_count": 6}]}
"""

_RETRY_NUDGE = """\

Your previous response could not be used: {reason}
Return ONLY the JSON object described above: {{"records": [ ... ]}} with \
exactly {count} record(s).
"""

#: Fixed epoch for synthesised dates. A wall-clock "today" would make the same
#: seed produce different data tomorrow, which defeats the point of the seed.
_EPOCH = datetime.date(2020, 1, 1)

#: Reserved-by-RFC hosts and the fictional +1-555-01xx phone block, so nothing
#: synthesised here can resolve to a real address, mailbox or subscriber.
_SAFE_DOMAIN = "example.invalid"


class InstanceGenerationError(RuntimeError):
    """No usable records could be produced for an entity."""


def topological_order(schema: SchemaGraph) -> List[EntitySchema]:
    """Entities ordered so every parent precedes its children.

    Kahn's algorithm over the parent -> child edges. Stage 2 guarantees a
    forest, but a cycle is still handled rather than hung on: whatever cannot be
    settled is appended in declaration order, because refusing to populate a
    schema is worse than populating it in an imperfect order.
    """
    entities = {e.name: e for e in schema.entities}
    indegree = {name: 0 for name in entities}
    children: Dict[str, List[str]] = {name: [] for name in entities}
    for rel in schema.relationships:
        if rel.parent_entity not in entities or rel.child_entity not in entities:
            continue
        children[rel.parent_entity].append(rel.child_entity)
        indegree[rel.child_entity] += 1

    # Declaration order among the ready set keeps the output stable and makes
    # the ordering predictable to read against schema.json.
    ready = [name for name in entities if indegree[name] == 0]
    ordered: List[str] = []
    while ready:
        name = ready.pop(0)
        ordered.append(name)
        for child in children[name]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)

    if len(ordered) < len(entities):  # cycle remnants
        ordered.extend(n for n in entities if n not in set(ordered))
    return [entities[n] for n in ordered]


class ParametricInstanceGenerator:
    """Populates a :class:`SchemaGraph` with linked records.

    The client is injected so tests never need a server; ``seed`` drives every
    structural choice, so a seeded run is reproducible even though the model's
    values are not pinned by it alone.
    """

    def __init__(self, client: Any = None, *,
                 seed: Optional[int] = None,
                 max_attempts: int = 3) -> None:
        if client is None:
            from .llm_bridge import build_client
            client = build_client(seed=seed)
        self.client = client
        self.seed = seed
        self.max_attempts = max(1, int(max_attempts))
        self.rng = random.Random(seed)

    # -- public api --------------------------------------------------------- #
    def generate_instances(self, schema: SchemaGraph, records_per_entity: int,
                           null_prob: float,
                           orphan_rate: float) -> InstanceGraph:
        """Generate ``records_per_entity`` rows of every entity, and link them.

        Raises :class:`ValueError` for out-of-range parameters and
        :class:`InstanceGenerationError` if an entity never produced usable
        values.
        """
        if records_per_entity < 1:
            raise ValueError(
                f"records_per_entity must be >= 1, got {records_per_entity}")
        if not 0.0 <= null_prob <= 1.0:
            raise ValueError(f"null_probability must be in [0, 1], got {null_prob}")
        if not 0.0 <= orphan_rate <= 1.0:
            raise ValueError(f"orphan_rate must be in [0, 1], got {orphan_rate}")

        # Reset so one generator instance reused across calls still reproduces.
        self.rng = random.Random(self.seed)
        graph = InstanceGraph(schema_domain=schema.domain)
        order = topological_order(schema)
        log.info("populating %d entities in order: %s", len(order),
                 " -> ".join(e.name for e in order))

        # child entity -> the foreign keys it must bind, from Stage 2's graph.
        incoming: Dict[str, List[Relationship]] = {}
        for rel in schema.relationships:
            incoming.setdefault(rel.child_entity, []).append(rel)

        for entity in order:
            relationships = incoming.get(entity.name, [])
            values = self._request_values(schema.domain, entity,
                                          records_per_entity, graph,
                                          relationships)
            for stranded in self._stranded_keys(entity, relationships):
                graph.warnings.append(
                    f"{entity.name}.{stranded.name}: id-typed column with no "
                    f"relationship behind it in the schema; assigned locally "
                    f"so it cannot be mistaken for a join")
            plans = self._plan_foreign_keys(entity, relationships, graph,
                                            records_per_entity, orphan_rate)
            for index, row in enumerate(values):
                record = Record(
                    id=self._record_id(entity, index),
                    entity_name=entity.name,
                )
                record.attributes[entity.primary_key] = record.id
                self._fill_attributes(record, entity, row, relationships,
                                      null_prob, graph)
                for column, (target, orphaned) in plans[index].items():
                    record.foreign_keys[column] = target
                    if orphaned:
                        record.orphaned_keys.append(column)
                graph.add(record)

        graph.metadata = self._metadata(schema, records_per_entity, null_prob,
                                        orphan_rate, order)
        for warning in graph.warnings:
            log.warning("instance repair: %s", warning)
        return graph

    # -- identity ----------------------------------------------------------- #
    @staticmethod
    def _record_id(entity: EntitySchema, index: int) -> str:
        """``Customer`` row 0 -> ``customer-001``. Stable and human-readable."""
        return f"{snake_case(entity.name)}-{index + 1:03d}"

    def _metadata(self, schema: SchemaGraph, records_per_entity: int,
                  null_prob: float, orphan_rate: float,
                  order: Sequence[EntitySchema]) -> Dict[str, Any]:
        client = self.client
        return {
            "stage": "3-4:instance_graph",
            "domain": schema.domain,
            "records_per_entity": records_per_entity,
            "null_probability": null_prob,
            "orphan_rate": orphan_rate,
            "seed": self.seed,
            "topological_order": [e.name for e in order],
            "model": getattr(client, "model", None),
            "base_url": getattr(client, "base_url", None),
            "backend": getattr(client, "backend", None),
            "generated_at": datetime.datetime.now(
                datetime.timezone.utc).replace(microsecond=0).isoformat(),
        }

    # -- values from the model ---------------------------------------------- #
    @staticmethod
    def _value_attributes(entity: EntitySchema,
                          relationships: Sequence[Relationship]) -> List[Attribute]:
        """Attributes the model is asked for: everything except identifiers.

        The primary key is this module's to assign and the foreign keys are its
        to bind, so asking the model for either would only invite a value that
        has to be thrown away.

        Every *other* ``id``-typed column is withheld too, which is not
        cosmetic. A schema can carry a key column with no relationship behind it
        — Stage 2's forest rule drops the second parent of a child but keeps the
        column — and a live run showed exactly what happens then: asked for
        ``Enrollment.course_id``, the model returned "EDU-101", which reads as a
        join to Course and resolves to nothing. A scorer cannot tell that from a
        real reference, so identity stays wholly this module's: if a value looks
        like a key, this module assigned it.
        """
        bound = {rel.child_attribute for rel in relationships}
        return [a for a in entity.attributes
                if a.name != entity.primary_key and a.name not in bound
                and a.type != "id"]

    @staticmethod
    def _stranded_keys(entity: EntitySchema,
                       relationships: Sequence[Relationship]) -> List[Attribute]:
        """``id``-typed columns that are neither the primary key nor a join."""
        bound = {rel.child_attribute for rel in relationships}
        return [a for a in entity.attributes
                if a.name != entity.primary_key and a.name not in bound
                and a.type == "id"]

    def _request_values(self, domain: str, entity: EntitySchema, count: int,
                        graph: InstanceGraph,
                        relationships: Sequence[Relationship] = ()
                        ) -> List[Dict[str, Any]]:
        """One call per entity, returning exactly ``count`` coerced rows."""
        wanted = self._value_attributes(entity, relationships)
        if not wanted:
            # A key-only entity (a pure join table) needs no values at all.
            return [{} for _ in range(count)]

        user = self._build_user_prompt(domain, entity, wanted, count)
        reasons: List[str] = []
        rows: List[Dict[str, Any]] = []

        for attempt in range(1, self.max_attempts + 1):
            prompt = user if attempt == 1 else user + _RETRY_NUDGE.format(
                reason=reasons[-1], count=count)
            payload = self.client.complete_json(
                INSTANCE_GENERATION_SYSTEM_PROMPT, prompt)
            if payload is None:
                reasons.append(str(getattr(self.client, "last_error", None)
                                   or "no JSON in the response"))
                continue
            rows = self._extract_rows(payload)
            if not rows:
                reasons.append("no 'records' list in the response")
                continue
            break

        if not rows:
            raise InstanceGenerationError(
                f"no usable records for {entity.name} after "
                f"{self.max_attempts} attempt(s): " + "; ".join(reasons))

        if len(rows) > count:
            graph.warnings.append(
                f"{entity.name}: model returned {len(rows)} rows, {count} "
                f"requested; kept the first {count}")
            rows = rows[:count]
        while len(rows) < count:
            # Padding rather than a hard failure: every later stage is indexed
            # by record count, and a short entity would silently shrink the
            # benchmark. Loud, but recoverable.
            graph.warnings.append(
                f"{entity.name}: model returned {len(rows)} of {count} rows; "
                f"synthesised row {len(rows) + 1} locally")
            rows.append({})
        return rows

    @staticmethod
    def _build_user_prompt(domain: str, entity: EntitySchema,
                           wanted: Sequence[Attribute], count: int) -> str:
        lines = [f"Domain: {domain}",
                 f"Entity: {entity.name}"]
        if entity.description:
            lines.append(f"Description: {entity.description}")
        lines.append(f"Rows requested: {count}")
        lines.append("Attributes:")
        for attr in wanted:
            spec = f"  {attr.name} ({attr.type}"
            if attr.type == "enum" and attr.values:
                spec += ": one of " + ", ".join(f'"{v}"' for v in attr.values)
            spec += ")"
            if attr.description:
                spec += f" -- {attr.description}"
            lines.append(spec)
        lines.append("")
        lines.append(f'Return {count} record(s) with exactly these attribute '
                     f'names. Return the JSON object now.')
        return "\n".join(lines)

    @staticmethod
    def _extract_rows(payload: Any) -> List[Dict[str, Any]]:
        """Find the row list in whatever the model wrapped it in."""
        if isinstance(payload, list):
            candidate = payload
        elif isinstance(payload, dict):
            candidate = None
            for key in ("records", "rows", "data", "instances", "items"):
                if isinstance(payload.get(key), list):
                    candidate = payload[key]
                    break
            if candidate is None:
                # A lone record returned unwrapped.
                candidate = [payload] if payload else []
        else:
            return []
        return [row for row in candidate if isinstance(row, dict)]

    # -- attribute filling -------------------------------------------------- #
    def _fill_attributes(self, record: Record, entity: EntitySchema,
                         row: Dict[str, Any],
                         relationships: Sequence[Relationship],
                         null_prob: float, graph: InstanceGraph) -> None:
        """Write one record's non-key attributes, applying nulls and coercion.

        Nulls are drawn per field so the rate holds across the whole graph
        rather than per row. Required attributes and keys are never nulled: a
        null primary key is not noise, it is a broken record.
        """
        supplied = {snake_case(k): v for k, v in row.items()}
        bound = {rel.child_attribute for rel in relationships}
        asked = {a.name for a in self._value_attributes(entity, relationships)}

        for attr in entity.attributes:
            if attr.name == entity.primary_key or attr.name in bound:
                continue
            if not attr.required and self.rng.random() < null_prob:
                record.attributes[attr.name] = None
                continue
            if attr.name not in asked:
                # A stranded key column: never asked of the model, so filled
                # here. Reported once per entity by the caller, not per row.
                record.attributes[attr.name] = self._synthesize(attr, record.id)
                continue
            if attr.name in supplied and supplied[attr.name] is not None:
                record.attributes[attr.name] = self._coerce(
                    attr, supplied[attr.name], entity, graph)
            else:
                record.attributes[attr.name] = self._synthesize(attr, record.id)
                graph.warnings.append(
                    f"{record.id}.{attr.name}: absent from the model's row; "
                    f"value synthesised locally")

    def _coerce(self, attr: Attribute, value: Any, entity: EntitySchema,
                graph: InstanceGraph) -> Any:
        """Keep the model's formatting, but hold it to the declared type.

        Formatting is deliberately preserved — "$1,240.50" is what a document
        should show, and re-deriving it from 1240.5 would lose the currency. Only
        two things are enforced: an ``enum`` must be one of its declared values,
        and a ``boolean`` must be a bool.
        """
        if isinstance(value, (dict, list)):
            graph.warnings.append(
                f"{entity.name}.{attr.name}: nested value flattened to text")
            return " ".join(str(v) for v in
                            (value.values() if isinstance(value, dict) else value))
        if attr.type == "boolean" and not isinstance(value, bool):
            return str(value).strip().lower() in ("true", "yes", "y", "1")
        if attr.type == "enum" and attr.values:
            for choice in attr.values:
                if str(value).strip().lower() == str(choice).strip().lower():
                    return choice
            replacement = self.rng.choice(attr.values)
            graph.warnings.append(
                f"{entity.name}.{attr.name}: {value!r} is not a declared enum "
                f"value; replaced with {replacement!r}")
            return replacement
        return value

    def _synthesize(self, attr: Attribute, record_id: str) -> Any:
        """A type-appropriate placeholder, for a field the model omitted.

        Reserved hosts and the fictional +1-555-01xx block only, so nothing
        synthesised here can resolve to a real address, mailbox or subscriber.
        """
        rng = self.rng
        label = attr.name.replace("_", " ")
        if attr.type == "integer":
            return rng.randint(1, 999)
        if attr.type == "decimal":
            return round(rng.uniform(1, 1000), 2)
        if attr.type == "currency":
            return f"${rng.uniform(10, 5000):,.2f}"
        if attr.type == "percent":
            return f"{rng.randint(0, 100)}%"
        if attr.type == "boolean":
            return rng.choice([True, False])
        if attr.type in ("date", "datetime"):
            day = _EPOCH + datetime.timedelta(days=rng.randint(0, 2000))
            if attr.type == "date":
                return day.isoformat()
            return f"{day.isoformat()} {rng.randint(8, 18):02d}:{rng.randrange(0, 60, 5):02d}"
        if attr.type == "email":
            return f"{snake_case(record_id).replace('_', '.')}@{_SAFE_DOMAIN}"
        if attr.type == "phone":
            return f"+1-555-01{rng.randint(0, 99):02d}"
        if attr.type == "url":
            return f"https://{_SAFE_DOMAIN}/{snake_case(record_id)}"
        if attr.type == "address":
            return f"{rng.randint(1, 400)} Example Street, Example City"
        if attr.type == "enum":
            return rng.choice(attr.values) if attr.values else "unspecified"
        if attr.type == "id":
            return f"{snake_case(attr.name)}-{rng.randint(1000, 9999)}"
        if attr.type == "text":
            return (f"Placeholder {label} recorded against {record_id}. "
                    f"No value was supplied for this field.")
        return f"Unspecified {label}"

    # -- foreign keys ------------------------------------------------------- #
    def _plan_foreign_keys(self, entity: EntitySchema,
                           relationships: Sequence[Relationship],
                           graph: InstanceGraph, count: int,
                           orphan_rate: float
                           ) -> List[Dict[str, Tuple[Optional[str], bool]]]:
        """Decide every child row's parent up front.

        Round-robin over a shuffled parent list rather than an independent draw
        per row: with 3 parents and 3 children, independent draws leave a parent
        childless about 60% of the time, and a 1:m schema whose m is empty
        exercises nothing downstream. Round-robin guarantees the join is
        actually populated while the shuffle keeps which-parent-gets-which
        varied and seed-reproducible.
        """
        plans: List[Dict[str, Tuple[Optional[str], bool]]] = [
            {} for _ in range(count)]
        for rel in relationships:
            parent_ids = [r.id for r in graph.by_entity(rel.parent_entity)]
            if not parent_ids:
                graph.warnings.append(
                    f"{entity.name}.{rel.child_attribute}: no "
                    f"{rel.parent_entity} records exist to link to; left null")
                for plan in plans:
                    plan[rel.child_attribute] = (None, False)
                continue

            pool = list(parent_ids)
            self.rng.shuffle(pool)
            orphan_count = self._orphan_count(count, orphan_rate)
            orphan_rows = set(self.rng.sample(range(count), orphan_count)) \
                if orphan_count else set()

            for index in range(count):
                if index in orphan_rows:
                    dangling = (f"{rel.parent_entity.lower()}-{ORPHAN_MARKER}"
                                f"{index + 1}__")
                    plans[index][rel.child_attribute] = (dangling, True)
                else:
                    plans[index][rel.child_attribute] = (
                        pool[index % len(pool)], False)
            if orphan_rows:
                graph.warnings.append(
                    f"{entity.name}.{rel.child_attribute}: orphaned "
                    f"{len(orphan_rows)} of {count} foreign key(s) at "
                    f"orphan_rate={orphan_rate}")
        return plans

    def _orphan_count(self, count: int, orphan_rate: float) -> int:
        """How many of ``count`` rows to orphan.

        The fractional remainder is resolved by one RNG draw rather than
        rounded, so a rate of 0.1 over ten single-row entities orphans about one
        of them instead of always zero — the rate has to survive small entities
        to mean anything.
        """
        if orphan_rate <= 0.0 or count <= 0:
            return 0
        exact = count * orphan_rate
        whole = int(exact)
        if self.rng.random() < (exact - whole):
            whole += 1
        return min(whole, count)


def write_instances(graph: InstanceGraph, path: str) -> str:
    """Serialise ``graph`` to ``path`` as JSON. Returns the path written."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(graph.to_json())
        handle.write("\n")
    return path


__all__ = [
    "ParametricInstanceGenerator",
    "InstanceGenerationError",
    "INSTANCE_GENERATION_SYSTEM_PROMPT",
    "topological_order",
    "write_instances",
]
