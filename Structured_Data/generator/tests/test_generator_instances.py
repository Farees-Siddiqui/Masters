"""Tests for Stages 3 & 4: instance population and the relational graph.

Every LLM call is mocked. The model's only job in these stages is to supply
field values, so the fixtures return deliberately thin, malformed and oversized
value lists — what is being tested is that the *structure* (identifiers, joins,
nulls, orphans) is correct regardless of what came back.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typer.testing import CliRunner  # noqa: E402

from src.generator import cli as cli_module  # noqa: E402
from src.generator.instance_generator import (  # noqa: E402
    INSTANCE_GENERATION_SYSTEM_PROMPT, InstanceGenerationError,
    ParametricInstanceGenerator, topological_order, write_instances)
from src.generator.instance_types import (ORPHAN_MARKER,  # noqa: E402
                                          InstanceGraph, Record)
from src.generator.schema_types import SchemaGraph  # noqa: E402


def setUpModule() -> None:
    logging.disable(logging.WARNING)


def tearDownModule() -> None:
    logging.disable(logging.NOTSET)


def read_json(*parts: str) -> dict:
    with open(os.path.join(*parts), encoding="utf-8") as handle:
        return json.load(handle)


def make_runner() -> CliRunner:
    if "mix_stderr" in inspect.signature(CliRunner.__init__).parameters:
        return CliRunner(mix_stderr=False)
    return CliRunner()


def stderr_of(result) -> str:
    try:
        return result.stderr
    except (ValueError, AttributeError):  # pragma: no cover - mixed streams
        return result.output


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
#: Company -> {Employee, Customer} -> nothing. Deliberately declared with the
#: child first, so a working topological sort is the only thing that can put
#: Company ahead of it.
TWO_LEVEL_SCHEMA = {
    "domain": "small_business",
    "entities": [
        {"name": "Employee", "primary_key": "id", "attributes": [
            {"name": "id", "type": "id", "required": True},
            {"name": "company_id", "type": "id", "required": True},
            {"name": "full_name", "type": "string", "required": True},
            {"name": "email", "type": "email", "required": False},
            {"name": "job_title", "type": "string", "required": False}]},
        {"name": "Company", "primary_key": "id", "attributes": [
            {"name": "id", "type": "id", "required": True},
            {"name": "company_name", "type": "string", "required": True},
            {"name": "location", "type": "address", "required": True},
            {"name": "industry", "type": "enum", "required": False,
             "values": ["retail", "trades", "hospitality"]}]},
        {"name": "Customer", "primary_key": "id", "attributes": [
            {"name": "id", "type": "id", "required": True},
            {"name": "company_id", "type": "id", "required": True},
            {"name": "customer_name", "type": "string", "required": True},
            {"name": "phone", "type": "phone", "required": False}]},
    ],
    "relationships": ["Employee.company_id -> Company.id",
                      "Customer.company_id -> Company.id"],
}


#: One entity, no joins. For assertions about a single value's handling, where
#: a three-entity schema would drag in unrelated synthesis warnings.
COMPANY_ONLY_SCHEMA = {
    "domain": "small_business",
    "entities": [TWO_LEVEL_SCHEMA["entities"][1]],
    "relationships": [],
}


def schema(payload=None) -> SchemaGraph:
    return SchemaGraph.from_payload(payload or TWO_LEVEL_SCHEMA,
                                    domain="small_business")


COMPANY_ROWS = {"records": [
    {"company_name": "Halvorsen Freight", "location": "42 Quay Road, Hull",
     "industry": "trades"},
    {"company_name": "Kestrel Supplies", "location": "9 Mill Bank, Leeds",
     "industry": "retail"},
]}
EMPLOYEE_ROWS = {"records": [
    {"full_name": "Ingrid Sorensen", "email": "i.sorensen@example.invalid",
     "job_title": "Dispatcher"},
    {"full_name": "Tomas Rowe", "email": "t.rowe@example.invalid",
     "job_title": "Driver"},
]}
CUSTOMER_ROWS = {"records": [
    {"customer_name": "Northgate Cafe", "phone": "+1-555-0142"},
    {"customer_name": "Bell Lane Grocers", "phone": "+1-555-0177"},
]}


class FakeClient:
    """Stands in for SeededLLMClient. Returns queued payloads in order."""

    def __init__(self, *payloads, raises=None):
        self.payloads = list(payloads)
        self.raises = raises
        self.calls = []
        self.last_error = None
        self.model = "llama3.3:70b"
        self.base_url = "http://127.0.0.1:11434"
        self.backend = "ollama"
        self.temperature = 0.0

    def complete_json(self, system, user):
        self.calls.append((system, user))
        if self.raises:
            raise self.raises
        if not self.payloads:
            self.last_error = "no output"
            return None
        payload = self.payloads.pop(0)
        if payload is None:
            self.last_error = "model returned no usable JSON"
        return payload


class CyclingClient(FakeClient):
    """Returns the same payload for every call, however many there are."""

    def __init__(self, payload):
        super().__init__()
        self.payload = payload

    def complete_json(self, system, user):
        self.calls.append((system, user))
        return self.payload


def populate(client=None, records_per_entity=2, null_prob=0.0,
             orphan_rate=0.0, seed=1, schema_graph=None, **kwargs):
    """Run one population against a queued or cycling client."""
    if client is None:
        client = FakeClient(COMPANY_ROWS, EMPLOYEE_ROWS, CUSTOMER_ROWS)
    generator = ParametricInstanceGenerator(client=client, seed=seed, **kwargs)
    return generator.generate_instances(
        schema=schema_graph or schema(), records_per_entity=records_per_entity,
        null_prob=null_prob, orphan_rate=orphan_rate)


# --------------------------------------------------------------------------- #
# 1. Topological sorting
# --------------------------------------------------------------------------- #
class TestTopologicalOrder(unittest.TestCase):

    def test_parents_precede_children(self):
        order = [e.name for e in topological_order(schema())]
        self.assertEqual(order[0], "Company")
        self.assertLess(order.index("Company"), order.index("Employee"))
        self.assertLess(order.index("Company"), order.index("Customer"))
        self.assertEqual(sorted(order), ["Company", "Customer", "Employee"])

    def test_three_level_chain_is_fully_ordered(self):
        graph = SchemaGraph.from_payload({
            "entities": [{"name": n, "attributes": [{"name": "id",
                                                     "type": "id"}]}
                         for n in ("Visit", "Clinic", "Patient")],
            "relationships": ["Visit.patient_id -> Patient.id",
                              "Patient.clinic_id -> Clinic.id"],
        }, domain="medical")
        self.assertEqual([e.name for e in topological_order(graph)],
                         ["Clinic", "Patient", "Visit"])

    def test_declaration_order_breaks_ties(self):
        """Unlinked entities keep schema.json's order, so output is stable."""
        graph = SchemaGraph.from_payload({
            "entities": [{"name": n, "attributes": [{"name": "id",
                                                     "type": "id"}]}
                         for n in ("Zebra", "Aardvark", "Mongoose")],
            "relationships": [],
        }, domain="zoo")
        self.assertEqual([e.name for e in topological_order(graph)],
                         ["Zebra", "Aardvark", "Mongoose"])

    def test_every_entity_survives_a_cycle(self):
        """A cycle cannot be ordered, but it must not lose or hang on entities."""
        graph = SchemaGraph(domain="x", entities=[
            {"name": "A", "attributes": [{"name": "id", "type": "id"}]},
            {"name": "B", "attributes": [{"name": "id", "type": "id"}]}])
        graph.relationships = [
            r for r in SchemaGraph.from_payload(
                {"entities": [{"name": "A"}, {"name": "B"}],
                 "relationships": ["A.b_id -> B.id", "B.a_id -> A.id"]}
            ).relationships]
        order = [e.name for e in topological_order(graph)]
        self.assertEqual(sorted(order), ["A", "B"])

    def test_edges_onto_unknown_entities_are_ignored(self):
        graph = schema()
        graph.relationships.append(
            type(graph.relationships[0])(parent_entity="Ghost",
                                         child_entity="Employee",
                                         child_attribute="ghost_id"))
        order = [e.name for e in topological_order(graph)]
        self.assertEqual(sorted(order), ["Company", "Customer", "Employee"])

    def test_generation_order_is_recorded_in_metadata(self):
        graph = populate()
        self.assertEqual(graph.metadata["topological_order"],
                         ["Company", "Employee", "Customer"])
        # Records are emitted in that order too, so no forward references.
        self.assertEqual([r.entity_name for r in graph.records][:2],
                         ["Company", "Company"])


# --------------------------------------------------------------------------- #
# 2a. Record counts
# --------------------------------------------------------------------------- #
class TestRecordCounts(unittest.TestCase):

    def test_every_entity_gets_the_requested_count(self):
        graph = populate(records_per_entity=2)
        self.assertEqual(graph.counts_by_entity(),
                         {"Company": 2, "Employee": 2, "Customer": 2})
        self.assertEqual(graph.total_records, 6)
        self.assertEqual(len(graph.records), graph.total_records)

    def test_one_llm_call_per_entity(self):
        client = FakeClient(COMPANY_ROWS, EMPLOYEE_ROWS, CUSTOMER_ROWS)
        populate(client=client)
        self.assertEqual(len(client.calls), 3)
        for system, _ in client.calls:
            self.assertEqual(system, INSTANCE_GENERATION_SYSTEM_PROMPT)

    def test_short_response_is_padded_to_the_requested_count(self):
        client = CyclingClient({"records": [
            {"company_name": "Solo", "location": "1 Only Road",
             "industry": "retail"}]})
        graph = populate(client=client, records_per_entity=3)
        self.assertEqual(graph.counts_by_entity()["Company"], 3)
        self.assertTrue(any("synthesised row" in w for w in graph.warnings),
                        graph.warnings)

    def test_oversized_response_is_truncated(self):
        client = CyclingClient({"records": [
            {"company_name": f"Firm {i}", "location": f"{i} Road",
             "industry": "retail"} for i in range(9)]})
        graph = populate(client=client, records_per_entity=2)
        self.assertEqual(graph.counts_by_entity()["Company"], 2)
        self.assertTrue(any("kept the first 2" in w for w in graph.warnings),
                        graph.warnings)

    def test_ids_are_unique_stable_and_entity_scoped(self):
        graph = populate(records_per_entity=2)
        ids = [r.id for r in graph.records]
        self.assertEqual(len(set(ids)), len(ids))
        self.assertEqual([r.id for r in graph.by_entity("Company")],
                         ["company-001", "company-002"])
        # The primary key attribute carries the same value as the record id.
        for record in graph.records:
            self.assertEqual(record.attributes["id"], record.id)

    def test_alternate_row_wrappers_are_accepted(self):
        client = CyclingClient({"rows": [{"company_name": "Alt",
                                          "location": "1 Road",
                                          "industry": "retail"}]})
        graph = populate(client=client, records_per_entity=1)
        self.assertEqual(graph.by_entity("Company")[0]
                         .attributes["company_name"], "Alt")

    def test_unusable_output_raises_after_the_retries(self):
        client = FakeClient(None, None, None)
        with self.assertRaises(InstanceGenerationError):
            populate(client=client)
        self.assertEqual(len(client.calls), 3)

    def test_retry_then_succeed(self):
        client = FakeClient(None, COMPANY_ROWS, EMPLOYEE_ROWS, CUSTOMER_ROWS)
        graph = populate(client=client)
        self.assertEqual(graph.total_records, 6)
        self.assertIn("could not be used", client.calls[1][1])

    def test_invalid_parameters_raise_before_any_call(self):
        client = FakeClient(COMPANY_ROWS)
        generator = ParametricInstanceGenerator(client=client, seed=1)
        for kwargs in ({"records_per_entity": 0},
                       {"null_prob": 1.5},
                       {"orphan_rate": -0.1}):
            args = {"records_per_entity": 2, "null_prob": 0.0,
                    "orphan_rate": 0.0, **kwargs}
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                generator.generate_instances(schema=schema(), **args)
        self.assertEqual(client.calls, [])

    def test_prompt_carries_the_entity_and_domain(self):
        client = FakeClient(COMPANY_ROWS, EMPLOYEE_ROWS, CUSTOMER_ROWS)
        populate(client=client, records_per_entity=2)
        _, prompt = client.calls[0]
        self.assertIn("small_business", prompt)
        self.assertIn("Company", prompt)
        self.assertIn("Rows requested: 2", prompt)
        self.assertIn("location (address)", prompt)
        self.assertIn('"retail"', prompt)  # enum choices are listed

    def test_keys_are_never_asked_of_the_model(self):
        """The model cannot be trusted with identity, so it is never offered it."""
        client = FakeClient(COMPANY_ROWS, EMPLOYEE_ROWS, CUSTOMER_ROWS)
        populate(client=client)
        employee_prompt = client.calls[1][1]
        self.assertIn("full_name", employee_prompt)
        self.assertNotIn("company_id", employee_prompt)
        self.assertNotIn("id (id)", employee_prompt)

    def test_system_prompt_is_deprimed(self):
        lowered = INSTANCE_GENERATION_SYSTEM_PROMPT.lower()
        for leaked in ("patient", "invoice", "student", "company", "customer",
                       "employee"):
            self.assertNotIn(leaked, lowered,
                             f"{leaked!r} in the prompt primes the model")


# --------------------------------------------------------------------------- #
# 2b. Foreign key binding integrity
# --------------------------------------------------------------------------- #
class TestForeignKeyBinding(unittest.TestCase):

    def test_child_keys_bind_to_real_parent_ids(self):
        graph = populate(records_per_entity=2)
        parent_ids = {r.id for r in graph.by_entity("Company")}
        for child in graph.by_entity("Employee") + graph.by_entity("Customer"):
            self.assertEqual(list(child.foreign_keys), ["company_id"])
            self.assertIn(child.foreign_keys["company_id"], parent_ids)
        self.assertEqual(graph.dangling_foreign_keys(), [])
        self.assertEqual(graph.foreign_key_count(), 4)
        self.assertEqual(graph.resolved_foreign_key_count(), 4)

    def test_parents_carry_no_foreign_keys(self):
        graph = populate()
        for record in graph.by_entity("Company"):
            self.assertEqual(record.foreign_keys, {})

    def test_foreign_keys_are_not_duplicated_into_attributes(self):
        """Joins live in one place, so ground truth is unambiguous."""
        graph = populate()
        employee = graph.by_entity("Employee")[0]
        self.assertNotIn("company_id", employee.attributes)
        self.assertIn("company_id", employee.fields())

    def test_every_parent_gets_a_child(self):
        """Round-robin, not independent draws: an empty 'm' side tests nothing."""
        graph = populate(records_per_entity=3)
        linked = {r.foreign_keys["company_id"]
                  for r in graph.by_entity("Employee")}
        self.assertEqual(linked, {r.id for r in graph.by_entity("Company")})

    def test_two_children_of_one_parent_both_link(self):
        graph = populate(records_per_entity=2)
        by_parent = {}
        for child in graph.by_entity("Employee") + graph.by_entity("Customer"):
            by_parent.setdefault(child.foreign_keys["company_id"],
                                 []).append(child.entity_name)
        self.assertEqual(len(by_parent), 2)
        for parent_id, kinds in by_parent.items():
            self.assertEqual(sorted(kinds), ["Customer", "Employee"])

    def test_three_level_chain_binds_each_level(self):
        chain = SchemaGraph.from_payload({
            "entities": [
                {"name": "Clinic", "attributes": [
                    {"name": "id", "type": "id"},
                    {"name": "clinic_name", "type": "string",
                     "required": True}]},
                {"name": "Patient", "attributes": [
                    {"name": "id", "type": "id"},
                    {"name": "clinic_id", "type": "id"},
                    {"name": "date_of_birth", "type": "date",
                     "required": True}]},
                {"name": "Visit", "attributes": [
                    {"name": "id", "type": "id"},
                    {"name": "patient_id", "type": "id"},
                    {"name": "fee", "type": "currency", "required": True}]}],
            "relationships": ["Patient.clinic_id -> Clinic.id",
                              "Visit.patient_id -> Patient.id"],
        }, domain="medical")
        client = CyclingClient({"records": [
            {"clinic_name": "A", "date_of_birth": "1980-01-01", "fee": "$40.00"},
            {"clinic_name": "B", "date_of_birth": "1975-06-02", "fee": "$65.00"}]})
        graph = populate(client=client, schema_graph=chain, records_per_entity=2)
        patient_ids = {r.id for r in graph.by_entity("Patient")}
        clinic_ids = {r.id for r in graph.by_entity("Clinic")}
        for patient in graph.by_entity("Patient"):
            self.assertIn(patient.foreign_keys["clinic_id"], clinic_ids)
        for visit in graph.by_entity("Visit"):
            self.assertIn(visit.foreign_keys["patient_id"], patient_ids)
        self.assertEqual(graph.dangling_foreign_keys(), [])

    def test_seeded_runs_are_identical(self):
        first = populate(seed=7).to_dict()
        second = populate(seed=7).to_dict()
        for payload in (first, second):
            payload.pop("metadata")
        self.assertEqual(first, second)

    def test_flat_schema_produces_no_joins(self):
        flat = SchemaGraph.from_payload({
            "entities": [{"name": "Ledger", "attributes": [
                {"name": "id", "type": "id"},
                {"name": "opened_on", "type": "date", "required": True}]}],
            "relationships": [],
        }, domain="small_business")
        client = CyclingClient({"records": [{"opened_on": "2021-04-04"}]})
        graph = populate(client=client, schema_graph=flat, records_per_entity=1)
        self.assertEqual(graph.foreign_key_count(), 0)
        self.assertEqual(graph.total_records, 1)


class TestStrandedKeyColumns(unittest.TestCase):
    """A key column with no relationship behind it.

    Replays a real llama3.1:8b run for education: Stage 2's forest rule dropped
    Enrollment.course_id -> Course.id but kept the column, and the model, asked
    for a value, returned "EDU-101" -- indistinguishable from a real join and
    resolving to nothing.
    """

    STRANDED_SCHEMA = {
        "domain": "education",
        "entities": [
            {"name": "Course", "attributes": [
                {"name": "id", "type": "id"},
                {"name": "title", "type": "string", "required": True}]},
            {"name": "Student", "attributes": [
                {"name": "id", "type": "id"},
                {"name": "first_name", "type": "string", "required": True}]},
            {"name": "Enrollment", "attributes": [
                {"name": "id", "type": "id"},
                {"name": "student_id", "type": "id"},
                {"name": "course_id", "type": "id"},
                {"name": "start_date", "type": "date", "required": True}]}],
        # Only the student link survives; course_id is left stranded.
        "relationships": ["Enrollment.student_id -> Student.id"],
    }

    def populate_stranded(self, **kwargs):
        client = CyclingClient({"records": [
            {"title": "Diploma in Nursing", "first_name": "Ava",
             "start_date": "2022-09-01", "course_id": "EDU-101"}]})
        return populate(client=client,
                        schema_graph=schema(self.STRANDED_SCHEMA),
                        records_per_entity=1, **kwargs), client

    def test_stranded_column_is_never_asked_of_the_model(self):
        _, client = self.populate_stranded()
        enrollment_prompt = client.calls[-1][1]
        self.assertIn("start_date", enrollment_prompt)
        self.assertNotIn("course_id", enrollment_prompt)

    def test_model_supplied_key_value_is_discarded(self):
        graph, _ = self.populate_stranded()
        enrollment = graph.by_entity("Enrollment")[0]
        self.assertNotEqual(enrollment.attributes["course_id"], "EDU-101")
        self.assertTrue(enrollment.attributes["course_id"])

    def test_stranded_column_is_reported_once_per_entity(self):
        graph, _ = self.populate_stranded()
        reports = [w for w in graph.warnings if "course_id" in w]
        self.assertEqual(len(reports), 1, graph.warnings)
        self.assertIn("no relationship behind it", reports[0])

    def test_stranded_column_stays_out_of_foreign_keys(self):
        graph, _ = self.populate_stranded()
        enrollment = graph.by_entity("Enrollment")[0]
        self.assertEqual(list(enrollment.foreign_keys), ["student_id"])
        self.assertEqual(graph.dangling_foreign_keys(), [])

    def test_bound_keys_are_still_bound_not_stranded(self):
        graph, _ = self.populate_stranded()
        student_ids = {r.id for r in graph.by_entity("Student")}
        enrollment = graph.by_entity("Enrollment")[0]
        self.assertIn(enrollment.foreign_keys["student_id"], student_ids)
        self.assertNotIn("student_id", enrollment.attributes)


# --------------------------------------------------------------------------- #
# 2c. Null probability
# --------------------------------------------------------------------------- #
class TestNullProbability(unittest.TestCase):

    def test_zero_probability_leaves_nothing_null(self):
        graph = populate(null_prob=0.0)
        for record in graph.records:
            self.assertEqual(record.null_attributes(), [])

    def test_certain_probability_nulls_every_optional_field(self):
        graph = populate(null_prob=1.0)
        company = graph.by_entity("Company")[0]
        self.assertIsNone(company.attributes["industry"])       # optional
        self.assertIsNotNone(company.attributes["company_name"])  # required
        self.assertIsNotNone(company.attributes["location"])      # required
        self.assertEqual(company.attributes["id"], company.id)    # key

    def test_keys_and_required_fields_are_never_nulled(self):
        graph = populate(null_prob=1.0, records_per_entity=3)
        parent_ids = {r.id for r in graph.by_entity("Company")}
        for record in graph.records:
            self.assertIsNotNone(record.attributes["id"])
            for column, value in record.foreign_keys.items():
                self.assertIn(value, parent_ids, f"{record.id}.{column}")
        for employee in graph.by_entity("Employee"):
            self.assertIsNotNone(employee.attributes["full_name"])
            self.assertIsNone(employee.attributes["job_title"])

    def test_intermediate_rate_lands_between_the_extremes(self):
        client = CyclingClient(EMPLOYEE_ROWS)
        graph = populate(client=client, records_per_entity=20, null_prob=0.5,
                         seed=3)
        employees = graph.by_entity("Employee")
        optional = 2 * len(employees)  # email, job_title
        nulls = sum(len(r.null_attributes()) for r in employees)
        self.assertGreater(nulls, 0)
        self.assertLess(nulls, optional)

    def test_null_rate_is_reproducible_under_a_seed(self):
        def null_ids(seed):
            graph = populate(records_per_entity=6, null_prob=0.4, seed=seed)
            return [(r.id, tuple(r.null_attributes())) for r in graph.records]
        self.assertEqual(null_ids(11), null_ids(11))
        self.assertNotEqual(null_ids(11), null_ids(12))

    def test_missing_field_is_synthesised_not_left_absent(self):
        client = CyclingClient({"records": [{"company_name": "Partial"}]})
        graph = populate(client=client, records_per_entity=1)
        company = graph.by_entity("Company")[0]
        self.assertEqual(sorted(company.attributes),
                         ["company_name", "id", "industry", "location"])
        self.assertIsNotNone(company.attributes["location"])
        self.assertTrue(any("synthesised locally" in w
                            for w in graph.warnings), graph.warnings)

    def test_synthesised_values_use_reserved_hosts_only(self):
        client = CyclingClient({"records": [{}]})
        graph = populate(client=client, records_per_entity=1)
        employee = graph.by_entity("Employee")[0]
        self.assertTrue(employee.attributes["email"].endswith("example.invalid"))


# --------------------------------------------------------------------------- #
# 2d. Orphan rate
# --------------------------------------------------------------------------- #
class TestOrphanRate(unittest.TestCase):

    def test_zero_rate_leaves_no_dangling_keys(self):
        graph = populate(orphan_rate=0.0, records_per_entity=4)
        self.assertEqual(graph.dangling_foreign_keys(), [])
        self.assertEqual(graph.orphan_count(), 0)

    def test_full_rate_orphans_every_child_key(self):
        graph = populate(orphan_rate=1.0, records_per_entity=3)
        children = graph.by_entity("Employee") + graph.by_entity("Customer")
        self.assertEqual(len(children), 6)
        for child in children:
            self.assertEqual(child.orphaned_keys, ["company_id"])
            self.assertIn(ORPHAN_MARKER, child.foreign_keys["company_id"])
        self.assertEqual(graph.orphan_count(), 6)
        self.assertEqual(graph.resolved_foreign_key_count(), 0)

    def test_orphans_are_the_only_dangling_keys(self):
        """Anything dangling that was not declared an orphan is a generator bug."""
        graph = populate(orphan_rate=0.5, records_per_entity=4)
        declared = {f"{r.id}.{c}" for r in graph.records
                    for c in r.orphaned_keys}
        dangling = {entry.split(" -> ")[0]
                    for entry in graph.dangling_foreign_keys()}
        self.assertEqual(dangling, declared)
        self.assertGreater(len(declared), 0)

    def test_half_rate_orphans_about_half(self):
        graph = populate(orphan_rate=0.5, records_per_entity=8)
        employees = graph.by_entity("Employee")
        orphans = sum(1 for r in employees if r.orphaned_keys)
        self.assertEqual(orphans, 4)
        self.assertEqual(graph.foreign_key_count(), 16)

    def test_orphan_rate_survives_a_single_row_entity(self):
        """A rate that always rounds to zero would be silently unimplemented."""
        seen = set()
        for seed in range(12):
            graph = populate(records_per_entity=1, orphan_rate=0.5, seed=seed)
            seen.add(graph.orphan_count())
        self.assertIn(0, seen)
        self.assertTrue(any(count > 0 for count in seen), seen)

    def test_orphaned_keys_are_reported_in_warnings(self):
        graph = populate(orphan_rate=1.0, records_per_entity=2)
        self.assertTrue(any("orphaned 2 of 2" in w for w in graph.warnings),
                        graph.warnings)

    def test_orphans_are_reproducible_under_a_seed(self):
        def orphan_ids(seed):
            graph = populate(records_per_entity=6, orphan_rate=0.3, seed=seed)
            return [r.id for r in graph.records if r.orphaned_keys]
        self.assertEqual(orphan_ids(5), orphan_ids(5))


# --------------------------------------------------------------------------- #
# 2e. Schema compliance of the values
# --------------------------------------------------------------------------- #
class TestSchemaCompliance(unittest.TestCase):

    def test_attribute_names_match_the_schema_exactly(self):
        graph = populate()
        expected = {
            "Company": {"id", "company_name", "location", "industry"},
            "Employee": {"id", "full_name", "email", "job_title"},
            "Customer": {"id", "customer_name", "phone"},
        }
        for record in graph.records:
            self.assertEqual(set(record.attributes),
                             expected[record.entity_name], record.id)

    def test_unknown_keys_from_the_model_are_discarded(self):
        client = CyclingClient({"records": [
            {"company_name": "Kestrel", "location": "1 Road",
             "industry": "retail", "ceo_favourite_colour": "teal"}]})
        graph = populate(client=client, records_per_entity=1)
        self.assertNotIn("ceo_favourite_colour",
                         graph.by_entity("Company")[0].attributes)

    def test_model_keys_are_normalised_before_matching(self):
        client = CyclingClient({"records": [
            {"Company Name": "Kestrel", "Location": "1 Road",
             "Industry": "retail"}]})
        graph = populate(client=client, records_per_entity=1,
                         schema_graph=schema(COMPANY_ONLY_SCHEMA))
        company = graph.by_entity("Company")[0]
        self.assertEqual(company.attributes["company_name"], "Kestrel")
        self.assertEqual(graph.warnings, [])

    def test_enum_value_outside_the_declared_set_is_replaced(self):
        client = CyclingClient({"records": [
            {"company_name": "Kestrel", "location": "1 Road",
             "industry": "cryptocurrency"}]})
        graph = populate(client=client, records_per_entity=1)
        self.assertIn(graph.by_entity("Company")[0].attributes["industry"],
                      ["retail", "trades", "hospitality"])
        self.assertTrue(any("not a declared enum value" in w
                            for w in graph.warnings), graph.warnings)

    def test_enum_matching_is_case_insensitive_but_output_is_canonical(self):
        client = CyclingClient({"records": [
            {"company_name": "Kestrel", "location": "1 Road",
             "industry": "RETAIL"}]})
        graph = populate(client=client, records_per_entity=1,
                         schema_graph=schema(COMPANY_ONLY_SCHEMA))
        self.assertEqual(graph.by_entity("Company")[0].attributes["industry"],
                         "retail")
        self.assertEqual(graph.warnings, [])

    def test_formatting_of_values_is_preserved(self):
        """A currency must keep its symbol; a document has to show what it shows."""
        currency = SchemaGraph.from_payload({
            "entities": [{"name": "Order", "attributes": [
                {"name": "id", "type": "id"},
                {"name": "total", "type": "currency", "required": True}]}],
            "relationships": [],
        }, domain="small_business")
        client = CyclingClient({"records": [{"total": "$1,240.50"}]})
        graph = populate(client=client, schema_graph=currency,
                         records_per_entity=1)
        self.assertEqual(graph.records[0].attributes["total"], "$1,240.50")

    def test_nested_value_is_flattened_not_kept(self):
        client = CyclingClient({"records": [
            {"company_name": {"legal": "Kestrel", "trading": "Kestrel Ltd"},
             "location": "1 Road", "industry": "retail"}]})
        graph = populate(client=client, records_per_entity=1)
        value = graph.by_entity("Company")[0].attributes["company_name"]
        self.assertIsInstance(value, str)
        self.assertTrue(any("flattened" in w for w in graph.warnings))

    def test_boolean_strings_are_coerced(self):
        flag = SchemaGraph.from_payload({
            "entities": [{"name": "Account", "attributes": [
                {"name": "id", "type": "id"},
                {"name": "is_active", "type": "boolean", "required": True}]}],
            "relationships": [],
        }, domain="small_business")
        client = CyclingClient({"records": [{"is_active": "yes"}]})
        graph = populate(client=client, schema_graph=flag, records_per_entity=1)
        self.assertIs(graph.records[0].attributes["is_active"], True)


# --------------------------------------------------------------------------- #
# 3. Serialisation
# --------------------------------------------------------------------------- #
class TestSerialization(unittest.TestCase):

    def assert_complies_with(self, payload: dict, schema_payload: dict) -> None:
        """Every record must match its entity, and every join must resolve."""
        schema_graph = SchemaGraph.from_payload(schema_payload)
        ids = {r["id"] for r in payload["records"]}
        for record in payload["records"]:
            entity = schema_graph.entity(record["entity_name"])
            self.assertIsNotNone(entity, record["entity_name"])
            fks = {rel.child_attribute for rel in schema_graph.relationships
                   if rel.child_entity == entity.name}
            expected = {a.name for a in entity.attributes} - fks
            self.assertEqual(set(record["attributes"]), expected, record["id"])
            self.assertEqual(set(record["foreign_keys"]), fks, record["id"])
            self.assertEqual(record["attributes"][entity.primary_key],
                             record["id"])
            orphaned = set(record.get("orphaned_keys") or [])
            for column, value in record["foreign_keys"].items():
                if column not in orphaned:
                    self.assertIn(value, ids, f"{record['id']}.{column}")

    def test_written_file_is_json_and_schema_compliant(self):
        graph = populate(records_per_entity=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "instances.json")
            self.assertEqual(write_instances(graph, path), path)
            payload = read_json(path)
        self.assertEqual(payload["schema_domain"], "small_business")
        self.assertEqual(payload["total_records"], 6)
        self.assertEqual(payload["counts_by_entity"],
                         {"Company": 2, "Employee": 2, "Customer": 2})
        self.assertEqual(payload["foreign_keys"],
                         {"declared": 4, "resolved": 4, "orphaned": 0})
        self.assert_complies_with(payload, TWO_LEVEL_SCHEMA)

    def test_metadata_records_the_run(self):
        graph = populate(records_per_entity=3, null_prob=0.25,
                         orphan_rate=0.1, seed=42)
        meta = graph.to_dict()["metadata"]
        self.assertEqual(meta["records_per_entity"], 3)
        self.assertEqual(meta["null_probability"], 0.25)
        self.assertEqual(meta["orphan_rate"], 0.1)
        self.assertEqual(meta["seed"], 42)
        self.assertEqual(meta["model"], "llama3.3:70b")
        self.assertTrue(meta["generated_at"].endswith("+00:00"))

    def test_nulls_survive_the_round_trip_as_nulls(self):
        graph = populate(null_prob=1.0)
        restored = InstanceGraph.from_dict(json.loads(graph.to_json()))
        self.assertEqual(restored.to_dict(), graph.to_dict())
        self.assertIsNone(restored.by_entity("Company")[0]
                          .attributes["industry"])

    def test_round_trip_preserves_orphan_marks(self):
        graph = populate(orphan_rate=1.0)
        restored = InstanceGraph.from_dict(json.loads(graph.to_json()))
        self.assertEqual(restored.orphan_count(), graph.orphan_count())
        self.assertEqual(restored.to_dict(), graph.to_dict())

    def test_total_records_is_recomputed_not_trusted(self):
        graph = InstanceGraph.from_dict({
            "schema_domain": "x", "total_records": 999,
            "records": [{"id": "a-001", "entity_name": "A"}]})
        self.assertEqual(graph.total_records, 1)
        graph.add(Record(id="a-002", entity_name="A"))
        self.assertEqual(graph.total_records, 2)

    def test_summary_reports_counts_and_joins(self):
        summary = populate(records_per_entity=2, orphan_rate=0.5).summary()
        self.assertIn("Company: 2 record(s)", summary)
        self.assertIn("Employee: 2 record(s), 2 fk(s)", summary)
        self.assertIn("joins:", summary)
        self.assertIn("orphaned", summary)


# --------------------------------------------------------------------------- #
# 4. End to end through the CLI
# --------------------------------------------------------------------------- #
class TestCLI(unittest.TestCase):

    def setUp(self):
        self.runner = make_runner()
        self.original = cli_module.build_client

    def tearDown(self):
        cli_module.build_client = self.original

    def stub_client(self, *payloads):
        client = FakeClient(*payloads)
        cli_module.build_client = lambda **kw: client
        return client

    def test_generate_writes_both_artefacts(self):
        self.stub_client(TWO_LEVEL_SCHEMA, COMPANY_ROWS, EMPLOYEE_ROWS,
                         CUSTOMER_ROWS)
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "bench")
            result = self.runner.invoke(cli_module.app, [
                "generate", "--no-render", "--domain", "small_business",
                "--num-entities", "3", "--records-per-entity", "2",
                "--seed", "4", "--output-dir", out])
            self.assertEqual(result.exit_code, 0, stderr_of(result))
            schema_payload = read_json(out, "schema.json")
            instances = read_json(out, "instances.json")
            self.assertEqual(result.stdout.split(),
                             [os.path.join(out, "schema.json"),
                              os.path.join(out, "instances.json")])
        self.assertEqual(instances["total_records"], 6)
        self.assertEqual(instances["metadata"]["records_per_entity"], 2)
        TestSerialization.assert_complies_with(self, instances, schema_payload)

    def test_flags_reach_the_instance_stage(self):
        self.stub_client(TWO_LEVEL_SCHEMA, COMPANY_ROWS, EMPLOYEE_ROWS,
                         CUSTOMER_ROWS)
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "bench")
            result = self.runner.invoke(cli_module.app, [
                "generate", "--no-render", "--num-entities", "3",
                "--records-per-entity", "2", "--null-probability", "1.0",
                "--orphan-rate", "1.0", "--output-dir", out])
            self.assertEqual(result.exit_code, 0, stderr_of(result))
            instances = read_json(out, "instances.json")
        self.assertEqual(instances["metadata"]["null_probability"], 1.0)
        self.assertEqual(instances["foreign_keys"]["orphaned"], 4)
        self.assertEqual(instances["foreign_keys"]["resolved"], 0)
        nulls = [k for r in instances["records"]
                 for k, v in r["attributes"].items() if v is None]
        self.assertTrue(nulls)

    def test_summary_of_records_and_joins_goes_to_stderr(self):
        self.stub_client(TWO_LEVEL_SCHEMA, COMPANY_ROWS, EMPLOYEE_ROWS,
                         CUSTOMER_ROWS)
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "bench")
            result = self.runner.invoke(cli_module.app, [
                "generate", "--no-render", "--num-entities", "3",
                "--records-per-entity", "2", "--output-dir", out])
        err = stderr_of(result)
        self.assertIn("stage 3-4: instances", err)
        self.assertIn("Company: 2 record(s)", err)
        self.assertIn("joins: 4 resolved", err)
        self.assertIn("records=6", err)
        self.assertIn("wrote", err)

    def test_schema_only_skips_the_instance_stage(self):
        self.stub_client(TWO_LEVEL_SCHEMA)
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "bench")
            result = self.runner.invoke(cli_module.app, [
                "generate", "--schema-only", "--num-entities", "3",
                "--output-dir", out])
            self.assertEqual(result.exit_code, 0, stderr_of(result))
            self.assertTrue(os.path.isfile(os.path.join(out, "schema.json")))
            self.assertFalse(os.path.exists(os.path.join(out,
                                                         "instances.json")))
            self.assertEqual(result.stdout.strip(),
                             os.path.join(out, "schema.json"))

    def test_rejects_out_of_range_rates(self):
        for args in (["--null-probability", "1.5"], ["--orphan-rate", "-0.2"],
                     ["--records-per-entity", "0"]):
            result = self.runner.invoke(cli_module.app, ["generate"] + args)
            self.assertEqual(result.exit_code, 2, f"{args} should be rejected")

    def test_failed_instance_stage_keeps_the_schema(self):
        self.stub_client(TWO_LEVEL_SCHEMA, None, None, None)
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "bench")
            result = self.runner.invoke(cli_module.app, [
                "generate", "--no-render", "--num-entities", "3",
                "--output-dir", out, "--max-attempts", "3"])
            self.assertEqual(result.exit_code, 5)
            self.assertTrue(os.path.isfile(os.path.join(out, "schema.json")))
            self.assertFalse(os.path.exists(os.path.join(out,
                                                         "instances.json")))
        self.assertIn("was written; rerun to populate it", stderr_of(result))


if __name__ == "__main__":
    unittest.main(verbosity=2)
