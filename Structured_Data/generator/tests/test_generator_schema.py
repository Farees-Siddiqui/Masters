"""Tests for Stages 1 & 2: the Typer CLI and the parametric schema generator.

Every LLM call is mocked, so nothing here needs Ollama, a GPU or model weights.
The fixtures deliberately return *malformed* schemas as well as good ones,
because the generator's job is not to relay what the model said but to make the
result satisfy the parameters it was asked for.
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
from src.generator.llm_bridge import (SeededLLMClient,  # noqa: E402
                                      build_client)
from src.generator.schema_generator import (  # noqa: E402
    SCHEMA_GENERATION_SYSTEM_PROMPT, ParametricSchemaGenerator,
    SchemaGenerationError, build_user_prompt, write_schema)
from src.generator.schema_types import (PRIMITIVE_TYPES,  # noqa: E402
                                        Attribute, EntitySchema, Relationship,
                                        SchemaGraph, SchemaValidationError,
                                        normalize_type, pascal_case,
                                        snake_case)


def setUpModule() -> None:
    """Repairs are logged at WARNING; the tests assert on them, not read them."""
    logging.disable(logging.WARNING)


def tearDownModule() -> None:
    logging.disable(logging.NOTSET)


def read_json(*parts: str) -> dict:
    with open(os.path.join(*parts), encoding="utf-8") as handle:
        return json.load(handle)


def make_runner() -> CliRunner:
    """A runner that keeps stderr separate where the click version allows it.

    The CLI's contract is that stdout carries only the artefact path, so the
    tests have to be able to tell the two streams apart. click 8.1 needs
    ``mix_stderr=False``; 8.2 dropped the flag and always separates them.
    """
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
#: A well-formed two-level schema, in the exact shape the prompt asks for.
SMALL_BUSINESS = {
    "entities": [
        {"name": "Customer", "description": "A buyer.", "primary_key": "id",
         "attributes": [
             {"name": "id", "type": "id", "required": True},
             {"name": "full_name", "type": "string", "required": True},
             {"name": "billing_address", "type": "address", "required": True},
             {"name": "contact_email", "type": "email", "required": False},
             {"name": "joined_on", "type": "date", "required": True}]},
        {"name": "Invoice", "description": "A bill issued to a customer.",
         "primary_key": "id",
         "attributes": [
             {"name": "id", "type": "id", "required": True},
             {"name": "customer_id", "type": "id", "required": True},
             {"name": "issued_on", "type": "date", "required": True},
             {"name": "total", "type": "currency", "required": True},
             {"name": "status", "type": "enum", "required": True,
              "values": ["draft", "sent", "paid"]}]},
    ],
    "relationships": ["Invoice.customer_id -> Customer.id"],
}

#: Three levels deep. Asked for depth 2, the deepest link must not survive.
THREE_LEVELS = {
    "entities": [
        {"name": "Clinic", "primary_key": "id",
         "attributes": [{"name": "id", "type": "id"},
                        {"name": "clinic_name", "type": "string"}]},
        {"name": "Patient", "primary_key": "id",
         "attributes": [{"name": "id", "type": "id"},
                        {"name": "clinic_id", "type": "id"},
                        {"name": "date_of_birth", "type": "date"}]},
        {"name": "Visit", "primary_key": "id",
         "attributes": [{"name": "id", "type": "id"},
                        {"name": "patient_id", "type": "id"},
                        {"name": "fee", "type": "currency"}]},
    ],
    "relationships": ["Patient.clinic_id -> Clinic.id",
                      "Visit.patient_id -> Patient.id"],
}


class FakeClient:
    """Stands in for LocalLLMClient. Returns queued payloads in order."""

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


def generate(payload, domain="small_business", num_entities=2, max_depth=2,
             **kwargs):
    """Run one generation against a single queued payload."""
    gen = ParametricSchemaGenerator(client=FakeClient(payload), **kwargs)
    return gen.generate_schema(domain=domain, num_entities=num_entities,
                               max_depth=max_depth)


# --------------------------------------------------------------------------- #
# 1. CLI argument parsing and --help
# --------------------------------------------------------------------------- #
class TestCLI(unittest.TestCase):

    def setUp(self):
        self.runner = make_runner()

    def test_app_help_lists_generate(self):
        result = self.runner.invoke(cli_module.app, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("generate", result.stdout)

    def test_generate_help_lists_every_stage_flag(self):
        result = self.runner.invoke(cli_module.app, ["generate", "--help"])
        self.assertEqual(result.exit_code, 0)
        for flag in ("--domain", "--num-entities", "--max-depth", "--seed",
                     "--output-dir"):
            self.assertIn(flag, result.stdout, f"{flag} missing from --help")

    def test_generate_help_shows_documented_defaults(self):
        result = self.runner.invoke(cli_module.app, ["generate", "--help"])
        out = " ".join(result.stdout.split())  # help text is line-wrapped
        self.assertIn("small_business", out)
        self.assertIn("out_benchmark", out)
        self.assertIn("127.0.0.1", out)  # rich elides the port when wrapped

    def test_parsed_arguments_reach_the_generator(self):
        seen = {}

        def fake_build_client(**kwargs):
            seen.update(kwargs)
            return FakeClient(SMALL_BUSINESS)

        original = cli_module.build_client
        cli_module.build_client = fake_build_client
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out = os.path.join(tmp, "bench")
                result = self.runner.invoke(cli_module.app, [
                    "generate", "--schema-only", "--domain", "education",
                    "--num-entities", "2", "--max-depth", "2", "--seed", "7",
                    "--output-dir", out, "--model", "llama3.1:8b"])
                self.assertEqual(result.exit_code, 0, stderr_of(result))
                schema = read_json(out, "schema.json")
        finally:
            cli_module.build_client = original

        self.assertEqual(seen["seed"], 7)
        self.assertEqual(seen["model"], "llama3.1:8b")
        self.assertEqual(seen["base_url"], "http://127.0.0.1:11434")
        self.assertEqual(schema["domain"], "education")
        self.assertEqual(schema["metadata"]["requested_entities"], 2)
        self.assertEqual(schema["metadata"]["seed"], 7)

    def test_stdout_is_only_the_artefact_path(self):
        original = cli_module.build_client
        cli_module.build_client = lambda **kw: FakeClient(SMALL_BUSINESS)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out = os.path.join(tmp, "bench")
                result = self.runner.invoke(
                    cli_module.app,
                    ["generate", "--schema-only", "--num-entities", "2",
                     "--output-dir", out])
                self.assertEqual(result.exit_code, 0, stderr_of(result))
                self.assertEqual(result.stdout.strip(),
                                 os.path.join(out, "schema.json"))
                # The human-readable summary went to stderr instead.
                self.assertIn("Customer", stderr_of(result))
        finally:
            cli_module.build_client = original

    def test_summary_of_entities_and_joins_goes_to_stderr(self):
        original = cli_module.build_client
        cli_module.build_client = lambda **kw: FakeClient(SMALL_BUSINESS)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                result = self.runner.invoke(cli_module.app, [
                    "generate", "--schema-only", "--num-entities", "2",
                    "--output-dir", os.path.join(tmp, "bench")])
        finally:
            cli_module.build_client = original
        err = stderr_of(result)
        self.assertIn("Invoice.customer_id -> Customer.id", err)
        self.assertIn("total:currency", err)
        self.assertIn("depth=2", err)

    def test_output_dir_is_created(self):
        original = cli_module.build_client
        cli_module.build_client = lambda **kw: FakeClient(SMALL_BUSINESS)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out = os.path.join(tmp, "deep", "nested", "bench")
                self.assertFalse(os.path.exists(out))
                result = self.runner.invoke(
                    cli_module.app,
                    ["generate", "--schema-only", "--num-entities", "2",
                     "--output-dir", out])
                self.assertEqual(result.exit_code, 0, stderr_of(result))
                self.assertTrue(os.path.isfile(os.path.join(out, "schema.json")))
        finally:
            cli_module.build_client = original

    def test_rejects_out_of_range_parameters(self):
        for args in (["--num-entities", "0"], ["--max-depth", "0"]):
            result = self.runner.invoke(cli_module.app, ["generate"] + args)
            self.assertEqual(result.exit_code, 2, f"{args} should be rejected")

    def test_rejects_unknown_backend(self):
        result = self.runner.invoke(
            cli_module.app, ["generate", "--backend", "carrier-pigeon"])
        self.assertEqual(result.exit_code, 2)
        self.assertIn("carrier-pigeon", stderr_of(result))

    def test_unusable_model_output_is_a_nonzero_exit(self):
        original = cli_module.build_client
        cli_module.build_client = lambda **kw: FakeClient(None, None, None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                result = self.runner.invoke(cli_module.app, [
                    "generate", "--output-dir", os.path.join(tmp, "b"),
                    "--max-attempts", "3"])
        finally:
            cli_module.build_client = original
        self.assertEqual(result.exit_code, 4)
        self.assertIn("no usable schema", stderr_of(result))


# --------------------------------------------------------------------------- #
# 2a. Parsing a mocked response
# --------------------------------------------------------------------------- #
class TestSchemaParsing(unittest.TestCase):

    def test_parses_entities_attributes_and_relationships(self):
        graph = generate(SMALL_BUSINESS)
        self.assertEqual(graph.entity_names, ["Customer", "Invoice"])
        self.assertEqual(graph.domain, "small_business")
        invoice = graph.entity("Invoice")
        self.assertEqual(invoice.attribute_names,
                         ["id", "customer_id", "issued_on", "total", "status"])
        self.assertEqual(len(graph.relationships), 1)
        rel = graph.relationships[0]
        self.assertEqual(rel.child_entity, "Invoice")
        self.assertEqual(rel.child_attribute, "customer_id")
        self.assertEqual(rel.parent_entity, "Customer")
        self.assertEqual(rel.parent_attribute, "id")
        self.assertEqual(rel.cardinality, "1:m")
        self.assertEqual(graph.warnings, [])

    def test_prompt_carries_the_stage_one_parameters(self):
        client = FakeClient(SMALL_BUSINESS)
        gen = ParametricSchemaGenerator(client=client)
        gen.generate_schema(domain="veterinary", num_entities=2, max_depth=2)
        system, user = client.calls[0]
        self.assertEqual(system, SCHEMA_GENERATION_SYSTEM_PROMPT)
        self.assertIn("veterinary", user)
        self.assertIn("exactly 2", user)
        self.assertIn("2 levels", user)

    def test_depth_one_prompt_forbids_relationships(self):
        prompt = build_user_prompt("education", 3, 1)
        self.assertIn("NO relationships", prompt)

    def test_system_prompt_is_deprimed(self):
        """The example must not seed a target domain's vocabulary."""
        lowered = SCHEMA_GENERATION_SYSTEM_PROMPT.lower()
        for leaked in ("patient", "invoice", "student", "clinic", "customer",
                       "enrolment", "prescription"):
            self.assertNotIn(leaked, lowered,
                             f"{leaked!r} in the prompt primes the model")

    def test_normalizes_entity_and_attribute_names(self):
        graph = generate({
            "entities": [{"name": "purchase order",
                          "attributes": [{"name": "Order Total",
                                          "type": "currency"},
                                         {"name": "2024 Quarter",
                                          "type": "string"}]}],
            "relationships": [],
        }, num_entities=1)
        entity = graph.entities[0]
        self.assertEqual(entity.name, "PurchaseOrder")
        self.assertIn("order_total", entity.attribute_names)
        self.assertIn("n2024_quarter", entity.attribute_names)

    def test_accepts_the_object_relationship_form(self):
        payload = json.loads(json.dumps(SMALL_BUSINESS))
        payload["relationships"] = [{"child": "Invoice",
                                     "child_attribute": "customer_id",
                                     "parent": "Customer",
                                     "parent_attribute": "id"}]
        graph = generate(payload)
        self.assertEqual(graph.relationships[0].as_fk(),
                         "Invoice.customer_id -> Customer.id")

    def test_accepts_a_wrapped_payload(self):
        graph = generate({"schema": SMALL_BUSINESS})
        self.assertEqual(graph.entity_names, ["Customer", "Invoice"])

    def test_accepts_alternate_key_names(self):
        graph = generate({
            "tables": [{"table": "Ledger",
                        "columns": [{"column": "id", "data_type": "uuid"}]}],
            "foreign_keys": [],
        }, num_entities=1)
        self.assertEqual(graph.entity_names, ["Ledger"])
        self.assertEqual(graph.entity("Ledger").attribute("id").type, "id")

    def test_unreadable_relationship_is_dropped_not_fatal(self):
        payload = json.loads(json.dumps(SMALL_BUSINESS))
        payload["relationships"] = ["Invoice.customer_id -> Customer.id",
                                    "this is not a foreign key"]
        graph = generate(payload)
        self.assertEqual(len(graph.relationships), 1)
        self.assertTrue(any("unreadable relationship" in w
                            for w in graph.warnings))

    def test_no_entities_is_fatal(self):
        with self.assertRaises(SchemaGenerationError):
            generate({"entities": [], "relationships": []})

    def test_retries_then_succeeds(self):
        client = FakeClient(None, {"nonsense": True}, SMALL_BUSINESS)
        gen = ParametricSchemaGenerator(client=client, max_attempts=3)
        graph = gen.generate_schema(domain="small_business", num_entities=2,
                                    max_depth=2)
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(graph.metadata["attempts"], 3)
        # The retry names the previous failure back to the model.
        self.assertIn("could not be used", client.calls[1][1])

    def test_gives_up_after_max_attempts(self):
        client = FakeClient(None, None)
        gen = ParametricSchemaGenerator(client=client, max_attempts=2)
        with self.assertRaises(SchemaGenerationError):
            gen.generate_schema(domain="medical", num_entities=3, max_depth=2)
        self.assertEqual(len(client.calls), 2)

    def test_invalid_parameters_raise_before_any_call(self):
        client = FakeClient(SMALL_BUSINESS)
        gen = ParametricSchemaGenerator(client=client)
        with self.assertRaises(ValueError):
            gen.generate_schema(domain="x", num_entities=0, max_depth=2)
        with self.assertRaises(ValueError):
            gen.generate_schema(domain="x", num_entities=2, max_depth=0)
        self.assertEqual(client.calls, [])


# --------------------------------------------------------------------------- #
# 2b. Attribute typing
# --------------------------------------------------------------------------- #
class TestAttributeTyping(unittest.TestCase):

    def test_primitive_vocabulary_covers_the_documented_types(self):
        for expected in ("string", "currency", "date", "address", "integer"):
            self.assertIn(expected, PRIMITIVE_TYPES)

    def test_known_aliases_map_onto_primitives(self):
        for raw, expected in (("VARCHAR", "string"), ("int", "integer"),
                              ("money", "currency"), ("timestamp", "datetime"),
                              ("uuid", "id"), ("decimal(10,2)", "decimal"),
                              ("Postal Address", "address"), ("bool", "boolean")):
            with self.subTest(raw=raw):
                coerced, known = normalize_type(raw)
                self.assertEqual(coerced, expected)
                self.assertTrue(known)

    def test_unknown_type_degrades_to_string_and_is_reported(self):
        coerced, known = normalize_type("hieroglyph")
        self.assertEqual(coerced, "string")
        self.assertFalse(known)

        graph = generate({
            "entities": [{"name": "Widget",
                          "attributes": [{"name": "id", "type": "id"},
                                         {"name": "hue", "type": "hieroglyph"}]}],
            "relationships": [],
        }, num_entities=1)
        self.assertEqual(graph.entity("Widget").attribute("hue").type, "string")
        self.assertTrue(any("hieroglyph" in w for w in graph.warnings),
                        graph.warnings)

    def test_types_survive_onto_the_dataclasses(self):
        graph = generate(SMALL_BUSINESS)
        customer, invoice = graph.entities
        self.assertEqual(customer.attribute("billing_address").type, "address")
        self.assertEqual(customer.attribute("contact_email").type, "email")
        self.assertEqual(customer.attribute("joined_on").type, "date")
        self.assertEqual(invoice.attribute("total").type, "currency")
        self.assertEqual(invoice.attribute("status").values,
                         ["draft", "sent", "paid"])

    def test_attribute_shorthand_forms(self):
        self.assertEqual(Attribute.from_payload("total").name, "total")
        self.assertEqual(Attribute.from_payload({"total": "money"}).type,
                         "currency")
        self.assertTrue(Attribute.from_payload(
            {"name": "id", "nullable": False}).required)

    def test_duplicate_attributes_are_collapsed(self):
        entity = EntitySchema(name="Order", attributes=[
            {"name": "id", "type": "id"},
            {"name": "id", "type": "integer"},
            {"name": "ID", "type": "string"}])
        self.assertEqual(entity.attribute_names, ["id"])
        self.assertEqual(entity.attribute("id").type, "id")


# --------------------------------------------------------------------------- #
# 2c. Depth and parameter enforcement
# --------------------------------------------------------------------------- #
class TestDepthEnforcement(unittest.TestCase):

    def test_depth_is_measured_in_levels_from_the_roots(self):
        graph = generate(SMALL_BUSINESS)
        self.assertEqual(graph.depth(), 2)
        self.assertEqual(graph.roots(), ["Customer"])
        self.assertEqual(graph.depths(), {"Customer": 1, "Invoice": 2})

    def test_over_deep_chain_is_truncated_to_max_depth(self):
        graph = generate(THREE_LEVELS, domain="medical", num_entities=3,
                         max_depth=2)
        self.assertEqual(graph.depth(), 2)
        self.assertEqual([r.as_fk() for r in graph.relationships],
                         ["Patient.clinic_id -> Clinic.id"])
        self.assertTrue(any("max_depth" in w for w in graph.warnings),
                        graph.warnings)
        # The orphaned child keeps its attributes and becomes a second root.
        self.assertIn("Visit", graph.entity_names)
        self.assertEqual(sorted(graph.roots()), ["Clinic", "Visit"])

    def test_three_levels_are_kept_when_three_are_allowed(self):
        graph = generate(THREE_LEVELS, domain="medical", num_entities=3,
                         max_depth=3)
        self.assertEqual(graph.depth(), 3)
        self.assertEqual(len(graph.relationships), 2)
        self.assertEqual(graph.warnings, [])

    def test_max_depth_one_removes_every_relationship(self):
        graph = generate(THREE_LEVELS, domain="medical", num_entities=3,
                         max_depth=1)
        self.assertEqual(graph.relationships, [])
        self.assertEqual(graph.depth(), 1)

    def test_foreign_key_onto_a_missing_entity_is_dropped(self):
        payload = json.loads(json.dumps(SMALL_BUSINESS))
        payload["relationships"].append("Invoice.vendor_id -> Vendor.id")
        graph = generate(payload)
        self.assertEqual(len(graph.relationships), 1)
        self.assertTrue(any("Vendor" in w and "not in the schema" in w
                            for w in graph.warnings), graph.warnings)
        self.assertEqual(graph.entity_names, ["Customer", "Invoice"])

    def test_self_reference_is_dropped(self):
        payload = json.loads(json.dumps(SMALL_BUSINESS))
        payload["relationships"] = ["Customer.parent_id -> Customer.id"]
        graph = generate(payload)
        self.assertEqual(graph.relationships, [])
        self.assertTrue(any("own parent" in w for w in graph.warnings))

    def test_cycle_cannot_survive_enforcement(self):
        graph = generate({
            "entities": [{"name": "A", "attributes": [{"name": "id",
                                                       "type": "id"}]},
                         {"name": "B", "attributes": [{"name": "id",
                                                       "type": "id"}]}],
            "relationships": ["A.b_id -> B.id", "B.a_id -> A.id"],
        }, num_entities=2, max_depth=2)
        self.assertEqual(len(graph.relationships), 1)
        self.assertLessEqual(graph.depth(), 2)

    def test_cycle_dies_even_when_max_depth_would_allow_it(self):
        """A 2-cycle implies depth 4, so a generous max_depth must not save it."""
        graph = generate({
            "entities": [{"name": "A", "attributes": [{"name": "id",
                                                       "type": "id"}]},
                         {"name": "B", "attributes": [{"name": "id",
                                                       "type": "id"}]},
                         {"name": "C", "attributes": [{"name": "id",
                                                       "type": "id"}]}],
            "relationships": ["A.b_id -> B.id", "B.c_id -> C.id",
                              "C.a_id -> A.id"],
        }, num_entities=3, max_depth=9)
        self.assertEqual([r.as_fk() for r in graph.relationships],
                         ["A.b_id -> B.id", "B.c_id -> C.id"])
        self.assertTrue(any("own ancestor" in w for w in graph.warnings),
                        graph.warnings)
        self.assertEqual(graph.depth(), 3)
        self.assertEqual(graph.roots(), ["C"])

    def test_second_parent_is_dropped_to_keep_a_forest(self):
        graph = generate({
            "entities": [{"name": "Ward", "attributes": [{"name": "id",
                                                          "type": "id"}]},
                         {"name": "Doctor", "attributes": [{"name": "id",
                                                            "type": "id"}]},
                         {"name": "Bed", "attributes": [{"name": "id",
                                                         "type": "id"}]}],
            "relationships": ["Bed.ward_id -> Ward.id",
                              "Bed.doctor_id -> Doctor.id"],
        }, num_entities=3, max_depth=2)
        self.assertEqual([r.as_fk() for r in graph.relationships],
                         ["Bed.ward_id -> Ward.id"])
        self.assertTrue(any("already has a parent" in w
                            for w in graph.warnings), graph.warnings)

    def test_trim_keeps_the_hierarchy_over_standalone_entities(self):
        """Replays a real llama3.3:70b response for small_business.

        Six entities came back, four of them parentless, and only two carried
        any structure. Keeping the four roots would satisfy --num-entities 4
        while quietly making --max-depth 2 unreachable.
        """
        payload = {
            "entities": [
                {"name": n, "attributes": [{"name": "id", "type": "id"}]}
                for n in ("Customer", "Product", "Supplier", "Shipment",
                          "Order", "OrderItem")],
            "relationships": ["Order.customer_id -> Customer.id",
                              "OrderItem.order_id -> Order.id",
                              "OrderItem.product_id -> Product.id"],
        }
        graph = generate(payload, num_entities=4, max_depth=2)
        self.assertEqual(sorted(graph.entity_names),
                         ["Customer", "Order", "OrderItem", "Product"])
        self.assertEqual([r.as_fk() for r in graph.relationships],
                         ["Order.customer_id -> Customer.id"])
        self.assertEqual(graph.depth(), 2)

    def test_excess_entities_are_trimmed_deepest_first(self):
        graph = generate(THREE_LEVELS, domain="medical", num_entities=2,
                         max_depth=2)
        self.assertEqual(graph.entity_names, ["Clinic", "Patient"])
        self.assertTrue(any("dropped Visit" in w for w in graph.warnings),
                        graph.warnings)
        for rel in graph.relationships:
            self.assertIn(rel.child_entity, graph.entity_names)
            self.assertIn(rel.parent_entity, graph.entity_names)

    def test_undercount_is_reported_not_repaired(self):
        graph = generate(SMALL_BUSINESS, num_entities=5)
        self.assertEqual(len(graph.entities), 2)
        self.assertTrue(any("5 requested" in w for w in graph.warnings),
                        graph.warnings)

    def test_duplicate_entity_definitions_are_merged(self):
        graph = generate({
            "entities": [{"name": "Order", "attributes": [{"name": "id",
                                                           "type": "id"}]},
                         {"name": "order", "attributes": [{"name": "total",
                                                           "type": "currency"}]}],
            "relationships": [],
        }, num_entities=1)
        self.assertEqual(graph.entity_names, ["Order"])
        self.assertEqual(graph.entity("Order").attribute_names, ["id", "total"])


# --------------------------------------------------------------------------- #
# 2d. Key repair
# --------------------------------------------------------------------------- #
class TestKeyRepair(unittest.TestCase):

    def test_missing_primary_key_is_synthesised_first(self):
        graph = generate({
            "entities": [{"name": "Shift",
                          "attributes": [{"name": "starts_at",
                                          "type": "datetime"}]}],
            "relationships": [],
        }, num_entities=1)
        entity = graph.entities[0]
        self.assertEqual(entity.attribute_names, ["id", "starts_at"])
        self.assertEqual(entity.attribute("id").type, "id")
        self.assertTrue(entity.attribute("id").unique)
        self.assertTrue(any("missing primary key" in w
                            for w in graph.warnings), graph.warnings)

    def test_missing_foreign_key_column_is_added_to_the_child(self):
        payload = json.loads(json.dumps(SMALL_BUSINESS))
        invoice = payload["entities"][1]
        invoice["attributes"] = [a for a in invoice["attributes"]
                                 if a["name"] != "customer_id"]
        graph = generate(payload)
        fk = graph.entity("Invoice").attribute("customer_id")
        self.assertIsNotNone(fk)
        self.assertEqual(fk.type, "id")
        self.assertTrue(fk.required)
        self.assertTrue(any("missing foreign key" in w
                            for w in graph.warnings), graph.warnings)

    def test_foreign_key_column_is_retyped_to_id(self):
        payload = json.loads(json.dumps(SMALL_BUSINESS))
        payload["entities"][1]["attributes"][1]["type"] = "string"
        graph = generate(payload)
        self.assertEqual(graph.entity("Invoice").attribute("customer_id").type,
                         "id")
        self.assertTrue(any("coerced to 'id'" in w for w in graph.warnings))

    def test_unknown_parent_attribute_falls_back_to_the_primary_key(self):
        payload = json.loads(json.dumps(SMALL_BUSINESS))
        payload["relationships"] = ["Invoice.customer_id -> Customer.cust_no"]
        graph = generate(payload)
        rel = graph.relationships[0]
        self.assertEqual(rel.parent_attribute, "id")
        self.assertTrue(any("retargeted" in w for w in graph.warnings),
                        graph.warnings)

    def test_named_primary_key_other_than_id_is_honoured(self):
        graph = generate({
            "entities": [{"name": "Ledger", "primary_key": "ledger_no",
                          "attributes": [{"name": "ledger_no", "type": "string"},
                                         {"name": "opened_on", "type": "date"}]}],
            "relationships": [],
        }, num_entities=1)
        entity = graph.entities[0]
        self.assertEqual(entity.primary_key, "ledger_no")
        self.assertEqual(entity.attribute("ledger_no").type, "id")
        self.assertEqual(len(entity.attributes), 2)


# --------------------------------------------------------------------------- #
# 3. Serialisation and foreign key integrity
# --------------------------------------------------------------------------- #
class TestSerialization(unittest.TestCase):

    def assert_fk_integrity(self, payload: dict) -> None:
        """Every FK in a serialised schema must resolve inside that schema."""
        entities = {e["name"]: e for e in payload["entities"]}
        for rel in payload["relationships"]:
            parent = entities.get(rel["parent_entity"])
            child = entities.get(rel["child_entity"])
            self.assertIsNotNone(parent, f"unknown parent in {rel}")
            self.assertIsNotNone(child, f"unknown child in {rel}")
            parent_attrs = {a["name"]: a for a in parent["attributes"]}
            child_attrs = {a["name"]: a for a in child["attributes"]}
            self.assertIn(rel["parent_attribute"], parent_attrs)
            self.assertIn(rel["child_attribute"], child_attrs)
            self.assertEqual(child_attrs[rel["child_attribute"]]["type"], "id")
            self.assertEqual(rel["cardinality"], "1:m")
            self.assertEqual(
                rel["fk"],
                f"{rel['child_entity']}.{rel['child_attribute']} -> "
                f"{rel['parent_entity']}.{rel['parent_attribute']}")

    def test_written_file_is_json_and_fk_complete(self):
        graph = generate(SMALL_BUSINESS)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "schema.json")
            self.assertEqual(write_schema(graph, path), path)
            payload = read_json(path)
        self.assertEqual(payload["domain"], "small_business")
        self.assertEqual(payload["depth"], 2)
        self.assertEqual(payload["roots"], ["Customer"])
        self.assertEqual(payload["entity_depths"],
                         {"Customer": 1, "Invoice": 2})
        self.assert_fk_integrity(payload)

    def test_metadata_records_the_run(self):
        gen = ParametricSchemaGenerator(client=FakeClient(SMALL_BUSINESS),
                                        seed=42)
        graph = gen.generate_schema(domain="medical", num_entities=2,
                                    max_depth=2)
        meta = graph.to_dict()["metadata"]
        self.assertEqual(meta["domain"], "medical")
        self.assertEqual(meta["requested_entities"], 2)
        self.assertEqual(meta["max_depth"], 2)
        self.assertEqual(meta["seed"], 42)
        self.assertEqual(meta["model"], "llama3.3:70b")
        self.assertEqual(meta["base_url"], "http://127.0.0.1:11434")
        self.assertTrue(meta["generated_at"].endswith("+00:00"))

    def test_round_trip_through_dict(self):
        graph = generate(SMALL_BUSINESS)
        restored = SchemaGraph.from_dict(json.loads(graph.to_json()))
        self.assertEqual(restored.to_dict(), graph.to_dict())

    def test_repaired_schema_is_still_fk_complete(self):
        """The integrity guarantee has to hold on a schema that needed repair."""
        payload = {
            "entities": [
                {"name": "Course",
                 "attributes": [{"name": "title", "type": "string"}]},
                {"name": "Enrolment",
                 "attributes": [{"name": "grade", "type": "percent"}]},
                {"name": "Assignment",
                 "attributes": [{"name": "weight", "type": "percent"}]},
            ],
            "relationships": ["Enrolment.course_id -> Course.code",
                              "Assignment.enrolment_id -> Enrolment.id",
                              "Enrolment.ghost_id -> Ghost.id"],
        }
        graph = generate(payload, domain="education", num_entities=3,
                         max_depth=2)
        serialised = json.loads(graph.to_json())
        self.assert_fk_integrity(serialised)
        self.assertEqual(serialised["depth"], 2)
        self.assertTrue(serialised["warnings"])

    def test_cli_output_passes_the_same_integrity_check(self):
        original = cli_module.build_client
        cli_module.build_client = lambda **kw: FakeClient(THREE_LEVELS)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out = os.path.join(tmp, "bench")
                runner = make_runner()
                result = runner.invoke(cli_module.app, [
                    "generate", "--schema-only", "--domain", "medical",
                    "--num-entities", "3", "--max-depth", "2",
                    "--output-dir", out])
                self.assertEqual(result.exit_code, 0, stderr_of(result))
                payload = read_json(out, "schema.json")
        finally:
            cli_module.build_client = original
        self.assert_fk_integrity(payload)
        self.assertEqual(payload["depth"], 2)


# --------------------------------------------------------------------------- #
# Seeding
# --------------------------------------------------------------------------- #
class TestSeeding(unittest.TestCase):

    def test_seed_reaches_the_ollama_payload_and_pins_temperature(self):
        client = build_client(seed=11)
        self.assertEqual(client.temperature, 0.0)
        _, payload = client._body("sys", "user")
        self.assertEqual(payload["options"]["seed"], 11)

    def test_no_seed_leaves_the_sampler_free(self):
        client = build_client()
        self.assertGreater(client.temperature, 0.0)
        _, payload = client._body("sys", "user")
        self.assertNotIn("seed", payload["options"])

    def test_openai_backend_takes_a_top_level_seed(self):
        client = build_client(backend="openai", seed=5)
        path, payload = client._body("sys", "user")
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(payload["seed"], 5)

    def test_default_endpoint_is_the_local_llama_server(self):
        client = build_client()
        self.assertEqual(client.base_url, "http://127.0.0.1:11434")
        self.assertEqual(client.backend, "ollama")
        self.assertIsInstance(client, SeededLLMClient)


if __name__ == "__main__":
    unittest.main(verbosity=2)
