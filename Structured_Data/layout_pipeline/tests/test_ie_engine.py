"""Tests for the Dynamic IE engine.

Every LLM call is mocked, so nothing here needs Ollama, a GPU or model weights.
The point of the exercise is that *arbitrary* JSON maps onto the tree, so the
fixtures deliberately use keys no code has ever seen.
"""

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dual_extractor import BlockType, SemanticBlock  # noqa: E402
from src.ie_engine.dynamic_extractor import (  # noqa: E402
    DynamicInformationExtractor, chunk_text, render_blocks)
from src.ie_engine.llm_client import (  # noqa: E402
    SCHEMA_DISCOVERY_SYSTEM_PROMPT, LLMUnavailable, LocalLLMClient,
    extract_json)
from src.ie_engine.node_schema import (DynamicDocument,  # noqa: E402
                                       DynamicElement, document_from_json,
                                       element_from_json, sanitize_tag)

REPORT_CARD = {
    "report_card": {
        "student": {
            "last_name": "Siddiqui",
            "first_name": "Farees",
            "address": {"street": "14 Fake Street", "city": "Milton"},
        },
        "evaluation": {"grade": "90", "standing": "Satisfactory"},
        "courses": [
            {"code": "MATH101", "mark": "88"},
            {"code": "PHYS201", "mark": "92"},
        ],
    }
}

#: The shape the StudentRecord corpus should come back as: one nested object per
#: composite, an array for the repeated thing. Mirrors ``gold/doc01.xml``.
STUDENT_RECORD_NESTED = {
    "student_record": {
        "name": {"lastname": "Siddiqui", "firstname": "Farees"},
        "address": {"street": "14 Fake Street", "city": "Milton"},
        "evaluation": {"grade": "90", "remark": "Satisfactory"},
        "units": [{"code": "MATH101", "mark": "88"},
                  {"code": "PHYS201", "mark": "92"}],
    }
}

#: What the 5 Aug run actually returned for ``doc01_email`` -- the same facts,
#: every one of them a root attribute, two of them welded into a single string.
STUDENT_RECORD_FLAT = {
    "student_record": {"name": "Farees Siddiqui",
                       "address": "14 Fake Street, Milton",
                       "grade": "90", "evaluation": "Satisfactory"}
}


class FakeClient:
    """Stands in for LocalLLMClient. Returns queued payloads in order."""

    def __init__(self, *payloads, raises=None):
        self.payloads = list(payloads)
        self.raises = raises
        self.calls = []
        self.last_error = None

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


def blk(btype, content, block_id=0):
    return SemanticBlock(block_id=block_id, block_type=btype,
                         bbox=[0, 0, 10, 10], parsed_content=content,
                         metadata={"page": 0, "status": "extracted"})


class TagSanitising(unittest.TestCase):
    """Tags must survive becoming XML names downstream."""

    def test_spaces_and_case(self):
        self.assertEqual(sanitize_tag("Student Record"), "student-record")

    def test_leading_digit_is_prefixed_not_dropped(self):
        self.assertEqual(sanitize_tag("2024 grades"), "n2024-grades")

    def test_illegal_characters_collapse(self):
        self.assertEqual(sanitize_tag("grade (%)"), "grade")
        self.assertEqual(sanitize_tag("a//b__c"), "a-b__c")

    def test_empty_falls_back(self):
        for value in ("", "   ", None, "!!!"):
            self.assertEqual(sanitize_tag(value), "element", repr(value))

    def test_already_valid_is_untouched(self):
        self.assertEqual(sanitize_tag("report_card"), "report_card")

    def test_overlong_names_are_truncated(self):
        self.assertLessEqual(len(sanitize_tag("x" * 200)), 64)


class JsonToTree(unittest.TestCase):
    """Requirement 2: open JSON maps to the tree, whatever its keys."""

    def test_scalars_become_attributes(self):
        el = element_from_json({"grade": "90", "count": 3, "ok": True}, "eval")
        self.assertEqual(el.tag_name, "eval")
        self.assertEqual(el.attributes,
                         {"grade": "90", "count": "3", "ok": "true"})
        self.assertEqual(el.children, [])

    def test_nested_objects_become_children(self):
        el = element_from_json({"address": {"city": "Milton"}}, "student")
        self.assertEqual([c.tag_name for c in el.children], ["address"])
        self.assertEqual(el.children[0].attributes, {"city": "Milton"})

    def test_arrays_become_repeated_children(self):
        el = element_from_json(
            {"courses": [{"code": "A"}, {"code": "B"}]}, "record")
        self.assertEqual([c.tag_name for c in el.children], ["courses", "courses"])
        self.assertEqual([c.attributes["code"] for c in el.children], ["A", "B"])

    def test_array_of_scalars_becomes_text_children(self):
        el = element_from_json({"tags": ["alpha", "beta"]}, "doc")
        self.assertEqual([c.text_content for c in el.children], ["alpha", "beta"])

    def test_bare_scalar_becomes_text_content(self):
        self.assertEqual(element_from_json("just text", "note").text_content,
                         "just text")

    def test_null_and_empty_values_are_skipped(self):
        el = element_from_json({"a": None, "b": "", "c": "keep"}, "x")
        self.assertEqual(el.attributes, {"c": "keep"})

    def test_unexpected_keys_do_not_raise(self):
        """The whole point: keys no code has declared must just work."""
        weird = {"☃ snowman": 1, "": 2, "123": 3, "a b/c": {"d": 4}}
        el = element_from_json(weird, "weird")
        self.assertIsInstance(el, DynamicElement)
        self.assertTrue(el.attributes or el.children)

    def test_deeply_nested_input_is_bounded(self):
        payload, cursor = {}, None
        for i in range(60):
            node = {}
            if cursor is None:
                payload["root"] = node
            else:
                cursor[f"level{i}"] = node
            cursor = node
        cursor["leaf"] = "value"
        el = element_from_json(payload, "deep")
        self.assertLessEqual(el.depth, 30)

    def test_full_report_card_shape(self):
        doc = document_from_json(REPORT_CARD, source="doc08.pdf")
        self.assertEqual(doc.root.tag_name, "report_card")
        student = doc.root.find("student")
        self.assertEqual(student.attributes["last_name"], "Siddiqui")
        self.assertEqual(student.find("address").attributes["city"], "Milton")
        self.assertEqual(len(doc.root.find_all("courses")), 2)
        self.assertTrue(doc.ok)

    def test_single_key_root_is_unwrapped(self):
        self.assertEqual(document_from_json({"email": {"to": "a@b.c"}}).root.tag_name,
                         "email")

    def test_multi_key_payload_keeps_generic_root(self):
        doc = document_from_json({"a": {"x": 1}, "b": {"y": 2}})
        self.assertEqual(doc.root.tag_name, "document")
        self.assertEqual(len(doc.root.children), 2)

    def test_tree_round_trips_through_dict(self):
        doc = document_from_json(REPORT_CARD)
        restored = DynamicDocument.from_dict(json.loads(json.dumps(doc.to_dict())))
        self.assertEqual(restored.root.to_dict(), doc.root.to_dict())

    def test_navigation_helpers(self):
        doc = document_from_json(REPORT_CARD)
        self.assertIsNone(doc.root.find("nonexistent"))
        self.assertEqual(doc.root.find_all("nonexistent"), [])
        self.assertGreater(doc.elements, 4)
        self.assertGreaterEqual(doc.root.depth, 3)


class NestedEntities(unittest.TestCase):
    """Composite things must arrive as child elements, not root attributes.

    The failure this guards against is real and was measured: the 5 Aug run over
    ``StudentRecord/`` returned every document's facts welded onto the root --
    ``address="14 Fake Street, Milton"``, ``name="Farees Siddiqui"`` -- so a
    six-leaf record came back one node deep with two of its six leaves fused
    into strings. Both halves are tested here: that the tree builder splits a
    nested payload into child elements, and that the prompt actually asks for
    one.
    """

    def test_prompt_states_all_three_nesting_rules(self):
        prompt = SCHEMA_DISCOVERY_SYSTEM_PROMPT.lower()
        self.assertIn("nesting rules", prompt)
        # composite -> nested object
        self.assertIn("nested object", prompt)
        # repeats -> array of objects
        self.assertIn("array of objects", prompt)
        # and the two anti-patterns that were actually observed
        self.assertIn("never weld them into one string", prompt)
        self.assertIn("thing_1", prompt)

    def test_prompt_example_is_itself_nested(self):
        """The shipped example must demonstrate the shape it claims.

        Asserted structurally rather than by string match: the example is parsed
        and run through the real tree builder, so an example that drifts flat
        fails here instead of silently teaching the model the wrong shape.
        """
        right, _, wrong = SCHEMA_DISCOVERY_SYSTEM_PROMPT.partition("WRONG")
        right_json = extract_json(right[right.index("RIGHT"):])
        root = document_from_json(right_json).root

        self.assertGreaterEqual(root.depth, 3, "example is not deeply nested")
        self.assertTrue(root.children, "example has no child elements at all")
        # a composite nested two levels down, and a repeated entity
        repeated = max((t for t in {c.tag_name for c in root.children}),
                       key=lambda t: len(root.find_all(t)))
        self.assertEqual(len(root.find_all(repeated)), 2,
                         "example does not show a repeated entity")
        self.assertTrue(any(c.children for c in root.children),
                        "example shows no composite inside a composite")

    def test_prompt_counter_example_is_flat(self):
        """The contrast only teaches if the WRONG half really is the failure."""
        _, _, wrong = SCHEMA_DISCOVERY_SYSTEM_PROMPT.partition("WRONG")
        root = document_from_json(extract_json(wrong)).root
        self.assertEqual(root.children, [])
        self.assertEqual(root.depth, 1)

    def test_composite_address_becomes_a_child_element(self):
        root = document_from_json(STUDENT_RECORD_NESTED).root
        address = root.find("address")
        self.assertIsNotNone(address, "address did not become an element")
        self.assertIn(address, root.children)
        self.assertEqual(address.attributes,
                         {"street": "14 Fake Street", "city": "Milton"})
        # and its parts did not also leak onto the root
        self.assertNotIn("address", root.attributes)
        self.assertNotIn("street", root.attributes)

    def test_composite_evaluation_becomes_a_child_element(self):
        root = document_from_json(STUDENT_RECORD_NESTED).root
        evaluation = root.find("evaluation")
        self.assertIn(evaluation, root.children)
        self.assertEqual(evaluation.attributes["grade"], "90")
        self.assertEqual(evaluation.attributes["remark"], "Satisfactory")
        self.assertNotIn("grade", root.attributes)

    def test_repeated_units_become_sibling_elements(self):
        root = document_from_json(STUDENT_RECORD_NESTED).root
        units = root.find_all("units")
        self.assertEqual(len(units), 2)
        self.assertEqual([u.attributes["code"] for u in units],
                         ["MATH101", "PHYS201"])

    def test_no_composite_survives_as_a_welded_string(self):
        """The specific 5 Aug defect: two leaves fused into one attribute."""
        for element in document_from_json(STUDENT_RECORD_NESTED).root.walk():
            for key, value in element.attributes.items():
                self.assertNotIn(", ", value,
                                 f"{element.tag_name}/{key} is welded: {value!r}")

    def test_every_leaf_of_the_record_is_reachable(self):
        """All six gold leaves survive as separately addressable values."""
        root = document_from_json(STUDENT_RECORD_NESTED).root
        leaves = {(el.tag_name, k): v
                  for el in root.walk() for k, v in el.attributes.items()}
        self.assertEqual(leaves[("name", "lastname")], "Siddiqui")
        self.assertEqual(leaves[("name", "firstname")], "Farees")
        self.assertEqual(leaves[("address", "street")], "14 Fake Street")
        self.assertEqual(leaves[("address", "city")], "Milton")
        self.assertEqual(leaves[("evaluation", "grade")], "90")
        self.assertEqual(leaves[("evaluation", "remark")], "Satisfactory")

    def test_nesting_survives_the_full_extractor_path(self):
        """End to end through the extractor, not just the tree builder."""
        doc = DynamicInformationExtractor(
            client=FakeClient(STUDENT_RECORD_NESTED)).extract(
                [blk(BlockType.TEXT, "Farees Siddiqui, 14 Fake Street")],
                source="doc01_email")
        self.assertEqual(doc.root.tag_name, "student_record")
        self.assertEqual(doc.root.find("address").attributes["city"], "Milton")
        self.assertGreaterEqual(doc.root.depth, 2)
        self.assertEqual(doc.metadata["status"], "extracted")

    def test_flat_output_is_still_accepted_and_visibly_flat(self):
        """The engine must not start rejecting documents that really are flat.

        Nesting is asked of the model, not enforced by the parser -- a genuinely
        flat document is a legitimate extraction, and depth is the signal that
        tells the two apart.
        """
        root = document_from_json(STUDENT_RECORD_FLAT).root
        self.assertEqual(root.depth, 1)
        self.assertEqual(root.attributes["address"], "14 Fake Street, Milton")


class JsonRecovery(unittest.TestCase):
    """Requirement 3: unparseable output must not explode."""

    def test_plain_json(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})

    def test_markdown_fence_is_stripped(self):
        self.assertEqual(extract_json('```json\n{"a": 1}\n```'), {"a": 1})

    def test_prose_prefix_is_skipped(self):
        self.assertEqual(
            extract_json('Sure! Here is the JSON:\n{"a": 1}\nHope that helps.'),
            {"a": 1})

    def test_braces_inside_strings_do_not_confuse_the_scan(self):
        self.assertEqual(extract_json('x {"a": "} not the end", "b": 2} y'),
                         {"a": "} not the end", "b": 2})

    def test_array_payload(self):
        self.assertEqual(extract_json("[1, 2]"), [1, 2])

    def test_unrecoverable_inputs(self):
        for bad in ("", "   ", None, "no json here", "{broken", "{'a': 1"):
            self.assertIsNone(extract_json(bad), repr(bad))


class Rendering(unittest.TestCase):
    def test_blocks_render_in_order_with_headings_marked(self):
        blocks = [blk(BlockType.TITLE, "Report Card", 0),
                  blk(BlockType.TEXT, "Grade: 90", 1)]
        self.assertEqual(render_blocks(blocks), "## Report Card\n\nGrade: 90")

    def test_figures_are_excluded(self):
        blocks = [blk(BlockType.TEXT, "keep", 0),
                  blk(BlockType.VISION, "![Figure](f.png)", 1)]
        self.assertEqual(render_blocks(blocks), "keep")

    def test_tables_are_included(self):
        self.assertIn("| a |", render_blocks([blk(BlockType.TABLE, "| a |", 0)]))

    def test_ocr_fallback_text_is_used_when_content_is_empty(self):
        b = SemanticBlock(block_id=0, block_type=BlockType.TABLE, bbox=[0, 0, 1, 1],
                          parsed_content="",
                          metadata={"ocr_fallback_text": "raw ocr"})
        self.assertEqual(render_blocks([b]), "raw ocr")

    def test_chunking_splits_on_paragraphs(self):
        text = "\n\n".join(["para " + "x" * 100 for _ in range(10)])
        chunks = chunk_text(text, limit=300)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 400 for c in chunks))
        self.assertEqual("".join(chunks).replace("\n", ""),
                         text.replace("\n", ""))

    def test_short_text_is_one_chunk(self):
        self.assertEqual(chunk_text("short"), ["short"])
        self.assertEqual(chunk_text("   "), [])


class Extraction(unittest.TestCase):
    """Requirement 1: mocked LLM response -> tree."""

    def test_arbitrary_nested_json_becomes_a_tree(self):
        client = FakeClient(REPORT_CARD)
        doc = DynamicInformationExtractor(client=client).extract(
            [blk(BlockType.TEXT, "Farees Siddiqui, grade 90")], source="doc08")
        self.assertEqual(doc.root.tag_name, "report_card")
        self.assertEqual(doc.source, "doc08")
        self.assertEqual(doc.metadata["status"], "extracted")
        self.assertEqual(doc.root.find("evaluation").attributes["grade"], "90")

    def test_document_text_reaches_the_prompt(self):
        client = FakeClient(REPORT_CARD)
        DynamicInformationExtractor(client=client).extract(
            [blk(BlockType.TEXT, "UNIQUE-MARKER-42")])
        self.assertIn("UNIQUE-MARKER-42", client.calls[0][1])

    def test_no_target_schema_is_sent(self):
        """Schema-agnostic by construction.

        The prompt shows JSON *shape* using a deliberately distant domain
        (shipping manifests). It must never name a field or value from the
        document domains this engine is pointed at -- "Siddiqui" appears
        verbatim in the StudentRecord corpus, so an example containing it could
        prime the model to emit it.

        The second group below is every term that survived screening the example
        vocabulary against ``StudentRecord/tex/``. They are banned here so the
        screening is enforced rather than remembered: the nesting example needs
        a composite-location analogue and an evaluation analogue, and the
        obvious words for both ("municipality", "score", "mark") are words the
        corpus itself uses.
        """
        client = FakeClient(REPORT_CARD)
        DynamicInformationExtractor(client=client).extract(
            [blk(BlockType.TEXT, "text")])
        prompt = " ".join(client.calls[0])
        for leaked in ("last_name", "report_card", "grade", "standing",
                       "Siddiqui", "student", "address", "email"):
            self.assertNotIn(leaked, prompt, f"{leaked} leaked into the prompt")
        # Whole words, not substrings: "mark" must not prime the model, but
        # "markdown" in the output rules is not the corpus's word.
        for leaked in ("municipality", "score", "mark", "remark", "assessment",
                       "surname", "evaluation", "term", "college"):
            self.assertIsNone(
                re.search(rf"\b{leaked}\b", prompt, re.IGNORECASE),
                f"{leaked} leaked into the prompt")

    def test_a_different_document_yields_a_different_schema(self):
        email = {"email": {"from": "r@x.ca", "subject": "year-end file"}}
        doc = DynamicInformationExtractor(client=FakeClient(email)).extract(
            [blk(BlockType.TEXT, "From: r@x.ca")])
        self.assertEqual(doc.root.tag_name, "email")
        self.assertEqual(doc.root.attributes["subject"], "year-end file")

    def test_empty_input_is_reported_not_crashed(self):
        doc = DynamicInformationExtractor(client=FakeClient()).extract([])
        self.assertEqual(doc.metadata["status"], "empty")
        self.assertFalse(doc.ok)

    def test_unparseable_output_degrades(self):
        doc = DynamicInformationExtractor(client=FakeClient(None)).extract(
            [blk(BlockType.TEXT, "text")])
        self.assertEqual(doc.metadata["status"], "failed")
        self.assertFalse(doc.ok)
        self.assertIn("reason", doc.metadata)

    def test_client_exception_does_not_propagate(self):
        client = FakeClient(raises=LLMUnavailable("connection refused"))
        doc = DynamicInformationExtractor(client=client).extract(
            [blk(BlockType.TEXT, "text")])
        self.assertEqual(doc.metadata["status"], "failed")
        self.assertIn("connection refused", str(doc.metadata["reason"]))

    def test_multi_chunk_documents_keep_every_chunk(self):
        client = FakeClient({"a": {"x": "1"}}, {"b": {"y": "2"}})
        long_text = "\n\n".join("para " + "z" * 200 for _ in range(20))
        doc = DynamicInformationExtractor(
            client=client, chunk_chars=500).extract_from_text(long_text)
        self.assertGreaterEqual(len(client.calls), 2)
        self.assertEqual(doc.metadata["status"], "partial"
                         if doc.metadata.get("errors") else "extracted")
        self.assertGreaterEqual(len(doc.root.children), 1)

    def test_partial_failure_is_recorded_but_keeps_good_chunks(self):
        client = FakeClient({"a": {"x": "1"}}, None)
        long_text = "\n\n".join("para " + "z" * 200 for _ in range(20))
        doc = DynamicInformationExtractor(
            client=client, chunk_chars=500).extract_from_text(long_text)
        self.assertEqual(doc.metadata["status"], "partial")
        self.assertTrue(doc.ok)
        self.assertIn("errors", doc.metadata)

    def test_extract_pages_flattens(self):
        client = FakeClient(REPORT_CARD)
        doc = DynamicInformationExtractor(client=client).extract_pages(
            [[blk(BlockType.TEXT, "a")], [blk(BlockType.TEXT, "b")]])
        self.assertEqual(doc.metadata["blocks"], 2)

    def test_parse_response_handles_raw_text(self):
        doc = DynamicInformationExtractor.parse_response(
            '```json\n{"invoice": {"total": "9.99"}}\n```')
        self.assertEqual(doc.root.tag_name, "invoice")
        self.assertEqual(doc.root.attributes["total"], "9.99")

    def test_parse_response_on_garbage(self):
        doc = DynamicInformationExtractor.parse_response("not json at all")
        self.assertEqual(doc.metadata["status"], "failed")


class ClientConfig(unittest.TestCase):
    """No network: construction and payload shaping only."""

    def test_ollama_payload(self):
        c = LocalLLMClient(model="llama3.3:70b")
        path, body = c._body("sys", "user")
        self.assertEqual(path, "/api/chat")
        self.assertEqual(body["format"], "json")
        self.assertEqual(body["options"]["temperature"], 0.0)
        self.assertEqual(body["messages"][0]["role"], "system")

    def test_openai_payload(self):
        c = LocalLLMClient(backend="openai", model="meta-llama/Llama-3.3-70B")
        path, body = c._body("sys", "user")
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(body["response_format"], {"type": "json_object"})

    def test_json_mode_can_be_disabled(self):
        _, body = LocalLLMClient(json_mode=False)._body("s", "u")
        self.assertNotIn("format", body)

    def test_unknown_backend_rejected(self):
        with self.assertRaises(ValueError):
            LocalLLMClient(backend="carrier-pigeon")

    def test_content_extraction_for_both_wire_formats(self):
        self.assertEqual(
            LocalLLMClient._content({"message": {"content": "ollama"}}), "ollama")
        self.assertEqual(
            LocalLLMClient._content(
                {"choices": [{"message": {"content": "openai"}}]}), "openai")
        self.assertEqual(LocalLLMClient._content({}), "")

    def test_unreachable_endpoint_is_reported_not_raised(self):
        c = LocalLLMClient(base_url="http://127.0.0.1:9", timeout=1)
        self.assertFalse(c.is_available())
        self.assertIsNone(c.complete_json("sys", "user"))
        self.assertIsNotNone(c.last_error)


if __name__ == "__main__":
    unittest.main()
