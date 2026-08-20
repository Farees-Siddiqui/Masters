"""Tests for Stage 5: LLM-written LaTeX, compilation, and the repair loop.

Every model call is mocked. Compilation is tested two ways: through a scripted
stand-in for pdflatex, so the self-correction loop can be driven without a TeX
installation, and against the real binary where one is present. The tests that
need TeX are skipped with a clear reason when it is absent.
"""

from __future__ import annotations

import inspect
import json
import re
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typer.testing import CliRunner  # noqa: E402

from src.generator import cli as cli_module  # noqa: E402
from src.generator.instance_types import InstanceGraph, Record  # noqa: E402
from src.generator.latex_generator import (  # noqa: E402
    ALLOWED_PACKAGES, HYPHENATION_GUARD, LATEX_GENERATION_SYSTEM_PROMPT,
    LATEX_REPAIR_SYSTEM_PROMPT, LATEX_RESTORE_SYSTEM_PROMPT, NULL_GLYPH,
    LatexText, LaTeXGenerationError, LLMLaTeXGenerator, escape_latex,
    extract_latex, harden_source, humanize, layout_instruction,
    leaked_examples, missing_values, normalize_for_comparison,
    PROMPT_EXAMPLE_VALUES, recorded_values, unescape_latex)
from src.generator.renderer import (CONCRETE_LAYOUTS,  # noqa: E402
                                    LAYOUT_STYLES, MANIFEST_FILENAME,
                                    CompileResult, DocumentScope, LaTeXRenderer,
                                    RenderError, document_scopes, log_errors)
from src.generator.schema_types import SchemaGraph  # noqa: E402

HAVE_TEX = LaTeXRenderer(generator=object()).resolve_engine() is not None
needs_tex = unittest.skipUnless(
    HAVE_TEX, "no LaTeX engine on PATH (pdflatex/xelatex/lualatex)")


def setUpModule() -> None:
    logging.disable(logging.WARNING)


def tearDownModule() -> None:
    logging.disable(logging.NOTSET)


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
SCHEMA_PAYLOAD = {
    "entities": [
        {"name": "Customer", "primary_key": "id", "attributes": [
            {"name": "id", "type": "id"},
            {"name": "company_name", "type": "string", "required": True},
            {"name": "billing_address", "type": "address", "required": True},
            {"name": "contact_email", "type": "email"}]},
        {"name": "Order", "primary_key": "id", "attributes": [
            {"name": "id", "type": "id"},
            {"name": "customer_id", "type": "id", "required": True},
            {"name": "order_date", "type": "date", "required": True},
            {"name": "total", "type": "currency", "required": True},
            {"name": "status", "type": "enum", "required": True,
             "values": ["draft", "paid"]}]}],
    "relationships": ["Order.customer_id -> Customer.id"],
}


def schema() -> SchemaGraph:
    return SchemaGraph.from_payload(SCHEMA_PAYLOAD, domain="small_business")


def customer(index: int, name: str, email=None) -> Record:
    return Record(id=f"customer-{index:03d}", entity_name="Customer",
                  attributes={"id": f"customer-{index:03d}",
                              "company_name": name,
                              "billing_address": f"{index} Mill Bank, Leeds",
                              "contact_email": email})


def order(index: int, parent: str, total: str, orphan: bool = False) -> Record:
    return Record(id=f"order-{index:03d}", entity_name="Order",
                  attributes={"id": f"order-{index:03d}",
                              "order_date": "2022-01-15",
                              "total": total, "status": "paid"},
                  foreign_keys={"customer_id": parent},
                  orphaned_keys=["customer_id"] if orphan else [])


def graph(*records: Record) -> InstanceGraph:
    out = InstanceGraph(schema_domain="small_business")
    out.extend(records)
    return out


def standard_graph() -> InstanceGraph:
    """Two customers, three orders between them, and one orphaned order."""
    return graph(
        customer(1, "Halvorsen Freight"),
        customer(2, "Kestrel Supplies", "k@example.invalid"),
        order(1, "customer-001", "$1,240.50"),
        order(2, "customer-001", "$98.00"),
        order(3, "customer-002", "$450.00"),
        order(4, "customer-__orphan_4__", "$12.00", orphan=True))


def faithful_document(records, preamble: str = "") -> str:
    """A compilable document carrying every recorded value, properly escaped.

    Stands in for a well-behaved model: what the real one is asked for, minus
    the layout flair, so fidelity and compilation can both be checked against a
    known-good baseline.
    """
    body = []
    for record in records:
        body.append(r"\section*{%s}" % escape_latex(record.id))
        for name, value in record.attributes.items():
            if value is None:
                body.append(r"%s: %s\\" % (escape_latex(humanize(name)),
                                           NULL_GLYPH))
            else:
                body.append(r"%s: %s\\" % (escape_latex(humanize(name)),
                                           escape_latex(value)))
    return ("\\documentclass{article}\n"
            "\\usepackage[T1]{fontenc}\n" + preamble +
            "\\begin{document}\n" + "\n".join(body) +
            "\n\\end{document}\n")


def unescaped_document(records) -> str:
    """A document that will not compile: a reserved character left raw."""
    return faithful_document(records).replace(r"\$", "$", 1)


class FakeLatexClient:
    """Stands in for SeededLLMClient. ``complete`` returns queued strings."""

    def __init__(self, *responses, raises=None):
        self.responses = list(responses)
        self.raises = raises
        self.calls = []
        self.last_error = None
        self.model = "llama3.3:70b"
        self.base_url = "http://127.0.0.1:11434"
        self.backend = "ollama"
        self.temperature = 0.0
        self.json_mode = False

    def complete(self, system, user):
        self.calls.append((system, user))
        if self.raises:
            raise self.raises
        if not self.responses:
            self.last_error = "no output"
            return ""
        response = self.responses.pop(0)
        if response is None:
            self.last_error = "model returned nothing"
            return ""
        return response


class CyclingLatexClient(FakeLatexClient):
    """Answers every call with a faithful document, varied per call.

    The variation matters: the renderer abandons a document whose repair came
    back byte-identical, since recompiling the same input can only fail the same
    way. A real repair changes something, so this fake does too.
    """

    def __init__(self, records):
        super().__init__()
        self.records = records
        self.passes = 0

    def complete(self, system, user):
        self.calls.append((system, user))
        self.passes += 1
        return faithful_document(
            self.records, preamble=f"% generated on pass {self.passes}\n")


def generator(*responses, **kwargs) -> LLMLaTeXGenerator:
    kwargs.setdefault("schema", schema())
    kwargs.setdefault("seed", 3)
    return LLMLaTeXGenerator(client=FakeLatexClient(*responses), **kwargs)

class TestPromptLeakage(unittest.TestCase):
    """A value from the prompt's own examples appearing on the page as data.

    Observed on a real llama3.1:8b run: the letter layout's example sentence
    ("The account balance on file is \\$1,240.50.") was copied onto a student
    enrolment letter, which then stated an account balance no record held. This
    is the most dangerous of the failure modes because the manifest cannot be
    wrong about it -- it simply does not mention it, so a consumer of the corpus
    has no way to tell the invented fact from a real one.
    """

    def render(self, source, records):
        instance = graph(*records)
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        engine = scripted(responses=[source, source])
        engine.render_documents(instance, Path(tmp), layout_style="form")
        return engine.manifest, engine

    # -- the prompt itself ---------------------------------------------- #
    def test_no_layout_instruction_hands_over_a_ready_made_value(self):
        """A complete example sentence with a value in it invites a copy."""
        for style in CONCRETE_LAYOUTS:
            instruction = layout_instruction(style)
            for example in PROMPT_EXAMPLE_VALUES:
                with self.subTest(style=style, example=example):
                    self.assertNotIn(example, instruction)

    def test_generation_prompt_forbids_reusing_its_own_examples(self):
        prompt = LATEX_GENERATION_SYSTEM_PROMPT.lower()
        self.assertIn("show form only", prompt)
        self.assertIn("they are not data", prompt)

    def test_no_layout_instruction_offers_a_form_to_fill_in(self):
        """A fill-in-the-blank sentence gets filled in literally.

        The first attempt at fixing the leak replaced the worked example with a
        placeholder template, "The <field label> on file is <the exact value>."
        llama3.1:8b wrote the angle brackets onto the page HTML-escaped
        (``\\&lt;2021-02-15\\&gt;``) and filled the slots with the wrong
        values. A template is an instruction to copy; the wording has to
        describe the requirement instead.
        """
        for style in CONCRETE_LAYOUTS:
            with self.subTest(style=style):
                self.assertNotIn("<", layout_instruction(style))

    # -- the detector ---------------------------------------------------- #
    def test_a_value_no_record_holds_is_a_leak(self):
        records = [customer(1, "Halvorsen Freight")]
        source = faithful_document(records).replace(
            r"\begin{document}",
            r"\begin{document}" + "\nThe account balance on file is \\$1,240.50.\n")
        self.assertEqual(leaked_examples(records, source), ["$1,240.50"])

    def test_a_value_a_record_actually_holds_is_not_a_leak(self):
        """An education corpus may legitimately contain "Early Learning"."""
        records = [customer(1, "Early Learning"),
                   order(1, "customer-001", "$1,240.50")]
        self.assertEqual(leaked_examples(records, faithful_document(records)),
                         [])

    def test_a_faithful_document_leaks_nothing(self):
        records = [customer(1, "Halvorsen Freight")]
        self.assertEqual(leaked_examples(records, faithful_document(records)),
                         [])

    def test_detection_survives_the_models_own_escaping(self):
        records = [customer(1, "Halvorsen Freight")]
        source = faithful_document(records).replace(
            r"\begin{document}",
            r"\begin{document}" + "\nSpecial offer: Bell \\& Co. 50\\% off\\_now.\n")
        self.assertEqual(sorted(leaked_examples(records, source)),
                         ["50% off_now", "Bell & Co."])

    # -- reporting -------------------------------------------------------- #
    def test_a_leak_is_recorded_against_the_document(self):
        records = [customer(1, "Halvorsen Freight")]
        source = faithful_document(records).replace(
            r"\begin{document}",
            r"\begin{document}" + "\nThe account balance on file is \\$1,240.50.\n")
        manifest, engine = self.render(source, records)
        fidelity = manifest["documents"][0]["fidelity"]
        self.assertEqual(fidelity["examples_leaked"], 1)
        self.assertEqual(fidelity["leaked"], ["$1,240.50"])
        self.assertEqual(manifest["summary"]["examples_leaked"], 1)
        self.assertTrue(any("came from the prompt" in w
                            for w in engine.warnings))

    def test_a_leak_does_not_fail_the_document(self):
        """It is a usable page with a known extra fact, not a broken one."""
        records = [customer(1, "Halvorsen Freight")]
        source = faithful_document(records).replace(
            r"\begin{document}",
            r"\begin{document}" + "\nThe account balance on file is \\$1,240.50.\n")
        manifest, _ = self.render(source, records)
        self.assertEqual(manifest["documents"][0]["status"], "compiled")

    def test_a_clean_corpus_reports_zero_leaks(self):
        records = [customer(1, "Halvorsen Freight")]
        manifest, engine = self.render(faithful_document(records), records)
        self.assertEqual(manifest["documents"][0]["fidelity"]["examples_leaked"],
                         0)
        self.assertNotIn("leaked", manifest["documents"][0]["fidelity"])
        self.assertEqual(manifest["summary"]["examples_leaked"], 0)

    def test_leaks_are_counted_alongside_a_missing_value(self):
        """Both fidelity checks run on the same document, not one or the other."""
        records = [customer(1, "Halvorsen Freight")]
        source = faithful_document(records).replace(
            "Halvorsen Freight", "").replace(
            r"\begin{document}",
            r"\begin{document}" + "\nThe account balance on file is \\$1,240.50.\n")
        manifest, _ = self.render(source, records)
        fidelity = manifest["documents"][0]["fidelity"]
        self.assertEqual(fidelity["examples_leaked"], 1)
        self.assertGreaterEqual(fidelity["values_missing_before_repair"], 1)

    def test_stderr_summary_mentions_leaks(self):
        records = [customer(1, "Halvorsen Freight")]
        _, engine = self.render(faithful_document(records), records)
        self.assertIn("prompt example(s) leaked", engine.summary())



# --------------------------------------------------------------------------- #
# A scripted compiler, so the repair loop can be driven without TeX
# --------------------------------------------------------------------------- #
class ScriptedRenderer(LaTeXRenderer):
    """``LaTeXRenderer`` with pdflatex replaced by a list of outcomes.

    Each outcome is either ``True`` (a PDF comes out) or a string (that error,
    no PDF). Outcomes are consumed per compilation attempt, so a script of
    ``[error, error, True]`` exercises two repairs.
    """

    def __init__(self, *args, outcomes=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.outcomes = list(outcomes)
        #: The source handed to each compilation, in order.
        self.compiled_sources = []

    def resolve_engine(self):
        return "scripted-engine"

    def _compile_once(self, source, stem, tmp_dir, engine):
        self.compiled_sources.append(source)
        # Written even in the stand-in: the real one writes here too, and the
        # test that nothing leaks out of the temporary directory depends on it.
        (Path(tmp_dir) / f"{stem}.tex").write_text(source, encoding="utf-8")
        outcome = self.outcomes.pop(0) if self.outcomes else True
        if outcome is True:
            return CompileResult(pdf=b"%PDF-1.5 scripted\n%%EOF\n",
                                 log="Output written")
        log_text = (f"This is a scripted run.\n! {outcome}\n"
                    f"l.42 the offending line\n")
        return CompileResult(pdf=None, log=log_text,
                             errors=f"! {outcome} l.42 the offending line")


def scripted(*, outcomes=(), responses=None, records=None, **kwargs):
    """A ScriptedRenderer whose model answers with faithful documents."""
    client = (FakeLatexClient(*responses) if responses is not None
              else CyclingLatexClient(records or []))
    kwargs.setdefault("schema", schema())
    return ScriptedRenderer(
        outcomes=outcomes,
        generator=LLMLaTeXGenerator(client=client, schema=kwargs.pop("schema"),
                                    seed=1),
        seed=1, **kwargs)


# --------------------------------------------------------------------------- #
# 1. LaTeX escaping
# --------------------------------------------------------------------------- #
class TestEscaping(unittest.TestCase):

    def test_every_reserved_character(self):
        for raw, expected in (
                ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                ("~", r"\textasciitilde{}"),
                ("^", r"\textasciicircum{}"),
                ("\\", r"\textbackslash{}")):
            with self.subTest(raw=raw):
                self.assertEqual(escape_latex(raw), expected)

    def test_backslash_is_not_re_escaped_by_the_brace_rules(self):
        self.assertEqual(escape_latex("a\\b"), r"a\textbackslash{}b")
        self.assertNotIn(r"\textbackslash\{", escape_latex("\\"))

    def test_mixed_string(self):
        self.assertEqual(
            escape_latex("Bell & Co. 50% $5 _x_ {y} #1"),
            r"Bell \& Co. 50\% \$5 \_x\_ \{y\} \#1")

    def test_escaping_is_idempotent(self):
        once = escape_latex("100% & rising")
        self.assertEqual(escape_latex(once), once)
        self.assertIsInstance(once, LatexText)

    def test_none_and_booleans(self):
        self.assertEqual(escape_latex(None), NULL_GLYPH)
        self.assertEqual(escape_latex(True), "Yes")

    def test_unescape_round_trips(self):
        for raw in ("Bell & Co. 50%", "a_b$c#d{e}", "back\\slash", "~tilde^"):
            with self.subTest(raw=raw):
                self.assertEqual(unescape_latex(escape_latex(raw)), raw)

    def test_escaping_rules_are_stated_in_the_system_prompt(self):
        """The model does its own escaping now, so it has to be told the rules."""
        for char in ("&", "%", "$", "#", "_"):
            with self.subTest(char=char):
                self.assertIn(f"\\{char}", LATEX_GENERATION_SYSTEM_PROMPT)
        self.assertIn(r"\textbackslash{}", LATEX_GENERATION_SYSTEM_PROMPT)
        for char in ("&", "%", "$", "#", "_"):
            self.assertIn(f"\\{char}", LATEX_REPAIR_SYSTEM_PROMPT)

    def test_humanize(self):
        self.assertEqual(humanize("company_name"), "Company Name")
        self.assertEqual(humanize("order-date"), "Order Date")


# --------------------------------------------------------------------------- #
# 2. Extracting a document from a model response
# --------------------------------------------------------------------------- #
class TestExtractLatex(unittest.TestCase):

    DOC = "\\documentclass{article}\n\\begin{document}\nx\n\\end{document}"

    def test_bare_source_passes_through(self):
        self.assertEqual(extract_latex(self.DOC).strip(), self.DOC)

    def test_markdown_fence_is_stripped(self):
        for fence in ("```latex", "```tex", "```"):
            with self.subTest(fence=fence):
                wrapped = f"{fence}\n{self.DOC}\n```"
                self.assertEqual(extract_latex(wrapped).strip(), self.DOC)

    def test_commentary_around_the_document_is_dropped(self):
        noisy = (f"Sure! Here is the document you asked for:\n\n{self.DOC}\n\n"
                 f"Let me know if you would like a different layout.")
        self.assertEqual(extract_latex(noisy).strip(), self.DOC)

    def test_missing_documentclass_is_unusable(self):
        self.assertIsNone(extract_latex("\\begin{document}x\\end{document}"))

    def test_truncated_document_is_unusable(self):
        self.assertIsNone(extract_latex("\\documentclass{article}\\begin{doc"))

    def test_empty_and_none(self):
        self.assertIsNone(extract_latex(""))
        self.assertIsNone(extract_latex(None))
        self.assertIsNone(extract_latex("   \n "))

    def test_hyphenation_guard_is_added_when_absent(self):
        """A hyphen LaTeX inserts into a value cannot be read back off the page."""
        hardened = harden_source(self.DOC)
        self.assertIn(r"\hyphenpenalty=10000", hardened)
        self.assertLess(hardened.index("hyphenpenalty"),
                        hardened.index(r"\begin{document}"))

    def test_hyphenation_guard_is_not_duplicated(self):
        once = harden_source(self.DOC)
        self.assertEqual(harden_source(once), once)

    def test_generated_source_is_hardened(self):
        source = generator(self.DOC).generate_latex_source(
            [customer(1, "A")], "form", "small_business")
        self.assertIn(r"\hyphenpenalty=10000", source)


# --------------------------------------------------------------------------- #
# 3. Prompt construction
# --------------------------------------------------------------------------- #
class TestPrompts(unittest.TestCase):

    def records(self):
        return [customer(1, "Halvorsen Freight"),
                order(1, "customer-001", "$1,240.50")]

    def test_records_are_described_with_names_types_and_values(self):
        described = generator().describe_records(self.records())
        self.assertIn("MAIN RECORD -- Customer [identifier: customer-001]", described)
        self.assertIn("Company Name (string) = Halvorsen Freight", described)
        self.assertIn("Total (currency) = $1,240.50", described)
        self.assertIn("LINKED RECORD -- Order", described)
        self.assertIn("linked to customer-001", described)

    def test_values_are_given_raw_not_escaped(self):
        """Pre-escaped values would be escaped twice by the model."""
        described = generator().describe_records(
            [customer(1, "Bell & Co. 50%")])
        self.assertIn("Bell & Co. 50%", described)
        self.assertNotIn(r"\&", described)

    def test_nulls_are_marked_as_blank(self):
        described = generator().describe_records([customer(1, "A")])
        self.assertIn("Contact Email (email) = (blank", described)

    def test_identifier_is_not_repeated_as_a_field(self):
        described = generator().describe_records([customer(1, "A")])
        self.assertEqual(described.count("customer-001"), 1)

    def test_foreign_key_value_is_withheld(self):
        """The join is expressed by containment; printing it gives it away."""
        described = generator().describe_records(self.records())
        self.assertNotIn("Customer Id =", described)

    def test_orphaned_key_is_described_as_unmatchable(self):
        described = generator().describe_records(
            [order(4, "customer-__orphan_4__", "$12.00", orphan=True)])
        self.assertIn("customer-__orphan_4__", described)
        self.assertIn("matches no record", described)

    def test_layout_instruction_differs_per_style(self):
        table, form, letter = (layout_instruction(s) for s in CONCRETE_LAYOUTS)
        self.assertIn("booktabs", table)
        self.assertIn("form", form.lower())
        self.assertIn("prose", letter)
        self.assertNotIn("tabular", letter.replace("Do not use a tabular", ""))

    def test_unknown_layout_is_rejected(self):
        with self.assertRaises(ValueError):
            layout_instruction("origami")
        with self.assertRaises(ValueError):
            generator().generate_latex_source([customer(1, "A")], "origami", "d")

    def test_user_prompt_carries_domain_layout_and_records(self):
        prompt = generator().build_user_prompt(self.records(), "table",
                                               "small_business")
        self.assertIn("Domain: small_business", prompt)
        self.assertIn("booktabs", prompt)
        self.assertIn("Halvorsen Freight", prompt)
        self.assertIn("2 in total", prompt)

    def test_observed_compile_failures_are_named_in_both_prompts(self):
        """Each rule here is a failure seen on a real run, not a precaution."""
        for phrase in ("Environment letter undefined", "Misplaced \\noalign",
                       "There's no line here to end"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, LATEX_GENERATION_SYSTEM_PROMPT)
                self.assertIn(phrase.split()[-1].rstrip("."),
                              LATEX_REPAIR_SYSTEM_PROMPT)

    def test_system_prompt_states_the_fidelity_rules(self):
        prompt = LATEX_GENERATION_SYSTEM_PROMPT
        self.assertIn("EXACTLY", prompt)
        self.assertIn("Do not invent", prompt)
        for package in ("geometry", "booktabs", "longtable"):
            self.assertIn(package, prompt)
        self.assertIn("Do not use \\write18", prompt)

    def test_allowed_packages_are_the_ones_named_in_the_prompt(self):
        for package in ALLOWED_PACKAGES:
            if package in ("fontenc", "inputenc"):
                continue
            with self.subTest(package=package):
                self.assertIn(package, LATEX_GENERATION_SYSTEM_PROMPT)

    def test_empty_record_list_is_rejected(self):
        with self.assertRaises(ValueError):
            generator().describe_records([])


# --------------------------------------------------------------------------- #
# 4. Generation retries
# --------------------------------------------------------------------------- #
class TestGeneration(unittest.TestCase):

    DOC = "\\documentclass{article}\n\\begin{document}\nx\n\\end{document}"

    def test_unusable_response_is_retried(self):
        gen = generator("I would rather not.", "", self.DOC, max_attempts=3)
        source = gen.generate_latex_source([customer(1, "A")], "form", "d")
        self.assertIn(r"\documentclass", source)
        self.assertEqual(len(gen.client.calls), 3)
        self.assertIn("could not be used", gen.client.calls[1][1])

    def test_gives_up_after_max_attempts(self):
        gen = generator("no", "no", max_attempts=2)
        with self.assertRaises(LaTeXGenerationError):
            gen.generate_latex_source([customer(1, "A")], "form", "d")
        self.assertEqual(len(gen.client.calls), 2)

    def test_json_mode_is_off_for_latex(self):
        """ollama's format=json would return a JSON object, not a document."""
        from src.generator.llm_bridge import build_client
        client = build_client(json_mode=False, seed=1)
        _, body = client._body("s", "u")
        self.assertNotIn("format", body)

    def test_repair_returns_the_original_when_the_model_fails(self):
        gen = generator(None)
        original = self.DOC
        self.assertEqual(gen.repair_latex_source(original, "! error"), original)

    def test_repair_prompt_carries_the_log_and_the_source(self):
        gen = generator(self.DOC)
        gen.repair_latex_source("BROKEN SOURCE", "! Missing $ inserted.",
                                [customer(1, "Halvorsen Freight")])
        system, user = gen.client.calls[0]
        self.assertEqual(system, LATEX_REPAIR_SYSTEM_PROMPT)
        self.assertIn("! Missing $ inserted.", user)
        self.assertIn("BROKEN SOURCE", user)
        self.assertIn("Halvorsen Freight", user)

    def test_restore_prompt_lists_the_missing_values(self):
        gen = generator(self.DOC)
        gen.restore_values("SOURCE", [("customer-001", "company_name",
                                       "Halvorsen Freight")])
        system, user = gen.client.calls[0]
        self.assertEqual(system, LATEX_RESTORE_SYSTEM_PROMPT)
        self.assertIn("record customer-001, field company_name", user)
        self.assertIn("Halvorsen Freight", user)

    def test_restore_is_a_no_op_with_nothing_missing(self):
        gen = generator()
        self.assertEqual(gen.restore_values("SOURCE", []), "SOURCE")
        self.assertEqual(gen.client.calls, [])


# --------------------------------------------------------------------------- #
# 5. Fidelity
# --------------------------------------------------------------------------- #
class TestFidelity(unittest.TestCase):

    def test_recorded_values_cover_ids_and_attributes(self):
        values = recorded_values([customer(1, "Halvorsen Freight")])
        fields = {field for _, field, _ in values}
        self.assertEqual(fields, {"id", "company_name", "billing_address"})

    def test_nulls_are_not_expected_on_the_page(self):
        values = recorded_values([customer(1, "A")])
        self.assertNotIn("contact_email", {f for _, f, _ in values})

    def test_foreign_keys_are_not_expected_on_the_page(self):
        values = recorded_values([order(1, "customer-001", "$8.00")])
        self.assertNotIn("customer_id", {f for _, f, _ in values})

    def test_a_faithful_document_is_complete(self):
        records = [customer(1, "Halvorsen Freight"),
                   order(1, "customer-001", "$1,240.50")]
        self.assertEqual(missing_values(records, faithful_document(records)), [])

    def test_a_dropped_value_is_detected(self):
        records = [customer(1, "Halvorsen Freight")]
        source = faithful_document(records).replace("Halvorsen Freight", "")
        missing = missing_values(records, source)
        self.assertEqual([(f, v) for _, f, v in missing],
                         [("company_name", "Halvorsen Freight")])

    def test_a_rounded_value_is_detected(self):
        """The failure mode a compiler cannot see: $1,240.50 -> $1240.5."""
        records = [order(1, "customer-001", "$1,240.50")]
        source = faithful_document(records).replace(r"\$1,240.50", r"\$1240.5")
        self.assertEqual([f for _, f, _ in missing_values(records, source)],
                         ["total"])

    def test_a_missing_record_is_detected(self):
        records = [customer(1, "A"), order(1, "customer-001", "$8.00")]
        source = faithful_document(records[:1])
        missing = {r for r, _, _ in missing_values(records, source)}
        self.assertEqual(missing, {"order-001"})

    def test_formatting_around_a_value_does_not_count_as_missing(self):
        records = [customer(1, "Halvorsen Freight")]
        source = faithful_document(records).replace(
            "Halvorsen Freight", r"\textbf{Halvorsen Freight}")
        self.assertEqual(missing_values(records, source), [])

    def test_a_value_split_across_lines_does_not_count_as_missing(self):
        records = [customer(1, "Halvorsen Freight")]
        source = faithful_document(records).replace(
            "Halvorsen Freight", "Halvorsen\n    Freight")
        self.assertEqual(missing_values(records, source), [])

    def test_a_tie_inside_a_value_does_not_count_as_missing(self):
        records = [customer(1, "Halvorsen Freight")]
        source = faithful_document(records).replace(
            "Halvorsen Freight", "Halvorsen~Freight")
        self.assertEqual(missing_values(records, source), [])

    def test_differently_escaped_value_does_not_count_as_missing(self):
        records = [customer(1, "Bell & Co")]
        source = faithful_document(records).replace(r"Bell \& Co",
                                                    r"Bell {\&} Co")
        self.assertEqual(missing_values(records, source), [])

    def test_normalize_flattens_escapes_breaks_and_whitespace(self):
        self.assertIn("50% off",
                      normalize_for_comparison("50\\%\\\\\n    off"))

    def test_a_literal_backslash_in_data_is_not_read_as_a_break(self):
        """"\\textbackslash{}" is data; "\\\\" is a line break."""
        self.assertIn("a\\b", normalize_for_comparison(r"a\textbackslash{}b"))

    def test_a_value_broken_by_a_row_break_is_present(self):
        records = [customer(1, "Halvorsen Freight")]
        source = faithful_document(records).replace(
            "Halvorsen Freight", "Halvorsen\\\\\nFreight")
        self.assertEqual(missing_values(records, source), [])


class TestFidelityLoop(unittest.TestCase):
    """The restore-once behaviour inside the renderer."""

    def render(self, responses, records=None, **kwargs):
        instance = graph(*(records or [customer(1, "Halvorsen Freight")]))
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        engine = scripted(responses=responses, **kwargs)
        engine.render_documents(instance, Path(tmp), layout_style="form")
        return engine.manifest, Path(tmp), engine

    def test_complete_document_needs_no_restoration(self):
        records = [customer(1, "Halvorsen Freight")]
        manifest, _, engine = self.render([faithful_document(records)], records)
        fidelity = manifest["documents"][0]["fidelity"]
        self.assertEqual(fidelity["values_missing"], 0)
        self.assertFalse(fidelity["restored"])
        self.assertEqual(len(engine.generator.client.calls), 1)

    def test_dropped_value_triggers_one_restoration(self):
        records = [customer(1, "Halvorsen Freight")]
        broken = faithful_document(records).replace("Halvorsen Freight", "")
        manifest, _, engine = self.render(
            [broken, faithful_document(records)], records)
        fidelity = manifest["documents"][0]["fidelity"]
        self.assertEqual(fidelity["values_missing_before_repair"], 1)
        self.assertEqual(fidelity["values_missing"], 0)
        self.assertTrue(fidelity["restored"])
        self.assertEqual(len(engine.generator.client.calls), 2)
        self.assertEqual(manifest["summary"]["documents_value_complete"], 1)

    def test_restoration_is_attempted_only_once(self):
        """A model that dropped it twice will drop it again; report instead."""
        records = [customer(1, "Halvorsen Freight")]
        broken = faithful_document(records).replace("Halvorsen Freight", "")
        manifest, _, engine = self.render([broken, broken], records)
        fidelity = manifest["documents"][0]["fidelity"]
        self.assertEqual(fidelity["values_missing"], 1)
        self.assertFalse(fidelity["restored"])
        self.assertEqual(len(engine.generator.client.calls), 2)

    def test_unrestored_values_are_named_in_the_manifest(self):
        records = [customer(1, "Halvorsen Freight")]
        broken = faithful_document(records).replace("Halvorsen Freight", "")
        manifest, _, engine = self.render([broken, broken], records)
        entry = manifest["documents"][0]
        self.assertEqual(entry["fidelity"]["missing"],
                         [{"record": "customer-001", "field": "company_name",
                           "value": "Halvorsen Freight"}])
        self.assertEqual(manifest["summary"]["values_missing"], 1)
        self.assertEqual(manifest["summary"]["documents_value_complete"], 0)
        self.assertTrue(any("not on the page" in w for w in engine.warnings))

    def test_a_page_with_missing_values_still_compiles_and_is_kept(self):
        """It is a usable document with a recorded gap, not a failure."""
        records = [customer(1, "Halvorsen Freight")]
        broken = faithful_document(records).replace("Halvorsen Freight", "")
        manifest, out, _ = self.render([broken, broken], records)
        self.assertEqual(manifest["documents"][0]["status"], "compiled")
        self.assertTrue((out / manifest["documents"][0]["pdf"]).is_file())


# --------------------------------------------------------------------------- #
# 6. The self-correction loop
# --------------------------------------------------------------------------- #
class TestRepairLoop(unittest.TestCase):

    def render(self, outcomes, records=None, **kwargs):
        records = records or [customer(1, "Halvorsen Freight")]
        instance = graph(*records)
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        engine = scripted(outcomes=outcomes, records=records, **kwargs)
        pdfs = engine.render_documents(instance, Path(tmp),
                                       layout_style="form",
                                       max_retries=kwargs.pop("max_retries", 2))
        return engine.manifest, Path(tmp), engine, pdfs

    def test_first_pass_success_needs_no_repair(self):
        manifest, _, engine, pdfs = self.render([True])
        entry = manifest["documents"][0]
        self.assertEqual(entry["status"], "compiled")
        self.assertEqual(entry["attempts"], 1)
        self.assertNotIn("repairs", entry)
        self.assertEqual(len(pdfs), 1)
        self.assertEqual(manifest["summary"]["repaired"], 0)

    def test_failure_then_success_records_one_repair(self):
        manifest, _, engine, pdfs = self.render(
            ["Undefined control sequence.", True])
        entry = manifest["documents"][0]
        self.assertEqual(entry["status"], "compiled")
        self.assertEqual(entry["attempts"], 2)
        self.assertEqual(len(entry["repairs"]), 1)
        self.assertEqual(entry["repairs"][0]["attempt"], 1)
        self.assertIn("Undefined control sequence", entry["repairs"][0]["errors"])
        self.assertEqual(len(pdfs), 1)
        self.assertEqual(manifest["summary"]["repaired"], 1)
        self.assertEqual(manifest["summary"]["compilations"], 2)

    def test_two_failures_then_success_uses_both_retries(self):
        manifest, _, engine, pdfs = self.render(["Bad 1", "Bad 2", True])
        entry = manifest["documents"][0]
        self.assertEqual(entry["status"], "compiled")
        self.assertEqual(entry["attempts"], 3)
        self.assertEqual(len(entry["repairs"]), 2)
        self.assertEqual(len(pdfs), 1)

    def test_retries_are_bounded(self):
        manifest, _, engine, pdfs = self.render(
            ["Bad 1", "Bad 2", "Bad 3", "Bad 4", True])
        entry = manifest["documents"][0]
        self.assertEqual(entry["status"], "failed")
        self.assertEqual(entry["attempts"], 3)  # 1 + max_retries
        self.assertEqual(pdfs, [])
        self.assertIn("Bad 3", entry["failure"])

    def test_zero_retries_disables_repair(self):
        records = [customer(1, "Halvorsen Freight")]
        instance = graph(*records)
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        engine = scripted(outcomes=["Bad", True], records=records)
        pdfs = engine.render_documents(instance, Path(tmp),
                                       layout_style="form", max_retries=0)
        self.assertEqual(pdfs, [])
        self.assertEqual(engine.manifest["documents"][0]["attempts"], 1)
        self.assertNotIn("repairs", engine.manifest["documents"][0])

    def test_the_repaired_source_is_what_gets_recompiled(self):
        records = [customer(1, "Halvorsen Freight")]
        marker = faithful_document(records).replace(
            r"\begin{document}", "\\begin{document}\n% REPAIRED\n")
        instance = graph(*records)
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        engine = scripted(outcomes=["Bad", True],
                          responses=[faithful_document(records), marker],
                          records=records)
        engine.render_documents(instance, Path(tmp), layout_style="form")
        self.assertEqual(len(engine.compiled_sources), 2)
        self.assertNotIn("% REPAIRED", engine.compiled_sources[0])
        self.assertIn("% REPAIRED", engine.compiled_sources[1])

    def test_repair_prompt_receives_the_error_log(self):
        records = [customer(1, "Halvorsen Freight")]
        instance = graph(*records)
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        engine = scripted(outcomes=["Missing $ inserted.", True],
                          records=records)
        engine.render_documents(instance, Path(tmp), layout_style="form")
        repair_call = [c for c in engine.generator.client.calls
                       if c[0] == LATEX_REPAIR_SYSTEM_PROMPT]
        self.assertEqual(len(repair_call), 1)
        self.assertIn("Missing $ inserted.", repair_call[0][1])
        self.assertIn("l.42 the offending line", repair_call[0][1])

    def test_an_unchanged_repair_abandons_the_document(self):
        """Recompiling identical input would only burn another pass."""
        records = [customer(1, "Halvorsen Freight")]
        instance = graph(*records)
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        engine = scripted(outcomes=["Bad", "Bad", "Bad"],
                          responses=[faithful_document(records), None, None],
                          records=records)
        pdfs = engine.render_documents(instance, Path(tmp),
                                       layout_style="form", max_retries=2)
        self.assertEqual(pdfs, [])
        entry = engine.manifest["documents"][0]
        self.assertEqual(entry["attempts"], 1)
        self.assertFalse(entry["repairs"][0]["changed"])

    def test_generation_failure_is_recorded_without_compiling(self):
        records = [customer(1, "Halvorsen Freight")]
        instance = graph(*records)
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        engine = scripted(outcomes=[True],
                          responses=["not latex", "still not", "nope"],
                          records=records)
        pdfs = engine.render_documents(instance, Path(tmp), layout_style="form")
        entry = engine.manifest["documents"][0]
        self.assertEqual(pdfs, [])
        self.assertEqual(entry["status"], "failed")
        self.assertEqual(entry["attempts"], 0)
        self.assertEqual(engine.compiled_sources, [])
        self.assertIn("no usable LaTeX", entry["failure"])

    def test_one_failure_does_not_stop_the_corpus(self):
        records = [customer(1, "A"), customer(2, "B")]
        instance = graph(*records)
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        engine = scripted(outcomes=["Bad", "Bad", "Bad", True], records=records)
        pdfs = engine.render_documents(instance, Path(tmp),
                                       layout_style="form", max_retries=2)
        statuses = [d["status"] for d in engine.manifest["documents"]]
        self.assertEqual(statuses, ["failed", "compiled"])
        self.assertEqual(len(pdfs), 1)

    def test_negative_retries_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                scripted().render_documents(standard_graph(), Path(tmp),
                                            max_retries=-1)

    def test_intermediates_do_not_survive_a_failed_run(self):
        manifest, out, _, _ = self.render(["Bad", "Bad", "Bad"])
        names = sorted(p.name for p in out.iterdir())
        self.assertNotIn(".aux", {Path(n).suffix for n in names})
        # The failing source and its log are kept: the diagnostic is useless
        # without them.
        self.assertTrue(any(n.endswith(".tex") for n in names), names)
        self.assertTrue(any(n.endswith(".log") for n in names), names)


# --------------------------------------------------------------------------- #
# 7. Document scoping
# --------------------------------------------------------------------------- #
class TestDocumentScopes(unittest.TestCase):

    def test_parent_and_children_form_one_scope(self):
        scopes = document_scopes(standard_graph())
        self.assertEqual([(s.root.id, [c.id for c in s.children])
                          for s in scopes],
                         [("customer-001", ["order-001", "order-002"]),
                          ("customer-002", ["order-003"]),
                          ("order-004", [])])

    def test_every_record_appears_in_exactly_one_scope(self):
        instance = standard_graph()
        placed = [rid for s in document_scopes(instance) for rid in s.record_ids]
        self.assertEqual(sorted(placed), sorted(r.id for r in instance.records))
        self.assertEqual(len(placed), len(set(placed)))

    def test_orphaned_child_becomes_its_own_document(self):
        scopes = document_scopes(standard_graph())
        orphan = [s for s in scopes if s.root.id == "order-004"]
        self.assertEqual(len(orphan), 1)
        self.assertEqual(orphan[0].children, [])

    def test_joins_are_recorded_as_tuples(self):
        scope = document_scopes(standard_graph())[0]
        self.assertEqual(scope.joins,
                         [("order-001", "customer_id", "customer-001"),
                          ("order-002", "customer_id", "customer-001")])

    def test_three_level_chain_collapses_into_one_scope(self):
        instance = graph(
            Record(id="clinic-001", entity_name="Clinic",
                   attributes={"id": "clinic-001", "clinic_name": "Ridgeway"}),
            Record(id="patient-001", entity_name="Patient",
                   attributes={"id": "patient-001", "dob": "1980-01-01"},
                   foreign_keys={"clinic_id": "clinic-001"}),
            Record(id="visit-001", entity_name="Visit",
                   attributes={"id": "visit-001", "fee": "$40.00"},
                   foreign_keys={"patient_id": "patient-001"}))
        scopes = document_scopes(instance)
        self.assertEqual(len(scopes), 1)
        self.assertEqual(scopes[0].record_ids,
                         ["clinic-001", "patient-001", "visit-001"])
        self.assertEqual(len(scopes[0].joins), 2)

    def test_children_group_by_entity(self):
        scope = document_scopes(standard_graph())[0]
        self.assertEqual([(n, [r.id for r in rs]) for n, rs in scope.groups()],
                         [("Order", ["order-001", "order-002"])])

    def test_self_referencing_key_does_not_swallow_its_own_record(self):
        instance = graph(Record(id="node-001", entity_name="Node",
                                attributes={"id": "node-001", "label": "x"},
                                foreign_keys={"parent_id": "node-001"}))
        scopes = document_scopes(instance)
        self.assertEqual(len(scopes), 1)
        self.assertEqual(scopes[0].record_ids, ["node-001"])


# --------------------------------------------------------------------------- #
# 8. Layout selection
# --------------------------------------------------------------------------- #
class TestLayoutSelection(unittest.TestCase):

    def test_explicit_style_is_used_everywhere(self):
        engine = scripted()
        for style in CONCRETE_LAYOUTS:
            for scope in document_scopes(standard_graph()):
                self.assertEqual(engine._select_layout(scope, style), style)

    def test_auto_mixes_layouts_across_a_corpus(self):
        engine = scripted()
        chosen = {engine._select_layout(s, "auto")
                  for s in document_scopes(standard_graph())}
        self.assertGreater(len(chosen), 1, "auto rendered everything the same")

    def test_auto_never_tables_a_childless_scope(self):
        engine = scripted()
        for scope in document_scopes(graph(customer(1, "A"), customer(2, "B"))):
            self.assertIn(engine._select_layout(scope, "auto"),
                          ("form", "letter"))

    def test_layout_reaches_the_prompt(self):
        records = [customer(1, "A")]
        instance = graph(*records)
        with tempfile.TemporaryDirectory() as tmp:
            engine = scripted(records=records)
            engine.render_documents(instance, Path(tmp), layout_style="letter")
        self.assertIn("formal letter", engine.generator.client.calls[0][1])

    def test_unknown_style_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                scripted().render_documents(standard_graph(), Path(tmp),
                                            layout_style="origami")

    def test_style_list_matches_the_layouts(self):
        self.assertEqual(set(LAYOUT_STYLES), {"auto"} | set(CONCRETE_LAYOUTS))


# --------------------------------------------------------------------------- #
# 9. Real compilation
# --------------------------------------------------------------------------- #
@needs_tex
class TestRealCompilation(unittest.TestCase):

    def render(self, responses, records=None, **kwargs):
        records = records or [customer(1, "Halvorsen Freight"),
                              order(1, "customer-001", "$1,240.50")]
        instance = graph(*records)
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        engine = LaTeXRenderer(
            schema=schema(), seed=1,
            generator=LLMLaTeXGenerator(client=FakeLatexClient(*responses),
                                        schema=schema(), seed=1),
            **kwargs)
        pdfs = engine.render_documents(instance, Path(tmp),
                                      layout_style="form")
        return engine, Path(tmp), pdfs

    def test_a_good_document_compiles(self):
        records = [customer(1, "Halvorsen Freight"),
                   order(1, "customer-001", "$1,240.50")]
        engine, out, pdfs = self.render([faithful_document(records)], records)
        self.assertEqual(len(pdfs), 1, engine.warnings)
        self.assertEqual(pdfs[0].read_bytes()[:5], b"%PDF-")
        self.assertGreater(pdfs[0].stat().st_size, 1000)
        self.assertEqual(engine.manifest["documents"][0]["attempts"], 1)

    def test_unescaped_character_is_repaired_and_compiles(self):
        """The real loop: pdflatex fails, the model fixes it, the PDF appears."""
        records = [customer(1, "Halvorsen Freight"),
                   order(1, "customer-001", "$1,240.50")]
        engine, out, pdfs = self.render(
            [unescaped_document(records), faithful_document(records)], records)
        self.assertEqual(len(pdfs), 1, engine.warnings)
        entry = engine.manifest["documents"][0]
        self.assertEqual(entry["attempts"], 2)
        self.assertEqual(len(entry["repairs"]), 1)
        self.assertTrue(entry["repairs"][0]["changed"])

    def test_a_document_that_cannot_be_fixed_is_reported(self):
        records = [customer(1, "Halvorsen Freight")]
        broken = faithful_document(records).replace(
            r"\begin{document}", "\\begin{document}\n\\notARealCommand\n")
        engine, out, pdfs = self.render([broken, broken, broken], records)
        self.assertEqual(pdfs, [])
        entry = engine.manifest["documents"][0]
        self.assertEqual(entry["status"], "failed")
        self.assertIn("notARealCommand", entry["failure"])
        names = sorted(p.name for p in out.iterdir())
        self.assertTrue(any(n.endswith(".log") for n in names), names)

    def test_values_survive_into_the_pdf(self):
        if not shutil.which("pdftotext"):
            self.skipTest("pdftotext not available to read the PDF back")
        records = [customer(1, "Bell & Co. 50% Ltd_#1"),
                   order(1, "customer-001", "$1,240.50")]
        engine, out, pdfs = self.render([faithful_document(records)], records)
        text = subprocess.run(["pdftotext", "-layout", str(pdfs[0]), "-"],
                              capture_output=True, text=True).stdout
        flat = " ".join(text.split())
        for value in ("Bell & Co. 50% Ltd_#1", "$1,240.50", "customer-001",
                      "order-001"):
            with self.subTest(value=value):
                self.assertIn(value, flat)

    def test_long_values_are_not_hyphenated(self):
        """A hyphen LaTeX inserts is a character that was never in the data."""
        if not shutil.which("pdftotext"):
            self.skipTest("pdftotext not available to read the PDF back")
        records = [Record(id="company-002", entity_name="Customer", attributes={
            "id": "company-002",
            "company_name": "Bake My Day Bakery",
            "billing_address": "456 Elm St, Othertown, NY 67890"})]
        engine, out, pdfs = self.render([faithful_document(records)], records)
        text = subprocess.run(["pdftotext", "-layout", str(pdfs[0]), "-"],
                              capture_output=True, text=True).stdout
        self.assertIn("456 Elm St, Othertown, NY 67890", " ".join(text.split()))

    def test_intermediate_files_are_purged(self):
        records = [customer(1, "A")]
        engine, out, _ = self.render([faithful_document(records)], records)
        suffixes = {p.suffix for p in out.iterdir()}
        self.assertEqual(suffixes, {".pdf", ".json"})

    def test_keep_tex_retains_the_source(self):
        records = [customer(1, "A")]
        engine, out, _ = self.render([faithful_document(records)], records,
                                     keep_tex=True)
        self.assertEqual(sorted({p.suffix for p in out.iterdir()}),
                         [".json", ".pdf", ".tex"])
        self.assertEqual(engine.manifest["documents"][0]["tex"],
                         "doc-001-customer_001.tex")

    def test_no_temporary_directories_are_left_behind(self):
        before = set(Path(tempfile.gettempdir()).glob("generator-tex-*"))
        records = [customer(1, "A")]
        self.render([faithful_document(records)], records)
        after = set(Path(tempfile.gettempdir()).glob("generator-tex-*"))
        self.assertEqual(after - before, set())

    def test_timeout_is_reported_as_a_failure(self):
        records = [customer(1, "A")]
        engine, out, pdfs = self.render(
            [faithful_document(records)] * 3, records, timeout=0.001)
        self.assertEqual(pdfs, [])
        self.assertTrue(any("timeout" in w for w in engine.warnings),
                        engine.warnings)


class TestMissingEngine(unittest.TestCase):
    """Behaviour on a machine with no TeX installed."""

    def test_tex_is_written_and_the_run_continues(self):
        records = [customer(1, "Halvorsen Freight")]
        instance = graph(*records)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            engine = LaTeXRenderer(
                schema=schema(), engine="definitely-not-a-real-binary", seed=1,
                generator=LLMLaTeXGenerator(client=CyclingLatexClient(records),
                                            schema=schema(), seed=1))
            self.assertIsNone(engine.resolve_engine())
            pdfs = engine.render_documents(instance, out)
            names = sorted(p.name for p in out.iterdir())
            manifest = json.loads((out / MANIFEST_FILENAME).read_text())
        self.assertEqual(pdfs, [])
        self.assertEqual([n for n in names if n.endswith(".tex")],
                         ["doc-001-customer_001.tex"])
        self.assertEqual(manifest["documents"][0]["status"], "not_compiled")
        self.assertTrue(any("no LaTeX engine" in w for w in engine.warnings))
        # The ground truth is still complete, so a later run can compile it.
        self.assertEqual(manifest["summary"]["records_covered"], 1)
        self.assertEqual(manifest["documents"][0]["fidelity"]["values_missing"],
                         0)


# --------------------------------------------------------------------------- #
# 10. Manifest accuracy
# --------------------------------------------------------------------------- #
class TestManifest(unittest.TestCase):

    def build(self, instance=None, **kwargs):
        instance = instance or standard_graph()
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        engine = scripted(records=list(instance.records), **kwargs)
        engine.render_documents(instance, Path(tmp))
        with open(Path(tmp) / MANIFEST_FILENAME, encoding="utf-8") as handle:
            return json.load(handle), Path(tmp), engine

    def test_manifest_is_written_and_names_the_domain(self):
        manifest, out, _ = self.build()
        self.assertTrue((out / MANIFEST_FILENAME).is_file())
        self.assertEqual(manifest["schema_domain"], "small_business")
        self.assertEqual(len(manifest["documents"]), 3)

    def test_every_record_is_claimed_by_exactly_one_document(self):
        manifest, _, _ = self.build()
        claimed = [rid for d in manifest["documents"] for rid in d["record_ids"]]
        self.assertEqual(sorted(claimed),
                         sorted(r.id for r in standard_graph().records))
        self.assertEqual(len(claimed), len(set(claimed)))
        self.assertEqual(manifest["summary"]["records_covered"],
                         manifest["summary"]["records_total"])

    def test_ground_truth_values_are_raw_not_escaped(self):
        instance = graph(customer(1, "Bell & Co. 50%"),
                         order(1, "customer-001", "$8.00"))
        manifest, _, _ = self.build(instance)
        attributes = manifest["documents"][0]["records"][0]["attributes"]
        self.assertEqual(attributes["company_name"], "Bell & Co. 50%")
        self.assertNotIn("\\", json.dumps(attributes))

    def test_nulls_are_recorded_as_null(self):
        manifest, _, _ = self.build()
        record = [r for d in manifest["documents"]
                  for r in d["records"] if r["id"] == "customer-001"][0]
        self.assertIsNone(record["attributes"]["contact_email"])

    def test_joins_are_full_tuples_and_resolve_inside_the_document(self):
        manifest, _, _ = self.build()
        for document in manifest["documents"]:
            ids = set(document["record_ids"])
            for child, column, parent in document["joins"]:
                self.assertIn(child, ids)
                self.assertIn(parent, ids)
                record = [r for r in document["records"] if r["id"] == child][0]
                self.assertEqual(record["foreign_keys"][column], parent)
        self.assertEqual(sum(len(d["joins"]) for d in manifest["documents"]), 3)
        self.assertEqual(manifest["summary"]["joins"], 3)

    def test_orphaned_keys_are_flagged_and_not_counted_as_joins(self):
        manifest, _, _ = self.build()
        orphan_doc = [d for d in manifest["documents"]
                      if d["root_record"] == "order-004"][0]
        self.assertEqual(orphan_doc["joins"], [])
        self.assertEqual(orphan_doc["orphaned_foreign_keys"],
                         [{"record": "order-004", "column": "customer_id",
                           "value": "customer-__orphan_4__"}])
        self.assertEqual(manifest["summary"]["orphaned_foreign_keys"], 1)

    def test_each_entry_points_at_a_file_that_exists(self):
        manifest, out, _ = self.build()
        for document in manifest["documents"]:
            if document["status"] == "compiled":
                self.assertTrue((out / document["pdf"]).is_file())
            else:
                self.assertIsNone(document["pdf"])

    def test_metadata_records_the_run(self):
        manifest, _, _ = self.build(keep_tex=True)
        meta = manifest["metadata"]
        self.assertEqual(meta["stage"], "5:rendered_documents")
        self.assertEqual(meta["source"], "llm_generated_latex")
        self.assertEqual(meta["layout_style"], "auto")
        self.assertEqual(meta["max_retries"], 2)
        self.assertTrue(meta["keep_tex"])
        self.assertEqual(meta["model"], "llama3.3:70b")
        self.assertEqual(sum(meta["layouts"].values()), 3)

    def test_document_ids_are_unique(self):
        manifest, _, _ = self.build()
        ids = [d["document_id"] for d in manifest["documents"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_summary_counts_add_up(self):
        manifest, _, _ = self.build()
        info = manifest["summary"]
        self.assertEqual(info["documents"], info["compiled"] + info["failed"])
        self.assertEqual(info["compilations"],
                         sum(d.get("attempts", 0)
                             for d in manifest["documents"]))
        self.assertEqual(info["documents_value_complete"], info["compiled"])

    def test_stderr_summary_mentions_fidelity(self):
        _, _, engine = self.build()
        summary = engine.summary()
        self.assertIn("value-complete", summary)
        self.assertIn("compiled=3", summary)


# --------------------------------------------------------------------------- #
# 11. Log parsing
# --------------------------------------------------------------------------- #
class TestLogErrors(unittest.TestCase):

    def test_error_lines_are_extracted_with_their_context(self):
        log = ("This is pdfTeX\n"
               "! Undefined control sequence.\n"
               "l.12 \\notARealCommand\n"
               "                     \n")
        self.assertEqual(log_errors(log),
                         "! Undefined control sequence. l.12 \\notARealCommand")

    def test_no_errors_is_none(self):
        self.assertIsNone(log_errors("Output written on doc.pdf (1 page)."))
        self.assertIsNone(log_errors(""))

    def test_many_errors_are_capped_and_counted(self):
        log = "\n".join(f"! Error {i}." for i in range(6))
        parsed = log_errors(log)
        self.assertIn("+3 more", parsed)


# --------------------------------------------------------------------------- #
# 12. End to end through the CLI
# --------------------------------------------------------------------------- #
class DualModeClient:
    """One stand-in for both clients the CLI builds.

    Stages 1-4 call ``complete_json``; Stage 5 calls ``complete``. Keeping them
    on one object lets a single stub cover a whole pipeline run.
    """

    def __init__(self, *json_payloads, latex_records=()):
        self.json_payloads = list(json_payloads)
        self.latex_records = list(latex_records)
        self.last_error = None
        self.model, self.backend = "llama3.3:70b", "ollama"
        self.base_url, self.temperature = "http://127.0.0.1:11434", 0.0
        self.latex_calls = 0

    def complete_json(self, system, user):
        if not self.json_payloads:
            self.last_error = "no output"
            return None
        return self.json_payloads.pop(0)

    def complete(self, system, user):
        self.latex_calls += 1
        # Echo back exactly what the prompt described. Reading the values out of
        # the prompt rather than hard-coding them keeps this fake faithful to
        # whatever Stages 3-4 actually produced, which is what makes the
        # fidelity check meaningful here instead of vacuous.
        body = []
        for identifier in re.findall(r"\[identifier: ([^\],]+)", user):
            body.append(r"\section*{%s}" % escape_latex(identifier.strip()))
        for label, value in re.findall(r"^  (.+?) = (.*)$", user, re.MULTILINE):
            if value.startswith("(blank"):
                body.append(r"%s: %s\\" % (escape_latex(label), NULL_GLYPH))
            else:
                body.append(r"%s: %s\\" % (escape_latex(label),
                                             escape_latex(value)))
        return ("\\documentclass{article}\n\\begin{document}\n"
                + "\n".join(body) + "\n\\end{document}\n")


CUSTOMER_ROWS = {"records": [
    {"company_name": "Halvorsen Freight", "billing_address": "42 Quay Road",
     "contact_email": "h@example.invalid"},
    {"company_name": "Kestrel Supplies", "billing_address": "9 Mill Bank",
     "contact_email": "k@example.invalid"}]}
ORDER_ROWS = {"records": [
    {"order_date": "2022-01-15", "total": "$1,240.50", "status": "paid"},
    {"order_date": "2022-02-20", "total": "$98.00", "status": "draft"}]}


class TestCLI(unittest.TestCase):

    def setUp(self):
        self.runner = make_runner()
        self.original = cli_module.build_client

    def tearDown(self):
        cli_module.build_client = self.original

    def run_generate(self, *extra):
        expected = [
            customer(1, "Halvorsen Freight", "h@example.invalid"),
            customer(2, "Kestrel Supplies", "k@example.invalid"),
            order(1, "customer-001", "$1,240.50"),
            order(2, "customer-002", "$98.00")]
        client = DualModeClient(SCHEMA_PAYLOAD, CUSTOMER_ROWS, ORDER_ROWS,
                                latex_records=expected)
        cli_module.build_client = lambda **kw: client
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        out = Path(tmp) / "bench"
        result = self.runner.invoke(cli_module.app, [
            "generate", "--num-entities", "2", "--records-per-entity", "2",
            "--seed", "4", "--output-dir", str(out), *extra])
        return result, out, client

    @needs_tex
    def test_generate_produces_pdfs_and_a_manifest(self):
        result, out, client = self.run_generate()
        self.assertEqual(result.exit_code, 0, stderr_of(result))
        manifest = json.loads((out / MANIFEST_FILENAME).read_text())
        pdfs = sorted(p.name for p in out.glob("*.pdf"))
        self.assertEqual(len(pdfs), manifest["summary"]["compiled"])
        self.assertTrue(pdfs)
        self.assertEqual(manifest["metadata"]["source"], "llm_generated_latex")
        self.assertEqual(manifest["summary"]["records_covered"], 4)
        lines = result.stdout.split()
        self.assertEqual(lines[0], str(out / "schema.json"))
        self.assertEqual(lines[1], str(out / "instances.json"))
        self.assertEqual(lines[2], str(out / MANIFEST_FILENAME))
        self.assertEqual(len(lines), 3 + len(pdfs))

    @needs_tex
    def test_one_llm_call_per_document(self):
        result, out, client = self.run_generate()
        manifest = json.loads((out / MANIFEST_FILENAME).read_text())
        self.assertEqual(client.latex_calls, len(manifest["documents"]))

    @needs_tex
    def test_max_retries_reaches_the_manifest(self):
        result, out, _ = self.run_generate("--max-retries", "5")
        self.assertEqual(result.exit_code, 0, stderr_of(result))
        manifest = json.loads((out / MANIFEST_FILENAME).read_text())
        self.assertEqual(manifest["metadata"]["max_retries"], 5)

    @needs_tex
    def test_layout_style_flag_is_honoured(self):
        result, out, _ = self.run_generate("--layout-style", "letter")
        self.assertEqual(result.exit_code, 0, stderr_of(result))
        manifest = json.loads((out / MANIFEST_FILENAME).read_text())
        self.assertEqual(manifest["metadata"]["layout_style"], "letter")
        self.assertEqual(list(manifest["metadata"]["layouts"]), ["letter"])

    @needs_tex
    def test_keep_tex_flag_retains_sources(self):
        result, out, _ = self.run_generate("--keep-tex")
        self.assertEqual(result.exit_code, 0, stderr_of(result))
        self.assertTrue(sorted(out.glob("*.tex")))

    @needs_tex
    def test_default_purges_intermediates(self):
        result, out, _ = self.run_generate()
        self.assertEqual(result.exit_code, 0, stderr_of(result))
        for pattern in ("*.tex", "*.aux", "*.log"):
            self.assertEqual(sorted(out.glob(pattern)), [], pattern)

    @needs_tex
    def test_summary_reaches_stderr(self):
        result, _, _ = self.run_generate()
        err = stderr_of(result)
        self.assertIn("stage 5: documents", err)
        self.assertIn("compiled=", err)
        self.assertIn("value-complete", err)
        self.assertIn(MANIFEST_FILENAME, err)

    def test_no_render_stops_after_stage_four(self):
        result, out, client = self.run_generate("--no-render")
        self.assertEqual(result.exit_code, 0, stderr_of(result))
        self.assertTrue((out / "instances.json").is_file())
        self.assertFalse((out / MANIFEST_FILENAME).exists())
        self.assertEqual(client.latex_calls, 0)

    def test_unknown_layout_style_is_rejected(self):
        result, _, _ = self.run_generate("--layout-style", "origami")
        self.assertEqual(result.exit_code, 2)
        self.assertIn("origami", stderr_of(result))

    def test_negative_retries_is_rejected(self):
        result, _, _ = self.run_generate("--max-retries", "-1")
        self.assertEqual(result.exit_code, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
