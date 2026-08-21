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
    ALLOWED_PACKAGES, AUTO_LAYOUT, DESIGN_AXES, DESIGN_DEVICES,
    HYPHENATION_GUARD, LATEX_GENERATION_SYSTEM_PROMPT,
    LATEX_REPAIR_SYSTEM_PROMPT, LATEX_RESTORE_SYSTEM_PROMPT,
    LAYOUT_DECLARATION_PREFIX, MAX_DECLARATION_LENGTH, NULL_GLYPH,
    LatexText, LaTeXGenerationError, LLMLaTeXGenerator, escape_latex,
    extract_latex, harden_source, humanize, is_auto, layout_declaration,
    layout_directive, leaked_examples, missing_values,
    normalize_for_comparison, normalize_layout_hint, PROMPT_EXAMPLE_VALUES,
    recorded_values, strip_comments, unescape_latex)
from src.generator.renderer import (AUTO_LAYOUT as RENDERER_AUTO,  # noqa: E402
                                    MANIFEST_FILENAME,
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


#: Layout hints a caller might actually type. Not one of them is a value the
#: generator knows about, which is the whole point: there is no list to be off.
FREEFORM_HINTS = (
    "1990s technical spec sheet with dense grid lines",
    "a hand-annotated field inspection sheet, carbon copy",
    "wide-margin municipal notice, two columns, heavy rules",
    "terse internal memo on headed paper",
)

#: What a page declares itself to be. Stands in for a sentence the model wrote.
DECLARATION = "A ruled dispatch note with a shaded status panel."


def faithful_document(records, preamble: str = "",
                      declaration: str = DECLARATION) -> str:
    """A compilable document carrying every recorded value, properly escaped.

    Stands in for a well-behaved model: what the real one is asked for, minus
    the layout flair, so fidelity and compilation can both be checked against a
    known-good baseline. It carries a ``% LAYOUT:`` line because a real response
    is asked for one; pass ``declaration=""`` for a page that did not bother.
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
    declared = f"{LAYOUT_DECLARATION_PREFIX} {declaration}\n" if declaration \
        else ""
    return ("\\documentclass{article}\n" + declared +
            "\\usepackage[T1]{fontenc}\n" + preamble +
            "\\begin{document}\n" + "\n".join(body) +
            "\n\\end{document}\n")


def invented_document(records, declaration: str = DECLARATION) -> str:
    """A page of the kind the dynamic prompt actually asks for.

    Nothing about its shape is drawn from a known layout: a shaded panel, a
    two-column body, a ruled grid of line items and a run of prose, all in one
    document, with the values threaded through wherever they happen to land.
    The fidelity checks have to hold against *this* as firmly as against a
    tidy list of labelled fields, because this is what an invented layout looks
    like and there is no longer a tidy list to fall back on.
    """
    root, children = records[0], records[1:]
    panel = "\n".join(
        r"\textsc{%s} & %s\\" % (escape_latex(humanize(name)),
                                escape_latex(value))
        for name, value in root.attributes.items())
    rows = []
    for child in children:
        cells = " & ".join(escape_latex(v) for v in child.attributes.values())
        rows.append(cells + r" \\")
    columns = max(1, max((len(c.attributes) for c in children), default=1))
    grid = ("\\begin{tabular}{|" + "l|" * columns + "}\n\\hline\n"
            + "\n\\hline\n".join(rows) + "\n\\hline\n\\end{tabular}\n"
            ) if rows else ""
    prose = " ".join(
        "Logged %s as %s." % (humanize(name), escape_latex(value))
        for name, value in root.attributes.items() if value is not None)
    declared = f"{LAYOUT_DECLARATION_PREFIX} {declaration}\n" if declaration \
        else ""
    return ("\\documentclass{article}\n" + declared +
            "\\usepackage[T1]{fontenc}\n"
            "\\usepackage{multicol}\n\\usepackage{xcolor}\n"
            "\\usepackage{geometry}\n"
            "\\begin{document}\n"
            "\\colorbox{gray!15}{\\parbox{0.9\\linewidth}{\\Large "
            + escape_latex(root.id) + "}}\n\n"
            "\\begin{tabular}{ll}\n" + panel + "\n\\end{tabular}\n\n"
            + grid + "\n"
            "\\begin{multicols}{2}\n" + prose + "\n\\end{multicols}\n"
            + "\n".join(r"\fbox{%s}" % escape_latex(c.id) for c in children)
            + "\n\\end{document}\n")


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
        engine.render_documents(instance, Path(tmp), layout_hint=AUTO_LAYOUT)
        return engine.manifest, engine

    # -- the prompt itself ---------------------------------------------- #
    def sample_directives(self):
        """A directive per hint and per rotation, since all of them vary now."""
        for hint in (AUTO_LAYOUT,) + FREEFORM_HINTS:
            for variation in range(len(DESIGN_DEVICES)):
                yield hint, variation, layout_directive(hint,
                                                        variation=variation)

    def test_no_layout_directive_hands_over_a_ready_made_value(self):
        """A complete example sentence with a value in it invites a copy.

        Wider than it was: the directive is now assembled per document out of a
        rotating device list and a rotating axis, so "the prompt" is a family of
        prompts and every member of it has to be clean.
        """
        for hint, variation, directive in self.sample_directives():
            for example in PROMPT_EXAMPLE_VALUES:
                with self.subTest(hint=hint, variation=variation,
                                  example=example):
                    self.assertNotIn(example, directive)

    def test_the_device_list_names_no_facts(self):
        """A device is a shape. A shape cannot be copied onto a page as data.

        This is what makes it safe to widen the list: three named layouts became
        twelve named devices, and the leak the corpus actually suffered came
        from a *value* in an example, not from the number of examples.
        """
        for device in DESIGN_DEVICES + DESIGN_AXES:
            with self.subTest(device=device):
                self.assertFalse(re.search(r"\d", device),
                                 "a number in a device reads as data")
                self.assertNotIn("$", device)

    def test_generation_prompt_forbids_reusing_its_own_examples(self):
        prompt = LATEX_GENERATION_SYSTEM_PROMPT.lower()
        self.assertIn("show form only", prompt)
        self.assertIn("they are not data", prompt)

    def test_no_layout_directive_offers_a_form_to_fill_in(self):
        """A fill-in-the-blank sentence gets filled in literally.

        The first attempt at fixing the leak replaced the worked example with a
        placeholder template, "The <field label> on file is <the exact value>."
        llama3.1:8b wrote the angle brackets onto the page HTML-escaped
        (``\\&lt;2021-02-15\\&gt;``) and filled the slots with the wrong
        values. A template is an instruction to copy; the wording has to
        describe the requirement instead. The same holds for the invention
        directive, which describes a *task* rather than showing a page.
        """
        for hint, variation, directive in self.sample_directives():
            with self.subTest(hint=hint, variation=variation):
                self.assertNotIn("<", directive)

    def test_the_declaration_instruction_is_not_a_slot_to_fill(self):
        """"% LAYOUT: ..." with a bracketed placeholder would be pasted back."""
        self.assertIn(LAYOUT_DECLARATION_PREFIX,
                      LATEX_GENERATION_SYSTEM_PROMPT)
        self.assertIn("not a slot to fill in", LATEX_GENERATION_SYSTEM_PROMPT)

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

    DOC = "\\documentclass{article}\n\\begin{document}\nx\n\\end{document}"

    def records(self):
        return [customer(1, "Halvorsen Freight"),
                order(1, "customer-001", "$1,240.50")]

    def test_no_prompt_carries_a_stray_line_continuation(self):
        """A soft-wrapped raw string must not leak its own backslashes.

        The prompts are raw strings, because they are largely LaTeX and every
        backslash in them has to reach the model intact. A raw string does not
        honour backslash-newline as a line continuation, so a prompt wrapped for
        readability and used as written keeps a literal "\\" mid-sentence -- in
        a prompt whose whole subject is LaTeX, where a stray backslash reads as
        the start of a command. It also breaks every sentence in two, which is
        why the fidelity rules could not be matched on. _unfold is what joins
        them; this is the guard that it ran.
        """
        for name, prompt in (
                ("generation", LATEX_GENERATION_SYSTEM_PROMPT),
                ("repair", LATEX_REPAIR_SYSTEM_PROMPT),
                ("restore", LATEX_RESTORE_SYSTEM_PROMPT)):
            with self.subTest(prompt=name):
                folded = [line for line in prompt.split("\n")
                          if re.search(r"(?<!\\)\\$", line)]
                self.assertEqual(folded, [])
                # Whole sentences, not fragments ending in a wrap.
                self.assertNotIn("\\\n", prompt)

    def test_unfolding_keeps_the_latex_the_prompt_is_teaching(self):
        """The fix must not cost the backslashes the raw string was for."""
        prompt = LATEX_GENERATION_SYSTEM_PROMPT
        for literal in (r"\documentclass", r"\&", r"\%", r"\_",
                        r"\textbackslash{}", r"\toprule", r"\begin{letter}",
                        r"\\"):
            with self.subTest(literal=literal):
                self.assertIn(literal, prompt)

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

    def test_auto_asks_the_model_to_invent_rather_than_choose(self):
        directive = layout_directive(AUTO_LAYOUT)
        self.assertIn("invent", directive.lower())
        self.assertIn("not a menu", directive)
        # The three shapes the corpus used to have. None of them is named now,
        # because naming one is what made every corpus contain it.
        for gone in ("Layout: a tabular document", "Layout: a completed form",
                     "Layout: a formal letter"):
            self.assertNotIn(gone, directive)

    def test_a_freeform_hint_reaches_the_model_verbatim(self):
        for hint in FREEFORM_HINTS:
            with self.subTest(hint=hint):
                self.assertIn(hint, layout_directive(hint))

    def test_a_freeform_hint_is_marked_as_form_not_fact(self):
        """A brief mentioning "1990s" must not put 1990 on the page as data."""
        directive = layout_directive(FREEFORM_HINTS[0])
        self.assertIn("describes form only", directive)
        self.assertIn("unless a record below supplies it", directive)

    def test_a_multiline_hint_survives_as_one_quoted_block(self):
        hint = "spec sheet\n  dense grid lines\n\nno colour"
        directive = layout_directive(hint)
        for line in ("spec sheet", "dense grid lines", "no colour"):
            self.assertIn(f"    {line}", directive)

    def test_no_hint_is_rejected_however_odd(self):
        """There is no list to be off. That is the point of the refactor."""
        for hint in ("origami", "", "   ", None, "table", "!!", "x" * 500):
            with self.subTest(hint=hint):
                self.assertIsInstance(layout_directive(
                    normalize_layout_hint(hint)), str)
        generator(self.DOC).generate_latex_source(
            [customer(1, "A")], "origami", "d")

    def test_a_blank_hint_means_auto(self):
        for blank in (None, "", "   ", "\n"):
            with self.subTest(blank=blank):
                self.assertEqual(normalize_layout_hint(blank), AUTO_LAYOUT)
                self.assertTrue(is_auto(blank))
        self.assertFalse(is_auto(FREEFORM_HINTS[0]))
        self.assertTrue(is_auto("AUTO"))

    def test_consecutive_documents_are_asked_for_different_pages(self):
        """--seed forces greedy decoding, so variety has to come from here.

        An identical prompt returns an identical page. If every document in a
        run were sent the same directive, a seeded corpus would be one layout
        repeated however many times, which is the failure the fixed enum had.
        """
        directives = [layout_directive(AUTO_LAYOUT, variation=i)
                      for i in range(len(DESIGN_DEVICES))]
        self.assertEqual(len(set(directives)), len(directives))

    def test_the_rotation_is_stable_for_a_given_document(self):
        """Same document index, same wording: a rerun reproduces the corpus."""
        self.assertEqual(layout_directive(AUTO_LAYOUT, variation=3),
                         layout_directive(AUTO_LAYOUT, variation=3))

    def test_the_rotation_wraps_rather_than_running_out(self):
        wide = layout_directive(AUTO_LAYOUT, variation=len(DESIGN_DEVICES) * 4)
        self.assertEqual(wide, layout_directive(AUTO_LAYOUT, variation=0))
        for device in DESIGN_DEVICES:
            self.assertIn(device, wide)

    def test_user_prompt_carries_domain_layout_and_records(self):
        prompt = generator().build_user_prompt(self.records(),
                                               FREEFORM_HINTS[0],
                                               "small_business")
        self.assertIn("Domain: small_business", prompt)
        self.assertIn(FREEFORM_HINTS[0], prompt)
        self.assertIn("Halvorsen Freight", prompt)
        self.assertIn("2 in total", prompt)

    def test_the_generators_own_hint_is_the_default_for_every_document(self):
        gen = LLMLaTeXGenerator(client=FakeLatexClient(self.DOC),
                                schema=schema(),
                                layout_hint=FREEFORM_HINTS[1])
        self.assertIn(FREEFORM_HINTS[1],
                      gen.build_user_prompt(self.records()))
        self.assertIn(FREEFORM_HINTS[2],
                      gen.build_user_prompt(self.records(), FREEFORM_HINTS[2]))

    def test_the_directive_sent_is_kept_for_the_manifest(self):
        gen = generator(self.DOC)
        gen.generate_latex_source([customer(1, "A")], FREEFORM_HINTS[0], "d",
                                  variation=2)
        self.assertEqual(gen.last_layout_directive,
                         layout_directive(FREEFORM_HINTS[0], variation=2))
        self.assertIn(gen.last_layout_directive, gen.client.calls[0][1])

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

    def test_fidelity_is_stated_as_surviving_the_invented_layout(self):
        """The design is free; what is on the page is not.

        A model told to invent a page will invent one that suits the space it
        has, and the cheapest way to make a design fit is to leave a field out.
        The rule has to say so in the same breath as the licence to design.
        """
        prompt = LATEX_GENERATION_SYSTEM_PROMPT
        self.assertIn("100% of what you were given has to be on it", prompt)
        self.assertIn("EVERY record you are given appears on the page", prompt)
        self.assertIn("the layout is wrong", prompt)
        self.assertIn("A linked child record does not lose fields", prompt)

    def test_forbidden_inventions_are_named_not_implied(self):
        prompt = LATEX_GENERATION_SYSTEM_PROMPT
        for invention in ("totals", "reference numbers", "signatures",
                          "legal wording"):
            with self.subTest(invention=invention):
                self.assertIn(invention, prompt)
        self.assertIn("Headings, captions, column titles", prompt)

    def test_the_prompt_licenses_a_design_rather_than_a_template(self):
        prompt = LATEX_GENERATION_SYSTEM_PROMPT
        self.assertIn("You are not filling in a template", prompt)
        self.assertIn("document designer", prompt)

    def test_the_page_is_asked_to_declare_its_own_layout(self):
        self.assertIn(LAYOUT_DECLARATION_PREFIX,
                      LATEX_GENERATION_SYSTEM_PROMPT)
        self.assertIn("straight after \\documentclass",
                      LATEX_GENERATION_SYSTEM_PROMPT)

    def test_a_comment_is_named_as_not_being_the_page(self):
        """Twice over: a value in a comment is a dropped value, not a kept one."""
        self.assertIn("not on the page", LATEX_GENERATION_SYSTEM_PROMPT)
        self.assertIn("comments are not", LATEX_RESTORE_SYSTEM_PROMPT.lower())

    def test_the_widened_package_set_is_the_one_offered(self):
        """A model told to shade a box needs the package that shades boxes."""
        for package in ("multicol", "xcolor", "colortbl", "ragged2e",
                        "setspace"):
            with self.subTest(package=package):
                self.assertIn(package, ALLOWED_PACKAGES)
                self.assertIn(package, LATEX_GENERATION_SYSTEM_PROMPT)
                self.assertIn(package, LATEX_REPAIR_SYSTEM_PROMPT)

    def test_repair_is_told_not_to_redesign_the_page(self):
        """The layout is the document's own; simplifying it is not a fix."""
        self.assertIn("Do not redesign the page", LATEX_REPAIR_SYSTEM_PROMPT)
        self.assertIn(LAYOUT_DECLARATION_PREFIX, LATEX_REPAIR_SYSTEM_PROMPT)

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
        engine.render_documents(instance, Path(tmp), layout_hint=AUTO_LAYOUT)
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
                                       layout_hint=AUTO_LAYOUT,
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
                                       layout_hint=AUTO_LAYOUT, max_retries=0)
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
        engine.render_documents(instance, Path(tmp), layout_hint=AUTO_LAYOUT)
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
        engine.render_documents(instance, Path(tmp), layout_hint=AUTO_LAYOUT)
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
                                       layout_hint=AUTO_LAYOUT, max_retries=2)
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
        pdfs = engine.render_documents(instance, Path(tmp), layout_hint=AUTO_LAYOUT)
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
                                       layout_hint=AUTO_LAYOUT, max_retries=2)
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
# 8. Open-ended layout hints
# --------------------------------------------------------------------------- #
class TestLayoutHints(unittest.TestCase):
    """There is no layout to select any more, only a hint to pass on."""

    def render(self, hint, records=None, **kwargs):
        records = records or [customer(1, "A")]
        instance = graph(*records)
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        engine = scripted(records=records, **kwargs)
        engine.render_documents(instance, Path(tmp), layout_hint=hint)
        return engine

    def test_a_freeform_hint_reaches_the_prompt_verbatim(self):
        for hint in FREEFORM_HINTS:
            with self.subTest(hint=hint):
                engine = self.render(hint)
                self.assertIn(hint, engine.generator.client.calls[0][1])

    def test_an_arbitrary_hint_is_not_rejected(self):
        """"origami" used to be the test for an invalid style. It is now valid."""
        engine = self.render("origami, folded, printed on one side")
        self.assertEqual(engine.manifest["documents"][0]["status"], "compiled")
        self.assertEqual(engine.manifest["metadata"]["layout_hint"],
                         "origami, folded, printed on one side")

    def test_a_blank_hint_is_recorded_as_auto(self):
        engine = self.render("   ")
        self.assertEqual(engine.manifest["metadata"]["layout_hint"],
                         AUTO_LAYOUT)
        self.assertEqual(engine.manifest["metadata"]["layout_mode"],
                         "invented")

    def test_a_brief_is_marked_as_a_brief_not_an_invention(self):
        engine = self.render(FREEFORM_HINTS[0])
        self.assertEqual(engine.manifest["metadata"]["layout_mode"], "brief")

    def test_a_hint_that_is_not_text_is_rejected(self):
        """The one mistake left: passing the old enum as a list or a dict."""
        with tempfile.TemporaryDirectory() as tmp:
            for bad in (["form"], {"style": "form"}, 3):
                with self.subTest(bad=bad):
                    with self.assertRaises(ValueError):
                        scripted().render_documents(standard_graph(), Path(tmp),
                                                    layout_hint=bad)

    def test_each_document_in_a_corpus_gets_a_different_directive(self):
        """Three scopes, three prompts. Under a seed this is the only variety."""
        instance = standard_graph()
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        engine = scripted(records=list(instance.records))
        engine.render_documents(instance, Path(tmp))
        prompts = [entry["layout_prompt"]
                   for entry in engine.manifest["documents"]]
        self.assertEqual(len(prompts), 3)
        self.assertEqual(len(set(prompts)), 3, "every document asked the same")
        for entry, prompt in zip(engine.manifest["documents"], prompts):
            sent = [u for _, u in engine.generator.client.calls if prompt in u]
            self.assertTrue(sent, f"{entry['document_id']} was sent something "
                                  f"other than the directive recorded for it")

    def test_the_variation_follows_the_document_not_the_call_order(self):
        """A rerun reproduces the corpus, document by document."""
        instance = standard_graph()
        first = scripted(records=list(instance.records))
        second = scripted(records=list(instance.records))
        for engine in (first, second):
            tmp = tempfile.mkdtemp()
            self.addCleanup(shutil.rmtree, tmp)
            engine.render_documents(instance, Path(tmp))
        self.assertEqual([e["layout_prompt"] for e in first.manifest["documents"]],
                         [e["layout_prompt"] for e in second.manifest["documents"]])


class TestLayoutDeclaration(unittest.TestCase):
    """Reading back the layout the model invented, since nothing else knows it."""

    def test_a_declaration_is_read_off_the_page(self):
        source = faithful_document([customer(1, "A")],
                                   declaration="A tinted two-column docket.")
        self.assertEqual(layout_declaration(source),
                         "A tinted two-column docket.")

    def test_it_is_found_however_the_model_spaced_it(self):
        for line in ("%LAYOUT:A docket.", "%%  layout :  A docket.",
                     "   % Layout:   A docket.   "):
            with self.subTest(line=line):
                self.assertEqual(
                    layout_declaration(f"\\documentclass{{article}}\n{line}\n"),
                    "A docket.")

    def test_a_page_that_never_declared_itself_is_not_an_error(self):
        source = faithful_document([customer(1, "A")], declaration="")
        self.assertIsNone(layout_declaration(source))
        self.assertIsNone(layout_declaration(""))
        self.assertIsNone(layout_declaration(None))

    def test_an_essay_is_trimmed_rather_than_stored_whole(self):
        long = "A docket " * 200
        source = f"\\documentclass{{article}}\n% LAYOUT: {long}\n"
        declared = layout_declaration(source)
        self.assertLessEqual(len(declared), MAX_DECLARATION_LENGTH + 3)
        self.assertTrue(declared.endswith("..."))

    def test_the_declaration_is_what_the_manifest_records(self):
        records = [customer(1, "Halvorsen Freight")]
        instance = graph(*records)
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        source = invented_document(records,
                                   declaration="A shaded dispatch panel.")
        engine = scripted(responses=[source, source])
        engine.render_documents(instance, Path(tmp))
        entry = engine.manifest["documents"][0]
        self.assertEqual(entry["layout_declared"], "A shaded dispatch panel.")
        self.assertEqual(entry["layout"], "A shaded dispatch panel.")
        self.assertEqual(entry["layout_hint"], AUTO_LAYOUT)

    def test_an_undeclared_page_falls_back_to_the_hint_and_says_so(self):
        records = [customer(1, "Halvorsen Freight")]
        instance = graph(*records)
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        source = faithful_document(records, declaration="")
        engine = scripted(responses=[source, source])
        engine.render_documents(instance, Path(tmp))
        entry = engine.manifest["documents"][0]
        self.assertIsNone(entry["layout_declared"])
        self.assertEqual(entry["layout"], AUTO_LAYOUT)
        self.assertTrue(any("did not declare its layout" in w
                            for w in engine.warnings))

    def test_a_failed_document_still_records_what_it_was_asked_to_be(self):
        records = [customer(1, "A")]
        instance = graph(*records)
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        engine = scripted(outcomes=[True],
                          responses=["not latex", "still not", "nope"],
                          records=records)
        engine.render_documents(instance, Path(tmp),
                                layout_hint=FREEFORM_HINTS[0])
        entry = engine.manifest["documents"][0]
        self.assertEqual(entry["status"], "failed")
        self.assertEqual(entry["layout_hint"], FREEFORM_HINTS[0])
        self.assertIn(FREEFORM_HINTS[0], entry["layout_prompt"])


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
                                      layout_hint=AUTO_LAYOUT)
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
        # What the run asked for, not a style it chose from: "auto" is the one
        # hint that is not a brief, and it means the layout was invented per
        # document. `layout_mode` is what tells the two apart.
        self.assertEqual(meta["layout_hint"], "auto")
        self.assertEqual(meta["layout_mode"], "invented")
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
# 10b. Several layouts of one subgraph (--layouts-per-graph)
# --------------------------------------------------------------------------- #
class PerScopeClient(FakeLatexClient):
    """A well-behaved model: answers with the records it was actually given.

    ``CyclingLatexClient`` echoes one fixed record list into every document,
    which is fine when every test renders the whole graph as one page but wrong
    here: a variant of subgraph 2 that printed subgraph 1's values would be a
    leak, and the leak check would be measuring the fake rather than the code.
    So this reads the identifiers and values back out of the prompt it was sent,
    which is also what makes it faithful *per variant*.
    """

    #: One declaration per call, so the variants of one subgraph do not all
    #: declare themselves identically and `layouts` stays a meaningful tally.
    def __init__(self):
        super().__init__()
        self.prompts = []

    def complete(self, system, user):
        self.calls.append((system, user))
        self.prompts.append(user)
        body = []
        for identifier in re.findall(r"\[identifier: ([^\],]+)", user):
            body.append(r"\section*{%s}" % escape_latex(identifier.strip()))
        for label, value in re.findall(r"^  (.+?) = (.*)$", user, re.MULTILINE):
            printed = NULL_GLYPH if value.startswith("(blank") \
                else escape_latex(value)
            body.append(r"%s: %s\\" % (escape_latex(label), printed))
        declaration = f"Variant {len(self.prompts)} of one subgraph."
        return ("\\documentclass{article}\n"
                f"{LAYOUT_DECLARATION_PREFIX} {declaration}\n"
                "\\usepackage[T1]{fontenc}\n"
                "\\begin{document}\n" + "\n".join(body) +
                "\n\\end{document}\n")


class TestLayoutsPerGraph(unittest.TestCase):
    """One relational subgraph, several pages, identical ground truth.

    The point of the flag is that the records are held fixed while the layout
    moves, so an extractor that recovers the tree from every variant has read
    the data and one that manages it for a single variant has learnt a shape.
    Every test here therefore checks the invariant *and* that the variants are
    actually different.
    """

    #: Two subgraphs: a customer with one order, and a childless customer.
    def two_subgraphs(self) -> InstanceGraph:
        return graph(customer(1, "Halvorsen Freight"),
                     order(1, "customer-001", "$1,240.50"),
                     customer(2, "Kestrel Supplies", "k@example.invalid"))

    def render(self, layouts_per_graph, instance=None, **kwargs):
        instance = instance or self.two_subgraphs()
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        client = PerScopeClient()
        engine = ScriptedRenderer(
            generator=LLMLaTeXGenerator(client=client, schema=schema(), seed=1),
            seed=1, **kwargs)
        pdfs = engine.render_documents(
            instance, Path(tmp), layouts_per_graph=layouts_per_graph)
        return engine.manifest, engine, client, pdfs, Path(tmp)

    # -- the count ---------------------------------------------------------- #
    def test_three_layouts_of_two_subgraphs_is_six_generations(self):
        """The headline arithmetic: 2 subgraphs x 3 layouts = 6 documents."""
        manifest, _, client, pdfs, _ = self.render(3)
        self.assertEqual(len(client.prompts), 6)
        self.assertEqual(len(manifest["documents"]), 6)
        self.assertEqual(len(pdfs), 6)
        self.assertEqual(manifest["summary"]["documents"], 6)
        self.assertEqual(manifest["summary"]["documents_expected"], 6)
        self.assertEqual(manifest["metadata"]["layouts_per_graph"], 3)
        self.assertEqual(manifest["metadata"]["subgraphs"], 2)

    def test_corpus_size_is_the_product_for_any_setting(self):
        """Predictable scaling is the other half of what the flag is for."""
        for per_graph in (1, 2, 4):
            with self.subTest(layouts_per_graph=per_graph):
                manifest, _, client, _, _ = self.render(per_graph)
                self.assertEqual(len(manifest["documents"]), 2 * per_graph)
                self.assertEqual(len(client.prompts), 2 * per_graph)

    def test_one_layout_renders_exactly_what_it_always_did(self):
        """The default has to be a no-op, filenames included."""
        manifest, _, _, pdfs, _ = self.render(1)
        self.assertEqual([e["document_id"] for e in manifest["documents"]],
                         ["doc-001-customer_001", "doc-002-customer_002"])
        self.assertEqual([p.name for p in pdfs],
                         ["doc-001-customer_001.pdf",
                          "doc-002-customer_002.pdf"])
        for entry in manifest["documents"]:
            with self.subTest(document=entry["document_id"]):
                self.assertEqual(entry["layout_variant_index"], 0)
                self.assertEqual(entry["layout_variants"], 1)

    # -- the manifest ------------------------------------------------------- #
    def test_manifest_logs_every_variant_with_its_subgraph_and_index(self):
        manifest, _, _, _, _ = self.render(3)
        seen = [(e["subgraph_id"], e["layout_variant_index"])
                for e in manifest["documents"]]
        self.assertEqual(seen, [("subgraph-001", 0), ("subgraph-001", 1),
                                ("subgraph-001", 2), ("subgraph-002", 0),
                                ("subgraph-002", 1), ("subgraph-002", 2)])
        for entry in manifest["documents"]:
            with self.subTest(document=entry["document_id"]):
                self.assertEqual(entry["layout_variants"], 3)
                self.assertEqual(entry["status"], "compiled")

    def test_document_ids_and_files_stay_unique_across_variants(self):
        manifest, _, _, pdfs, out = self.render(3)
        ids = [e["document_id"] for e in manifest["documents"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids, ["doc-001-customer_001-v1",
                               "doc-001-customer_001-v2",
                               "doc-001-customer_001-v3",
                               "doc-002-customer_002-v1",
                               "doc-002-customer_002-v2",
                               "doc-002-customer_002-v3"])
        self.assertEqual(len({p.name for p in pdfs}), 6)
        for entry in manifest["documents"]:
            with self.subTest(document=entry["document_id"]):
                self.assertTrue((out / entry["pdf"]).is_file())

    def test_variants_of_one_subgraph_carry_identical_ground_truth(self):
        """The invariant the whole feature rests on."""
        manifest, _, _, _, _ = self.render(3)
        by_subgraph = {}
        for entry in manifest["documents"]:
            by_subgraph.setdefault(entry["subgraph_id"], []).append(entry)
        self.assertEqual(len(by_subgraph), 2)
        for subgraph_id, variants in by_subgraph.items():
            with self.subTest(subgraph=subgraph_id):
                self.assertEqual(len(variants), 3)
                # Same records, same root, same joins - only the page differs.
                self.assertEqual({tuple(v["record_ids"]) for v in variants},
                                 {tuple(variants[0]["record_ids"])})
                self.assertEqual({v["root_record"] for v in variants},
                                 {variants[0]["root_record"]})
                self.assertEqual(
                    {json.dumps(v["joins"], sort_keys=True) for v in variants},
                    {json.dumps(variants[0]["joins"], sort_keys=True)})

    def test_records_are_covered_once_per_variant_not_once_overall(self):
        """Coverage counts records, not pages, however many pages carry them."""
        manifest, _, _, _, _ = self.render(3)
        self.assertEqual(manifest["summary"]["records_covered"], 3)
        self.assertEqual(manifest["summary"]["records_total"], 3)

    def test_a_failed_variant_does_not_take_its_siblings_with_it(self):
        """Sibling independence, and why the count is not a division.

        The first subgraph's second variant fails every attempt; its two
        siblings and the whole of the second subgraph still render, so five of
        six documents survive — but subgraph 1 is no longer usable as an
        invariance case, which is what `subgraphs_fully_rendered` says.
        """
        outcomes = [True,                    # sg1 v1
                    "! Undefined control sequence.",  # sg1 v2, and its repairs
                    "! Undefined control sequence.",
                    "! Undefined control sequence.",
                    True, True, True]        # sg1 v3, sg2 v1-v3
        manifest, _, _, pdfs, _ = self.render(3, outcomes=outcomes)
        statuses = {e["document_id"]: e["status"]
                    for e in manifest["documents"]}
        self.assertEqual(statuses["doc-001-customer_001-v2"], "failed")
        self.assertEqual(len(pdfs), 5)
        self.assertEqual(manifest["summary"]["compiled"], 5)
        self.assertEqual(manifest["summary"]["failed"], 1)
        # 5 // 3 would claim 1; only subgraph 2 is actually complete.
        self.assertEqual(manifest["summary"]["subgraphs_fully_rendered"], 1)

    # -- the prompts -------------------------------------------------------- #
    def test_each_variant_is_asked_for_a_different_page(self):
        """Distinct prompts are the mechanism, not a nicety.

        ``--seed`` forces greedy decoding, so two variants sent the same prompt
        come back as the same page and the invariance test is vacuous.
        """
        manifest, _, client, _, _ = self.render(3)
        self.assertEqual(len(set(client.prompts)), 6)
        prompts = [e["layout_prompt"] for e in manifest["documents"]]
        self.assertEqual(len(set(prompts)), 6)

    def test_the_prompt_tells_the_model_it_is_one_of_several(self):
        _, _, client, _, _ = self.render(3)
        first = client.prompts[0]
        self.assertIn("This is layout 1 of 3", first)
        self.assertIn("must not resemble its siblings", first)
        # And that the shared records are not negotiable.
        self.assertIn("as completely as on any other variant", first)
        self.assertIn("This is layout 2 of 3", client.prompts[1])

    def test_a_single_layout_run_is_told_nothing_about_variants(self):
        _, _, client, _, _ = self.render(1)
        for prompt in client.prompts:
            with self.subTest():
                self.assertNotIn("This is layout", prompt)
                self.assertNotIn("siblings", prompt)

    def test_no_variant_is_handed_a_named_layout(self):
        """The flag must not smuggle the old enum back in one level down.

        Naming variant 1 an invoice and variant 2 a memo would cap the corpus at
        as many shapes as the list had names, which is the failure Stage 5 was
        rewritten to remove.
        """
        _, _, client, _, _ = self.render(3)
        for position, prompt in enumerate(client.prompts, start=1):
            with self.subTest(variant=position):
                variant_clause = prompt.split("For a sense of the range")[0]
                for named in ("invoice", "memo", "letter", "form",
                              "spec sheet", "receipt", "statement"):
                    # Whole words: "memo" must not match "memorised", which is
                    # a sentence about what the corpus is for, not a layout.
                    self.assertIsNone(
                        re.search(rf"\b{named}\b", variant_clause,
                                  re.IGNORECASE),
                        f"variant clause names a layout: {named!r}")

    # -- fidelity, per variant ---------------------------------------------- #
    def test_fidelity_is_checked_independently_for_every_variant(self):
        manifest, _, _, _, _ = self.render(3)
        for entry in manifest["documents"]:
            with self.subTest(document=entry["document_id"]):
                fidelity = entry["fidelity"]
                self.assertEqual(fidelity["values_missing"], 0)
                self.assertEqual(fidelity["examples_leaked"], 0)
                # Every variant was measured, not just the first of the set.
                self.assertGreater(fidelity["values_expected"], 0)
        self.assertEqual(manifest["summary"]["values_missing"], 0)
        self.assertEqual(manifest["summary"]["examples_leaked"], 0)
        self.assertEqual(manifest["summary"]["documents_value_complete"], 6)

    def test_a_leak_in_one_variant_is_reported_against_that_variant(self):
        """``leaked_examples`` runs per page, so one bad variant is one report.

        The leaking client copies a prompt example onto every page it writes;
        only the subgraph whose records do not hold that value can leak it, and
        each of its variants is charged separately.
        """
        leaked = PROMPT_EXAMPLE_VALUES[0]

        class LeakingClient(PerScopeClient):
            def complete(self, system, user):
                source = super().complete(system, user)
                sentence = (r"\par The balance on file is %s.\par"
                            % escape_latex(leaked))
                return source.replace(r"\end{document}",
                                      sentence + "\n" + r"\end{document}")

        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        client = LeakingClient()
        engine = ScriptedRenderer(
            generator=LLMLaTeXGenerator(client=client, schema=schema(), seed=1),
            seed=1)
        engine.render_documents(self.two_subgraphs(), Path(tmp),
                                layouts_per_graph=3)
        charged = {e["document_id"]: e["fidelity"]["examples_leaked"]
                   for e in engine.manifest["documents"]}
        # customer-001's subgraph holds $1,240.50 on its order, so printing it
        # is data. customer-002's does not, so all three of its variants leak.
        for variant in ("v1", "v2", "v3"):
            with self.subTest(variant=variant):
                self.assertEqual(charged[f"doc-001-customer_001-{variant}"], 0)
                self.assertEqual(charged[f"doc-002-customer_002-{variant}"], 1)
        self.assertEqual(engine.manifest["summary"]["examples_leaked"], 3)

    # -- validation --------------------------------------------------------- #
    def test_zero_or_negative_is_rejected(self):
        for bad in (0, -1):
            with self.subTest(layouts_per_graph=bad):
                with self.assertRaises(ValueError) as caught:
                    self.render(bad)
                self.assertIn("layouts_per_graph", str(caught.exception))

    def test_a_non_integer_is_rejected(self):
        for bad in ("3", 2.5, None, True):
            with self.subTest(layouts_per_graph=bad):
                with self.assertRaises(ValueError):
                    self.render(bad)


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
        self.assertEqual(lines[3], str(out / cli_module.RUN_CONFIG_FILENAME))
        self.assertEqual(len(lines), 4 + len(pdfs))

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
    def test_freeform_layout_brief_reaches_every_document(self):
        """An arbitrary sentence is a brief, and each PDF records it.

        The flag no longer names a template, so "honoured" cannot mean "the
        letter branch ran". It means the brief was passed through unaltered and
        quoted into the directive that produced each page.
        """
        brief = FREEFORM_HINTS[0]
        result, out, _ = self.run_generate("--layout-style", brief)
        self.assertEqual(result.exit_code, 0, stderr_of(result))
        manifest = json.loads((out / MANIFEST_FILENAME).read_text())
        meta = manifest["metadata"]
        self.assertEqual(meta["layout_hint"], brief)
        self.assertEqual(meta["layout_mode"], "brief")
        self.assertTrue(manifest["documents"])
        for entry in manifest["documents"]:
            with self.subTest(document=entry["document_id"]):
                self.assertEqual(entry["layout_hint"], brief)
                # The brief itself, verbatim, inside the prompt that was sent.
                self.assertIn(brief, entry["layout_prompt"])

    @needs_tex
    def test_run_config_records_the_layout_of_each_pdf(self):
        """``run_config.json`` answers "how does each of these PDFs look?".

        The manifest carries it too, but run_config is the file that describes
        the *run*, and a run whose every document has an invented layout records
        nothing useful about them if it only keeps the word "auto".
        """
        result, out, _ = self.run_generate()
        self.assertEqual(result.exit_code, 0, stderr_of(result))
        config = json.loads((out / cli_module.RUN_CONFIG_FILENAME).read_text())
        self.assertEqual(config["parameters"]["layout_hint"], AUTO_LAYOUT)
        manifest = json.loads((out / MANIFEST_FILENAME).read_text())
        self.assertEqual(len(config["documents"]), len(manifest["documents"]))
        for entry in config["documents"]:
            with self.subTest(document=entry["document_id"]):
                self.assertEqual(entry["layout_hint"], AUTO_LAYOUT)
                # The directive actually sent for this one PDF, not the hint.
                self.assertIn("Layout:", entry["layout_prompt"])
                self.assertTrue(entry["layout"])

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

    def test_any_layout_text_is_accepted_not_validated(self):
        """The inverse of the test this replaces.

        "origami" used to be exit code 2, because the flag was a choice of
        three. There is no list to be off any more: a hint is a description, a
        description cannot be invalid, and the run records the word it was
        given. Rendering is skipped so this holds without a TeX engine.
        """
        result, out, _ = self.run_generate("--no-render",
                                          "--layout-style", "origami")
        self.assertEqual(result.exit_code, 0, stderr_of(result))
        config = json.loads((out / cli_module.RUN_CONFIG_FILENAME).read_text())
        self.assertEqual(config["parameters"]["layout_hint"], "origami")

    def test_blank_layout_text_means_auto(self):
        """Whitespace states no preference, which is what "auto" asks for."""
        result, out, _ = self.run_generate("--no-render",
                                          "--layout-style", "   ")
        self.assertEqual(result.exit_code, 0, stderr_of(result))
        config = json.loads((out / cli_module.RUN_CONFIG_FILENAME).read_text())
        self.assertEqual(config["parameters"]["layout_hint"], AUTO_LAYOUT)

    def test_negative_retries_is_rejected(self):
        result, _, _ = self.run_generate("--max-retries", "-1")
        self.assertEqual(result.exit_code, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
