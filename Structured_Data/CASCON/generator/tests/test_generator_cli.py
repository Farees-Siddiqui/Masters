"""Tests for the CLI surface and :class:`GeneratorConfig`.

The other test modules cover a stage each and reach the CLI only where a stage
needs it. This one is about the arguments themselves: what the flags accept,
what the resolved configuration says, and whether ``run_config.json`` records
the run faithfully enough to reproduce it.

That last point is why the config is tested at all. It is the single list of
what a run was asked to do, and a parameter that reaches the pipeline without
reaching ``run_config.json`` makes the file quietly wrong about the corpus
sitting next to it — which is worse than it being absent, because it is
believed.

Every model call is mocked, so nothing here needs Ollama or a GPU. The one test
that compiles PDFs is skipped when no TeX engine is installed.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typer.testing import CliRunner  # noqa: E402

from src.generator import cli as cli_module  # noqa: E402
from src.generator.config import BACKENDS, GeneratorConfig  # noqa: E402
from src.generator.llm_bridge import (DEFAULT_BASE_URL,  # noqa: E402
                                      DEFAULT_MODEL)
from src.generator.renderer import (MANIFEST_FILENAME,  # noqa: E402
                                    LaTeXRenderer)

HAVE_TEX = LaTeXRenderer(generator=object()).resolve_engine() is not None
needs_tex = unittest.skipUnless(
    HAVE_TEX, "no LaTeX engine on PATH (pdflatex/xelatex/lualatex)")


def setUpModule() -> None:
    logging.disable(logging.WARNING)


def tearDownModule() -> None:
    logging.disable(logging.NOTSET)


def make_runner() -> CliRunner:
    """A runner that keeps stderr separate, across Click versions."""
    try:
        return CliRunner(mix_stderr=False)
    except TypeError:  # pragma: no cover - Click >= 8.2 split them already
        return CliRunner()


def stderr_of(result) -> str:
    try:
        return result.stderr or ""
    except ValueError:  # pragma: no cover - stderr not captured separately
        return ""


# --------------------------------------------------------------------------- #
# Fixtures: a two-entity schema and enough records to make two subgraphs
# --------------------------------------------------------------------------- #
SCHEMA_PAYLOAD = {
    "domain": "small_business",
    "entities": [
        {"name": "Customer", "primary_key": "id", "level": 1,
         "attributes": [{"name": "id", "type": "id"},
                        {"name": "company_name", "type": "string"},
                        {"name": "contact_email", "type": "email"}]},
        {"name": "Order", "primary_key": "id", "level": 2,
         "attributes": [{"name": "id", "type": "id"},
                        {"name": "customer_id", "type": "id"},
                        {"name": "total", "type": "currency"}]},
    ],
    "relationships": [{"child": "Order", "parent": "Customer",
                       "column": "customer_id", "cardinality": "1:m"}],
}

CUSTOMER_ROWS = {"records": [
    {"company_name": "Halvorsen Freight", "contact_email": "h@example.invalid"},
    {"company_name": "Kestrel Supplies", "contact_email": "k@example.invalid"}]}
ORDER_ROWS = {"records": [{"total": "$1,240.50"}, {"total": "$98.00"}]}


class PipelineClient:
    """One stand-in for both clients the CLI builds.

    Stages 1-4 call ``complete_json``; Stage 5 calls ``complete`` and gets back
    the records named in the prompt, so each document is faithful to its own
    subgraph without the fake needing to know how the graph was partitioned.
    """

    def __init__(self, *json_payloads):
        self.json_payloads = list(json_payloads)
        self.latex_prompts = []
        self.last_error = None
        self.model, self.backend = "llama3.3:70b", "ollama"
        self.base_url, self.temperature = DEFAULT_BASE_URL, 0.0

    def complete_json(self, system, user):
        if not self.json_payloads:
            self.last_error = "no output"
            return None
        return self.json_payloads.pop(0)

    def complete(self, system, user):
        self.latex_prompts.append(user)
        body = []
        for identifier in re.findall(r"\[identifier: ([^\],]+)", user):
            body.append(r"\section*{%s}" % identifier.strip().replace("_", r"\_"))
        for label, value in re.findall(r"^  (.+?) = (.*)$", user, re.MULTILINE):
            printed = "---" if value.startswith("(blank") else value
            for raw, escaped in (("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                                 ("#", r"\#"), ("_", r"\_")):
                printed = printed.replace(raw, escaped)
                label = label.replace(raw, escaped)
            body.append(r"%s: %s\\" % (label, printed))
        return ("\\documentclass{article}\n"
                f"% LAYOUT: Variant {len(self.latex_prompts)}, plainly set.\n"
                "\\usepackage[T1]{fontenc}\n"
                "\\begin{document}\n" + "\n".join(body) +
                "\n\\end{document}\n")


# --------------------------------------------------------------------------- #
# 1. GeneratorConfig
# --------------------------------------------------------------------------- #
class TestGeneratorConfig(unittest.TestCase):

    def test_defaults_are_a_single_layout_per_subgraph(self):
        """The feature is opt-in: an unset flag renders what it always did."""
        self.assertEqual(GeneratorConfig().layouts_per_graph, 1)

    def test_layouts_per_graph_must_be_at_least_one(self):
        for bad in (0, -1, -10):
            with self.subTest(layouts_per_graph=bad):
                with self.assertRaises(ValueError) as caught:
                    GeneratorConfig(layouts_per_graph=bad)
                message = str(caught.exception)
                self.assertIn("layouts_per_graph", message)
                self.assertIn(">= 1", message)

    def test_layouts_per_graph_must_be_a_whole_number(self):
        """``True`` is an int to Python and is not a count of layouts."""
        for bad in (2.5, "3", None, True):
            with self.subTest(layouts_per_graph=bad):
                with self.assertRaises(TypeError):
                    GeneratorConfig(layouts_per_graph=bad)

    def test_any_positive_count_is_accepted(self):
        for good in (1, 2, 3, 50):
            with self.subTest(layouts_per_graph=good):
                config = GeneratorConfig(layouts_per_graph=good)
                self.assertEqual(config.layouts_per_graph, good)

    def test_corpus_size_is_knowable_before_the_first_model_call(self):
        """Predictable scaling, which is half of what the flag is for."""
        config = GeneratorConfig(layouts_per_graph=3)
        self.assertEqual(config.documents_for(20), 60)
        self.assertEqual(config.documents_for(0), 0)
        self.assertEqual(GeneratorConfig().documents_for(20), 20)

    def test_other_bounds_are_enforced_too(self):
        """The config is constructible without the CLI, so it validates itself.

        A bound enforced only by Typer's ``min=`` is not enforced at all for a
        caller driving the pipeline as a library.
        """
        for field, bad in (("num_entities", 0), ("max_depth", 0),
                           ("records_per_entity", 0), ("max_retries", -1),
                           ("max_attempts", 0), ("null_probability", 1.5),
                           ("orphan_rate", -0.2)):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    GeneratorConfig(**{field: bad})

    def test_unknown_backend_is_rejected(self):
        with self.assertRaises(ValueError):
            GeneratorConfig(backend="carrier-pigeon")
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                self.assertEqual(GeneratorConfig(backend=backend).backend,
                                 backend)

    def test_blank_layout_hint_normalises_to_auto(self):
        for blank in ("", "   ", "\n", None):
            with self.subTest(layout_hint=blank):
                self.assertEqual(GeneratorConfig(layout_hint=blank).layout_hint,
                                 "auto")

    def test_freeform_layout_hint_is_kept_as_written(self):
        brief = "1990s technical spec sheet with dense grid lines"
        self.assertEqual(GeneratorConfig(layout_hint=brief).layout_hint, brief)

    def test_to_dict_is_json_serialisable_and_complete(self):
        config = GeneratorConfig(layouts_per_graph=4,
                                 output_dir=Path("out_somewhere"))
        payload = config.to_dict()
        # Round-trips through JSON: this is what run_config.json holds.
        restored = json.loads(json.dumps(payload))
        self.assertEqual(restored["layouts_per_graph"], 4)
        self.assertEqual(restored["output_dir"], "out_somewhere")
        self.assertIsInstance(restored["output_dir"], str)
        # Every field of the config is present, so nothing can reach the
        # pipeline without being recorded.
        for name in ("domain", "num_entities", "max_depth", "seed",
                     "records_per_entity", "null_probability", "orphan_rate",
                     "layout_hint", "layouts_per_graph", "keep_tex",
                     "max_retries", "schema_only", "no_render", "output_dir",
                     "model", "base_url", "backend", "max_attempts"):
            with self.subTest(field=name):
                self.assertIn(name, restored)

    def test_defaults_match_the_cli_defaults(self):
        """Two lists of defaults that disagree is a run recorded wrongly."""
        config = GeneratorConfig()
        self.assertEqual(config.model, DEFAULT_MODEL)
        self.assertEqual(config.base_url, DEFAULT_BASE_URL)
        self.assertEqual(config.domain, "small_business")
        self.assertEqual(config.num_entities, 5)
        self.assertEqual(config.records_per_entity, 5)
        self.assertEqual(config.max_retries, 2)
        self.assertEqual(config.output_dir, Path("out_benchmark"))


# --------------------------------------------------------------------------- #
# 2. The flag
# --------------------------------------------------------------------------- #
class TestLayoutsPerGraphFlag(unittest.TestCase):

    def setUp(self):
        self.runner = make_runner()
        self.original = cli_module.build_client

    def tearDown(self):
        cli_module.build_client = self.original

    def run_generate(self, *extra, payloads=None):
        client = PipelineClient(*(payloads or (SCHEMA_PAYLOAD, CUSTOMER_ROWS,
                                               ORDER_ROWS)))
        cli_module.build_client = lambda **kw: client
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        out = Path(tmp) / "bench"
        result = self.runner.invoke(cli_module.app, [
            "generate", "--num-entities", "2", "--records-per-entity", "2",
            "--seed", "4", "--output-dir", str(out), *extra])
        return result, out, client

    def read_run_config(self, out: Path) -> dict:
        with open(out / cli_module.RUN_CONFIG_FILENAME, encoding="utf-8") as fh:
            return json.load(fh)

    # -- the flag exists and is documented ---------------------------------- #
    @staticmethod
    def declared_option(name: str):
        """One of ``generate``'s options, as Click sees it.

        Read off the command rather than out of ``--help``: Typer renders help
        through Rich, which interleaves ANSI codes and box-drawing characters
        with the text and truncates a long flag name to fit its column
        ("--layouts-per-gr…"). Asserting on that output tests the terminal
        renderer. The declaration is what the CLI actually offers.
        """
        import typer.main
        group = typer.main.get_command(cli_module.app)
        for param in group.commands["generate"].params:
            if param.name == name:
                return param
        raise AssertionError(f"generate has no {name!r} option")

    def test_flag_is_declared_with_the_documented_wording(self):
        option = self.declared_option("layouts_per_graph")
        self.assertIn("--layouts-per-graph", option.opts)
        self.assertEqual(option.default, 1)
        self.assertEqual(
            option.help,
            "Number of distinct visual layout variations to generate for each "
            "relational join subgraph. The records are identical across a "
            "subgraph's variants and only the page differs, which makes each "
            "set a layout-invariance case and the corpus size a product: 20 "
            "subgraphs at 3 is exactly 60 documents.")

    def test_help_renders_without_error_and_mentions_the_flag(self):
        """A looser check on the rendered page, tolerant of Rich's formatting."""
        result = self.runner.invoke(cli_module.app, ["generate", "--help"])
        self.assertEqual(result.exit_code, 0)
        # Strip ANSI, then the box-drawing characters Rich pads columns with.
        plain = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)
        collapsed = " ".join(re.sub(r"[│╭╮╰╯─]", " ", plain).split())
        self.assertIn("layout variations", collapsed)
        self.assertIn("relational", collapsed)
        self.assertIn("join subgraph", collapsed)

    def test_flag_defaults_to_one_and_is_recorded(self):
        result, out, _ = self.run_generate("--no-render")
        self.assertEqual(result.exit_code, 0, stderr_of(result))
        config = self.read_run_config(out)
        self.assertEqual(config["parameters"]["layouts_per_graph"], 1)

    def test_flag_value_reaches_run_config(self):
        for value in ("1", "2", "5"):
            with self.subTest(layouts_per_graph=value):
                result, out, _ = self.run_generate(
                    "--no-render", "--layouts-per-graph", value)
                self.assertEqual(result.exit_code, 0, stderr_of(result))
                config = self.read_run_config(out)
                self.assertEqual(config["parameters"]["layouts_per_graph"],
                                 int(value))

    def test_zero_and_negative_are_rejected_by_the_parser(self):
        for bad in ("0", "-1"):
            with self.subTest(layouts_per_graph=bad):
                result, _, _ = self.run_generate(
                    "--no-render", "--layouts-per-graph", bad)
                self.assertEqual(result.exit_code, 2)

    def test_a_non_integer_is_rejected_by_the_parser(self):
        result, _, _ = self.run_generate(
            "--no-render", "--layouts-per-graph", "three")
        self.assertEqual(result.exit_code, 2)

    def test_the_setting_is_echoed_to_stderr(self):
        """The run states what it is doing; stdout stays machine-readable."""
        result, _, _ = self.run_generate("--layouts-per-graph", "3",
                                         "--no-render")
        # --no-render skips stage 5, so the stage-5 line is absent by design.
        self.assertEqual(result.exit_code, 0, stderr_of(result))
        result, _, _ = self.run_generate("--layouts-per-graph", "3")
        self.assertIn("layouts_per_graph=3", stderr_of(result))

    # -- it actually multiplies the corpus ---------------------------------- #
    @needs_tex
    def test_three_layouts_of_two_subgraphs_produces_six_documents(self):
        result, out, client = self.run_generate("--layouts-per-graph", "3")
        self.assertEqual(result.exit_code, 0, stderr_of(result))
        with open(out / MANIFEST_FILENAME, encoding="utf-8") as handle:
            manifest = json.load(handle)
        # Two customers, each with one order: two subgraphs, six documents.
        self.assertEqual(manifest["metadata"]["subgraphs"], 2)
        self.assertEqual(manifest["metadata"]["layouts_per_graph"], 3)
        self.assertEqual(len(manifest["documents"]), 6)
        self.assertEqual(manifest["summary"]["documents_expected"], 6)
        # One model call per document, and each asked for a different page.
        self.assertEqual(len(client.latex_prompts), 6)
        self.assertEqual(len(set(client.latex_prompts)), 6)

    @needs_tex
    def test_run_config_pairs_every_pdf_with_its_subgraph_and_variant(self):
        result, out, _ = self.run_generate("--layouts-per-graph", "3")
        self.assertEqual(result.exit_code, 0, stderr_of(result))
        config = self.read_run_config(out)
        self.assertEqual(len(config["documents"]), 6)
        grouped = {}
        for document in config["documents"]:
            self.assertEqual(document["layout_variants"], 3)
            grouped.setdefault(document["subgraph_id"], []).append(
                document["layout_variant_index"])
        self.assertEqual(len(grouped), 2)
        for subgraph_id, variants in grouped.items():
            with self.subTest(subgraph=subgraph_id):
                self.assertEqual(sorted(variants), [0, 1, 2])

    @needs_tex
    def test_every_variant_of_a_subgraph_claims_the_same_records(self):
        """Identical ground truth is the invariant the feature is built on."""
        result, out, _ = self.run_generate("--layouts-per-graph", "3")
        self.assertEqual(result.exit_code, 0, stderr_of(result))
        config = self.read_run_config(out)
        grouped = {}
        for document in config["documents"]:
            grouped.setdefault(document["subgraph_id"], []).append(
                tuple(document["record_ids"]))
        for subgraph_id, record_sets in grouped.items():
            with self.subTest(subgraph=subgraph_id):
                self.assertEqual(len(set(record_sets)), 1)

    @needs_tex
    def test_pdf_names_carry_the_variant_and_stay_unique(self):
        result, out, _ = self.run_generate("--layouts-per-graph", "3")
        self.assertEqual(result.exit_code, 0, stderr_of(result))
        pdfs = sorted(p.name for p in out.glob("*.pdf"))
        self.assertEqual(len(pdfs), 6)
        self.assertEqual(len(set(pdfs)), 6)
        for name in pdfs:
            with self.subTest(pdf=name):
                self.assertRegex(name, r"^doc-\d{3}-\w+-v[123]\.pdf$")
        # Every PDF is on stdout, after the four fixed artefact lines.
        lines = result.stdout.split()
        self.assertEqual(len(lines), 4 + 6)

    @needs_tex
    def test_one_layout_leaves_the_filenames_alone(self):
        """The default must not rename an existing corpus's documents."""
        result, out, _ = self.run_generate("--layouts-per-graph", "1")
        self.assertEqual(result.exit_code, 0, stderr_of(result))
        for name in sorted(p.name for p in out.glob("*.pdf")):
            with self.subTest(pdf=name):
                self.assertNotIn("-v", name)

    @needs_tex
    def test_fidelity_is_reported_for_each_variant_separately(self):
        result, out, _ = self.run_generate("--layouts-per-graph", "3")
        self.assertEqual(result.exit_code, 0, stderr_of(result))
        with open(out / MANIFEST_FILENAME, encoding="utf-8") as handle:
            manifest = json.load(handle)
        for document in manifest["documents"]:
            with self.subTest(document=document["document_id"]):
                fidelity = document["fidelity"]
                self.assertGreater(fidelity["values_expected"], 0)
                self.assertEqual(fidelity["values_missing"], 0)
                self.assertEqual(fidelity["examples_leaked"], 0)
        self.assertEqual(manifest["summary"]["documents_value_complete"], 6)
        self.assertEqual(manifest["summary"]["examples_leaked"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
