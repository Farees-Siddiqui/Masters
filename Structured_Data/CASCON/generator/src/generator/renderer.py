"""Stage 5: compile LLM-written LaTeX into PDFs, repairing it when it fails.

    InstanceGraph -> scopes -> LLMLaTeXGenerator -> pdflatex -> *.pdf
                                     ^                  |
                                     +---- repair <-----+
                                        (error log)

One document per **scope**: a root record plus every record that hangs off it
through resolving foreign keys. A Customer and its three Orders become one
document; a Product with nothing under it becomes one document of its own. Every
record lands in exactly one scope, which is what lets the manifest state, per
PDF, the exact records an extractor is supposed to recover from it.

The *layout* is written by the model too, and per document: there is no list of
styles to pick from, only a ``layout_hint`` that is either ``"auto"`` — invent a
page that suits these records in this domain — or a freeform brief describing a
look. So the manifest cannot state a layout up front; it records, per PDF, the
hint that was sent, the exact directive that was built for that document, and
the layout the finished page declares itself to be.

The source is written by the model rather than by a template, so three kinds of
thing can now go wrong:

**It does not compile.** An unescaped ``&``, a package that is not installed, a
tabular row with the wrong number of ``&``. The log excerpt goes back to the
model with the source, up to ``max_retries`` times. This is the loop the design
calls for and it earns its keep — mis-escaping is the single commonest failure.

**It compiles, but the data is wrong.** A value reworded, rounded, truncated or
dropped. Nothing in a compiler catches this, and it is worse than a failed
compile: the manifest goes on claiming a value the page does not carry, so the
corpus is quietly corrupt rather than visibly short. So the source is read back
before it is compiled — :func:`latex_generator.missing_values` — and anything
absent is sent back once for restoration. Whatever is still missing is recorded
per document in the manifest, never silently accepted.

**It compiles, the data is all there, and there is extra.** The model states a
fact no record holds — measured, one that it copied out of the prompt's own
worked example. This is the worst of the three, because the manifest does not
mention the invented fact at all, so nothing downstream can distinguish it from
data. :func:`latex_generator.leaked_examples` catches the case that has actually
been observed, and reports it per document; there is no repair, because the leak
is a sentence the model wrote rather than a substitution.

Compilation happens inside a :func:`tempfile.TemporaryDirectory`, so the
``.aux``/``.log``/``.out`` litter is purged by the context manager rather than
swept up afterwards. Only the ``.pdf`` is moved out, plus the ``.tex`` when
``keep_tex`` asks for it.
"""

from __future__ import annotations

import datetime
import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .instance_types import InstanceGraph, Record
from .latex_generator import (AUTO_LAYOUT, LaTeXGenerationError,
                              LLMLaTeXGenerator, escape_latex, is_auto,
                              layout_declaration, leaked_examples,
                              missing_values, normalize_layout_hint,
                              recorded_values)
from .schema_types import SchemaGraph, snake_case

log = logging.getLogger(__name__)

MANIFEST_FILENAME = "benchmark_manifest.json"
#: Engines tried, in order, when ``engine="auto"``.
ENGINE_PREFERENCE = ("pdflatex", "xelatex", "lualatex")
DEFAULT_TIMEOUT = 120.0


class RenderError(RuntimeError):
    """No document could be produced at all."""


# --------------------------------------------------------------------------- #
# Document scoping
# --------------------------------------------------------------------------- #
@dataclass
class DocumentScope:
    """One document's worth of records: a root plus its descendants."""

    index: int
    root: Record
    children: List[Record] = field(default_factory=list)
    #: ``(child_id, column, parent_id)`` for every join inside this scope.
    joins: List[Tuple[str, str, str]] = field(default_factory=list)

    @property
    def records(self) -> List[Record]:
        return [self.root] + list(self.children)

    @property
    def record_ids(self) -> List[str]:
        return [r.id for r in self.records]

    def groups(self) -> List[Tuple[str, List[Record]]]:
        """Child records grouped by entity, in first-appearance order."""
        ordered: List[str] = []
        buckets: Dict[str, List[Record]] = {}
        for record in self.children:
            if record.entity_name not in buckets:
                buckets[record.entity_name] = []
                ordered.append(record.entity_name)
            buckets[record.entity_name].append(record)
        return [(name, buckets[name]) for name in ordered]


def document_scopes(graph: InstanceGraph) -> List[DocumentScope]:
    """Partition the graph into one scope per root record.

    A record is a root when no foreign key of its own resolves to a record in
    the graph — which deliberately includes a child whose key was orphaned at
    ``--orphan-rate``. Such a record still has to appear in the corpus, as a
    document about an order whose customer cannot be identified, so it becomes
    a root of its own rather than being dropped.

    Every record lands in exactly one scope; the partition is asserted, not
    assumed, because a record silently omitted from every document would be a
    fact the manifest claims is somewhere and is not.
    """
    known = graph.ids()
    parent_of: Dict[str, str] = {}
    children_of: Dict[str, List[Record]] = {}
    edges: Dict[str, List[Tuple[str, str, str]]] = {}

    for record in graph.records:
        for column, value in record.foreign_keys.items():
            if value in known and value != record.id:
                # Stage 2 guarantees one parent per entity, so the first
                # resolving key is the scope-forming one.
                parent_of.setdefault(record.id, value)
                if parent_of[record.id] == value:
                    children_of.setdefault(value, []).append(record)
                    edges.setdefault(value, []).append(
                        (record.id, column, value))

    scopes: List[DocumentScope] = []
    placed: Set[str] = set()
    for record in graph.records:
        if record.id in parent_of or record.id in placed:
            continue
        scope = DocumentScope(index=len(scopes), root=record)
        queue = [record.id]
        placed.add(record.id)
        while queue:
            current = queue.pop(0)
            for child in children_of.get(current, []):
                if child.id in placed:
                    continue
                placed.add(child.id)
                scope.children.append(child)
                queue.append(child.id)
            scope.joins.extend(edges.get(current, []))
        scopes.append(scope)

    missed = [r.id for r in graph.records if r.id not in placed]
    if missed:  # pragma: no cover - only reachable via a foreign key cycle
        raise RenderError(
            f"{len(missed)} record(s) belong to no document scope, which would "
            f"leave them unrepresented in the corpus: {', '.join(missed[:5])}")
    return scopes


# --------------------------------------------------------------------------- #
# Compilation result
# --------------------------------------------------------------------------- #
@dataclass
class CompileResult:
    """One pdflatex pass: whether a PDF came out, and what the log said."""

    pdf: Optional[bytes] = None
    log: str = ""
    errors: Optional[str] = None
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return bool(self.pdf)


# --------------------------------------------------------------------------- #
# Renderer
# --------------------------------------------------------------------------- #
class LaTeXRenderer:
    """Renders an :class:`InstanceGraph` into compiled PDFs plus a manifest.

    ``generator`` is injected so tests never need a model; left out, it defaults
    to :class:`LLMLaTeXGenerator` against the local Llama 3 endpoint with JSON
    mode off.
    """

    def __init__(self, schema: Optional[SchemaGraph] = None, *,
                 generator: Optional[LLMLaTeXGenerator] = None,
                 client: Any = None,
                 engine: str = "auto",
                 keep_tex: bool = False,
                 timeout: float = DEFAULT_TIMEOUT,
                 seed: Optional[int] = None) -> None:
        self.schema = schema
        self.generator = generator or LLMLaTeXGenerator(
            client=client, schema=schema, seed=seed)
        self.engine = engine
        self.keep_tex = keep_tex
        self.timeout = timeout
        self.seed = seed
        #: Filled by :meth:`render_documents`; the manifest as written.
        self.manifest: Dict[str, Any] = {}
        self.warnings: List[str] = []

    # -- engine ------------------------------------------------------------- #
    def resolve_engine(self) -> Optional[str]:
        """The LaTeX binary to use, or ``None`` if none is installed."""
        if self.engine and self.engine != "auto":
            return self.engine if shutil.which(self.engine) else None
        for candidate in ENGINE_PREFERENCE:
            if shutil.which(candidate):
                return candidate
        return None

    # -- public api --------------------------------------------------------- #
    def render_documents(self, instance_graph: InstanceGraph,
                         output_dir: Path,
                         layout_hint: Any = AUTO_LAYOUT,
                         max_retries: int = 2,
                         layouts_per_graph: int = 1) -> List[Path]:
        """Render ``layouts_per_graph`` documents per scope, write the manifest.

        ``layout_hint`` is freeform. ``"auto"`` — the default — has the model
        invent a layout per document from the records themselves; any other text
        is a stylistic brief handed to the model as written. There is no set of
        accepted values, so the only thing rejected here is a hint that is not
        text at all: a caller who passes a list or a dict has made a mistake
        that would otherwise reach the model as ``"['form']"``.

        ``layouts_per_graph`` renders each subgraph more than once, the same
        records laid out differently each time. One is the default and behaves
        exactly as a single-layout run always did, filenames included. Above one,
        the corpus size is the product — 20 subgraphs at 3 is 60 documents,
        known before the first model call — and each set of variants is a
        layout-invariance test: identical ground truth, different page.

        Returns the PDFs that compiled. A document that failed every attempt is
        still recorded in the manifest with ``status="failed"``, so the corpus is
        auditable rather than quietly short. A failed variant does not cancel its
        siblings; each is generated, checked and compiled on its own.
        """
        if layout_hint is not None and not isinstance(layout_hint, str):
            raise ValueError(
                f"layout_hint must be text, got {type(layout_hint).__name__}; "
                f"pass 'auto' to have the layout invented, or a sentence "
                f"describing the look you want")
        layout_hint = normalize_layout_hint(layout_hint)
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")
        if not isinstance(layouts_per_graph, int) or \
                isinstance(layouts_per_graph, bool):
            raise ValueError(
                f"layouts_per_graph must be an integer, got "
                f"{type(layouts_per_graph).__name__}")
        if layouts_per_graph < 1:
            raise ValueError(
                f"layouts_per_graph must be >= 1, got {layouts_per_graph}; "
                f"1 renders each subgraph once")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.warnings = []

        scopes = document_scopes(instance_graph)
        engine = self.resolve_engine()
        if engine is None:
            self.warnings.append(
                f"no LaTeX engine found on PATH (tried "
                f"{', '.join(ENGINE_PREFERENCE)}); wrote .tex sources only")
            log.warning("%s", self.warnings[-1])

        entries: List[Dict[str, Any]] = []
        rendered: List[Path] = []
        for scope in scopes:
            for variant in range(layouts_per_graph):
                entry, pdf_path = self._render_one(
                    instance_graph, scope, output_dir, layout_hint,
                    max_retries, engine, variant=variant,
                    variant_count=layouts_per_graph)
                entries.append(entry)
                if pdf_path is not None:
                    rendered.append(pdf_path)

        self.manifest = self._build_manifest(instance_graph, entries,
                                             layout_hint, engine, max_retries,
                                             layouts_per_graph, len(scopes))
        self._write_manifest(output_dir)
        return rendered

    # -- one document ------------------------------------------------------- #
    def _render_one(self, graph: InstanceGraph, scope: DocumentScope,
                    output_dir: Path, layout_hint: str, max_retries: int,
                    engine: Optional[str], *, variant: int = 0,
                    variant_count: int = 1
                    ) -> Tuple[Dict[str, Any], Optional[Path]]:
        stem = self._document_stem(scope, variant, variant_count)
        # Distinct per (subgraph, variant), which is the whole mechanism behind
        # --layouts-per-graph: --seed forces greedy decoding, so two variants
        # sent the same prompt come back as the same page. Rotating the device
        # and axis per variant is what makes them differ while the records stay
        # byte-identical.
        variation = scope.index * max(1, variant_count) + variant
        # The wording this document is asked for, built before the call and kept
        # whatever the call does: a document that failed still has to say what it
        # was asked to be, or a failed run cannot be diagnosed.
        directive = self.generator.build_layout_directive(
            layout_hint, variation=variation, variant=variant,
            variant_count=variant_count)
        entry: Dict[str, Any] = {
            "document_id": stem,
            "layout_hint": layout_hint,
            "layout_prompt": directive,
            # Overwritten with what the page declares itself to be, once there
            # is a page. Until then the hint is the best statement available.
            "layout": layout_hint,
            # Which relational subgraph this page is about, and which of its
            # layouts this one is. Every variant of one subgraph carries the
            # same subgraph_id and the same record_ids, and differs only in
            # layout_variant_index -- that pairing is what a layout-invariance
            # check groups on.
            "subgraph_id": self._subgraph_id(scope),
            "subgraph_index": scope.index,
            "layout_variant_index": variant,
            "layout_variants": variant_count,
            "root_record": scope.root.id,
            "root_entity": scope.root.entity_name,
            "record_ids": scope.record_ids,
            "records": [r.to_dict() for r in scope.records],
            "joins": [list(j) for j in scope.joins],
            "orphaned_foreign_keys": [
                {"record": r.id, "column": c, "value": r.foreign_keys.get(c)}
                for r in scope.records for c in r.orphaned_keys],
            "pdf": None,
            "tex": None,
        }

        try:
            source = self.generator.generate_latex_source(
                scope.records, layout_hint, graph.schema_domain,
                variation=variation, variant=variant,
                variant_count=variant_count)
        except LaTeXGenerationError as exc:
            entry.update(status="failed", failure=str(exc), attempts=0)
            self.warnings.append(f"{stem}: {exc}")
            log.warning("%s: %s", stem, exc)
            return entry, None

        # What the page says it turned out to be, which is the only statement of
        # the layout that exists once the model invents one per document. A page
        # that never declared itself keeps the hint, and is flagged: with "auto"
        # that leaves the manifest saying nothing about how the PDF looks.
        entry["layout_declared"] = layout_declaration(source)
        if entry["layout_declared"]:
            entry["layout"] = entry["layout_declared"]
        elif is_auto(layout_hint):
            self.warnings.append(
                f"{stem}: the page did not declare its layout, so the manifest "
                f"records the hint ('{layout_hint}') rather than the invented "
                f"layout")

        source, fidelity = self._enforce_fidelity(scope, source, stem)
        entry["fidelity"] = fidelity

        if engine is None:
            entry.update(status="not_compiled", attempts=0,
                         tex=self._write_tex(source, stem, output_dir))
            return entry, None

        pdf_path, attempts, repairs, diagnostic = self._compile_with_repair(
            source, scope, stem, output_dir, engine, layout_hint,
            graph.schema_domain, max_retries)
        entry["attempts"] = attempts
        if repairs:
            entry["repairs"] = repairs
        if pdf_path is None:
            entry.update(status="failed", failure=diagnostic)
            self.warnings.append(
                f"{stem}: compilation failed after {attempts} attempt(s); "
                f"{diagnostic}")
            log.warning("%s failed to compile: %s", stem, diagnostic)
            entry["tex"] = self._tex_name(stem, output_dir)
            return entry, None

        entry.update(status="compiled", pdf=pdf_path.name,
                     tex=self._tex_name(stem, output_dir))
        if diagnostic:
            # A PDF can appear despite errors under nonstopmode. Saying so is
            # the difference between a usable corpus and one with silently
            # garbled pages.
            entry["log_errors"] = diagnostic
            self.warnings.append(
                f"{stem}: compiled, but LaTeX reported errors; {diagnostic}")
        if repairs:
            self.warnings.append(
                f"{stem}: compiled after {len(repairs)} repair(s)")
        return entry, pdf_path

    # -- fidelity ----------------------------------------------------------- #
    def _enforce_fidelity(self, scope: DocumentScope, source: str, stem: str
                          ) -> Tuple[str, Dict[str, Any]]:
        """Check the recorded values are in the source; restore them once.

        One restoration attempt, not a loop: a model that dropped a value twice
        will drop it a third time, and an unbounded loop here would trade a
        recorded, auditable gap for an unbounded bill. What is still missing
        afterwards is written into the manifest, because a corpus that knows
        which pages are incomplete is usable and one that does not is not.
        """
        expected = len(recorded_values(scope.records))
        missing = missing_values(scope.records, source)
        info: Dict[str, Any] = {"values_expected": expected,
                                "values_missing_before_repair": len(missing),
                                "restored": False}
        if not missing:
            info["values_missing"] = 0
            self._note_leaks(scope, source, stem, info)
            return source, info

        log.info("%s: %d of %d recorded value(s) absent from the generated "
                 "source; asking for restoration", stem, len(missing), expected)
        restored = self.generator.restore_values(source, missing, scope.records)
        still = missing_values(scope.records, restored)
        if len(still) < len(missing):
            source, info["restored"] = restored, True
            missing = still
        info["values_missing"] = len(missing)
        if missing:
            info["missing"] = [{"record": r, "field": f, "value": v}
                               for r, f, v in missing]
            self.warnings.append(
                f"{stem}: {len(missing)} recorded value(s) are not on the page "
                f"(" + ", ".join(f"{r}.{f}" for r, f, _ in missing[:4])
                + ("..." if len(missing) > 4 else "") + ")")
            log.warning("%s", self.warnings[-1])
        self._note_leaks(scope, source, stem, info)
        return source, info

    def _note_leaks(self, scope: DocumentScope, source: str, stem: str,
                    info: Dict[str, Any]) -> None:
        """Record any prompt example the model copied onto the page as data.

        Reported, not repaired. A leak is a sentence the model wrote, not a
        substitution, so there is no mechanical fix — and unlike a missing
        value, leaving it unrecorded would hide it completely: the manifest
        never mentions the invented fact, so nothing downstream could tell it
        from a real one.
        """
        leaked = leaked_examples(scope.records, source)
        info["examples_leaked"] = len(leaked)
        if not leaked:
            return
        info["leaked"] = leaked
        self.warnings.append(
            f"{stem}: the page states {len(leaked)} value(s) that came from "
            f"the prompt's own examples, not from any record "
            f"({', '.join(leaked)}); this document's ground truth is "
            f"incomplete about what is on it")
        log.warning("%s", self.warnings[-1])

    # -- compile / repair loop ---------------------------------------------- #
    def _compile_with_repair(self, source: str, scope: DocumentScope, stem: str,
                             output_dir: Path, engine: str, layout_hint: str,
                             domain: str, max_retries: int
                             ) -> Tuple[Optional[Path], int, List[Dict[str, Any]],
                                        Optional[str]]:
        """Compile, and on failure send the log back to the model to fix.

        ``max_retries`` bounds the *repairs*, so a document gets at most
        ``max_retries + 1`` compilations. The whole exchange happens inside one
        temporary directory: the source of every attempt, and every ``.aux`` and
        ``.log`` LaTeX writes, live and die in there.
        """
        repairs: List[Dict[str, Any]] = []
        attempts = 0
        result = CompileResult()

        with tempfile.TemporaryDirectory(prefix="generator-tex-") as tmp:
            tmp_dir = Path(tmp)
            for attempt in range(max_retries + 1):
                attempts += 1
                result = self._compile_once(source, stem, tmp_dir, engine)
                if result.ok:
                    break
                diagnostic = result.errors or "no PDF was produced"
                if attempt == max_retries:
                    break
                log.info("%s: attempt %d failed (%s); asking for a repair",
                         stem, attempts, diagnostic)
                repaired = self.generator.repair_latex_source(
                    source, self._repair_excerpt(result), scope.records,
                    layout_hint, domain)
                repairs.append({"attempt": attempts, "errors": diagnostic,
                                "changed": repaired != source})
                if repaired == source:
                    # The model returned nothing usable, so a further pass would
                    # feed pdflatex the identical input. Stop and report.
                    log.warning("%s: repair returned an unchanged source; "
                                "abandoning", stem)
                    break
                source = repaired

            if not result.ok:
                self._write_tex(source, stem, output_dir)
                if result.log:
                    (output_dir / f"{stem}.log").write_text(
                        result.log, encoding="utf-8")
                return None, attempts, repairs, (
                    result.errors or "no PDF was produced")

            if self.keep_tex:
                self._write_tex(source, stem, output_dir)
            destination = output_dir / f"{stem}.pdf"
            destination.write_bytes(result.pdf or b"")
            return destination, attempts, repairs, result.errors
        # The TemporaryDirectory is gone here: nothing to clean up by hand.

    def _compile_once(self, source: str, stem: str, tmp_dir: Path, engine: str
                      ) -> CompileResult:
        """One compilation, entirely inside ``tmp_dir``.

        Returns the PDF as bytes rather than a path so the caller need not move
        a file out before the temporary directory is torn down. Overridable:
        the tests drive the repair loop through this method without a TeX
        installation.
        """
        tex_path = tmp_dir / f"{stem}.tex"
        tex_path.write_text(source, encoding="utf-8")
        log_path = tmp_dir / f"{stem}.log"
        pdf_path = tmp_dir / f"{stem}.pdf"
        for stale in (log_path, pdf_path):
            if stale.exists():
                stale.unlink()

        log_text = ""
        for pass_no in (1, 2):
            try:
                subprocess.run(
                    [engine, "-interaction=nonstopmode", "-no-shell-escape",
                     "-halt-on-error", f"{stem}.tex"],
                    cwd=str(tmp_dir), capture_output=True, check=False,
                    timeout=self.timeout)
            except subprocess.TimeoutExpired:
                log_text = log_path.read_text(encoding="utf-8", errors="replace") \
                    if log_path.exists() else ""
                return CompileResult(
                    log=log_text, timed_out=True,
                    errors=f"{engine} exceeded the {self.timeout:g}s timeout")
            except OSError as exc:  # pragma: no cover - engine vanished
                return CompileResult(errors=f"could not run {engine}: {exc}")
            log_text = log_path.read_text(encoding="utf-8", errors="replace") \
                if log_path.exists() else ""
            # A second pass only when LaTeX itself asks for one, so a one-page
            # form is not compiled twice for nothing.
            if pass_no == 1 and "Rerun to get" in log_text:
                continue
            break

        produced = pdf_path.exists() and pdf_path.stat().st_size > 0
        return CompileResult(pdf=pdf_path.read_bytes() if produced else None,
                             log=log_text, errors=log_errors(log_text))

    @staticmethod
    def _repair_excerpt(result: CompileResult, limit: int = 3000) -> str:
        """What to show the model: the error lines, then the log tail.

        The whole log is mostly font paths and package banners; the errors and
        the lines around them are what a fix needs, and a short prompt is a
        cheaper and more accurate one.
        """
        parts: List[str] = []
        if result.errors:
            parts.append(result.errors)
        lines = result.log.splitlines()
        marked = [i for i, line in enumerate(lines) if line.startswith("! ")]
        if marked:
            for index in marked[:3]:
                window = lines[max(0, index - 2):index + 6]
                parts.append("\n".join(window))
        elif lines:
            parts.append("\n".join(lines[-40:]))
        return "\n\n".join(parts)[:limit] or "pdflatex produced no log output"

    @staticmethod
    def _subgraph_id(scope: DocumentScope) -> str:
        """A stable name for the join subgraph, shared by all of its variants."""
        return f"subgraph-{scope.index + 1:03d}"

    @staticmethod
    def _document_stem(scope: DocumentScope, variant: int = 0,
                       variant_count: int = 1) -> str:
        """A filesystem-safe, collision-free name derived from the root record.

        The ``-vN`` suffix appears only when there is more than one layout to
        tell apart, so a single-layout run writes the filenames it always wrote
        and an existing corpus does not have to be regenerated to be read.
        """
        stem = f"doc-{scope.index + 1:03d}-{snake_case(scope.root.id)}"
        if max(1, int(variant_count)) < 2:
            return stem
        return f"{stem}-v{int(variant) + 1}"

    # -- files -------------------------------------------------------------- #
    def _write_tex(self, source: str, stem: str, output_dir: Path) -> str:
        """Write the ``.tex`` beside the PDFs. Only when asked, or on failure."""
        path = Path(output_dir) / f"{stem}.tex"
        path.write_text(source, encoding="utf-8")
        return path.name

    @staticmethod
    def _tex_name(stem: str, output_dir: Path) -> Optional[str]:
        path = Path(output_dir) / f"{stem}.tex"
        return path.name if path.exists() else None

    # -- manifest ----------------------------------------------------------- #
    def _build_manifest(self, graph: InstanceGraph,
                        entries: Sequence[Dict[str, Any]],
                        layout_hint: str, engine: Optional[str],
                        max_retries: int, layouts_per_graph: int = 1,
                        subgraphs: int = 0) -> Dict[str, Any]:
        compiled = [e for e in entries if e["status"] == "compiled"]
        covered = {rid for e in entries for rid in e["record_ids"]}
        complete = [e for e in compiled
                    if not e.get("fidelity", {}).get("values_missing")]
        # A tally rather than a count of three named layouts: with the layout
        # invented per document these keys are sentences and mostly unique, and
        # a key appearing twice is the thing worth seeing — it means two pages
        # came out the same, which is the failure this stage exists to avoid.
        layouts: Dict[str, int] = {}
        for entry in entries:
            layouts[entry["layout"]] = layouts.get(entry["layout"], 0) + 1
        declared = [e for e in entries if e.get("layout_declared")]
        return {
            "schema_domain": graph.schema_domain,
            "metadata": {
                "stage": "5:rendered_documents",
                "source": "llm_generated_latex",
                "layout_hint": layout_hint,
                "layout_mode": "invented" if is_auto(layout_hint) else "brief",
                "layouts": layouts,
                "distinct_layouts": len(layouts),
                "layouts_declared": len(declared),
                "layouts_per_graph": layouts_per_graph,
                "subgraphs": subgraphs,
                "engine": engine,
                "max_retries": max_retries,
                "keep_tex": self.keep_tex,
                "seed": self.seed,
                "model": getattr(self.generator.client, "model", None),
                "generated_at": datetime.datetime.now(
                    datetime.timezone.utc).replace(microsecond=0).isoformat(),
            },
            "summary": {
                "documents": len(entries),
                "documents_expected": subgraphs * layouts_per_graph,
                "compiled": len(compiled),
                "failed": len(entries) - len(compiled),
                # A subgraph is only usable as a layout-invariance case when
                # every one of its variants compiled: comparing an extractor
                # across two shapes needs both shapes to exist.
                "subgraphs_fully_rendered": _fully_rendered(entries,
                                                            layouts_per_graph),
                "repaired": sum(1 for e in entries if e.get("repairs")),
                "compilations": sum(e.get("attempts", 0) for e in entries),
                "records_total": graph.total_records,
                "records_covered": len(covered),
                "joins": sum(len(e["joins"]) for e in entries),
                "orphaned_foreign_keys": sum(
                    len(e["orphaned_foreign_keys"]) for e in entries),
                # The number that decides whether the corpus is usable: a
                # compiled PDF carrying every value the manifest claims of it.
                "documents_value_complete": len(complete),
                "values_missing": sum(
                    e.get("fidelity", {}).get("values_missing", 0)
                    for e in entries),
                "examples_leaked": sum(
                    e.get("fidelity", {}).get("examples_leaked", 0)
                    for e in entries),
            },
            "documents": list(entries),
            "warnings": list(self.warnings),
        }

    def _write_manifest(self, output_dir: Path) -> Path:
        path = Path(output_dir) / MANIFEST_FILENAME
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.manifest, handle, indent=2, ensure_ascii=False,
                      default=str)
            handle.write("\n")
        return path

    def summary(self) -> str:
        """Stderr summary: what was rendered, repaired, and what is incomplete."""
        if not self.manifest:
            return "no documents rendered"
        info = self.manifest["summary"]
        meta = self.manifest["metadata"]
        lines = [f"documents={info['documents']} compiled={info['compiled']} "
                 f"failed={info['failed']} repaired={info['repaired']} "
                 f"engine={meta['engine']} "
                 f"layout_hint={meta['layout_hint']!r} "
                 f"distinct_layouts={meta['distinct_layouts']} of "
                 f"{info['documents']}",
                 f"  {info['compilations']} compilation(s) for "
                 f"{info['documents']} document(s); records covered "
                 f"{info['records_covered']} of {info['records_total']}, "
                 f"{info['joins']} join(s), "
                 f"{info['orphaned_foreign_keys']} orphaned key(s)",
                 f"  value-complete {info['documents_value_complete']} of "
                 f"{info['compiled']} compiled, "
                 f"{info['values_missing']} value(s) missing from pages, "
                 f"{info['examples_leaked']} prompt example(s) leaked onto "
                 f"pages"]
        for entry in self.manifest["documents"]:
            mark = "ok " if entry["status"] == "compiled" else "FAIL"
            missing = entry.get("fidelity", {}).get("values_missing", 0)
            note = f", {missing} value(s) MISSING" if missing else ""
            repairs = entry.get("repairs")
            fixed = f", {len(repairs)} repair(s)" if repairs else ""
            lines.append(f"  [{mark}] {entry['document_id']}.pdf "
                         f"({len(entry['record_ids'])} record(s), "
                         f"{len(entry['joins'])} join(s){fixed}{note})")
            # The invented layout on its own line and untruncated: it is a
            # sentence, and it is the only record of what the page looks like.
            lines.append(f"         layout: {entry['layout']}")
        return "\n".join(lines)


def _fully_rendered(entries: Sequence[Dict[str, Any]],
                    layouts_per_graph: int) -> int:
    """Subgraphs whose every layout variant compiled.

    Counted rather than assumed equal to ``compiled // layouts_per_graph``: the
    failures are not spread evenly, and three subgraphs that each lost one
    variant leave zero usable invariance cases while the division claims two.
    """
    wanted = max(1, int(layouts_per_graph))
    compiled: Dict[str, int] = {}
    for entry in entries:
        if entry.get("status") != "compiled":
            continue
        key = entry.get("subgraph_id") or entry.get("root_record") or ""
        compiled[key] = compiled.get(key, 0) + 1
    return sum(1 for count in compiled.values() if count >= wanted)


def log_errors(log_text: str) -> Optional[str]:
    """The first few LaTeX error lines from a run log, joined into one line."""
    if not log_text:
        return None
    lines = log_text.splitlines()
    errors = []
    for index, line in enumerate(lines):
        if not line.startswith("! "):
            continue
        # "! Undefined control sequence." on its own says nothing about *which*
        # sequence; LaTeX puts that on the following "l.<n> ..." line.
        context = next((c.strip() for c in lines[index + 1:index + 4]
                        if c.startswith("l.")), "")
        errors.append(f"{line.strip()} {context}".strip())
    if not errors:
        return None
    joined = "; ".join(errors[:3])
    if len(errors) > 3:
        joined += f" (+{len(errors) - 3} more)"
    return joined[:500]


__all__ = [
    "LaTeXRenderer",
    "RenderError",
    "DocumentScope",
    "CompileResult",
    "document_scopes",
    "log_errors",
    "escape_latex",
    "AUTO_LAYOUT",
    "MANIFEST_FILENAME",
]
