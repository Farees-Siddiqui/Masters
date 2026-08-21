"""Typer CLI for the data generation pipeline.

    python -m src.generator.cli generate --domain small_business --num-entities 4

``generate`` runs Stages 1 to 5 in one pass: ask local Llama 3 for an ER graph
and enforce the requested parameters on it, populate it with records, then have
the model write the LaTeX for each document and compile it — repairing the
source from pdflatex's own error log when it does not build. The artefacts are
``schema.json``, ``instances.json``, one PDF per document,
``benchmark_manifest.json`` and ``run_config.json``. ``--schema-only`` and
``--no-render`` stop early, and still write ``run_config.json``.

``--layout-hint`` is freeform text, not a choice: ``auto`` has the model invent a
document layout per PDF from the records themselves, and anything else is a
stylistic brief passed through to it verbatim. Because there is no longer a
layout named in the arguments, ``run_config.json`` records what each PDF was
asked to be and what it came out as, per document.

``--layouts-per-graph`` renders each join subgraph more than once, the same
records laid out differently each time, which is both a layout-invariance test
and the knob that makes the corpus size predictable: subgraphs times variants,
known before the first model call.

Everything human-readable goes to **stderr**, so the command composes: stdout
carries only the path of the artefact written, which means

    OUT=$(python -m src.generator.cli generate --domain medical)

leaves the summary on the terminal and the artefact paths in the variable.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import typer

# Support both `python -m src.generator.cli` and `python src/generator/cli.py`,
# matching the import-style shim in layout_pipeline/main.py.
if __package__ in (None, ""):  # pragma: no cover - import-style shim
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    from src.generator.llm_bridge import (DEFAULT_BASE_URL, DEFAULT_MODEL,
                                          LLMUnavailable, build_client)
    from src.generator.instance_generator import (InstanceGenerationError,
                                                  ParametricInstanceGenerator,
                                                  write_instances)
    from src.generator.latex_generator import (LLMLaTeXGenerator,
                                               normalize_layout_hint)
    from src.generator.config import GeneratorConfig
    from src.generator.renderer import (MANIFEST_FILENAME, LaTeXRenderer,
                                        RenderError)
    from src.generator.schema_generator import (ParametricSchemaGenerator,
                                                SchemaGenerationError,
                                                write_schema)
else:
    from .llm_bridge import (DEFAULT_BASE_URL, DEFAULT_MODEL, LLMUnavailable,
                             build_client)
    from .instance_generator import (InstanceGenerationError,
                                     ParametricInstanceGenerator,
                                     write_instances)
    from .latex_generator import LLMLaTeXGenerator, normalize_layout_hint
    from .config import GeneratorConfig
    from .renderer import MANIFEST_FILENAME, LaTeXRenderer, RenderError
    from .schema_generator import (ParametricSchemaGenerator,
                                   SchemaGenerationError, write_schema)

SCHEMA_FILENAME = "schema.json"
INSTANCES_FILENAME = "instances.json"
RUN_CONFIG_FILENAME = "run_config.json"

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Generate synthetic structured-document benchmarks: a parametric ER "
         "schema, and the records and documents derived from it.",
)

log = logging.getLogger("generator")


def _configure_logging(verbose: bool) -> None:
    """Send logs to stderr. stdout is reserved for machine-readable output."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.DEBUG if verbose else logging.INFO)


def _echo(message: str = "") -> None:
    typer.echo(message, err=True)


def _report_warnings(warnings: list, headline: str, limit: int = 40) -> None:
    """List warnings on stderr, capped so one noisy entity cannot bury the rest.

    The cap is announced rather than silent: a truncated list that reads as
    complete is worse than no list.
    """
    if not warnings:
        return
    _echo("")
    _echo(f"{len(warnings)} {headline}:")
    for warning in warnings[:limit]:
        _echo(f"  - {warning}")
    if len(warnings) > limit:
        _echo(f"  ... and {len(warnings) - limit} more (all of them are in the "
              f"artefact's own \"warnings\" list)")


def _write_run_config(path: Path, parameters: dict, artefacts: list,
                      documents: Optional[list] = None) -> Path:
    """Write ``run_config.json``: what was asked for, and what each PDF became.

    Written on every run, including the ones that stop early, because the file
    answers "what produced this directory" and a half-finished directory is
    exactly when that is hard to reconstruct from the artefacts themselves.

    The per-document block is the part that matters now that Stage 5 invents
    layouts. ``layout_hint`` is what the run asked for — one string for the
    whole corpus — while ``layout_prompt`` is the exact directive built for that
    one PDF and ``layout`` is what the finished page declared itself to be.
    Without those two, a corpus of a hundred documents records only the word
    "auto" about how any of them look.
    """
    payload = {
        "tool": "src.generator.cli generate",
        "generated_at": datetime.datetime.now(
            datetime.timezone.utc).replace(microsecond=0).isoformat(),
        "parameters": parameters,
        "artefacts": [str(a) for a in artefacts],
        "documents": list(documents or []),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
        handle.write("\n")
    return path


def _document_layouts(manifest: dict) -> list:
    """The layout record for each PDF, lifted out of the manifest.

    ``subgraph_id`` and ``layout_variant_index`` travel with it under
    ``--layouts-per-graph``: several PDFs then describe the same records, and
    without the pair there is nothing in this file saying which of them are
    variants of each other rather than separate documents.
    """
    return [{"document_id": entry["document_id"],
             "pdf": entry.get("pdf"),
             "status": entry["status"],
             "subgraph_id": entry.get("subgraph_id"),
             "layout_variant_index": entry.get("layout_variant_index"),
             "layout_variants": entry.get("layout_variants"),
             "record_ids": entry.get("record_ids"),
             "layout_hint": entry.get("layout_hint"),
             "layout": entry.get("layout"),
             "layout_declared": entry.get("layout_declared"),
             "layout_prompt": entry.get("layout_prompt")}
            for entry in manifest.get("documents", [])]


# An explicit callback keeps `generate` addressable as a subcommand. Without
# one, Typer promotes a lone command to the app root, and `generate` would stop
# being a word you type the moment a second stage command is added.
@app.callback()
def cli() -> None:
    """Synthetic benchmark generation: schema, records, documents."""


@app.command()
def generate(
    domain: str = typer.Option(
        "small_business", "--domain",
        help="Target domain vertical, e.g. medical, small_business, education."),
    num_entities: int = typer.Option(
        5, "--num-entities", min=1,
        help="Number of relational schema tables/classes to request."),
    max_depth: int = typer.Option(
        2, "--max-depth", min=1,
        help="Maximum hierarchical nesting depth of 1:m foreign key links. "
             "A parentless entity is level 1, so 1 means a flat schema with no "
             "foreign keys at all."),
    seed: Optional[int] = typer.Option(
        None, "--seed",
        help="Pseudo-random seed. Pins the decoder seed and forces greedy "
             "decoding (temperature 0) for a reproducible schema; without it "
             "the model samples, so repeated runs give a varied corpus."),
    records_per_entity: int = typer.Option(
        5, "--records-per-entity", min=1,
        help="Target number of concrete data instances to generate per entity "
             "type."),
    null_probability: float = typer.Option(
        0.05, "--null-probability", min=0.0, max=1.0,
        help="Probability that an optional non-key field is set to null. Keys "
             "and required fields are never nulled."),
    orphan_rate: float = typer.Option(
        0.0, "--orphan-rate", min=0.0, max=1.0,
        help="Fraction of child foreign keys pointed at a parent that does not "
             "exist, for simulating noisy relational graphs. Each orphaned "
             "column is listed in the record's own 'orphaned_keys', so injected "
             "noise stays distinguishable from an extraction error."),
    layout_hint: str = typer.Option(
        "auto", "--layout-hint", "--layout-style",
        help="Freeform. 'auto' (the default) has the model read each "
             "document's records and invent a layout that suits them, so a run "
             "produces as many shapes as it has documents. Any other text is a "
             "stylistic brief handed to the model as written, e.g. "
             "--layout-hint '1990s technical spec sheet with dense grid "
             "lines'. There is no list of accepted values."),
    layouts_per_graph: int = typer.Option(
        1, "--layouts-per-graph", min=1,
        help="Number of distinct visual layout variations to generate for each "
             "relational join subgraph. The records are identical across a "
             "subgraph's variants and only the page differs, which makes each "
             "set a layout-invariance case and the corpus size a product: 20 "
             "subgraphs at 3 is exactly 60 documents."),
    keep_tex: bool = typer.Option(
        False, "--keep-tex/--no-keep-tex",
        help="Retain the .tex source beside each PDF for debugging. Off by "
             "default; a failed document's source and log are kept either way."),
    max_retries: int = typer.Option(
        2, "--max-retries", min=0,
        help="Maximum compilation repair attempts per document. On a failed "
             "compile the pdflatex errors and the source go back to the model "
             "to be fixed; 0 disables repair."),
    schema_only: bool = typer.Option(
        False, "--schema-only",
        help="Stop after Stages 1 & 2: write schema.json and skip instance "
             "population and rendering."),
    no_render: bool = typer.Option(
        False, "--no-render",
        help="Stop after Stages 3 & 4: skip LaTeX rendering and the manifest."),
    output_dir: Path = typer.Option(
        Path("out_benchmark"), "--output-dir",
        help="Directory for benchmark artefacts. Created if absent."),
    model: str = typer.Option(
        DEFAULT_MODEL, "--model",
        help="Model tag served by the local endpoint."),
    base_url: str = typer.Option(
        DEFAULT_BASE_URL, "--base-url",
        help="Local inference endpoint."),
    backend: str = typer.Option(
        "ollama", "--backend",
        help="Wire protocol: 'ollama' (/api/chat) or 'openai' "
             "(/v1/chat/completions, e.g. vLLM)."),
    max_attempts: int = typer.Option(
        3, "--max-attempts", min=1,
        help="Attempts before giving up on an unusable model response."),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Debug-level logging on stderr."),
) -> None:
    """Stages 1-5: generate a schema, populate it, and render documents.

    Writes schema.json, instances.json, one PDF per document and
    benchmark_manifest.json into --output-dir.
    """
    _configure_logging(verbose)

    if backend not in ("ollama", "openai"):
        _echo(f"error: unknown backend {backend!r}; expected 'ollama' or 'openai'")
        raise typer.Exit(code=2)
    # One object rather than a dict assembled beside the signature: the config
    # is what gets recorded, so a parameter that reaches the pipeline without
    # reaching run_config.json should not be expressible. No validation of the
    # layout hint, deliberately — it is a description, and the whole point of
    # Stage 5's rewrite is that there is no list to be off. Blank means "no
    # preference", which is what 'auto' already asks for, and the config
    # normalises it.
    config = GeneratorConfig(
        domain=domain, num_entities=num_entities, max_depth=max_depth,
        seed=seed, records_per_entity=records_per_entity,
        null_probability=null_probability, orphan_rate=orphan_rate,
        layout_hint=layout_hint, layouts_per_graph=layouts_per_graph,
        keep_tex=keep_tex, max_retries=max_retries, schema_only=schema_only,
        no_render=no_render, output_dir=output_dir, model=model,
        base_url=base_url, backend=backend, max_attempts=max_attempts)
    layout_hint = config.layout_hint

    output_dir.mkdir(parents=True, exist_ok=True)
    schema_path = output_dir / SCHEMA_FILENAME
    instances_path = output_dir / INSTANCES_FILENAME
    run_config_path = output_dir / RUN_CONFIG_FILENAME
    parameters = config.to_dict()

    _echo(f"domain={domain} entities={num_entities} max_depth={max_depth} "
          f"seed={seed if seed is not None else 'unset'}")
    if not schema_only:
        _echo(f"records_per_entity={records_per_entity} "
              f"null_probability={null_probability} "
              f"orphan_rate={orphan_rate}")
    if not schema_only and not no_render:
        _echo(f"layout_hint={layout_hint!r} "
              f"layouts_per_graph={layouts_per_graph} keep_tex={keep_tex} "
              f"max_retries={max_retries}")
    _echo(f"model={model} endpoint={base_url} backend={backend}")

    client = build_client(model=model, base_url=base_url, backend=backend,
                         seed=seed)
    generator = ParametricSchemaGenerator(client=client, seed=seed,
                                         max_attempts=max_attempts)

    try:
        graph = generator.generate_schema(domain=domain,
                                         num_entities=num_entities,
                                         max_depth=max_depth)
    except LLMUnavailable as exc:
        _echo(f"error: local model unreachable at {base_url}: {exc}")
        _echo("hint: start the server (`ollama serve`) and confirm the model "
              f"is pulled (`ollama pull {model}`).")
        raise typer.Exit(code=3)
    except SchemaGenerationError as exc:
        _echo(f"error: {exc}")
        raise typer.Exit(code=4)
    except ValueError as exc:
        _echo(f"error: {exc}")
        raise typer.Exit(code=2)

    write_schema(graph, str(schema_path))

    _echo("")
    _echo("stage 1-2: schema")
    _echo(graph.summary())
    _report_warnings(graph.warnings,
                     "repair(s) applied to satisfy the requested parameters")
    _echo("")
    _echo(f"wrote {schema_path}")

    if schema_only:
        _write_run_config(run_config_path, parameters, [schema_path])
        typer.echo(str(schema_path))
        typer.echo(str(run_config_path))
        return

    populator = ParametricInstanceGenerator(client=client, seed=seed,
                                           max_attempts=max_attempts)
    try:
        instances = populator.generate_instances(
            schema=graph, records_per_entity=records_per_entity,
            null_prob=null_probability, orphan_rate=orphan_rate)
    except LLMUnavailable as exc:
        _echo(f"error: local model unreachable at {base_url}: {exc}")
        _echo(f"note: {schema_path} was written; rerun to populate it.")
        raise typer.Exit(code=3)
    except InstanceGenerationError as exc:
        _echo(f"error: {exc}")
        _echo(f"note: {schema_path} was written; rerun to populate it.")
        raise typer.Exit(code=5)
    except ValueError as exc:
        _echo(f"error: {exc}")
        raise typer.Exit(code=2)

    write_instances(instances, str(instances_path))

    _echo("")
    _echo("stage 3-4: instances")
    _echo(instances.summary())
    _report_warnings(instances.warnings,
                     "note(s) while populating the instance graph")
    _echo("")
    _echo(f"wrote {instances_path}")

    if no_render:
        _write_run_config(run_config_path, parameters,
                          [schema_path, instances_path])
        typer.echo(str(schema_path))
        typer.echo(str(instances_path))
        typer.echo(str(run_config_path))
        return

    # A second client, with JSON mode off: Stage 5 asks for LaTeX, and ollama's
    # "format": "json" would make the model return JSON instead of a document.
    latex_client = build_client(model=model, base_url=base_url,
                               backend=backend, seed=seed, json_mode=False,
                               max_tokens=6144)
    renderer = LaTeXRenderer(
        schema=graph, keep_tex=keep_tex, seed=seed,
        generator=LLMLaTeXGenerator(client=latex_client, schema=graph,
                                    layout_hint=layout_hint, seed=seed))
    engine = renderer.resolve_engine()
    if engine is None:
        _echo("")
        _echo("warning: no LaTeX engine on PATH (tried pdflatex, xelatex, "
              "lualatex); .tex sources will be written but nothing compiled.")
    try:
        pdfs = renderer.render_documents(
            instances, output_dir, layout_hint=layout_hint,
            max_retries=max_retries,
            layouts_per_graph=config.layouts_per_graph)
    except RenderError as exc:
        _echo(f"error: {exc}")
        raise typer.Exit(code=6)
    except ValueError as exc:
        _echo(f"error: {exc}")
        raise typer.Exit(code=2)

    manifest_path = output_dir / MANIFEST_FILENAME
    _echo("")
    _echo("stage 5: documents")
    _echo(renderer.summary())
    _report_warnings(renderer.warnings, "rendering note(s)")
    _echo("")
    _write_run_config(run_config_path, parameters,
                      [schema_path, instances_path, manifest_path] + list(pdfs),
                      _document_layouts(renderer.manifest))
    _echo(f"wrote {len(pdfs)} PDF(s), {manifest_path} and {run_config_path}")

    # stdout: the artefact paths, one per line, and nothing else.
    typer.echo(str(schema_path))
    typer.echo(str(instances_path))
    typer.echo(str(manifest_path))
    typer.echo(str(run_config_path))
    for pdf in pdfs:
        typer.echo(str(pdf))


def main(argv: Optional[list] = None) -> None:
    """Entry point. ``argv`` is accepted so tests can drive it directly."""
    app(args=argv)


if __name__ == "__main__":  # pragma: no cover - process entry point
    main()
