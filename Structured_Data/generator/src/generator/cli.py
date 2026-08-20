"""Typer CLI for the data generation pipeline.

    python -m src.generator.cli generate --domain small_business --num-entities 4

``generate`` runs Stages 1 to 5 in one pass: ask local Llama 3 for an ER graph
and enforce the requested parameters on it, populate it with records, then have
the model write the LaTeX for each document and compile it — repairing the
source from pdflatex's own error log when it does not build. The artefacts are
``schema.json``, ``instances.json``, one PDF per document and
``benchmark_manifest.json``. ``--schema-only`` and ``--no-render`` stop early.

Everything human-readable goes to **stderr**, so the command composes: stdout
carries only the path of the artefact written, which means

    OUT=$(python -m src.generator.cli generate --domain medical)

leaves the summary on the terminal and the artefact paths in the variable.
"""

from __future__ import annotations

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
    from src.generator.latex_generator import LLMLaTeXGenerator
    from src.generator.renderer import (LAYOUT_STYLES, MANIFEST_FILENAME,
                                        LaTeXRenderer, RenderError)
    from src.generator.schema_generator import (ParametricSchemaGenerator,
                                                SchemaGenerationError,
                                                write_schema)
else:
    from .llm_bridge import (DEFAULT_BASE_URL, DEFAULT_MODEL, LLMUnavailable,
                             build_client)
    from .instance_generator import (InstanceGenerationError,
                                     ParametricInstanceGenerator,
                                     write_instances)
    from .latex_generator import LLMLaTeXGenerator
    from .renderer import (LAYOUT_STYLES, MANIFEST_FILENAME, LaTeXRenderer,
                           RenderError)
    from .schema_generator import (ParametricSchemaGenerator,
                                   SchemaGenerationError, write_schema)

SCHEMA_FILENAME = "schema.json"
INSTANCES_FILENAME = "instances.json"

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
    layout_style: str = typer.Option(
        "auto", "--layout-style",
        help="Layout the model is asked to write: auto, table, form or "
             "letter. 'auto' alternates between the layouts that suit each "
             "document's shape, so one run produces a mix — a corpus rendered "
             "every document the same way would not test layout invariance."),
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
    if layout_style not in LAYOUT_STYLES:
        _echo(f"error: unknown layout style {layout_style!r}; expected one of "
              f"{', '.join(LAYOUT_STYLES)}")
        raise typer.Exit(code=2)

    output_dir.mkdir(parents=True, exist_ok=True)
    schema_path = output_dir / SCHEMA_FILENAME
    instances_path = output_dir / INSTANCES_FILENAME

    _echo(f"domain={domain} entities={num_entities} max_depth={max_depth} "
          f"seed={seed if seed is not None else 'unset'}")
    if not schema_only:
        _echo(f"records_per_entity={records_per_entity} "
              f"null_probability={null_probability} "
              f"orphan_rate={orphan_rate}")
    if not schema_only and not no_render:
        _echo(f"layout_style={layout_style} keep_tex={keep_tex} "
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
        typer.echo(str(schema_path))
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
        typer.echo(str(schema_path))
        typer.echo(str(instances_path))
        return

    # A second client, with JSON mode off: Stage 5 asks for LaTeX, and ollama's
    # "format": "json" would make the model return JSON instead of a document.
    latex_client = build_client(model=model, base_url=base_url,
                               backend=backend, seed=seed, json_mode=False,
                               max_tokens=6144)
    renderer = LaTeXRenderer(
        schema=graph, keep_tex=keep_tex, seed=seed,
        generator=LLMLaTeXGenerator(client=latex_client, schema=graph,
                                    seed=seed))
    engine = renderer.resolve_engine()
    if engine is None:
        _echo("")
        _echo("warning: no LaTeX engine on PATH (tried pdflatex, xelatex, "
              "lualatex); .tex sources will be written but nothing compiled.")
    try:
        pdfs = renderer.render_documents(instances, output_dir,
                                        layout_style=layout_style,
                                        max_retries=max_retries)
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
    _echo(f"wrote {len(pdfs)} PDF(s) and {manifest_path}")

    # stdout: the artefact paths, one per line, and nothing else.
    typer.echo(str(schema_path))
    typer.echo(str(instances_path))
    typer.echo(str(manifest_path))
    for pdf in pdfs:
        typer.echo(str(pdf))


def main(argv: Optional[list] = None) -> None:
    """Entry point. ``argv`` is accepted so tests can drive it directly."""
    app(args=argv)


if __name__ == "__main__":  # pragma: no cover - process entry point
    main()
