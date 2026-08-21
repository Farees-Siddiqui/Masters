"""Synthetic structured-document benchmark generator.

    parameters -> ER schema -> records -> documents -> ground truth

Stages 1 to 5 are implemented.

:class:`ParametricSchemaGenerator` (Stages 1 & 2) asks the local Llama 3 server
for an Entity-Relationship graph in a requested domain, then enforces the
requested entity count and nesting depth on what comes back. The result is a
:class:`SchemaGraph`, serialised to ``schema.json`` as the contract every later
stage reads.

:class:`ParametricInstanceGenerator` (Stages 3 & 4) populates that schema
parent-before-child: the model supplies field values, while identifiers, joins,
nulls and injected orphans are all decided here off one seeded RNG, so the
resulting :class:`InstanceGraph` in ``instances.json`` is ground truth by
construction.

:class:`LaTeXRenderer` (Stage 5) renders that graph into compiled PDFs — one per
root record and its descendants — and writes ``benchmark_manifest.json``, which
states for each PDF the exact records, attribute values and foreign key tuples it
contains, along with the layout that PDF turned out to have. There is no list of
layouts: each document's is invented for it from the records it carries, or from
a freeform ``layout_hint`` describing the look wanted.

The LaTeX itself is written by the model, not by a template
(:class:`LLMLaTeXGenerator`), and repaired from the compiler's own error log when
it does not build. Because a model can alter a value as easily as it can format
one, the generated source is read back against the records before it is
compiled, and anything missing is reported per document in the manifest.
"""

from .config import BACKENDS, GeneratorConfig
from .instance_generator import (INSTANCE_GENERATION_SYSTEM_PROMPT,
                                 InstanceGenerationError,
                                 ParametricInstanceGenerator,
                                 topological_order, write_instances)
from .instance_types import InstanceGraph, Record
from .latex_generator import (AUTO_LAYOUT, LATEX_GENERATION_SYSTEM_PROMPT,
                              LATEX_REPAIR_SYSTEM_PROMPT,
                              LaTeXGenerationError, LLMLaTeXGenerator,
                              escape_latex, extract_latex, layout_declaration,
                              layout_directive, variant_directive,
                              leaked_examples,
                              missing_values, normalize_layout_hint)
from .renderer import (MANIFEST_FILENAME, CompileResult, DocumentScope,
                       LaTeXRenderer, RenderError, document_scopes)
from .schema_generator import (SCHEMA_GENERATION_SYSTEM_PROMPT,
                               ParametricSchemaGenerator,
                               SchemaGenerationError, build_user_prompt,
                               write_schema)
from .schema_types import (CARDINALITY, PRIMITIVE_TYPES, Attribute,
                           EntitySchema, Relationship, SchemaGraph,
                           SchemaValidationError, normalize_type, pascal_case,
                           snake_case)

__all__ = [
    "GeneratorConfig",
    "BACKENDS",
    "ParametricSchemaGenerator",
    "ParametricInstanceGenerator",
    "InstanceGenerationError",
    "INSTANCE_GENERATION_SYSTEM_PROMPT",
    "topological_order",
    "write_instances",
    "InstanceGraph",
    "Record",
    "LaTeXRenderer",
    "RenderError",
    "DocumentScope",
    "document_scopes",
    "escape_latex",
    "extract_latex",
    "missing_values",
    "leaked_examples",
    "CompileResult",
    "LLMLaTeXGenerator",
    "LaTeXGenerationError",
    "LATEX_GENERATION_SYSTEM_PROMPT",
    "LATEX_REPAIR_SYSTEM_PROMPT",
    "AUTO_LAYOUT",
    "layout_directive",
    "variant_directive",
    "layout_declaration",
    "normalize_layout_hint",
    "MANIFEST_FILENAME",
    "SchemaGenerationError",
    "SCHEMA_GENERATION_SYSTEM_PROMPT",
    "build_user_prompt",
    "write_schema",
    "Attribute",
    "EntitySchema",
    "Relationship",
    "SchemaGraph",
    "SchemaValidationError",
    "PRIMITIVE_TYPES",
    "CARDINALITY",
    "normalize_type",
    "snake_case",
    "pascal_case",
]
