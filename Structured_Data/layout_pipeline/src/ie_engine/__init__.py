"""Dynamic Information Extraction: discover a document's schema, don't impose one.

    blocks -> DynamicInformationExtractor -> DynamicDocument (any tag, any depth)

The tree deliberately has no declared fields beyond tag/attributes/text/children,
because a fixed model is exactly what this engine must not have.
"""

from .dynamic_extractor import DynamicInformationExtractor, render_blocks
from .llm_client import (SCHEMA_DISCOVERY_SYSTEM_PROMPT, LLMUnavailable,
                         LocalLLMClient, extract_json)
from .node_schema import (DynamicDocument, DynamicElement, document_from_json,
                          element_from_json, sanitize_tag)
from .relational_exporter import (RelationalExporter, Table,  # noqa: F401
                                  coerce_element, export_semantic_xml,
                                  sql_ident)

__all__ = [
    "DynamicInformationExtractor",
    "render_blocks",
    "LocalLLMClient",
    "LLMUnavailable",
    "SCHEMA_DISCOVERY_SYSTEM_PROMPT",
    "extract_json",
    "DynamicDocument",
    "DynamicElement",
    "document_from_json",
    "element_from_json",
    "sanitize_tag",
    "RelationalExporter",
    "Table",
    "coerce_element",
    "export_semantic_xml",
    "sql_ident",
]
