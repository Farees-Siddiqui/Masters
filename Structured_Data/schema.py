"""Data model for open-schema key-value extraction.

A *record* is one key-value pair that captures some piece of a document's
essence. The key is often *inferred* (nobody wrote "Architecture:" in the
text) while the value is usually a span that appears in the text. We keep an
`evidence` quote so every record is traceable back to the source, and a
coarse `confidence` so a downstream filter can rank.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Confidence(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class Record(BaseModel):
    key: str = Field(
        description=(
            "The semantic type of the value. May be inferred and need not appear "
            "verbatim in the text. Examples: 'dataset', 'architecture', 'metric', "
            "'date', 'publishing_number', 'category', 'organization'. Prefer short, "
            "lowercase, snake_case keys so they cluster cleanly across a corpus."
        )
    )
    value: str = Field(description="The value for this key, as a concise string.")
    evidence: str = Field(
        description=(
            "A short verbatim quote copied from the document that this record is "
            "grounded in. Must appear character-for-character in the source text."
        )
    )
    confidence: Confidence = Field(
        description="How confident you are that this is a correct, meaningful record."
    )


class Extraction(BaseModel):
    """A document's full set of extracted key-value records."""

    records: list[Record]
