"""Data model for open-schema key-value extraction.

A *record* is one key-value pair that captures some piece of a document's
essence. The key is often *inferred* (nobody wrote "Architecture:" in the
text) while the value is a span that appears in the text.

Nothing here fixes what the keys may be. The extractor stays open-schema so an
unseen document type (an invoice, a discharge summary, a contract) produces its
own keys rather than being force-fit into a vocabulary written for papers. Two
things still make records comparable across runs:

- `raw_key` is whatever the model emitted; `key` is that string mapped onto a
  vocabulary *induced from the corpus* by `vocab.py`. The vocabulary is fitted
  from data, never authored — see that module for why.
- Every record resolves to **character spans** in the source. The model quotes
  evidence; `extract.resolve` locates that quote and derives the offsets. A
  record that cannot be located is not grounded, full stop — no fuzzy substring
  match papering over a hallucinated quote.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from enum import Enum


class Confidence(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class Record(BaseModel):
    """What the extraction model emits, before grounding or canonicalization."""

    key: str = Field(
        description=(
            "The semantic type of the value. May be inferred and need not appear "
            "verbatim in the text. Short, lowercase, snake_case, and reused for "
            "the same kind of thing so keys cluster cleanly across a corpus."
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


class Span(BaseModel):
    """One located mention of a value, as character offsets into the source."""

    start: int
    end: int
    text: str = Field(description="The exact source slice, so grounding is self-evident.")


class GroundedRecord(BaseModel):
    """A record after its evidence has been resolved to real source offsets.

    One record per distinct (key, value) *fact*; every place that fact appears
    is a `Span` in `spans`. Counting facts rather than mentions is what makes
    `record_count` a property of the document instead of a property of how
    talkative the model felt on a given chunk.
    """

    key: str = Field(description="Canonical key, or raw_key if no vocabulary applied.")
    raw_key: str = Field(description="Exactly what the model emitted. Never discarded.")
    value: str
    confidence: Confidence
    entity: str = Field(
        default="",
        description=(
            "What the record is *about*, when the document says so structurally — "
            "a table's row label. `{entity, key, value}` is one cell of a "
            "relational row, which is what lets records join across a corpus."
        ),
    )
    extractor: str = Field(
        default="llm",
        description="Which stage produced this record: 'table' (parsed) or 'llm' (inferred).",
    )
    spans: list[Span] = Field(default_factory=list)
    evidence: str = Field(default="", description="Representative evidence quote.")
    grounded_by: str = Field(
        default="none",
        description=(
            "How the span was located: value_in_evidence | value_in_document | "
            "evidence_only | none. Only the first two anchor the value itself; "
            "evidence_only means the value was never found in the source."
        ),
    )

    @property
    def grounded(self) -> bool:
        """Any span at all. Prefer `value_grounded` — this over-counts."""
        return bool(self.spans)

    @property
    def value_grounded(self) -> bool:
        """The strict claim: the value itself was located in the source."""
        return self.grounded_by in ("value_in_evidence", "value_in_document")

    @property
    def mentions(self) -> int:
        return len(self.spans)


class Extraction(BaseModel):
    """A document's full set of extracted key-value records."""

    records: list[Record]
