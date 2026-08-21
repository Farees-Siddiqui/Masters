"""The resolved parameters of one generation run, in one object.

    CLI options -> GeneratorConfig -> run_config.json

Every stage of the pipeline is driven by arguments the CLI resolved, and until
now those arguments lived twice: once in ``generate``'s signature and once in a
hand-built ``parameters`` dict assembled beside it for ``run_config.json``. Two
copies of the same list is one copy too many — a parameter added to the flag and
forgotten in the dict is a run whose recorded configuration is quietly wrong,
and a wrong ``run_config.json`` is worse than a missing one, because it is
believed. :class:`GeneratorConfig` is the single list; the CLI builds one and
serialises it.

Plain ``dataclasses`` rather than pydantic, matching :mod:`schema_types` and the
rest of this repo's stdlib-only path. The validation wanted here is a handful of
range checks, and ``__post_init__`` does them in one pass; adding a dependency
to express ``ge=1`` would buy nothing the standard library does not already do.

Validation lives here rather than only in Typer's ``min=``/``max=`` because the
config is constructible without going through the CLI — the tests do it, and so
would any caller driving the pipeline as a library. A bound enforced only by the
argument parser is not enforced at all for them.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .llm_bridge import DEFAULT_BASE_URL, DEFAULT_MODEL

#: Wire protocols :mod:`llm_bridge` can speak. Checked here so a typo is a
#: message about the backend rather than a connection error against a URL that
#: was never going to answer.
BACKENDS = ("ollama", "openai")


@dataclass
class GeneratorConfig:
    """What one ``generate`` run was asked to do.

    Field order matches the order the flags are declared in the CLI, so the two
    can be read side by side.
    """

    # -- stages 1-2: schema ------------------------------------------------- #
    domain: str = "small_business"
    num_entities: int = 5
    max_depth: int = 2
    seed: Optional[int] = None

    # -- stages 3-4: records ------------------------------------------------ #
    records_per_entity: int = 5
    null_probability: float = 0.05
    orphan_rate: float = 0.0

    # -- stage 5: documents ------------------------------------------------- #
    layout_hint: str = "auto"
    #: How many visually distinct documents to render *per join subgraph*. The
    #: records are identical across the variants and only the layout differs,
    #: which is what makes a set of them a layout-invariance test: an extractor
    #: that recovers the same tree from all of them is reading the data, and one
    #: that recovers it from only the tabular variant has learnt the shape of a
    #: table. It is also the knob that scales the corpus predictably — a run of
    #: 20 subgraphs at 3 gives exactly 60 documents.
    layouts_per_graph: int = 1
    keep_tex: bool = False
    max_retries: int = 2

    # -- what to run, and where -------------------------------------------- #
    schema_only: bool = False
    no_render: bool = False
    output_dir: Path = field(default_factory=lambda: Path("out_benchmark"))

    # -- inference ---------------------------------------------------------- #
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    backend: str = "ollama"
    max_attempts: int = 3

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.layout_hint = _normalized_hint(self.layout_hint)

        # `ge=1` and friends, as one pass. Named bounds rather than a loop so a
        # failure says which parameter and what it was given.
        self._at_least("num_entities", 1)
        self._at_least("max_depth", 1)
        self._at_least("records_per_entity", 1)
        self._at_least("layouts_per_graph", 1)
        self._at_least("max_retries", 0)
        self._at_least("max_attempts", 1)
        self._within("null_probability", 0.0, 1.0)
        self._within("orphan_rate", 0.0, 1.0)

        if self.backend not in BACKENDS:
            raise ValueError(
                f"backend must be one of {', '.join(BACKENDS)}, got "
                f"{self.backend!r}")

    # -- validation helpers ------------------------------------------------- #
    def _at_least(self, name: str, minimum: int) -> None:
        value = getattr(self, name)
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer, got "
                            f"{type(value).__name__}")
        if value < minimum:
            raise ValueError(f"{name} must be >= {minimum}, got {value}")

    def _within(self, name: str, low: float, high: float) -> None:
        value = getattr(self, name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{name} must be a number, got "
                            f"{type(value).__name__}")
        if not low <= float(value) <= high:
            raise ValueError(f"{name} must be between {low} and {high}, got "
                             f"{value}")

    # -- derived ------------------------------------------------------------ #
    def documents_for(self, subgraphs: int) -> int:
        """How many documents ``subgraphs`` join subgraphs will produce.

        The whole point of ``layouts_per_graph`` being a multiplier rather than
        a target: the corpus size is known before the first model call.
        """
        return max(0, int(subgraphs)) * self.layouts_per_graph

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready, for ``run_config.json``. ``Path`` becomes its string."""
        payload = dataclasses.asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return payload


def _normalized_hint(layout_hint: Any) -> str:
    """``layout_hint`` as :mod:`latex_generator` wants it.

    Imported lazily: :mod:`latex_generator` pulls in the record and schema
    types, and a config object should stay cheap to construct.
    """
    from .latex_generator import normalize_layout_hint
    return normalize_layout_hint(layout_hint)


__all__ = ["BACKENDS", "GeneratorConfig"]
