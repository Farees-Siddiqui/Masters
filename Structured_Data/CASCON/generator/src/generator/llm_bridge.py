"""Reuse ``layout_pipeline``'s ``LocalLLMClient`` from this package.

The extraction side of this repo already has a stdlib-only client for the local
Ollama server (``layout_pipeline/src/ie_engine/llm_client.py``): backends for
``/api/chat`` and ``/v1/chat/completions``, forgiving JSON extraction, and an
``is_available`` probe. Generation talks to the same server, so it uses the same
client rather than a second copy that can drift from it.

Importing it is the awkward part: ``layout_pipeline`` is not an installed
package and has no top-level ``__init__.py``, and its inner package is *also*
called ``src``, which is the name this package's own root already occupies. So
the module is loaded from its file path under a private name instead of being
put on ``sys.path``, which would shadow ``src.generator``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from types import ModuleType
from typing import Any, Dict, Optional, Tuple

#: Where ``llm_client.py`` sits inside ``Structured_Data/``.
_CLIENT_RELPATH = os.path.join("layout_pipeline", "src", "ie_engine",
                               "llm_client.py")


def _find_structured_data() -> str:
    """Walk up from this file to the ancestor that holds ``layout_pipeline/``.

    This package used to sit directly inside ``Structured_Data/``, so a fixed
    four-level climb found it. It now lives under ``CASCON/``, and searching
    upwards keeps the import working wherever the package is moved next.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.isfile(os.path.join(here, _CLIENT_RELPATH)):
            return here
        parent = os.path.dirname(here)
        if parent == here:  # filesystem root: let the caller report the miss
            return ""
        here = parent


_STRUCTURED_DATA = _find_structured_data()
_CLIENT_PATH = os.path.join(_STRUCTURED_DATA, _CLIENT_RELPATH)
#: Private module name: not importable as ``src.ie_engine.llm_client`` here, and
#: must not collide with anything the host process already imported.
_MODULE_NAME = "_generator_vendored_llm_client"

_module: Optional[ModuleType] = None


def _load_client_module() -> ModuleType:
    """Load ``llm_client.py`` once, by path, and cache it in ``sys.modules``."""
    global _module
    if _module is not None:
        return _module
    if _MODULE_NAME in sys.modules:
        _module = sys.modules[_MODULE_NAME]
        return _module

    # If the extraction package happens to be importable normally (running from
    # inside layout_pipeline, or installed), prefer that over the path load so
    # there is only one copy of the class in the process.
    try:  # pragma: no cover - depends on the host process's sys.path
        from src.ie_engine import llm_client as already_importable  # type: ignore
        _module = already_importable
        return _module
    except Exception:  # noqa: BLE001 - absence is the normal case
        pass

    if not os.path.isfile(_CLIENT_PATH):
        raise ImportError(
            f"cannot find LocalLLMClient at {_CLIENT_PATH!r}. The generator "
            "expects to live beside layout_pipeline/ inside Structured_Data/.")
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _CLIENT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load a module spec for {_CLIENT_PATH!r}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    _module = module
    return module


_client_module = _load_client_module()

LocalLLMClient = _client_module.LocalLLMClient
LLMUnavailable = _client_module.LLMUnavailable
extract_json = _client_module.extract_json
DEFAULT_BASE_URL = _client_module.DEFAULT_BASE_URL
DEFAULT_TIMEOUT = _client_module.DEFAULT_TIMEOUT

#: Generation is a long single call rather than many short ones, and the schema
#: prompt is small, so the extraction default (llama3.3:70b) is kept but the
#: context is sized for one ER graph plus its instructions.
DEFAULT_MODEL = os.environ.get("GEN_MODEL",
                               os.environ.get("IE_MODEL", "llama3.3:70b"))
DEFAULT_NUM_CTX = 8192
DEFAULT_MAX_TOKENS = 4096
#: With no ``--seed`` the generator is *meant* to vary: a benchmark corpus of
#: identical schemas measures nothing. A seed pins it back to greedy decoding.
DEFAULT_TEMPERATURE = 0.4


class SeededLLMClient(LocalLLMClient):  # type: ignore[misc,valid-type]
    """``LocalLLMClient`` plus a decoder seed.

    ``--seed`` has to reach the sampler, and the base client's payload builder
    has no hook for extra options, so the payload is amended here instead of
    being reinvented. Ollama takes ``options.seed``; OpenAI-compatible servers
    take a top-level ``seed``.
    """

    def __init__(self, *args: Any, seed: Optional[int] = None,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.seed = seed

    def _body(self, system: str, user: str) -> Tuple[str, Dict[str, Any]]:
        path, payload = super()._body(system, user)
        if self.seed is None:
            return path, payload
        if self.backend == "ollama":
            payload.setdefault("options", {})["seed"] = int(self.seed)
        else:
            payload["seed"] = int(self.seed)
        return path, payload

    def __repr__(self) -> str:
        return (f"SeededLLMClient(backend={self.backend!r}, "
                f"model={self.model!r}, base_url={self.base_url!r}, "
                f"seed={self.seed!r})")


def build_client(model: str = DEFAULT_MODEL,
                 base_url: str = DEFAULT_BASE_URL,
                 backend: str = "ollama",
                 seed: Optional[int] = None,
                 temperature: Optional[float] = None,
                 timeout: float = DEFAULT_TIMEOUT,
                 num_ctx: int = DEFAULT_NUM_CTX,
                 max_tokens: int = DEFAULT_MAX_TOKENS,
                 json_mode: bool = True) -> SeededLLMClient:
    """A client aimed at the local Llama 3 server, seeded when asked.

    A seed implies determinism, which means greedy decoding as well: a fixed
    seed at temperature 0.4 still wanders if the server's batching changes, so
    ``temperature`` drops to 0.0 unless the caller overrides it explicitly.

    ``json_mode`` must be turned **off** for anything that is not JSON. Stages 1
    to 4 ask for JSON and want it enforced; Stage 5 asks for LaTeX, and ollama's
    ``"format": "json"`` would make the model wrap the source in a JSON string
    or, worse, emit a JSON object instead of a document.
    """
    if temperature is None:
        temperature = 0.0 if seed is not None else DEFAULT_TEMPERATURE
    return SeededLLMClient(base_url=base_url, model=model, backend=backend,
                           temperature=temperature, timeout=timeout,
                           json_mode=json_mode, num_ctx=num_ctx,
                           max_tokens=max_tokens, seed=seed)


__all__ = [
    "LocalLLMClient",
    "LLMUnavailable",
    "SeededLLMClient",
    "build_client",
    "extract_json",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TIMEOUT",
]
