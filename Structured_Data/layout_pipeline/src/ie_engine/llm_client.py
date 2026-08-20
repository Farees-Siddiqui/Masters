"""Local LLM backend for schema discovery.

Talks to a model running on this machine's own GPUs. Two wire protocols:

* ``ollama``  — Ollama's native ``/api/chat`` (the default; a server is already
  running here with ``llama3.3:70b`` and ``llama3.1:8b`` pulled).
* ``openai``  — any OpenAI-compatible ``/v1/chat/completions``, which covers
  vLLM, TGI and llama.cpp's server, so swapping inference stacks is a flag.

Implemented on ``urllib`` from the standard library rather than the ``openai``
SDK or ``requests``. Neither is installed in any of this project's virtualenvs,
and both would have to be added to several of them; the JSON-over-HTTP contract
here is small enough that the dependency buys nothing.

Extraction runs locally on purpose, for the reason already recorded in the
repo's ``llm.py``: a hosted endpoint's served weights, quantisation and batching
can change without notice, and that shows up downstream as unexplained swings in
what gets extracted.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.environ.get("IE_MODEL", "llama3.3:70b")
DEFAULT_TIMEOUT = 600.0

#: The engine's whole premise lives in this prompt: no target schema is given,
#: because inventing one per document is the job.
#:
#: The nesting rules are the load-bearing part. Left to itself the model returns
#: the document's *surface* — an email flattens into one object of root-level
#: keys, a table into one object per row — so composite things arrive welded
#: into single strings ("14 Fake Street, Milton") and the tree that
#: ``element_from_json`` builds is one node deep. Naming the failure and showing
#: it beside the correct shape is what moves it; a bare "group related facts"
#: instruction did not.
#:
#: Every identifier in the examples was screened against the target corpora: a
#: word that appears in a document under extraction cannot appear here, or the
#: example stops demonstrating shape and starts supplying content.
SCHEMA_DISCOVERY_SYSTEM_PROMPT = """\
You are an information extraction engine. You are given the text of one \
document. Infer the structure that document actually has and return it as JSON.

There is no fixed schema. Do not force the content into any predefined shape. \
Read the document and decide for yourself what the entities, their attributes \
and their groupings are.

Output rules:
- Return ONE JSON object and nothing else. No prose, no markdown fences.
- Name the single top-level key after what the document *is*. Use snake_case.
- Use key names taken from the document's own vocabulary, not generic ones like \
"field1" or "value".
- Extract only what the text states. Do not infer, complete or invent values. \
Omit a key entirely rather than guessing it.
- Ignore conversational filler, greetings, sign-offs and boilerplate. Keep the \
facts they surround.
- Preserve values verbatim, including their units and formatting.

Nesting rules. Return a tree, not a flat list of keys. How the document happens \
to present a fact -- running prose, a table row, a form field, a letterhead -- \
must not change the tree you return; two documents stating the same facts in \
different layouts must come back with the same nesting.

1. A composite thing becomes its own nested object. When two or more facts are \
parts of one real-world thing, put them in a nested object named for that \
thing, one key per part. Do not lift those parts up onto the parent object, and \
never weld them into one string: several parts means several keys inside one \
object, never one joined value. Layout does not decide this -- a composite \
written on one line, in one cell, or as one uninterrupted run of words is still \
a composite, and is still split into its parts. If the parts of a value are \
separately meaningful, and especially if the document names those parts \
separately anywhere else, split them.
2. A thing that occurs more than once becomes an array of objects -- one object \
per occurrence, every occurrence carrying the same keys. Never number keys \
(thing_1, thing_2), never concatenate the occurrences into one value, and use \
an array even when only one occurrence is present.
3. A scalar fact stays a key/value pair on the object it describes -- the \
innermost object it belongs to, not the root. Only facts about the document as \
a whole belong on the top-level object.

Shape only, from an unrelated domain -- do not reuse these names.

RIGHT -- composites nested, repeats in an array, scalars on their own object. \
Note that "Ingrid Sorensen" reads as one run of words in the source and is \
still split, because its parts are separately meaningful:
{"shipping_manifest": {"vessel_name": "Kestrel", "voyage": "V-114",
 "master": {"forename": "Ingrid", "patronymic": "Sorensen"},
 "consignee": {"company": "Halvorsen Freight",
               "depot": {"quay": "42 Dock Road", "harbour": "Rotterdam"}},
 "clearance": {"tally": "88", "verdict": "Cleared"},
 "cargo": [{"container_id": "MSKU4412", "tonnage": "18.4"},
           {"container_id": "TGHU9087", "tonnage": "12.1"}]}}

WRONG -- the same facts, flattened: composites hoisted to the top level and \
welded into strings, the repeated thing collapsed into one value:
{"shipping_manifest": {"vessel_name": "Kestrel", "voyage": "V-114",
 "master": "Ingrid Sorensen", "consignee": "Halvorsen Freight",
 "depot": "42 Dock Road, Rotterdam", "clearance": "88 Cleared",
 "cargo": "MSKU4412, TGHU9087"}}
"""


class LLMUnavailable(RuntimeError):
    """The configured endpoint could not be reached."""


def extract_json(text: str) -> Optional[Any]:
    """Pull the first JSON value out of a model response.

    Even in JSON mode a model occasionally wraps output in a markdown fence or
    prefixes it with a sentence, so this is more forgiving than ``json.loads``:
    it strips fences, then falls back to scanning for the first balanced
    ``{...}`` or ``[...]``.
    """
    if not text or not text.strip():
        return None
    body = text.strip()

    fence = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```$", body, re.DOTALL)
    if fence:
        body = fence.group(1).strip()

    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass

    for opener, closer in (("{", "}"), ("[", "]")):
        start = body.find(opener)
        if start < 0:
            continue
        depth, in_str, escape = 0, False, False
        for i in range(start, len(body)):
            ch = body[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(body[start:i + 1])
                    except json.JSONDecodeError:
                        break
    return None


class LocalLLMClient:
    """Chat completion against a locally-served model.

    ``complete`` returns raw text; ``complete_json`` returns a parsed object or
    ``None``. Neither raises on a bad response — the extractor's contract is to
    degrade, not to abort a document because one call misbehaved. A genuinely
    unreachable endpoint does raise :class:`LLMUnavailable`, because that is a
    configuration error rather than a bad generation.
    """

    def __init__(self, base_url: str = DEFAULT_BASE_URL,
                 model: str = DEFAULT_MODEL,
                 backend: str = "ollama",
                 temperature: float = 0.0,
                 timeout: float = DEFAULT_TIMEOUT,
                 json_mode: bool = True,
                 num_ctx: Optional[int] = 8192,
                 api_key: Optional[str] = None,
                 max_tokens: Optional[int] = None):
        if backend not in ("ollama", "openai"):
            raise ValueError(f"unknown backend {backend!r}")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.backend = backend
        # Zero temperature: a schema that changes between identical runs is not
        # a schema.
        self.temperature = temperature
        self.timeout = timeout
        self.json_mode = json_mode
        self.num_ctx = num_ctx
        self.api_key = api_key or os.environ.get("IE_API_KEY")
        self.max_tokens = max_tokens
        self.last_error: Optional[str] = None

    # -- transport ---------------------------------------------------------- #
    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(url, data=data, headers=headers,
                                     method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise LLMUnavailable(f"HTTP {exc.code} from {url}: {detail}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise LLMUnavailable(f"cannot reach {url}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise LLMUnavailable(f"non-JSON response from {url}: {exc}") from exc

    def _body(self, system: str, user: str) -> tuple:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        if self.backend == "ollama":
            payload: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": self.temperature},
            }
            if self.num_ctx:
                payload["options"]["num_ctx"] = self.num_ctx
            if self.max_tokens:
                payload["options"]["num_predict"] = self.max_tokens
            if self.json_mode:
                payload["format"] = "json"
            return "/api/chat", payload

        payload = {"model": self.model, "messages": messages,
                   "temperature": self.temperature, "stream": False}
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        return "/v1/chat/completions", payload

    @staticmethod
    def _content(response: dict) -> str:
        # Ollama: {"message": {"content": ...}} / OpenAI: {"choices":[{"message":...}]}
        msg = response.get("message")
        if isinstance(msg, dict) and msg.get("content") is not None:
            return msg["content"]
        choices = response.get("choices") or []
        if choices:
            inner = choices[0].get("message") or {}
            if inner.get("content") is not None:
                return inner["content"]
            if choices[0].get("text") is not None:
                return choices[0]["text"]
        return response.get("response") or ""

    # -- api ----------------------------------------------------------------- #
    def complete(self, system: str, user: str) -> str:
        """Raw text from the model. Empty string if the call produced nothing."""
        self.last_error = None
        path, payload = self._body(system, user)
        response = self._post(path, payload)
        return self._content(response) or ""

    def complete_json(self, system: str, user: str) -> Optional[Any]:
        """Parsed JSON, or ``None`` if the model returned nothing usable."""
        try:
            text = self.complete(system, user)
        except LLMUnavailable as exc:
            self.last_error = str(exc)
            log.warning("LLM unavailable: %s", exc)
            return None
        if not text.strip():
            self.last_error = "model returned empty output"
            return None
        parsed = extract_json(text)
        if parsed is None:
            self.last_error = f"unparseable model output: {text[:200]!r}"
            log.warning("%s", self.last_error)
        return parsed

    # -- diagnostics ---------------------------------------------------------- #
    def is_available(self) -> bool:
        """Whether the endpoint answers at all. Never raises."""
        url = f"{self.base_url}/api/tags" if self.backend == "ollama" \
            else f"{self.base_url}/v1/models"
        try:
            with urllib.request.urlopen(url, timeout=min(self.timeout, 10)):
                return True
        except Exception as exc:  # noqa: BLE001 - a probe must not throw
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False

    def available_models(self) -> list:
        try:
            url = f"{self.base_url}/api/tags" if self.backend == "ollama" \
                else f"{self.base_url}/v1/models"
            with urllib.request.urlopen(url, timeout=min(self.timeout, 10)) as r:
                payload = json.loads(r.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            return []
        if "models" in payload:
            return [m.get("name") or m.get("model") for m in payload["models"]]
        return [m.get("id") for m in payload.get("data", [])]

    def __repr__(self) -> str:
        return (f"LocalLLMClient(backend={self.backend!r}, model={self.model!r}, "
                f"base_url={self.base_url!r})")
