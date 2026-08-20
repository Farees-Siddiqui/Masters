"""Backend-agnostic structured extraction.

One function, `extract`, turns a (system, user) prompt into a validated
`Extraction`. It dispatches over two backends:

- ollama     : a local model on your own GPU (needs Ollama running). The
               default, and the only one used for real runs.
- anthropic  : Claude via the native SDK (needs ANTHROPIC_API_KEY, paid)

ollama goes through the OpenAI SDK in JSON mode and we validate the result with
pydantic; anthropic uses native structured outputs.

Extraction runs **locally on purpose**. A hosted endpoint adds variance nothing
downstream can account for — the served weights, quantization and batching are
all outside our control and can change between runs without notice, which shows
up as unexplained swings in record counts. Local inference pins every one of
those. It also removes daily token caps, which silently truncated runs.
"""

from __future__ import annotations

import json

from schema import Extraction

DEFAULT_MODEL = {
    "ollama": "llama3.3:70b",
    "anthropic": "claude-opus-4-8",
}

# Appended to the system prompt for JSON-mode backends so the model knows the
# exact shape to emit (the word "JSON" must appear for OpenAI-style JSON mode).
_JSON_INSTRUCTION = """\

Return ONLY a JSON object, no prose and no markdown fences, of the form:
{"records": [
  {"key": "<snake_case type>", "value": "<value>", "evidence": "<verbatim quote>", "confidence": "high|medium|low"}
]}
"""

_VALID_CONFIDENCE = {"high", "medium", "low"}


def _sanitize(raw: dict) -> dict:
    """Coerce a loosely-typed model response into our schema's shape."""
    records = raw.get("records", raw if isinstance(raw, list) else [])
    if isinstance(records, dict):  # some models wrap a single record
        records = [records]
    clean = []
    for r in records:
        if not isinstance(r, dict) or "key" not in r or "value" not in r:
            continue
        conf = str(r.get("confidence", "medium")).strip().lower()
        clean.append(
            {
                "key": str(r["key"]),
                "value": str(r["value"]),
                "evidence": str(r.get("evidence", "")),
                "confidence": conf if conf in _VALID_CONFIDENCE else "medium",
            }
        )
    return {"records": clean}


def _extract_ollama(system: str, user: str, model: str, base_url: str) -> Extraction:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="ollama")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system + _JSON_INSTRUCTION},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=8000,
        # Ollama defaults to OLLAMA_CONTEXT_LENGTH (4096) regardless of what the
        # model supports. A ~2.2k-token chunk plus the system prompt leaves under
        # 1.9k for output, so dense chunks had their record list silently closed
        # early — invisible in the JSON, visible only as a low record count. Ask
        # for the headroom explicitly.
        extra_body={"options": {"num_ctx": 32768}},
    )
    raw = json.loads(resp.choices[0].message.content)
    return Extraction.model_validate(_sanitize(raw))


def _extract_anthropic(system: str, user: str, model: str) -> Extraction:
    import anthropic

    client = anthropic.Anthropic()
    resp = client.messages.parse(
        model=model,
        max_tokens=8000,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_format=Extraction,
    )
    return resp.parsed_output or Extraction(records=[])


OLLAMA_URL = "http://localhost:11434/v1"


def extract(system: str, user: str, backend: str = "ollama", model: str | None = None) -> Extraction:
    model = model or DEFAULT_MODEL[backend]
    if backend == "ollama":
        return _extract_ollama(system, user, model, OLLAMA_URL)
    if backend == "anthropic":
        return _extract_anthropic(system, user, model)
    raise ValueError(f"Unknown backend: {backend!r}")
