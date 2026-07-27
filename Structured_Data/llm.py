"""Backend-agnostic structured extraction.

One function, `extract`, turns a (system, user) prompt into a validated
`Extraction`. It dispatches over three backends:

- groq       : hosted Llama 3.3 70B via Groq's free, OpenAI-compatible API
               (needs GROQ_API_KEY from https://console.groq.com)
- ollama     : a local model on your own GPU (needs Ollama running)
- anthropic  : Claude via the native SDK (needs ANTHROPIC_API_KEY, paid)

groq/ollama go through the OpenAI SDK in JSON mode and we validate the result
with pydantic; anthropic uses native structured outputs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from schema import Extraction


def _groq_key() -> str | None:
    """GROQ_API_KEY from the environment, falling back to GROQ_API_KEY.txt."""
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key.strip()
    key_file = Path(__file__).with_name("GROQ_API_KEY.txt")
    if key_file.exists():
        return key_file.read_text(encoding="utf-8").strip()
    return None

DEFAULT_MODEL = {
    "groq": "llama-3.3-70b-versatile",
    "ollama": "llama3.1:8b",
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


def _extract_openai(system: str, user: str, model: str, base_url: str, api_key: str) -> Extraction:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system + _JSON_INSTRUCTION},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0,
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


def extract(system: str, user: str, backend: str = "groq", model: str | None = None) -> Extraction:
    model = model or DEFAULT_MODEL[backend]
    if backend == "groq":
        key = _groq_key()
        if not key:
            raise RuntimeError(
                "No Groq key. Set GROQ_API_KEY or create GROQ_API_KEY.txt "
                "(free key at https://console.groq.com)"
            )
        return _extract_openai(system, user, model, "https://api.groq.com/openai/v1", key)
    if backend == "ollama":
        return _extract_openai(system, user, model, "http://localhost:11434/v1", "ollama")
    if backend == "anthropic":
        return _extract_anthropic(system, user, model)
    raise ValueError(f"Unknown backend: {backend!r}")
