"""Structured outputs: ask for JSON with a schema, validate it, and repair on failure.

A schema is a dict from field name to a spec:

    str, int, float, bool            a value of that type
    ["agree", "disagree"]            one of these strings
    (int, 1, 7)  /  (float, 0, 1)    a number in a range
    ("str", "what this field is")    any spec plus a description, as (spec, description)

    schema = {
        "analysis": (str, "your private reasoning; the other side never sees it"),
        "reply": str,
        "stance": ["support", "oppose", "neutral"],
        "confidence": (int, 1, 7),
    }

If the model's answer does not parse or validate, the error goes back to the
model and it tries again (`max_repairs` times), the same way Sotopia, AgentDojo
and tau2-bench repair bad outputs.
"""

from __future__ import annotations

import itertools
import json
import re
from typing import Any

from .llm import LLM, Message

Schema = dict[str, Any]


class StructuredOutputError(ValueError):
    pass


def _unpack(spec) -> tuple[Any, str]:
    if isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[1], str) and not isinstance(spec[0], str):
        return spec[0], spec[1]
    return spec, ""


def _describe_spec(spec) -> str:
    spec, _ = _unpack(spec)
    if isinstance(spec, list):
        return "one of " + ", ".join(json.dumps(s) for s in spec)
    if isinstance(spec, tuple):
        t, lo, hi = spec
        return f"{t.__name__} from {lo} to {hi}"
    return {str: "string", int: "integer", float: "number", bool: "true or false"}.get(spec, str(spec))


def format_instructions(schema: Schema) -> str:
    lines = []
    for name, spec in schema.items():
        _, desc = _unpack(spec)
        lines.append(f'  "{name}": {_describe_spec(spec)}' + (f"  // {desc}" if desc else ""))
    return "Respond with only a JSON object, no other text:\n{\n" + ",\n".join(lines) + "\n}"


def extract_json(text: str) -> dict[str, Any]:
    """The first JSON object in `text`, tolerating code fences, thinking tags and chatter around it."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    candidates = itertools.chain(
        [fence.group(1)] if fence else [], (_balanced(text, i) for i, ch in enumerate(text) if ch == "{")
    )
    for c in candidates:
        if not c:
            continue
        try:
            out = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(out, dict):
            return out
    raise StructuredOutputError("no JSON object found in the response")


def _balanced(text: str, start: int) -> str | None:
    """The brace-balanced substring starting at `start`, skipping braces inside strings."""
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def validate(data: dict[str, Any], schema: Schema) -> dict[str, Any]:
    out = {}
    for name, spec in schema.items():
        spec, _ = _unpack(spec)
        if name not in data:
            raise StructuredOutputError(f"missing field {name!r}")
        v = data[name]
        if isinstance(spec, list):
            match = next((s for s in spec if str(v).strip().lower() == str(s).lower()), None)
            if match is None:
                raise StructuredOutputError(f"{name!r} must be one of {spec}, got {v!r}")
            v = match
        elif isinstance(spec, tuple):
            t, lo, hi = spec
            try:
                v = t(v)
            except (TypeError, ValueError):
                raise StructuredOutputError(f"{name!r} must be a {t.__name__}, got {v!r}")
            if not lo <= v <= hi:
                raise StructuredOutputError(f"{name!r} must be between {lo} and {hi}, got {v}")
        elif spec is bool:
            if isinstance(v, str) and v.lower() in ("true", "false"):
                v = v.lower() == "true"
            if not isinstance(v, bool):
                raise StructuredOutputError(f"{name!r} must be true or false, got {v!r}")
        elif spec in (int, float):
            try:
                v = spec(v)
            except (TypeError, ValueError):
                raise StructuredOutputError(f"{name!r} must be a {spec.__name__}, got {v!r}")
        elif spec is str:
            v = v if isinstance(v, str) else json.dumps(v)
        out[name] = v
    return out


def parse(text: str, schema: Schema) -> dict[str, Any]:
    return validate(extract_json(text), schema)


def structured_chat(
    llm: LLM, messages: list[Message], schema: Schema, max_repairs: int = 2, add_instructions: bool = True
) -> tuple[dict[str, Any] | None, list[Message], list[str]]:
    """Call `llm` until its answer matches `schema`.

    Returns (parsed or None, the full message list including repair turns, error messages).
    """
    messages = list(messages)
    if add_instructions:
        last = dict(messages[-1])
        last["content"] = f"{last['content']}\n\n{format_instructions(schema)}"
        messages[-1] = last
    errors: list[str] = []
    for _ in range(max_repairs + 1):
        resp = llm.chat(messages)
        messages.append(resp.assistant_message())
        try:
            return parse(resp.text, schema), messages, errors
        except StructuredOutputError as e:
            errors.append(str(e))
            messages.append({
                "role": "user",
                "content": f"Your answer could not be used: {e}. {format_instructions(schema)}",
            })
    return None, messages, errors
