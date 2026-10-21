"""Model backends.

Every backend takes OpenAI-style chat messages and returns an `LLMResponse`.
One interface covers API models (OpenAI, Anthropic, Together, ...), locally
served open models (vLLM, SGLang, Ollama all speak the OpenAI protocol), and
`ScriptedLLM`, a fake model for free, deterministic tests.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

Message = dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None

    def assistant_message(self) -> Message:
        """The message to append to the history after this response."""
        msg: Message = {"role": "assistant", "content": self.text or None}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                }
                for c in self.tool_calls
            ]
        return msg


class LLM:
    """Base class. Subclasses implement `_chat`."""

    model: str = "unknown"

    def __init__(self, temperature: float = 0.7, max_tokens: int = 1024, seed: int | None = None):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self._usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self._lock = threading.Lock()

    def chat(self, messages: list[Message], tools: list[dict] | None = None, **kwargs) -> LLMResponse:
        resp = self._chat(messages, tools or None, **kwargs)
        with self._lock:
            self._usage["calls"] += 1
            self._usage["prompt_tokens"] += resp.usage.get("prompt_tokens", 0)
            self._usage["completion_tokens"] += resp.usage.get("completion_tokens", 0)
        return resp

    def complete(self, prompt: str, system: str | None = None, **kwargs) -> str:
        """Convenience for one-shot text calls (summaries, importance scores, judges)."""
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        return self.chat(messages, **kwargs).text

    @property
    def usage(self) -> dict[str, int]:
        return dict(self._usage)

    def describe(self) -> dict[str, Any]:
        return {
            "backend": type(self).__name__,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
        }

    def _chat(self, messages, tools, **kwargs) -> LLMResponse:  # pragma: no cover
        raise NotImplementedError


class OpenAIChat(LLM):
    """Any OpenAI-compatible endpoint: OpenAI, vLLM, SGLang, Ollama, Together, Anthropic's compat API."""

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None, **kwargs):
        super().__init__(**kwargs)
        from openai import OpenAI

        self.model = model
        self.base_url = base_url
        self.client = OpenAI(base_url=base_url, api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"))

    def _chat(self, messages, tools, **kwargs) -> LLMResponse:
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.seed is not None:
            params["seed"] = self.seed
        if tools:
            params["tools"] = tools
        params.update(kwargs)
        out = self.client.chat.completions.create(**params)
        choice = out.choices[0].message
        calls = []
        for c in choice.tool_calls or []:
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": c.function.arguments}
            calls.append(ToolCall(id=c.id, name=c.function.name, arguments=args))
        usage = {}
        if out.usage:
            usage = {"prompt_tokens": out.usage.prompt_tokens, "completion_tokens": out.usage.completion_tokens}
        return LLMResponse(text=choice.content or "", tool_calls=calls, usage=usage, raw=out)

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "base_url": self.base_url}


class LiteLLMChat(LLM):
    """Routes through LiteLLM (`pip install litellm`) for providers without an OpenAI-compatible API."""

    def __init__(self, model: str, **kwargs):
        super().__init__(**kwargs)
        import litellm

        self.model = model
        self._litellm = litellm

    def _chat(self, messages, tools, **kwargs) -> LLMResponse:
        params: dict[str, Any] = dict(
            model=self.model, messages=messages, temperature=self.temperature, max_tokens=self.max_tokens
        )
        if self.seed is not None:
            params["seed"] = self.seed
        if tools:
            params["tools"] = tools
        params.update(kwargs)
        out = self._litellm.completion(**params)
        choice = out.choices[0].message
        calls = [
            ToolCall(id=c.id, name=c.function.name, arguments=json.loads(c.function.arguments or "{}"))
            for c in (choice.tool_calls or [])
        ]
        usage = {}
        if getattr(out, "usage", None):
            usage = {"prompt_tokens": out.usage.prompt_tokens, "completion_tokens": out.usage.completion_tokens}
        return LLMResponse(text=choice.content or "", tool_calls=calls, usage=usage, raw=out)


class ScriptedLLM(LLM):
    """A fake model for tests and dry runs. Costs nothing.

    `script` is either a list of responses (str or LLMResponse) returned in order,
    or a function `(messages, tools) -> str | LLMResponse`.
    """

    model = "scripted"

    def __init__(self, script: list | Callable[[list[Message], list[dict] | None], Any] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.script = script if script is not None else (lambda m, t: "ok")
        self._i = 0
        self.calls: list[list[Message]] = []

    def _chat(self, messages, tools, **kwargs) -> LLMResponse:
        self.calls.append(list(messages))
        if callable(self.script):
            out = self.script(messages, tools)
        else:
            if self._i >= len(self.script):
                raise RuntimeError("ScriptedLLM ran out of scripted responses")
            out = self.script[self._i]
            self._i += 1
        return out if isinstance(out, LLMResponse) else LLMResponse(text=str(out))


_PROVIDERS = {
    "openai": (None, "OPENAI_API_KEY"),
    "anthropic": ("https://api.anthropic.com/v1/", "ANTHROPIC_API_KEY"),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "vllm": ("http://localhost:8000/v1", None),
    "ollama": ("http://localhost:11434/v1", None),
}


def make_llm(spec: str, **kwargs) -> LLM:
    """Build a backend from a short spec string.

        make_llm("openai:gpt-4o-mini")
        make_llm("anthropic:claude-haiku-4-5")
        make_llm("vllm:Qwen/Qwen2.5-7B-Instruct")                      # localhost:8000
        make_llm("vllm:Qwen/Qwen2.5-7B-Instruct@http://gpu012:8000/v1")
        make_llm("litellm:gemini/gemini-2.5-flash")
    """
    provider, _, rest = spec.partition(":")
    if not rest:
        raise ValueError(f"Expected 'provider:model', got {spec!r}")
    if provider == "litellm":
        return LiteLLMChat(rest, **kwargs)
    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown provider {provider!r}; choose from {sorted(_PROVIDERS) + ['litellm']}")
    model, _, url = rest.partition("@")
    default_url, key_env = _PROVIDERS[provider]
    api_key = os.environ.get(key_env) if key_env else "EMPTY"
    return OpenAIChat(model, base_url=url or default_url, api_key=api_key, **kwargs)
