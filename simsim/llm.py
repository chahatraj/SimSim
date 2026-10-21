"""Model backends.

Every backend takes OpenAI-style chat messages and returns an `LLMResponse`.
One interface covers API models (OpenAI, Anthropic, Together, ...), locally
served open models (vLLM, SGLang, Ollama all speak the OpenAI protocol), and
`ScriptedLLM`, a fake model for free, deterministic tests.

All backends retry transient failures (rate limits, timeouts, server errors)
with exponential backoff, and count tokens (and dollars, if you pass `price`).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import threading
import time
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
    reasoning: str = ""  # thinking-model reasoning, if the server returns it separately
    finish_reason: str = ""  # "length" means the answer was cut off by max_tokens
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


def parse_arguments(raw: str | None) -> dict[str, Any]:
    """Tool-call arguments as a dict. Malformed JSON is kept under `_raw` so the tool can report it."""
    try:
        args = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {"_raw": raw}
    return args if isinstance(args, dict) else {"_raw": raw}


class LLM:
    """Base class. Subclasses implement `_chat` (and optionally `_retryable`).

    Args:
        temperature: sampling temperature sent with every call.
        max_tokens: cap on generated tokens per call.
        seed: makes runs reproducible; each request gets a seed derived from this one
            and the prompt (see `request_seed`).
        retries: attempts after the first on transient errors.
        price: (dollars per 1M input tokens, dollars per 1M output tokens), to track cost.
        max_concurrency: at most this many calls in flight at once (across threads).
        rpm: at most this many calls per minute (for API rate limits).
    """

    model: str = "unknown"

    def __init__(
        self,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        seed: int | None = None,
        retries: int = 6,
        price: tuple[float, float] | None = None,
        max_concurrency: int | None = None,
        rpm: float | None = None,
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.retries = retries
        self.price = price
        self.max_concurrency = max_concurrency
        self.rpm = rpm
        self._usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "retries": 0}
        self._init_sync()

    def _init_sync(self):
        self._lock = threading.Lock()
        self._seed_counts: dict[str, int] = {}
        self._slots = threading.BoundedSemaphore(self.max_concurrency) if self.max_concurrency else None
        self._next_slot = 0.0

    def __getstate__(self):
        state = self.__dict__.copy()
        for k in ("_lock", "_slots", "client", "_litellm"):
            state.pop(k, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._init_sync()
        if hasattr(self, "_connect"):
            self._connect()

    def request_seed(self, messages: list[Message]) -> int | None:
        """A per-request seed derived from `seed`, the prompt, and how often this exact prompt
        was sent before. Runs stay reproducible, but different calls get different random
        draws (sending one fixed seed with every call makes samples correlated)."""
        if self.seed is None:
            return None
        key = hashlib.sha256(json.dumps(messages, sort_keys=True, default=str).encode()).hexdigest()
        with self._lock:
            n = self._seed_counts.get(key, 0)
            self._seed_counts[key] = n + 1
        digest = hashlib.sha256(f"{self.seed}:{key}:{n}".encode()).hexdigest()
        return int(digest[:8], 16)

    def _wait_rate(self):
        if not self.rpm:
            return
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next_slot)
            self._next_slot = start + 60.0 / self.rpm
        if start > now:
            time.sleep(start - now)

    def chat(self, messages: list[Message], tools: list[dict] | None = None, **kwargs) -> LLMResponse:
        for attempt in range(self.retries + 1):
            try:
                self._wait_rate()
                if self._slots:
                    with self._slots:
                        resp = self._chat(messages, tools or None, **kwargs)
                else:
                    resp = self._chat(messages, tools or None, **kwargs)
                break
            except Exception as e:
                if attempt == self.retries or not self._retryable(e):
                    raise
                with self._lock:
                    self._usage["retries"] += 1
                time.sleep(min(60.0, 2.0**attempt) * (0.5 + random.random()))
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

    def choice_probs(self, messages: list[Message], options: list[str]) -> dict[str, float]:
        """Probability of each option as the model's next token (renormalized over `options`).

        Used for survey-style probes: label options "A", "B", ... or "1".."7". Needs a
        backend that returns token log-probs (OpenAI, vLLM); others raise NotImplementedError.
        """
        raise NotImplementedError(f"{type(self).__name__} does not return log-probs")

    @property
    def usage(self) -> dict[str, float]:
        u: dict[str, float] = dict(self._usage)
        if self.price:
            u["cost_usd"] = round(
                (u["prompt_tokens"] * self.price[0] + u["completion_tokens"] * self.price[1]) / 1e6, 6
            )
        return u

    def describe(self) -> dict[str, Any]:
        return {
            "backend": type(self).__name__,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
        }

    def _retryable(self, e: Exception) -> bool:
        return not isinstance(e, (TypeError, ValueError, KeyError, AttributeError, NotImplementedError))

    def _chat(self, messages, tools, **kwargs) -> LLMResponse:  # pragma: no cover
        raise NotImplementedError


def _usage_of(out) -> dict[str, int]:
    u = getattr(out, "usage", None)
    return {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens} if u else {}


def _reasoning_of(msg) -> str:
    # vLLM's reasoning parsers put thinking in `reasoning_content` (older) or `reasoning` (newer).
    return getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""


class OpenAIChat(LLM):
    """Any OpenAI-compatible endpoint: OpenAI, vLLM, SGLang, Ollama, Together, Anthropic's compat API.

    `extra_body` is sent with every request, e.g. {"chat_template_kwargs": {"enable_thinking": False}}
    to turn off thinking on Qwen3-style models served by vLLM.
    """

    def __init__(
        self, model: str, base_url: str | None = None, api_key: str | None = None,
        extra_body: dict | None = None, timeout: float = 300, **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.base_url = base_url
        self.extra_body = extra_body or {}
        self._api_key, self._timeout = api_key, timeout
        self._connect()

    def _connect(self):
        from openai import OpenAI

        # Retries happen in LLM.chat, so the client itself does not retry.
        self.client = OpenAI(
            base_url=self.base_url, api_key=self._api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
            max_retries=0, timeout=self._timeout,
        )

    def _params(self, messages, **kwargs) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model, "messages": messages,
            "temperature": self.temperature, "max_tokens": self.max_tokens,
        }
        if self.seed is not None:
            params["seed"] = self.request_seed(messages)
        if self.extra_body:
            params["extra_body"] = self.extra_body
        params.update(kwargs)
        return params

    def _chat(self, messages, tools, **kwargs) -> LLMResponse:
        params = self._params(messages, **kwargs)
        if tools:
            params["tools"] = tools
        out = self.client.chat.completions.create(**params)
        msg = out.choices[0].message
        calls = [
            ToolCall(id=c.id, name=c.function.name, arguments=parse_arguments(c.function.arguments))
            for c in (msg.tool_calls or [])
        ]
        return LLMResponse((msg.content or "").strip(), calls, _usage_of(out), _reasoning_of(msg),
                           out.choices[0].finish_reason or "", out)

    def choice_probs(self, messages, options):
        params = self._params(messages, max_tokens=1, logprobs=True, top_logprobs=20, temperature=0)
        out = self.chat_raw(params)
        top = out.choices[0].logprobs.content[0].top_logprobs
        probs = {o: 0.0 for o in options}
        for lp in top:
            tok = lp.token.strip()
            if tok in probs:
                probs[tok] += math.exp(lp.logprob)
        z = sum(probs.values())
        return {o: p / z for o, p in probs.items()} if z else probs

    def chat_raw(self, params: dict[str, Any]):
        """One raw API call with this backend's retry policy (for features beyond chat)."""
        for attempt in range(self.retries + 1):
            try:
                out = self.client.chat.completions.create(**params)
                usage = _usage_of(out)
                with self._lock:
                    self._usage["calls"] += 1
                    for k in ("prompt_tokens", "completion_tokens"):
                        self._usage[k] += usage.get(k, 0)
                return out
            except Exception as e:
                if attempt == self.retries or not self._retryable(e):
                    raise
                time.sleep(min(60.0, 2.0**attempt) * (0.5 + random.random()))

    def _retryable(self, e):
        import openai

        if isinstance(e, (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError)):
            return True
        if isinstance(e, openai.APIStatusError):
            return e.status_code >= 500 or e.status_code in (408, 409, 429)
        return False

    def describe(self):
        return {**super().describe(), "base_url": self.base_url, "extra_body": self.extra_body}


class LiteLLMChat(LLM):
    """Routes through LiteLLM (`pip install litellm`) for providers without an OpenAI-compatible API."""

    def __init__(self, model: str, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self._connect()

    def _connect(self):
        import litellm

        self._litellm = litellm

    def _chat(self, messages, tools, **kwargs) -> LLMResponse:
        params: dict[str, Any] = dict(
            model=self.model, messages=messages, temperature=self.temperature, max_tokens=self.max_tokens
        )
        if self.seed is not None:
            params["seed"] = self.request_seed(messages)
        if tools:
            params["tools"] = tools
        params.update(kwargs)
        out = self._litellm.completion(**params)
        msg = out.choices[0].message
        calls = [
            ToolCall(id=c.id, name=c.function.name, arguments=parse_arguments(c.function.arguments))
            for c in (msg.tool_calls or [])
        ]
        return LLMResponse((msg.content or "").strip(), calls, _usage_of(out), _reasoning_of(msg),
                           out.choices[0].finish_reason or "", out)


class ScriptedLLM(LLM):
    """A fake model for tests and dry runs. Costs nothing and never retries.

    `script` is either a list of responses (str or LLMResponse) returned in order,
    or a function `(messages, tools) -> str | LLMResponse`.
    """

    model = "scripted"

    def __init__(self, script: list | Callable[[list[Message], list[dict] | None], Any] | None = None, **kwargs):
        kwargs.setdefault("retries", 0)
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

    def choice_probs(self, messages, options):
        text = self.chat(messages).text.strip()
        return {o: float(o == text) for o in options}


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
    api_key = kwargs.pop("api_key", None) or (os.environ.get(key_env) if key_env else "EMPTY")
    if key_env and not api_key and not url:
        raise ValueError(f"{provider} needs an API key: set {key_env} (or pass api_key=...)")
    return OpenAIChat(model, base_url=url or default_url, api_key=api_key, **kwargs)
