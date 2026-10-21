# Models

Every model backend takes OpenAI-style chat messages and returns an
`LLMResponse` (`text`, `tool_calls`, `usage`, `reasoning`, `finish_reason`).

```python
from simsim import make_llm
llm = make_llm("vllm:Qwen/Qwen3.5-9B@http://gpu012:8000/v1",
               temperature=0.7, max_tokens=512, seed=0,
               extra_body={"chat_template_kwargs": {"enable_thinking": False}},
               price=(0.15, 0.60), max_concurrency=32, rpm=None)
llm.complete("Say hi.")     # one-shot text
llm.usage                   # {"calls", "prompt_tokens", "completion_tokens", "retries", "cost_usd"}
```

| Argument | What it does |
|---|---|
| `temperature`, `max_tokens` | sampling settings sent with every call |
| `seed` | makes runs reproducible: each request gets its own seed derived from this one, the prompt, and how often that prompt was sent before |
| `retries` | transient errors (rate limits, timeouts, 5xx) are retried with exponential backoff; programming errors are not |
| `price` | `($ per 1M input tokens, $ per 1M output tokens)`; adds `cost_usd` to usage |
| `max_concurrency`, `rpm` | cap calls in flight and calls per minute, across all threads using this backend |
| `extra_body` | sent with every request (e.g. turn thinking off on Qwen3-style models) |

## Thinking models

With vLLM's reasoning parser (`--reasoning-parser qwen3`), a thinking model's
reasoning arrives separately and is stored in `turn.reasoning`. Thinking needs a
larger `max_tokens`; if the budget runs out before the answer, the turn records a
clear error ("response cut off at max_tokens ... thinking used the budget").
Surveys need thinking off, because they read the very first answer token.

## Caching

```python
from simsim import CachedLLM
with CachedLLM(make_llm("openai:gpt-4o-mini", seed=0), "cache/gpt4o-mini.sqlite") as llm:
    ...
```

The cache key is the full request. Identical requests are numbered in order, so
the n-th identical request gets the n-th stored answer: repeated sampling still
varies, and rerunning the same code replays the original run exactly. Change any
prompt or setting and those calls go to the model. Use it to add an evaluator or
fix analysis code and rerun without paying again.

## A fake model for tests

`ScriptedLLM` returns scripted answers, from a list or a function of the messages.
It costs nothing and is how SimSim's own tests run:

```python
from simsim import ScriptedLLM
llm = ScriptedLLM(lambda messages, tools: "I offer $150.")
```

## Your own backend

Subclass `LLM` and implement `_chat(messages, tools, **kwargs) -> LLMResponse`
(and optionally `choice_probs` for surveys and `_retryable` for retries).
