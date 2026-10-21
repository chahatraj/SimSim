"""Response caching: rerun a simulation without paying for (or waiting on) model calls.

    llm = CachedLLM(make_llm("openai:gpt-4o-mini", seed=0), "cache/gpt4o-mini.sqlite")

The cache key is the full request (model, messages, tools, sampling settings).
Identical requests are numbered in the order they occur, so the n-th identical
request gets the n-th stored response: repeated sampling still gives different
samples, and rerunning the same code replays the original run exactly. Change
anything (a prompt, the seed, the temperature) and those calls go to the model.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections import Counter
from pathlib import Path
from typing import Any

from .llm import LLM, LLMResponse, ToolCall


class CachedLLM(LLM):
    """Wraps any backend with a persistent SQLite cache (safe across threads and processes)."""

    def __init__(self, llm: LLM, path: str | Path, read: bool = True, write: bool = True):
        super().__init__(temperature=llm.temperature, max_tokens=llm.max_tokens, seed=llm.seed,
                         retries=0, price=llm.price)
        self.inner, self.path, self.read, self.write = llm, Path(path), read, write
        self.model = llm.model
        self.hits = 0
        self._seen: Counter[str] = Counter()
        self._db_lock = threading.Lock()
        self._open()

    def _open(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False, timeout=60)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, response TEXT)")
        self._db.commit()

    def _key(self, messages, tools, kwargs) -> str:
        req = {"llm": self.inner.describe(), "messages": messages, "tools": tools, "kwargs": kwargs}
        base = hashlib.sha256(json.dumps(req, sort_keys=True, default=str).encode()).hexdigest()
        with self._db_lock:
            self._seen[base] += 1
            return f"{base}:{self._seen[base]}"

    def _chat(self, messages, tools, **kwargs) -> LLMResponse:
        key = self._key(messages, tools, kwargs)
        if self.read:
            with self._db_lock:
                row = self._db.execute("SELECT response FROM responses WHERE key = ?", (key,)).fetchone()
            if row:
                self.hits += 1
                d = json.loads(row[0])
                d["tool_calls"] = [ToolCall(**c) for c in d["tool_calls"]]
                return LLMResponse(**d)
        resp = self.inner.chat(messages, tools, **kwargs)
        if self.write:
            d = {"text": resp.text, "tool_calls": [c.__dict__ for c in resp.tool_calls], "usage": resp.usage,
                 "reasoning": resp.reasoning, "finish_reason": resp.finish_reason}
            with self._db_lock:
                self._db.execute("INSERT OR REPLACE INTO responses VALUES (?, ?)", (key, json.dumps(d)))
                self._db.commit()
        return resp

    def choice_probs(self, messages, options):
        key = self._key(messages, None, {"choice_probs": options})
        with self._db_lock:
            row = self._db.execute("SELECT response FROM responses WHERE key = ?", (key,)).fetchone() if self.read else None
        if row:
            self.hits += 1
            return json.loads(row[0])
        probs = self.inner.choice_probs(messages, options)
        if self.write:
            with self._db_lock:
                self._db.execute("INSERT OR REPLACE INTO responses VALUES (?, ?)", (key, json.dumps(probs)))
                self._db.commit()
        return probs

    def close(self) -> None:
        """Close the database (also done automatically when used as a context manager)."""
        with self._db_lock:
            self._db.close()

    def __enter__(self) -> CachedLLM:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def usage(self) -> dict[str, Any]:
        """Tokens actually sent to the model (cache hits are free), plus the hit count."""
        return {**self.inner.usage, "cache_hits": self.hits}

    def describe(self):
        return {**self.inner.describe(), "cache": str(self.path)}

    def __getstate__(self):
        state = super().__getstate__()
        state.pop("_db", None)
        state.pop("_db_lock", None)
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._db_lock = threading.Lock()
        self._open()
