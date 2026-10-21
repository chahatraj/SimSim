"""Memory modules.

All memories share one small interface, so swapping memory is a one-line change
and can be an experimental condition:

    write(entry)   -> store what happened (called after every agent turn)
    read(query)    -> text placed in the agent's prompt for its next turn
    tools()        -> optional tools that let the agent manage memory itself

The environment decides what is *new* for an agent; memory decides what is
*remembered*. Keep those separate and you can vary one without touching the other.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .llm import LLM
from .tools import Tool, tool


@dataclass
class MemoryEntry:
    t: int
    text: str
    kind: str = "event"  # e.g. "observation", "action", "reflection"
    meta: dict = field(default_factory=dict)

    def render(self) -> str:
        return f"[t={self.t}] {self.text}"


class Memory:
    def write(self, entry: MemoryEntry) -> None:
        pass

    def read(self, query: str = "") -> str:
        return ""

    def tools(self) -> list[Tool]:
        return []

    def state(self) -> dict:
        """Serializable snapshot, saved in run logs."""
        return {"type": type(self).__name__}


class NoMemory(Memory):
    """The agent only sees the current observation."""


class FullHistory(Memory):
    """Everything that ever happened, in order. Optionally truncated to the last `max_chars`."""

    def __init__(self, max_chars: int | None = None):
        self.entries: list[MemoryEntry] = []
        self.max_chars = max_chars

    def write(self, entry):
        self.entries.append(entry)

    def read(self, query=""):
        text = "\n".join(e.render() for e in self.entries)
        if self.max_chars and len(text) > self.max_chars:
            text = "...\n" + text[-self.max_chars :]
        return text

    def state(self):
        return {"type": "FullHistory", "n": len(self.entries)}


class WindowMemory(FullHistory):
    """Only the last `k` entries."""

    def __init__(self, k: int = 10):
        super().__init__()
        self.k = k

    def read(self, query=""):
        return "\n".join(e.render() for e in self.entries[-self.k :])


class SummaryMemory(Memory):
    """A rolling LLM-written summary plus the most recent raw entries."""

    PROMPT = (
        "Update this memory summary with the new events. Keep facts, commitments, and "
        "impressions of others; drop small talk. Under {words} words.\n\n"
        "Current summary:\n{summary}\n\nNew events:\n{events}\n\nUpdated summary:"
    )

    def __init__(self, llm: LLM, every: int = 5, words: int = 200):
        self.llm, self.every, self.words = llm, every, words
        self.summary = ""
        self.buffer: list[MemoryEntry] = []

    def write(self, entry):
        self.buffer.append(entry)
        if len(self.buffer) >= self.every:
            events = "\n".join(e.render() for e in self.buffer)
            self.summary = self.llm.complete(
                self.PROMPT.format(words=self.words, summary=self.summary or "(empty)", events=events)
            ).strip()
            self.buffer = []

    def read(self, query=""):
        recent = "\n".join(e.render() for e in self.buffer)
        return f"Summary:\n{self.summary or '(none yet)'}\n\nRecent:\n{recent or '(none)'}"

    def state(self):
        return {"type": "SummaryMemory", "summary": self.summary}


class MarkdownMemory(Memory):
    """A notes file the agent edits itself through tools (like Claude Code's memory.md).

    Nothing is written automatically; the agent decides what to keep. Pass `path`
    to persist the file across runs, or `initial` to seed it (e.g. a planted memory).
    """

    def __init__(self, path: str | Path | None = None, initial: str = ""):
        self.path = Path(path) if path else None
        if self.path and self.path.exists():
            self.text = self.path.read_text()
        else:
            self.text = initial

    def read(self, query=""):
        return self.text or "(memory file is empty)"

    def _save(self):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(self.text)

    def tools(self):
        @tool
        def memory_append(note: str) -> str:
            """Append a line to your memory file. Use it for anything you want to remember later."""
            self.text = (self.text.rstrip("\n") + "\n" if self.text else "") + note.strip() + "\n"
            self._save()
            return "saved"

        @tool
        def memory_replace(old: str, new: str) -> str:
            """Replace exact text in your memory file (use an empty `new` to delete it)."""
            if old not in self.text:
                return "Error: text not found in memory file"
            self.text = self.text.replace(old, new, 1)
            self._save()
            return "updated"

        return [memory_append, memory_replace]

    def state(self):
        return {"type": "MarkdownMemory", "text": self.text}


def _words(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", s.lower()))


def keyword_relevance(query: str, text: str) -> float:
    q, d = _words(query), _words(text)
    return len(q & d) / math.sqrt(len(q) * len(d)) if q and d else 0.0


class ScoredMemory(Memory):
    """Generative Agents-style retrieval (Park et al., 2023).

    Each entry gets an importance score; `read` returns the top-k entries by
    recency + importance + relevance to the query.

    - `importance`: a function `text -> float in [0, 1]`. If None and `llm` is
      given, the LLM rates importance 1-10; if neither, every entry scores 0.5.
    - `relevance`: a function `(query, text) -> float in [0, 1]`; defaults to
      keyword overlap. Pass an embedding-based scorer for semantic retrieval.
    """

    IMPORTANCE_PROMPT = (
        "On a scale of 1 (mundane, e.g. brushing teeth) to 10 (poignant, e.g. a breakup), "
        "rate how important this memory is. Answer with one number.\n\nMemory: {text}\nRating:"
    )

    def __init__(
        self,
        k: int = 8,
        llm: LLM | None = None,
        importance: Callable[[str], float] | None = None,
        relevance: Callable[[str, str], float] = keyword_relevance,
        decay: float = 0.99,
        weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        self.k, self.llm, self.decay, self.weights = k, llm, decay, weights
        self.importance_fn = importance
        self.relevance = relevance
        self.entries: list[MemoryEntry] = []

    def _importance(self, text: str) -> float:
        if self.importance_fn:
            return self.importance_fn(text)
        if self.llm:
            m = re.search(r"\d+", self.llm.complete(self.IMPORTANCE_PROMPT.format(text=text), max_tokens=5))
            return min(max(int(m.group()) if m else 5, 1), 10) / 10
        return 0.5

    def write(self, entry):
        entry.meta.setdefault("importance", self._importance(entry.text))
        self.entries.append(entry)

    def read(self, query=""):
        if not self.entries:
            return ""
        now = max(e.t for e in self.entries)
        wr, wi, wv = self.weights

        def score(e: MemoryEntry) -> float:
            recency = self.decay ** (now - e.t)
            return wr * recency + wi * e.meta["importance"] + wv * self.relevance(query, e.text)

        top = sorted(self.entries, key=score, reverse=True)[: self.k]
        return "\n".join(e.render() for e in sorted(top, key=lambda e: e.t))

    def state(self):
        return {"type": "ScoredMemory", "n": len(self.entries)}
