"""Memory modules.

All memories share one small interface, so swapping memory is a one-line change
and can be an experimental condition:

    write(entry)   -> store what happened (called after every agent turn)
    read(query)    -> text placed in the agent's prompt for its next turn
    tools()        -> optional tools that let the agent manage memory itself
    dump() / load() -> save and restore the contents (agent state, checkpoints)

The environment decides what is *new* for an agent; memory decides what is
*remembered*. Keep those separate and you can vary one without touching the other.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .embed import Embedder
from .llm import LLM
from .tools import Tool, tool


@dataclass
class MemoryEntry:
    t: int
    text: str
    kind: str = "event"  # e.g. "turn", "probe", "whisper", "reflection"
    meta: dict = field(default_factory=dict)

    def render(self) -> str:
        stamp = self.meta.get("time") or f"t={self.t}"
        return f"[{stamp}] {self.text}"


class Memory:
    def write(self, entry: MemoryEntry) -> None:
        pass

    def read(self, query: str = "") -> str:
        return ""

    def peek(self, query: str = "") -> str:
        """Like `read`, but must not change the memory (used for probes)."""
        return self.read(query)

    def tools(self) -> list[Tool]:
        return []

    def state(self) -> dict:
        """Short summary for run logs."""
        return {"type": type(self).__name__}

    def dump(self) -> dict[str, Any]:
        """Full contents, JSON-serializable."""
        entries = getattr(self, "entries", None)
        return {"entries": [asdict(e) for e in entries]} if entries is not None else {}

    def load(self, data: dict[str, Any]) -> None:
        if "entries" in data:
            self.entries = [MemoryEntry(**e) for e in data["entries"]]


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

    def dump(self):
        return {"summary": self.summary, "buffer": [asdict(e) for e in self.buffer]}

    def load(self, data):
        self.summary = data.get("summary", "")
        self.buffer = [MemoryEntry(**e) for e in data.get("buffer", [])]


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

    def dump(self):
        return {"text": self.text}

    def load(self, data):
        self.text = data.get("text", "")
        self._save()


def _words(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", s.lower()))


def keyword_relevance(query: str, text: str) -> float:
    q, d = _words(query), _words(text)
    return len(q & d) / math.sqrt(len(q) * len(d)) if q and d else 0.0


IMPORTANCE_PROMPT = (
    "On a scale of 1 (mundane, e.g. brushing teeth) to 10 (poignant, e.g. a breakup), "
    "rate how important this memory is. Answer with one number.\n\nMemory: {text}\nRating:"
)


def llm_importance(llm: LLM) -> Callable[[str], float]:
    """Generative Agents' importance rating: the LLM scores each memory 1-10 (returned as 0.1-1)."""

    def score(text: str) -> float:
        m = re.search(r"\d+", llm.complete(IMPORTANCE_PROMPT.format(text=text), max_tokens=5))
        return min(max(int(m.group()) if m else 5, 1), 10) / 10

    return score


class ScoredMemory(Memory):
    """Generative Agents-style retrieval (Park et al., 2023).

    `read(query)` returns the top-k entries by a weighted sum of
        recency     decay ** (steps since the entry was last retrieved)
        importance  `importance(text)` in [0, 1]; with `llm`, the LLM rates 1-10
        relevance   cosine similarity of embeddings (with `embedder`), else keyword overlap
    Each component is min-max normalized over the candidates (as in the original),
    so the `weights` are comparable. Retrieved entries count as accessed.

    Args:
        k: entries returned.
        embedder: an `Embedder` for semantic relevance (recommended).
        llm: rates importance if `importance` is not given.
        importance: function `text -> [0, 1]`; default 0.5 for everything.
        decay: recency decay per step.
        weights: (recency, importance, relevance); Park et al. used (0.5, 3, 2).
    """

    def __init__(
        self,
        k: int = 8,
        embedder: Embedder | None = None,
        llm: LLM | None = None,
        importance: Callable[[str], float] | None = None,
        relevance: Callable[[str, str], float] | None = None,
        decay: float = 0.99,
        weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
        normalize: bool = True,
    ):
        self.k, self.decay, self.weights, self.normalize = k, decay, weights, normalize
        self.embedder = embedder
        self.importance_fn = importance or (llm_importance(llm) if llm else (lambda text: 0.5))
        self.relevance = relevance
        self.entries: list[MemoryEntry] = []
        self.now = 0

    def write(self, entry):
        entry.meta.setdefault("importance", self.importance_fn(entry.text))
        entry.meta.setdefault("last_access", entry.t)
        self.now = max(self.now, entry.t)
        self.entries.append(entry)
        if self.embedder:
            self.embedder.embed([entry.text])  # cache the vector now, not at read time

    def _relevance(self, query: str, entries: list[MemoryEntry]) -> list[float]:
        if not query:
            return [0.0] * len(entries)
        if self.relevance:
            return [self.relevance(query, e.text) for e in entries]
        if self.embedder:
            from .embed import cosine

            q, *vs = self.embedder.embed([query] + [e.text for e in entries])
            return [cosine(q, v) for v in vs]
        return [keyword_relevance(query, e.text) for e in entries]

    def retrieve(self, query: str = "", k: int | None = None, touch: bool = True) -> list[MemoryEntry]:
        if not self.entries:
            return []
        es = self.entries
        parts = [
            [self.decay ** (self.now - e.meta["last_access"]) for e in es],
            [e.meta["importance"] for e in es],
            self._relevance(query, es),
        ]
        if self.normalize:
            parts = [_minmax(p) for p in parts]
        scores = [sum(w * p[i] for w, p in zip(self.weights, parts)) for i in range(len(es))]
        top = [es[i] for i in sorted(range(len(es)), key=lambda i: scores[i], reverse=True)[: k or self.k]]
        if touch:
            for e in top:
                e.meta["last_access"] = self.now
        return sorted(top, key=lambda e: e.t)

    def read(self, query=""):
        return "\n".join(e.render() for e in self.retrieve(query))

    def peek(self, query=""):
        return "\n".join(e.render() for e in self.retrieve(query, touch=False))

    def state(self):
        return {"type": type(self).__name__, "n": len(self.entries)}

    def load(self, data):
        super().load(data)
        self.now = max((e.t for e in self.entries), default=0)


def _lines(text: str) -> list[str]:
    """Non-empty lines with list markers ("1.", "2)", "-", "*") removed."""
    return [re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s*", "", ln).strip() for ln in text.splitlines() if ln.strip()]


def _minmax(xs: list[float]) -> list[float]:
    lo, hi = min(xs), max(xs)
    return [0.5] * len(xs) if hi == lo else [(x - lo) / (hi - lo) for x in xs]


class ReflectiveMemory(ScoredMemory):
    """`ScoredMemory` plus Generative Agents-style reflection.

    When the summed importance of new entries passes `threshold`, the LLM asks
    itself `n_questions` high-level questions about recent memories, retrieves
    evidence for each, and writes `n_insights` insights back as memories
    (kind="reflection"), which can then be retrieved like any other memory.
    """

    QUESTIONS_PROMPT = (
        "Recent memories:\n{memories}\n\nGiven only these, what are the {n} most salient high-level "
        "questions we can answer about the subjects in them? One question per line, no numbering."
    )
    INSIGHTS_PROMPT = (
        "Memories:\n{memories}\n\nWhat {n} high-level insights can you infer from these memories? "
        "Write one insight per line, in the first person, no numbering."
    )

    def __init__(self, llm: LLM, threshold: float = 5.0, n_questions: int = 3, n_insights: int = 3,
                 recent: int = 50, **kwargs):
        kwargs.setdefault("llm", llm)
        super().__init__(**kwargs)
        self.reflect_llm, self.threshold = llm, threshold
        self.n_questions, self.n_insights, self.recent = n_questions, n_insights, recent
        self._budget = 0.0
        self._reflecting = False

    def write(self, entry):
        super().write(entry)
        if entry.kind == "reflection" or self._reflecting:
            return
        self._budget += entry.meta["importance"]
        if self._budget >= self.threshold:
            self._budget = 0.0
            self.reflect(entry.t)

    def reflect(self, t: int) -> list[str]:
        self._reflecting = True
        try:
            recent = "\n".join(e.render() for e in self.entries[-self.recent :])
            qs = _lines(self.reflect_llm.complete(self.QUESTIONS_PROMPT.format(memories=recent, n=self.n_questions)))
            insights: list[str] = []
            for q in qs[: self.n_questions]:
                evidence = "\n".join(e.render() for e in self.retrieve(q))
                lines = self.reflect_llm.complete(self.INSIGHTS_PROMPT.format(memories=evidence, n=self.n_insights))
                for line in _lines(lines)[: self.n_insights]:
                    insights.append(line)
                    self.write(MemoryEntry(t, line, kind="reflection", meta={"question": q}))
            return insights
        finally:
            self._reflecting = False
