"""Logging. Every LLM call, tool call, and agent turn becomes one JSON line."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


class Recorder:
    """Appends events to `path` (JSONL) and keeps them in memory as `events`.

    Pass `path=None` to keep events in memory only.
    """

    def __init__(self, path: str | Path | None = None, run_id: str = "run"):
        self.path = Path(path) if path else None
        self.run_id = run_id
        self.events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a")

    def log(self, event: str, **data: Any) -> None:
        rec = {"run": self.run_id, "event": event, "time": time.time(), **data}
        with self._lock:
            self.events.append(rec)
            if self.path:
                self._fh.write(json.dumps(rec, default=str) + "\n")
                self._fh.flush()

    def of(self, event: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["event"] == event]

    def close(self) -> None:
        if self.path and not self._fh.closed:
            self._fh.close()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
