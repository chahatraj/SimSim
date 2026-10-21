from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..agent import Agent, Turn
from ..tools import Tool
from .base import Environment


@dataclass
class Task:
    """One episode for a single agent.

    `score(turn, world)` returns a dict of metrics (e.g. {"success": 1.0}); `world`
    is the shared mutable state that tools read and write.
    """

    prompt: str
    score: Callable[[Turn, dict], dict[str, Any]] | None = None
    id: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


class TaskSequence(Environment):
    """One agent works through tasks in order; its memory carries across tasks.

    This is the usual setup for memory experiments: earlier tasks plant
    information, later tasks test whether (and how) the agent uses it.

    Args:
        tasks: the episodes, in order.
        tools: functions `world -> list[Tool]` or a plain list of tools.
        world: initial shared state (a dict) that tools mutate.
    """

    def __init__(
        self,
        tasks: list[Task],
        tools: Callable[[dict], list[Tool]] | list[Tool] = (),
        world: dict | None = None,
    ):
        self.tasks = tasks
        self._tools = tools
        self.initial_world = world or {}

    def reset(self, agents, rng):
        super().reset(agents, rng)
        import copy

        self.world = copy.deepcopy(self.initial_world)
        self.tool_list = self._tools(self.world) if callable(self._tools) else list(self._tools)
        self.i = 0
        self.results: list[dict[str, Any]] = []

    def observe(self, agent: Agent, t: int) -> str | None:
        return self.tasks[self.i].prompt if self.i < len(self.tasks) else None

    def tools_for(self, agent, t):
        return self.tool_list

    def apply(self, turn: Turn, t: int) -> None:
        task = self.tasks[self.i]
        scores = task.score(turn, self.world) if task.score else {}
        self.results.append({"task": task.id or self.i, "response": turn.text, "n_tool_calls": len(turn.tool_calls), **scores})
        self.i += 1

    def done(self, t: int) -> bool:
        return self.i >= len(self.tasks)

    def metrics(self) -> dict[str, Any]:
        numeric = {}
        for r in self.results:
            for k, v in r.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "task":
                    numeric.setdefault(k, []).append(v)
        means = {f"mean_{k}": sum(v) / len(v) for k, v in numeric.items()}
        return {**means, "tasks": self.results}
