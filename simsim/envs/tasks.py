from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..agent import Agent, Turn
from ..tools import Tool
from .base import Environment


@dataclass
class Task:
    """One episode.

    `score(turn, world)` returns a dict of metrics (e.g. {"success": 1.0}); `world`
    is the agent's mutable state that tools read and write.
    """

    prompt: str
    score: Callable[[Turn, dict], dict[str, Any]] | None = None
    id: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


class TaskSequence(Environment):
    """Each agent works through the tasks in order; its memory carries across tasks.

    This is the usual setup for memory experiments: earlier tasks plant
    information, later tasks test whether (and how) the agent uses it. With
    several agents, each gets its own copy of the world and its own progress.

    Args:
        tasks: the episodes, in order.
        tools: a function `world -> list[Tool]` (so tools can mutate the world)
            or a plain list of tools.
        world: initial state (a dict), deep-copied per agent.
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
        self.worlds: dict[str, dict] = {}
        self.tool_lists: dict[str, list[Tool]] = {}
        self.progress: dict[str, int] = {}
        self.results: list[dict[str, Any]] = []
        for a in agents:
            self.add_agent(a)

    def add_agent(self, agent):
        super().add_agent(agent)
        world = copy.deepcopy(self.initial_world)
        self.worlds[agent.name] = world
        self.tool_lists[agent.name] = self._tools(world) if callable(self._tools) else list(self._tools)
        self.progress[agent.name] = 0

    @property
    def world(self) -> dict:
        """The world of the only agent (convenience for single-agent runs)."""
        if len(self.worlds) != 1:
            raise AttributeError("several agents: use env.worlds[name]")
        return next(iter(self.worlds.values()))

    def observe(self, agent: Agent, t: int) -> str | None:
        i = self.progress[agent.name]
        return self.tasks[i].prompt if i < len(self.tasks) else None

    def tools_for(self, agent, t):
        return self.tool_lists[agent.name]

    def apply(self, turn: Turn, t: int) -> None:
        i = self.progress[turn.agent]
        task = self.tasks[i]
        scores = task.score(turn, self.worlds[turn.agent]) if task.score else {}
        self.results.append({
            "agent": turn.agent, "task": task.id or i, "response": turn.text,
            "n_tool_calls": len(turn.tool_calls), **scores,
        })
        self.progress[turn.agent] = i + 1
        if all(p >= len(self.tasks) for p in self.progress.values()):
            self.stop("all tasks done")

    def metrics(self) -> dict[str, Any]:
        numeric: dict[str, list[float]] = {}
        for r in self.results:
            for k, v in r.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "task":
                    numeric.setdefault(k, []).append(v)
        means = {f"mean_{k}": sum(v) / len(v) for k, v in numeric.items()}
        return {**means, "tasks": self.results}
