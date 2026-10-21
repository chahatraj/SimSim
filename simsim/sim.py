"""The simulation loop."""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .agent import Agent, Turn
from .envs.base import Environment
from .record import Recorder
from .schedule import Scheduler, round_robin


@dataclass
class Result:
    turns: list[Turn]
    metrics: dict[str, Any]
    usage: dict[str, dict[str, int]] = field(default_factory=dict)
    steps: int = 0


class Simulation:
    """
    Args:
        env: the environment.
        agents: the agents (names must be unique).
        scheduler: who acts each step; see `simsim.schedule`.
        max_steps: hard cap on steps.
        seed: seeds the scheduler/environment RNG.
        simultaneous: if True, all agents scheduled in a step observe the same
            world state and act in parallel threads; their actions are applied
            afterwards in schedule order. If False, each sees the previous
            agent's action.
    """

    def __init__(
        self,
        env: Environment,
        agents: list[Agent],
        scheduler: Scheduler = round_robin,
        max_steps: int = 20,
        seed: int = 0,
        simultaneous: bool = False,
    ):
        names = [a.name for a in agents]
        if len(set(names)) != len(names):
            raise ValueError(f"Agent names must be unique: {names}")
        self.env = env
        self.agents = agents
        self.scheduler = scheduler
        self.max_steps = max_steps
        self.seed = seed
        self.simultaneous = simultaneous

    def run(self, recorder: Recorder | None = None) -> Result:
        rec = recorder or Recorder()
        rng = random.Random(self.seed)
        self.env.reset(self.agents, rng)
        rec.log("start", seed=self.seed, max_steps=self.max_steps, agents=[a.describe() for a in self.agents])
        turns: list[Turn] = []
        t = 0
        for t in range(self.max_steps):
            if self.env.done(t):
                break
            active = self.scheduler(self.agents, t, rng)
            if self.simultaneous:
                pending = [(a, self.env.observe(a, t), self.env.tools_for(a, t)) for a in active]
                pending = [p for p in pending if p[1] is not None]
                with ThreadPoolExecutor(max_workers=max(1, len(pending))) as pool:
                    done = list(pool.map(lambda p: p[0].act(p[1], t, p[2], rec), pending))
                for turn in done:
                    self.env.apply(turn, t)
                    turns.append(turn)
            else:
                for a in active:
                    obs = self.env.observe(a, t)
                    if obs is None:
                        continue
                    turn = a.act(obs, t, self.env.tools_for(a, t), rec)
                    self.env.apply(turn, t)
                    turns.append(turn)
                    if self.env.done(t):
                        break
        else:
            t = self.max_steps
        metrics = self.env.metrics()
        usage = {a.name: a.llm.usage for a in self.agents}
        rec.log("end", steps=t, metrics=metrics, usage=usage, memory={a.name: a.memory.state() for a in self.agents})
        return Result(turns, metrics, usage, steps=t)
