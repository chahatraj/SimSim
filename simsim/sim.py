"""The simulation loop, plus interventions (things that happen to the simulation at chosen steps)."""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .agent import Agent, Turn
from .envs.base import Environment
from .memory import MemoryEntry
from .record import Recorder
from .schedule import Scheduler, round_robin
from .structured import Schema

END = -1  # pass as a step to `at` to run after the last step


@dataclass
class Result:
    turns: list[Turn]
    metrics: dict[str, Any]
    usage: dict[str, Any] = field(default_factory=dict)
    steps: int = 0
    termination: str = ""
    probes: list[dict[str, Any]] = field(default_factory=list)

    def transcript(self) -> str:
        return "\n".join(f"{t.agent}: {t.text}" for t in self.turns)


class Intervention:
    """Runs `fn(sim, t)` at the start of every step where `when(t)` is true (or at the end)."""

    def __init__(self, fn: Callable[["Simulation", int], None], when: Callable[[int], bool], at_end: bool = False):
        self.fn, self.when, self.at_end = fn, when, at_end


def at(steps: int | Iterable[int], fn: Callable[["Simulation", int], None]) -> Intervention:
    """Run `fn` at the given step(s). Use `END` for after the final step."""
    s = {steps} if isinstance(steps, int) else set(steps)
    return Intervention(fn, lambda t: t in s, at_end=END in s)


def every(k: int, fn: Callable[["Simulation", int], None], start: int = 0, at_end: bool = False) -> Intervention:
    """Run `fn` every `k` steps from `start` (and after the final step if `at_end`)."""
    return Intervention(fn, lambda t: t >= start and (t - start) % k == 0, at_end=at_end)


def whisper(agent: str, text: str) -> Callable[["Simulation", int], None]:
    """Plant a memory in an agent (Generative Agents' 'whisper'): it believes this happened."""

    def fn(sim: "Simulation", t: int) -> None:
        sim.agent(agent).memory.write(MemoryEntry(max(t, 0), text, kind="whisper"))

    return fn


def add_agent(agent: Agent) -> Callable[["Simulation", int], None]:
    """A new agent joins the simulation."""

    def fn(sim: "Simulation", t: int) -> None:
        sim.agents.append(agent)
        sim.env.add_agent(agent)

    return fn


def probe(question: str, schema: Schema | None = None, agents: list[str] | None = None,
          name: str = "probe") -> Callable[["Simulation", int], None]:
    """Interview agents (without changing their memory); answers go to `Result.probes`."""

    def fn(sim: "Simulation", t: int) -> None:
        for a in sim.agents_named(agents):
            answer = a.ask(question, schema=schema, t=t, recorder=sim.recorder)
            sim.probes.append({"name": name, "t": t, "agent": a.name, "question": question, "answer": answer})

    return fn


def survey(question: str, options: list[str] | dict[str, str], agents: list[str] | None = None,
           name: str = "survey") -> Callable[["Simulation", int], None]:
    """A closed-ended survey item answered with token probabilities; goes to `Result.probes`."""

    def fn(sim: "Simulation", t: int) -> None:
        for a in sim.agents_named(agents):
            probs = a.survey(question, options, t=t, recorder=sim.recorder)
            sim.probes.append({"name": name, "t": t, "agent": a.name, "question": question, "probs": probs})

    return fn


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
        interventions: things that happen at chosen steps (`at`, `every` with
            `whisper`, `add_agent`, `probe`, `survey`, or any `fn(sim, t)`).
        evaluators: {name: fn(result) -> dict}, run at the end and added to
            metrics under `name` (e.g. an LLM judge scoring the transcript).
    """

    def __init__(
        self,
        env: Environment,
        agents: list[Agent],
        scheduler: Scheduler = round_robin,
        max_steps: int = 20,
        seed: int = 0,
        simultaneous: bool = False,
        interventions: list[Intervention] = (),
        evaluators: dict[str, Callable[[Result], dict[str, Any]]] | None = None,
    ):
        names = [a.name for a in agents]
        if len(set(names)) != len(names):
            raise ValueError(f"Agent names must be unique: {names}")
        self.env = env
        self.agents = list(agents)
        self.scheduler = scheduler
        self.max_steps = max_steps
        self.seed = seed
        self.simultaneous = simultaneous
        self.interventions = list(interventions)
        self.evaluators = evaluators or {}

    def agent(self, name: str) -> Agent:
        for a in self.agents:
            if a.name == name:
                return a
        raise KeyError(name)

    def agents_named(self, names: list[str] | None) -> list[Agent]:
        return list(self.agents) if names is None else [self.agent(n) for n in names]

    def _intervene(self, t: int, end: bool = False) -> None:
        for iv in self.interventions:
            if (iv.at_end if end else iv.when(t)):
                self.recorder.log("intervention", t=t, fn=getattr(iv.fn, "__qualname__", repr(iv.fn)))
                iv.fn(self, t)

    def _step(self, t: int, turns: list[Turn]) -> None:
        active = self.scheduler(self.agents, t, self.rng)
        if self.simultaneous:
            pending = [(a, self.env.observe(a, t), self.env.tools_for(a, t)) for a in active]
            pending = [p for p in pending if p[1] is not None]
            with ThreadPoolExecutor(max_workers=max(1, len(pending))) as pool:
                done = list(pool.map(lambda p: p[0].act(p[1], t, p[2], self.recorder), pending))
            for turn in done:
                self.env.apply(turn, t)
                turns.append(turn)
            return
        for a in active:
            obs = self.env.observe(a, t)
            if obs is None:
                continue
            turn = a.act(obs, t, self.env.tools_for(a, t), self.recorder)
            self.env.apply(turn, t)
            turns.append(turn)
            if self.env.done(t):
                return

    def run(self, recorder: Recorder | None = None) -> Result:
        self.recorder = recorder or Recorder()
        self.rng = random.Random(self.seed)
        self.probes: list[dict[str, Any]] = []
        self.env.reset(self.agents, self.rng)
        self.recorder.log("start", seed=self.seed, max_steps=self.max_steps, agents=[a.describe() for a in self.agents])
        turns: list[Turn] = []
        steps, termination = self.max_steps, "max_steps"
        for t in range(self.max_steps):
            self._intervene(t)
            if self.env.done(t):
                steps, termination = t, self.env.stop_reason or "done"
                break
            self._step(t, turns)
            if self.env.done(t):
                steps, termination = t + 1, self.env.stop_reason or "done"
                break
        self._intervene(steps, end=True)

        result = Result(turns, self.env.metrics(), self._usage(turns), steps, termination, self.probes)
        for name, fn in self.evaluators.items():
            result.metrics[name] = fn(result)
            self.recorder.log("evaluation", name=name, result=result.metrics[name])
        self.recorder.log(
            "end", steps=steps, termination=termination, metrics=result.metrics, usage=result.usage,
            probes=self.probes, memory={a.name: a.memory.state() for a in self.agents},
        )
        return result

    def _usage(self, turns: list[Turn]) -> dict[str, Any]:
        per_agent: dict[str, dict[str, int]] = {}
        for turn in turns:
            u = per_agent.setdefault(turn.agent, {})
            for k, v in turn.usage.items():
                u[k] = u.get(k, 0) + v
        # Totals per distinct backend object, so agents sharing one LLM are not double counted.
        # These include memory, probe and judge calls made through the same backends.
        backends = {id(a.llm): a.llm for a in self.agents}
        return {"agents": per_agent, "backends": {f"{b.model}#{i}": b.usage for i, b in enumerate(backends.values())}}
