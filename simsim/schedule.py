"""Schedulers decide who acts at each step: `(agents, t, rng) -> list of agents`.

A scheduler may also accept a `sim` keyword argument to see the environment
(e.g. the conversation so far) or the clock; see `NextSpeaker` and `by_hour`.
"""

from __future__ import annotations

import inspect
import random
import re
from typing import TYPE_CHECKING, Callable

from .agent import Agent
from .llm import LLM

if TYPE_CHECKING:
    from .sim import Simulation

Scheduler = Callable[..., list[Agent]]


def call_scheduler(scheduler: Scheduler, agents: list[Agent], t: int, rng: random.Random, sim: "Simulation"):
    try:
        takes_sim = "sim" in inspect.signature(scheduler).parameters
    except (TypeError, ValueError):
        takes_sim = False
    return scheduler(agents, t, rng, sim=sim) if takes_sim else scheduler(agents, t, rng)


def round_robin(agents: list[Agent], t: int, rng: random.Random) -> list[Agent]:
    """One agent per step, in a fixed order. The usual choice for conversations."""
    return [agents[t % len(agents)]]


def everyone(agents: list[Agent], t: int, rng: random.Random) -> list[Agent]:
    """All agents act every step, in a fixed order."""
    return list(agents)


def everyone_shuffled(agents: list[Agent], t: int, rng: random.Random) -> list[Agent]:
    """All agents act every step, in a fresh random order (removes first-mover effects)."""
    order = list(agents)
    rng.shuffle(order)
    return order


def random_one(agents: list[Agent], t: int, rng: random.Random) -> list[Agent]:
    """One uniformly random agent per step."""
    return [rng.choice(agents)]


def bernoulli(p: float | Callable[[Agent], float]) -> Scheduler:
    """Each agent independently acts with probability `p` (or `p(agent)`) each step."""

    def schedule(agents, t, rng):
        return [a for a in agents if rng.random() < (p(a) if callable(p) else p)]

    return schedule


def zipf(s: float = 1.0, k: int = 1) -> Scheduler:
    """`k` agents per step, drawn with Zipfian activity: a few 'power users' act most.

    Ranks are assigned by agent order, so shuffle the agent list (with your seed)
    if rank should not follow construction order.
    """

    def schedule(agents: list[Agent], t: int, rng: random.Random) -> list[Agent]:
        weights = [1 / (i + 1) ** s for i in range(len(agents))]
        return rng.choices(agents, weights=weights, k=k)

    return schedule


def by_hour(activity: dict[int, float] | list[float], base: Scheduler | None = None) -> Scheduler:
    """Each agent acts with a probability that depends on the simulated hour (needs a `Clock`).

    `activity` maps hour (0-23) to probability, e.g. low at night, high in the evening,
    as in OASIS's hourly activity profiles.
    """
    probs = activity if isinstance(activity, dict) else dict(enumerate(activity))

    def schedule(agents, t, rng, sim=None):
        if sim is None or sim.clock is None:
            raise ValueError("by_hour needs Simulation(clock=Clock(...))")
        p = probs.get(sim.clock.time(t).hour, 0.0)
        return [a for a in agents if rng.random() < p]

    return schedule


class NextSpeaker:
    """An LLM moderator picks who speaks next from the conversation so far
    (Concordia's NextActing, MARBLE's planner). Works with `Conversation`.

    Falls back to round-robin order if the answer names no agent.
    """

    PROMPT = (
        "Here is a conversation between {names}.\n\n{transcript}\n\n"
        "{guidance}Who should speak next? Answer with exactly one name from: {names}."
    )

    def __init__(self, llm: LLM, guidance: str = "", allow_repeat: bool = False):
        self.llm, self.guidance, self.allow_repeat = llm, guidance, allow_repeat
        self.last: str | None = None

    def __call__(self, agents: list[Agent], t: int, rng: random.Random, sim: "Simulation" = None) -> list[Agent]:
        names = [a.name for a in agents if self.allow_repeat or a.name != self.last] or [a.name for a in agents]
        transcript = sim.env.render() if sim is not None and hasattr(sim.env, "render") else ""
        if not transcript:
            choice = agents[t % len(agents)].name
        else:
            answer = self.llm.complete(self.PROMPT.format(
                names=", ".join(names), transcript=transcript,
                guidance=f"{self.guidance}\n\n" if self.guidance else ""), max_tokens=20)
            found = [n for n in names if re.search(rf"\b{re.escape(n)}\b", answer)]
            choice = found[0] if found else names[t % len(names)]
        self.last = choice
        return [a for a in agents if a.name == choice]
