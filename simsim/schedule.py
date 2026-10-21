"""Schedulers decide who acts at each step: `(agents, t, rng) -> list of agents`."""

from __future__ import annotations

import random
from typing import Callable

from .agent import Agent

Scheduler = Callable[[list[Agent], int, random.Random], list[Agent]]


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


def zipf(s: float = 1.0, k: int = 1) -> Scheduler:
    """`k` agents per step, drawn with Zipfian activity: a few 'power users' act most.

    Ranks are assigned by agent order, so shuffle the agent list (with your seed)
    if rank should not follow construction order.
    """

    def schedule(agents: list[Agent], t: int, rng: random.Random) -> list[Agent]:
        weights = [1 / (i + 1) ** s for i in range(len(agents))]
        return rng.choices(agents, weights=weights, k=k)

    return schedule
