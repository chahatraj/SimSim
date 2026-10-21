from __future__ import annotations

import random
from typing import Any

from ..agent import Agent, Turn
from ..tools import Tool


class Environment:
    """The world the agents live in.

    The simulation loop calls, for each scheduled agent at step t:
        observe(agent, t)   -> what the agent sees now (None to skip its turn)
        tools_for(agent, t) -> extra tools the agent may use this turn
        apply(turn, t)      -> update the world with what the agent did
    and stops when done(t) is True. metrics() is the result saved per run.
    """

    def reset(self, agents: list[Agent], rng: random.Random) -> None:
        self.agents = {a.name: a for a in agents}
        self.rng = rng

    def observe(self, agent: Agent, t: int) -> str | None:
        raise NotImplementedError

    def tools_for(self, agent: Agent, t: int) -> list[Tool]:
        return []

    def apply(self, turn: Turn, t: int) -> None:
        pass

    def done(self, t: int) -> bool:
        return False

    def metrics(self) -> dict[str, Any]:
        return {}
