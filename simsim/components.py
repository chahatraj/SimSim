"""Agent components: pluggable pieces of an agent's mind.

A component adds a section to the agent's system prompt each turn (`context`) and
updates itself after each turn (`update`). Plans, impressions of others, goals,
and inner states such as emotions or beliefs are all components, so you can
add, remove, or swap them as experimental conditions:

    agent = Agent("Ana", llm, persona=..., components=[
        Goal("Sell the bike for at least $150."),
        Plan(every=4),
        Impressions(),
        InnerState({"mood": ["calm", "annoyed", "angry"], "trust": (int, 1, 7)}),
    ])

Every component's state after each turn is logged (`turn.components`), so you can
analyze trajectories such as trust over time. Components use the agent's own
model unless you pass `llm=`.
"""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any

from .llm import LLM
from .structured import Schema, structured_chat

if TYPE_CHECKING:
    from .agent import Agent, Turn
    from .record import Recorder


class Component:
    """Base class. `heading` titles the prompt section; empty context adds nothing.

    `context` may update the component (e.g. write a new plan); `view` must not,
    and is used for probes so interviewing an agent never changes it.
    """

    heading: str = ""

    def context(self, agent: Agent, observation: str, t: int) -> str:
        return ""

    def view(self, agent: Agent, observation: str, t: int) -> str:
        return self.context(agent, observation, t)

    def update(self, agent: Agent, turn: Turn, recorder: Recorder | None = None) -> None:
        pass

    def dump(self) -> Any:
        return None

    def load(self, data: Any) -> None:
        pass

    def _llm(self, agent: Agent) -> LLM:
        return getattr(self, "llm", None) or agent.llm


class Goal(Component):
    """A fixed goal shown every turn (e.g. a hidden objective in a social scenario)."""

    heading = "Your goal"

    def __init__(self, text: str):
        self.text = text

    def context(self, agent, observation, t):
        return self.text

    def dump(self):
        return self.text

    def load(self, data):
        self.text = data


class Plan(Component):
    """A plan the agent writes for itself and revises every `every` turns."""

    heading = "Your current plan"
    PROMPT = (
        "You are {name}.\n{persona}\n\nWhat you remember:\n{memory}\n\nWhat is happening now:\n{observation}\n\n"
        "Current plan: {plan}\n\nWrite your plan for {horizon}: 2-4 short bullet points. Output only the plan."
    )

    def __init__(self, every: int = 5, horizon: str = "the next few turns", llm: LLM | None = None):
        self.every, self.horizon, self.llm = every, horizon, llm
        self.plan = ""
        self._turns_since = None

    def context(self, agent, observation, t):
        if self._turns_since is None or self._turns_since >= self.every:
            self.plan = self._llm(agent).complete(self.PROMPT.format(
                name=agent.name, persona=agent.persona.render(), memory=agent.memory_text(observation) or "(nothing)",
                observation=observation, plan=self.plan or "(none yet)", horizon=self.horizon,
            )).strip()
            self._turns_since = 0
        return self.plan

    def view(self, agent, observation, t):
        return self.plan

    def update(self, agent, turn, recorder=None):
        self._turns_since = (self._turns_since or 0) + 1

    def dump(self):
        return {"plan": self.plan, "turns_since": self._turns_since}

    def load(self, data):
        self.plan, self._turns_since = data["plan"], data["turns_since"]


class Impressions(Component):
    """What the agent thinks of each other agent, updated after turns where they appear
    (Generative Agents summarizes the relationship before each conversation line)."""

    heading = "Your impressions of others"
    PROMPT = (
        "You are {name}.\n{persona}\n\nYour previous impression of {other}: {old}\n\n"
        "What just happened:\n{observation}\nYou said: {said}\n\n"
        "In one or two sentences (under {words} words), what is your impression of {other} now? "
        "Write in the first person. Output only the impression."
    )

    def __init__(self, words: int = 40, llm: LLM | None = None):
        self.words, self.llm = words, llm
        self.of: dict[str, str] = {}

    def context(self, agent, observation, t):
        return "\n".join(f"- {name}: {text}" for name, text in self.of.items())

    def update(self, agent, turn, recorder=None):
        for other in agent.peers:
            if other in turn.observation:
                self.of[other] = self._llm(agent).complete(self.PROMPT.format(
                    name=agent.name, persona=agent.persona.render(), other=other,
                    old=self.of.get(other, "(none yet)"), observation=turn.observation, said=turn.text,
                    words=self.words,
                )).strip()

    def dump(self):
        return dict(self.of)

    def load(self, data):
        self.of = dict(data)


class InnerState(Component):
    """Structured inner state (emotions, beliefs, trust, stance...) updated after every turn.

    `schema` uses the structured-output format (see `simsim.structured`), e.g.
    {"mood": ["calm", "anxious", "angry"], "trust_in_seller": (int, 1, 7)}.
    The current values are shown to the agent unless `visible=False` (then the
    state is only measured, not fed back).
    """

    heading = "Your current state"
    PROMPT = (
        "You are {name}.\n{persona}\n\nYour state before this turn: {state}\n\nWhat just happened:\n"
        "{observation}\nYou said: {said}\n\nUpdate your state to reflect how you are now."
    )

    def __init__(self, schema: Schema, initial: dict[str, Any] | None = None, visible: bool = True,
                 heading: str | None = None, llm: LLM | None = None):
        self.schema, self.visible, self.llm = schema, visible, llm
        self.values: dict[str, Any] = dict(initial or {})
        if heading:
            self.heading = heading

    def context(self, agent, observation, t):
        if not self.visible or not self.values:
            return ""
        return "\n".join(f"- {k}: {v}" for k, v in self.values.items())

    def update(self, agent, turn, recorder=None):
        prompt = self.PROMPT.format(name=agent.name, persona=agent.persona.render(),
                                    state=json.dumps(self.values) if self.values else "(not set)",
                                    observation=turn.observation, said=turn.text)
        data, _, errors = structured_chat(self._llm(agent), [{"role": "user", "content": prompt}], self.schema)
        if data is not None:
            self.values = data
        elif recorder:
            recorder.log("component_error", agent=agent.name, t=turn.t, component="InnerState", errors=errors)

    def dump(self):
        return copy.deepcopy(self.values)

    def load(self, data):
        self.values = dict(data)
