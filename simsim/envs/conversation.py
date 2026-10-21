from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..agent import Agent, Turn
from ..tools import Tool
from .base import Environment


class Conversation(Environment):
    """Agents talk in a shared room: negotiation, debate, persuasion, interviews,
    co-writing, or a user simulator talking to an assistant.

    Args:
        scenario: the setup every agent sees on its first turn.
        private: per-agent private info, e.g. {"buyer": "Your budget is $500."}.
        show: "new" shows each agent only messages since its last turn (memory
            keeps the rest); "all" shows the full transcript every turn.
        end_token: an agent ends the conversation by putting this on the last line
            of its message (mentions elsewhere, e.g. in quotes, do not count).
        enders: agents allowed to end it (default: anyone). For a user simulator,
            only the user should end.
        min_turns: end tokens before this many messages are removed and ignored
            (models otherwise tend to "end" in their first message).
        tools: per-agent tools, e.g. {"assistant": [lookup_order, refund]}, or per-agent
            factories `world -> list[Tool]` so each run's tools act on that run's `world`.
        max_turns: stop after this many messages.
        judge: optional `(transcript_text) -> bool`, checked every `judge_every`
            messages; True ends the conversation (e.g. an LLM deciding a deal is
            reached). See `simsim.judge.Judge.stopper`.
        score: optional `(transcript, world) -> dict` added to metrics (e.g. check
            what the assistant's tools did to `world`).
        world: shared mutable state for tools and `score`.
    """

    def __init__(
        self,
        scenario: str,
        private: dict[str, str] | None = None,
        show: str = "new",
        end_token: str | None = "[END]",
        enders: list[str] | None = None,
        min_turns: int = 2,
        tools: dict[str, list[Tool] | Callable[[dict], list[Tool]]] | None = None,
        max_turns: int | None = None,
        judge: Callable[[str], bool] | None = None,
        judge_every: int = 2,
        score: Callable[[list[dict], dict], dict[str, Any]] | None = None,
        world: dict | None = None,
    ):
        self.scenario = scenario
        self.private = private or {}
        self.show = show
        self.end_token = end_token
        self.enders = enders
        self.min_turns = min_turns
        self.tools = tools or {}
        self.max_turns = max_turns
        self.judge = judge
        self.judge_every = judge_every
        self.score = score
        self.initial_world = world or {}

    def reset(self, agents, rng):
        import copy

        super().reset(agents, rng)
        self.transcript: list[tuple[str, str]] = []
        self.seen = {a.name: 0 for a in agents}
        self.briefed: set[str] = set()
        self.ended_by: str | None = None
        self.world = copy.deepcopy(self.initial_world)
        self.agent_tools = {name: (ts(self.world) if callable(ts) else list(ts)) for name, ts in self.tools.items()}

    def add_agent(self, agent):
        super().add_agent(agent)
        self.seen.setdefault(agent.name, len(self.transcript) if self.show == "new" else 0)

    def render(self) -> str:
        return "\n".join(f"{n}: {m}" for n, m in self.transcript)

    def observe(self, agent: Agent, t: int) -> str:
        start = 0 if self.show == "all" else self.seen[agent.name]
        lines = [f"{n}: {m}" for n, m in self.transcript[start:]]
        self.seen[agent.name] = len(self.transcript)
        parts = []
        if agent.name not in self.briefed:
            self.briefed.add(agent.name)
            others = ", ".join(n for n in self.agents if n != agent.name)
            parts.append(f"{self.scenario}\nYou are talking with: {others}.")
            if agent.name in self.private:
                parts.append(f"Private information (only you know this): {self.private[agent.name]}")
            if self.end_token and (self.enders is None or agent.name in self.enders):
                parts.append(f"Only when the conversation has truly concluded (for example, an agreement is reached "
                             f"or talks have broken down), put {self.end_token} on the last line of your message. "
                             "Do not use it otherwise.")
        parts.append("Conversation so far:\n" + "\n".join(lines) if lines else "You speak first.")
        parts.append("Write your next message only.")
        return "\n\n".join(parts)

    def tools_for(self, agent, t):
        return self.agent_tools.get(agent.name, [])

    def _ends(self, text: str) -> bool:
        lines = [ln for ln in text.strip().splitlines() if ln.strip()]
        return bool(lines) and self.end_token in lines[-1]

    def apply(self, turn: Turn, t: int) -> None:
        text = turn.text.strip()
        has_token = bool(self.end_token) and self._ends(text)
        ends = has_token and (self.enders is None or turn.agent in self.enders) and len(self.transcript) + 1 >= self.min_turns
        if has_token:
            lines = text.splitlines()
            lines[-1] = lines[-1].replace(self.end_token, "").rstrip()
            text = "\n".join(lines).strip()
        self.transcript.append((turn.agent, text))
        self.seen[turn.agent] = len(self.transcript)
        if ends:
            self.ended_by = turn.agent
            self.stop(f"ended by {turn.agent}")
        elif self.max_turns is not None and len(self.transcript) >= self.max_turns:
            self.stop("max_turns")
        elif self.judge and len(self.transcript) % self.judge_every == 0 and self.judge(self.render()):
            self.stop("judge")

    def metrics(self) -> dict[str, Any]:
        transcript = [{"agent": n, "text": m} for n, m in self.transcript]
        out: dict[str, Any] = {"n_messages": len(self.transcript), "ended_by": self.ended_by, "transcript": transcript}
        if self.score:
            out.update(self.score(transcript, self.world))
        return out


USER_SIMULATOR_INSTRUCTIONS = """You are playing a user talking to an AI assistant. Stay in character.
Pursue your goal below. Reveal information only when it is needed, as a real user would.
Do not invent details beyond your goal and persona. Wait for the assistant to confirm what
it actually did. Only when your goal is met, or clearly cannot be met, end the conversation
with {end_token} on the last line; never in a message that asks for something.

Your goal: {goal}"""


def user_simulator(name: str, llm, goal: str, persona: str = "", end_token: str = "###STOP###", **kwargs) -> Agent:
    """An LLM that plays a user with a hidden goal, as in tau-bench and SimulatorArena.

    Pair it with an assistant agent in a `Conversation(..., end_token="###STOP###",
    enders=[name], tools={assistant: [...]})`.
    """
    instructions = USER_SIMULATOR_INSTRUCTIONS.format(goal=goal, end_token=end_token)
    return Agent(name, llm, persona=persona, instructions=instructions, **kwargs)
