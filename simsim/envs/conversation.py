from __future__ import annotations

from typing import Any

from ..agent import Agent, Turn
from .base import Environment


class Conversation(Environment):
    """Agents talk in a shared room: negotiation, debate, persuasion, interviews.

    Args:
        scenario: the setup every agent sees on its first turn.
        private: per-agent private info, e.g. {"buyer": "Your budget is $500."}.
        show: "new" shows each agent only messages since its last turn (memory
            keeps the rest); "all" shows the full transcript every turn.
        end_token: an agent can end the conversation by including this string.
        max_turns: stop after this many messages.
    """

    def __init__(
        self,
        scenario: str,
        private: dict[str, str] | None = None,
        show: str = "new",
        end_token: str | None = "[END]",
        max_turns: int | None = None,
    ):
        self.scenario = scenario
        self.private = private or {}
        self.show = show
        self.end_token = end_token
        self.max_turns = max_turns

    def reset(self, agents, rng):
        super().reset(agents, rng)
        self.transcript: list[tuple[str, str]] = []
        self.seen = {a.name: 0 for a in agents}
        self.briefed: set[str] = set()
        self.ended_by: str | None = None

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
            if self.end_token:
                parts.append(f"When the conversation is over, include {self.end_token} in your message.")
        parts.append("Conversation so far:\n" + "\n".join(lines) if lines else "You speak first.")
        parts.append("Write your next message only.")
        return "\n\n".join(parts)

    def apply(self, turn: Turn, t: int) -> None:
        self.transcript.append((turn.agent, turn.text.strip()))
        self.seen[turn.agent] = len(self.transcript)
        if self.end_token and self.end_token in turn.text and self.ended_by is None:
            self.ended_by = turn.agent

    def done(self, t: int) -> bool:
        if self.ended_by:
            return True
        return self.max_turns is not None and len(self.transcript) >= self.max_turns

    def metrics(self) -> dict[str, Any]:
        return {
            "n_messages": len(self.transcript),
            "ended_by": self.ended_by,
            "transcript": [{"agent": n, "text": m} for n, m in self.transcript],
        }
