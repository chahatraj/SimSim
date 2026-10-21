"""Agents: a model + a persona + a memory + tools."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .llm import LLM, Message
from .memory import Memory, MemoryEntry, NoMemory
from .record import Recorder
from .tools import Tool


@dataclass
class Persona:
    """Who the agent is. `description` is free text; `attributes` are structured fields
    (age, party, occupation...) that also appear in the prompt and in the logs."""

    name: str
    description: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        lines = [self.description] if self.description else []
        lines += [f"- {k}: {v}" for k, v in self.attributes.items()]
        return "\n".join(lines)

    @classmethod
    def load_jsonl(cls, path: str | Path) -> list["Persona"]:
        """One JSON object per line: {"name": ..., "description": ..., any other keys -> attributes}."""
        out = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    out.append(cls(d.pop("name"), d.pop("description", ""), d))
        return out


DEFAULT_TEMPLATE = """You are {name}.
{persona}

{instructions}

# Your memory
{memory}"""


@dataclass
class Turn:
    """Everything an agent did in one turn."""

    agent: str
    t: int
    observation: str
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)


class Agent:
    """
    Args:
        name: unique name in the simulation.
        llm: the model backend.
        persona: a `Persona` or plain text.
        memory: any `Memory`; defaults to `NoMemory`.
        tools: the agent's own tools (environments can add more per turn).
        instructions: task or role instructions, placed after the persona.
        template: system prompt template with {name}, {persona}, {instructions}, {memory}.
            Override it to control exactly what the model sees.
        max_tool_steps: cap on tool-call rounds per turn.
    """

    def __init__(
        self,
        name: str,
        llm: LLM,
        persona: Persona | str = "",
        memory: Memory | None = None,
        tools: Iterable[Tool] = (),
        instructions: str = "",
        template: str = DEFAULT_TEMPLATE,
        max_tool_steps: int = 8,
    ):
        self.name = name
        self.llm = llm
        self.persona = persona if isinstance(persona, Persona) else Persona(name, persona)
        self.memory = memory or NoMemory()
        self.tools = list(tools)
        self.instructions = instructions
        self.template = template
        self.max_tool_steps = max_tool_steps

    def system_prompt(self, query: str = "") -> str:
        text = self.template.format(
            name=self.name,
            persona=self.persona.render(),
            instructions=self.instructions,
            memory=self.memory.read(query) or "(nothing yet)",
        )
        if isinstance(self.memory, NoMemory) and self.template == DEFAULT_TEMPLATE:
            text = text.split("\n# Your memory")[0]
        return text.strip()

    def act(
        self,
        observation: str,
        t: int = 0,
        tools: Iterable[Tool] = (),
        recorder: Recorder | None = None,
    ) -> Turn:
        """Respond to `observation`, calling tools as needed, then store the turn in memory."""
        all_tools = {tl.name: tl for tl in [*self.tools, *self.memory.tools(), *tools]}
        schemas = [tl.schema() for tl in all_tools.values()] or None
        messages: list[Message] = [
            {"role": "system", "content": self.system_prompt(observation)},
            {"role": "user", "content": observation},
        ]
        calls: list[dict[str, Any]] = []
        text = ""
        for step in range(self.max_tool_steps + 1):
            # On the final step, withhold tools so the model must answer in text.
            resp = self.llm.chat(messages, tools=schemas if step < self.max_tool_steps else None)
            if recorder:
                recorder.log(
                    "llm_call", agent=self.name, t=t, step=step, messages=list(messages),
                    response=resp.text, tool_calls=[c.__dict__ for c in resp.tool_calls], usage=resp.usage,
                )
            messages.append(resp.assistant_message())
            if not resp.tool_calls:
                text = resp.text
                break
            for c in resp.tool_calls:
                tl = all_tools.get(c.name)
                result = tl(**c.arguments) if tl else f"Error: unknown tool {c.name!r}"
                calls.append({"name": c.name, "arguments": c.arguments, "result": result})
                messages.append({"role": "tool", "tool_call_id": c.id, "content": result})
                if recorder:
                    recorder.log("tool_call", agent=self.name, t=t, name=c.name, arguments=c.arguments, result=result)

        self.memory.write(MemoryEntry(t, f"Observed: {observation}", kind="observation"))
        did = "".join(f"Called {c['name']}({c['arguments']}) -> {c['result']}\n" for c in calls)
        self.memory.write(MemoryEntry(t, f"{did}I said: {text}".strip(), kind="action"))
        turn = Turn(self.name, t, observation, text, calls, messages)
        if recorder:
            recorder.log("turn", agent=self.name, t=t, observation=observation, text=text, tool_calls=calls)
        return turn

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "persona": {"description": self.persona.description, **self.persona.attributes},
            "memory": type(self.memory).__name__,
            "tools": [tl.name for tl in self.tools],
            "llm": self.llm.describe(),
        }
