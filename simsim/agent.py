"""Agents: a model + a persona + a memory + tools."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .llm import LLM, LLMResponse, Message, ScriptedLLM, ToolCall
from .memory import Memory, MemoryEntry, NoMemory
from .record import Recorder
from .structured import Schema, StructuredOutputError, extract_json, format_instructions, parse, structured_chat
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

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, **self.attributes}

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

REACT_INSTRUCTIONS = """# Tools
You can use these tools:
{tools}

To use a tool, write exactly:
Thought: <your reasoning>
Action: <tool name>
Action Input: <JSON object of arguments>

You will then receive an Observation. Use one tool at a time. When you are done, write:
Thought: <your reasoning>
Final Answer: <your answer>"""


@dataclass
class Turn:
    """Everything an agent did in one turn.

    `text` is what the agent said publicly. With an `output_schema`, `private`
    holds the other fields (e.g. reasoning the other agents never see).
    `reasoning` holds a thinking model's reasoning and ReAct "Thought:" lines.
    """

    agent: str
    t: int
    observation: str
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    private: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    error: str | None = None
    messages: list[Message] = field(default_factory=list)


def _add_usage(total: dict[str, int], resp: LLMResponse) -> None:
    for k, v in resp.usage.items():
        total[k] = total.get(k, 0) + v
    total["calls"] = total.get("calls", 0) + 1


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
        action_format: "tools" uses the API's native function calling; "react" uses
            Thought/Action/Final Answer text, for models or servers without tool calling.
        output_schema: if set, every reply must be JSON matching this schema (see
            `simsim.structured`). Bad replies are sent back for repair.
        public_field: with `output_schema`, the field that becomes the public `text`;
            the other fields are private (logged, optionally remembered, never shown to others).
        remember_private: store private fields in the agent's own memory.
        remember_observations: True stores the full observation in memory; an int keeps
            only its first N characters; False stores only what the agent did.
        memory_max_chars: cap on memory text in the prompt (keeps the most recent part).
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
        action_format: str = "tools",
        output_schema: Schema | None = None,
        public_field: str = "reply",
        remember_private: bool = True,
        remember_observations: bool | int = True,
        memory_max_chars: int | None = None,
        max_tool_steps: int = 8,
        max_repairs: int = 2,
    ):
        if action_format not in ("tools", "react"):
            raise ValueError("action_format must be 'tools' or 'react'")
        if output_schema is not None and public_field not in output_schema:
            raise ValueError(f"public_field {public_field!r} is not in output_schema")
        self.name = name
        self.llm = llm
        self.persona = persona if isinstance(persona, Persona) else Persona(name, persona)
        self.memory = memory or NoMemory()
        self.tools = list(tools)
        self.instructions = instructions
        self.template = template
        self.action_format = action_format
        self.output_schema = output_schema
        self.public_field = public_field
        self.remember_private = remember_private
        self.remember_observations = remember_observations
        self.memory_max_chars = memory_max_chars
        self.max_tool_steps = max_tool_steps
        self.max_repairs = max_repairs

    # ---- prompts ---------------------------------------------------------------

    def memory_text(self, query: str = "") -> str:
        text = self.memory.read(query)
        if self.memory_max_chars and len(text) > self.memory_max_chars:
            text = "...\n" + text[-self.memory_max_chars :]
        return text

    def system_prompt(self, query: str = "") -> str:
        text = self.template.format(
            name=self.name,
            persona=self.persona.render(),
            instructions=self.instructions,
            memory=self.memory_text(query) or "(nothing yet)",
        )
        if isinstance(self.memory, NoMemory) and self.template == DEFAULT_TEMPLATE:
            text = text.split("\n# Your memory")[0]
        return text.strip()

    def _react_prompt(self, tools: dict[str, Tool]) -> str:
        lines = [f"- {t.name}: {t.description} Arguments: {json.dumps(t.parameters['properties'])}" for t in tools.values()]
        return REACT_INSTRUCTIONS.format(tools="\n".join(lines))

    # ---- acting ----------------------------------------------------------------

    def act(
        self,
        observation: str,
        t: int = 0,
        tools: Iterable[Tool] = (),
        recorder: Recorder | None = None,
    ) -> Turn:
        """Respond to `observation`, calling tools as needed, then store the turn in memory."""
        all_tools = {tl.name: tl for tl in [*self.tools, *self.memory.tools(), *tools]}
        system = self.system_prompt(observation)
        if self.action_format == "react" and all_tools:
            system += "\n\n" + self._react_prompt(all_tools)
        user = observation
        if self.output_schema:
            where = "in your Final Answer" if self.action_format == "react" and all_tools else ""
            user += f"\n\n{format_instructions(self.output_schema)} {where}".rstrip()
        messages: list[Message] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        turn = Turn(self.name, t, observation, "")
        run = self._run_react if self.action_format == "react" else self._run_tools
        final = run(messages, all_tools, turn, recorder)
        self._finish(final, messages, turn, recorder)
        turn.messages = messages
        self._remember(turn)
        if recorder:
            recorder.log(
                "turn", agent=self.name, t=t, observation=observation, text=turn.text, private=turn.private,
                reasoning=turn.reasoning, tool_calls=turn.tool_calls, usage=turn.usage, error=turn.error,
            )
        return turn

    def _call(self, messages, schemas, turn: Turn, recorder, step: int) -> LLMResponse:
        resp = self.llm.chat(messages, tools=schemas)
        _add_usage(turn.usage, resp)
        if resp.finish_reason == "length":
            turn.error = (f"response cut off at max_tokens={self.llm.max_tokens}"
                          + (" (thinking used the budget; raise max_tokens or turn thinking off)" if resp.reasoning else ""))
        if resp.reasoning:
            turn.reasoning += resp.reasoning.strip() + "\n"
        if recorder:
            recorder.log(
                "llm_call", agent=self.name, t=turn.t, step=step, messages=list(messages), response=resp.text,
                reasoning=resp.reasoning, tool_calls=[c.__dict__ for c in resp.tool_calls], usage=resp.usage,
            )
        return resp

    def _run_tool(self, call: ToolCall, tools: dict[str, Tool], turn: Turn, recorder) -> str:
        tl = tools.get(call.name)
        if tl is None:
            result = f"Error: unknown tool {call.name!r}. Available: {', '.join(tools) or 'none'}"
        elif "_raw" in call.arguments:
            result = f"Error: arguments were not valid JSON: {call.arguments['_raw']!r}"
        else:
            result = tl(**call.arguments)
        turn.tool_calls.append({"name": call.name, "arguments": call.arguments, "result": result})
        if recorder:
            recorder.log("tool_call", agent=self.name, t=turn.t, name=call.name, arguments=call.arguments, result=result)
        return result

    def _run_tools(self, messages, tools, turn, recorder) -> str:
        schemas = [tl.schema() for tl in tools.values()] or None
        for step in range(self.max_tool_steps + 1):
            # On the final step, withhold tools so the model must answer in text.
            resp = self._call(messages, schemas if step < self.max_tool_steps else None, turn, recorder, step)
            messages.append(resp.assistant_message())
            if not resp.tool_calls:
                return resp.text
            for c in resp.tool_calls:
                result = self._run_tool(c, tools, turn, recorder)
                messages.append({"role": "tool", "tool_call_id": c.id, "content": result})
        return ""

    def _run_react(self, messages, tools, turn, recorder) -> str:
        for step in range(self.max_tool_steps + 1):
            resp = self._call(messages, None, turn, recorder, step)
            messages.append({"role": "assistant", "content": resp.text})
            kind, value, thought = parse_react(resp.text)
            if thought:
                turn.reasoning += thought + "\n"
            if kind == "final" or not tools:
                return value if kind == "final" else resp.text
            if kind == "action" and step < self.max_tool_steps:
                name, args = value
                result = self._run_tool(ToolCall(f"react-{step}", name, args), tools, turn, recorder)
                messages.append({"role": "user", "content": f"Observation: {result}"})
            elif step < self.max_tool_steps:
                messages.append({"role": "user", "content": f"Could not parse your response ({value}). "
                                 "Use 'Action:' + 'Action Input:' or 'Final Answer:'."})
            else:
                return resp.text
        return ""

    def _finish(self, final: str, messages, turn: Turn, recorder) -> None:
        if not self.output_schema:
            turn.text = final
            return
        try:
            data = parse(final, self.output_schema)
        except StructuredOutputError as e:
            messages.append({"role": "user", "content": f"Your answer could not be used: {e}. "
                             f"{format_instructions(self.output_schema)}"})
            data, repaired, errors = structured_chat(
                self.llm, messages, self.output_schema, max_repairs=self.max_repairs - 1, add_instructions=False
            )
            messages[:] = repaired
            if recorder:
                recorder.log("repair", agent=self.name, t=turn.t, errors=[str(e), *errors], ok=data is not None)
            if data is None:
                turn.error = turn.error or f"output did not match schema: {e}"
                turn.text = final
                return
        turn.error = None  # a valid answer arrived, possibly after a cut-off attempt
        turn.text = str(data[self.public_field])
        turn.private = {k: v for k, v in data.items() if k != self.public_field}

    def _remember(self, turn: Turn) -> None:
        parts = []
        if self.remember_observations:
            obs = turn.observation
            if isinstance(self.remember_observations, int) and not isinstance(self.remember_observations, bool):
                obs = obs[: self.remember_observations]
            parts.append(f"Observed: {obs}")
        parts += [f"Called {c['name']}({c['arguments']}) -> {c['result']}" for c in turn.tool_calls]
        if turn.private and self.remember_private:
            parts.append(f"My private notes: {json.dumps(turn.private)}")
        parts.append(f"I said: {turn.text}")
        self.memory.write(MemoryEntry(turn.t, "\n".join(parts), kind="turn"))

    # ---- probes: questions that do not change the agent ------------------------

    def ask(
        self,
        question: str,
        schema: Schema | None = None,
        remember: bool = False,
        t: int = -1,
        recorder: Recorder | None = None,
    ) -> str | dict[str, Any] | None:
        """Interview the agent: it answers in character, from its memory, without tools.

        Nothing is written to memory unless `remember=True`, so probing mid-run does
        not change what happens next. With `schema`, returns a validated dict (or None).
        """
        messages: list[Message] = [
            {"role": "system", "content": self.system_prompt(question)},
            {"role": "user", "content": question},
        ]
        if schema:
            answer, messages, errors = structured_chat(self.llm, messages, schema, self.max_repairs)
        else:
            answer, errors = self.llm.chat(messages).text, []
        if recorder:
            recorder.log("probe", agent=self.name, t=t, question=question, answer=answer, errors=errors)
        if remember:
            self.memory.write(MemoryEntry(t, f"I was asked: {question}\nI answered: {answer}", kind="probe"))
        return answer

    def survey(
        self,
        question: str,
        options: list[str] | dict[str, str],
        t: int = -1,
        recorder: Recorder | None = None,
    ) -> dict[str, float]:
        """A closed-ended survey item, answered with the model's next-token probabilities.

        `options` is a list of answer texts (labeled A, B, C...) or a dict {label: text},
        e.g. {"1": "Strongly disagree", ..., "7": "Strongly agree"}. Returns {label: prob}.
        Thinking models must have thinking turned off for this (see OpenAIChat extra_body).
        """
        labeled = options if isinstance(options, dict) else {chr(65 + i): o for i, o in enumerate(options)}
        listing = "\n".join(f"{k}. {v}" for k, v in labeled.items())
        prompt = f"{question}\n\n{listing}\n\nAnswer with the label only ({', '.join(labeled)})."
        messages = [{"role": "system", "content": self.system_prompt(question)}, {"role": "user", "content": prompt}]
        probs = self.llm.choice_probs(messages, list(labeled))
        if recorder:
            recorder.log("survey", agent=self.name, t=t, question=question, options=labeled, probs=probs)
        return probs

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": type(self).__name__,
            "persona": self.persona.to_dict(),
            "instructions": self.instructions,
            "memory": self.memory.state(),
            "tools": [tl.name for tl in self.tools],
            "action_format": self.action_format,
            "output_schema": list(self.output_schema) if self.output_schema else None,
            "llm": self.llm.describe(),
        }


def parse_react(text: str) -> tuple[str, Any, str]:
    """Parse ReAct text -> ("final", answer, thought) | ("action", (name, args), thought) | ("error", why, thought)."""
    thought = ""
    if "Thought:" in text:
        thought = text.split("Thought:", 1)[1].split("Action:")[0].split("Final Answer:")[0].strip()
    if "Final Answer:" in text:
        return "final", text.split("Final Answer:", 1)[1].strip(), thought
    if "Action:" in text:
        rest = text.split("Action:", 1)[1]
        name = rest.split("\n", 1)[0].strip().strip("`")
        args: dict[str, Any] = {}
        if "Action Input:" in rest:
            raw = rest.split("Action Input:", 1)[1].split("Observation:")[0].strip()
            try:
                args = extract_json(raw)
            except StructuredOutputError:
                return "error", f"Action Input is not a JSON object: {raw[:200]!r}", thought
        return "action", (name, args), thought
    return "error", "no 'Action:' or 'Final Answer:' found", thought


class ScriptedAgent(Agent):
    """A rule-based agent in the same simulation as LLM agents (confederates, bots, baselines).

    `policy` is a list of replies used in order (then repeated from the last), or a
    function `(observation, t) -> str`.
    """

    def __init__(self, name: str, policy: list[str] | Callable[[str, int], str], persona: Persona | str = "",
                 memory: Memory | None = None):
        super().__init__(name, ScriptedLLM(), persona=persona, memory=memory)
        self.policy = policy
        self._i = 0

    def act(self, observation, t=0, tools=(), recorder=None) -> Turn:
        if callable(self.policy):
            text = self.policy(observation, t)
        else:
            text = self.policy[min(self._i, len(self.policy) - 1)]
            self._i += 1
        turn = Turn(self.name, t, observation, text)
        self._remember(turn)
        if recorder:
            recorder.log("turn", agent=self.name, t=t, observation=observation, text=text, scripted=True)
        return turn


class HumanAgent(ScriptedAgent):
    """A person typing at the terminal takes this agent's turns."""

    def __init__(self, name: str, persona: Persona | str = "", input_fn: Callable[[str], str] = input):
        def policy(observation: str, t: int) -> str:
            print(f"\n===== {name}, step {t} =====\n{observation}\n")
            return input_fn(f"{name}> ")

        super().__init__(name, policy, persona=persona)
