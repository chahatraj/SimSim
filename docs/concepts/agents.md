# Agents

An agent is a model plus a persona, a memory, tools and components:

```python
from simsim import Agent, Persona, FullHistory, Plan

agent = Agent(
    "Ana", llm,
    persona=Persona("Ana", "A nurse from Ohio.", {"age": 46, "party": "Independent"}),
    instructions="You are taking part in a town hall discussion.",
    memory=FullHistory(),
    tools=[lookup_bill],
    components=[Plan(every=4)],
)
turn = agent.act("The moderator asks: what should the town do about the clinic?")
```

## What the model sees

Each turn the agent sends a system prompt and the observation:

```
You are Ana.
A nurse from Ohio.
- age: 46
- party: Independent

You are taking part in a town hall discussion.

# Your memory
[t=0] Observed: ...  I said: ...

# Current time            (if the simulation has a clock)
# Your current plan       (one section per component)
```

Change it with `template=`, using `{name}`, `{persona}`, `{instructions}` and
`{memory}`. The exact prompt of every call is in the logs, so you can always
check what a model actually received.

## Tools

```python
from simsim import tool

@tool
def lookup_bill(bill_id: str) -> str:
    """Look up the text of a town bill."""
    return BILLS[bill_id]
```

The signature and docstring become the schema the model sees. Errors inside a
tool, unknown tool names and malformed arguments are returned to the model as
error messages instead of crashing the run. `max_tool_steps` caps tool rounds per
turn; on the last round tools are withheld so the model must answer.

For models or servers without native function calling, use
`action_format="react"`: the agent writes `Thought: / Action: / Action Input:` and
`Final Answer:`, which SimSim parses (and asks to repair when malformed).

`emulated_tool(llm, name, description, parameters, world=...)` makes a tool whose
output an LLM generates (as in ToolEmu), for testing agents on actions that would
be dangerous or not yet built.

## Structured replies and private fields

```python
schema = {"thoughts": (str, "private; never shown to others"),
          "reply": str,
          "offer": (int, 0, 1000),
          "stance": ["accept", "counter", "walk away"]}
agent = Agent("Buyer", llm, output_schema=schema, public_field="reply")
turn = agent.act("Seller: $200, final.")
turn.text      # the reply, which is what other agents see
turn.private   # {"thoughts": ..., "offer": 150, "stance": "counter"}
```

Schema specs: a type (`str`, `int`, `float`, `bool`), a list of allowed strings,
a range `(int, 1, 7)`, or any of these with a description `(spec, "description")`.
Replies that are not valid JSON or break the schema are sent back with the error,
up to `max_repairs` times. If repair fails the turn keeps the raw text and records
`turn.error`, so one bad reply does not crash an experiment.

`remember_private=False` keeps private fields out of the agent's own memory.

## A turn

`agent.act(...)` returns a `Turn`: `text`, `private`, `reasoning`, `tool_calls`,
`components` (each component's state after the turn), `usage`, `error`, `time`,
and the full `messages`.

## Probes: questions that do not change the agent

```python
agent.ask("How do you feel about the seller?")                       # text
agent.ask("Rate your trust.", schema={"trust": (int, 1, 7)})         # dict
agent.survey("I trust the seller.", {"1": "Strongly disagree", "5": "Strongly agree"})
```

Probes read memory and components without side effects: nothing is stored
(unless `ask(..., remember=True)`), and retrieval does not count as access. The
agent answers in character, from what it remembers.

`survey` returns next-token probabilities over the option labels (OpenAI and vLLM
support this). Options can be a dict `{label: text}` or a list (labeled A, B, C…).

## Saving an agent

`agent.save(path)` writes its memory and component state as JSON;
`Agent(...).restore(path)` loads it, for example to continue the same agent in a
later experiment.

## Non-LLM agents

- `ScriptedAgent(name, ["line 1", "line 2"])` or `ScriptedAgent(name, fn)`:
  confederates, bots and baselines that follow a script.
- `HumanAgent(name)`: a person types the agent's turns at the terminal.

Both work in any environment next to LLM agents.
