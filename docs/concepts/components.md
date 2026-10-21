# Components

Components are pluggable parts of an agent's mind. Each one adds a section to the
agent's system prompt every turn (`context`) and updates itself after the turn
(`update`). Their state after every turn is saved in `turn.components` and in the
logs, so trajectories (trust over time, how plans change) are directly analyzable.

```python
from simsim import Agent, Goal, Plan, Impressions, InnerState

agent = Agent("Ana", llm, persona=..., components=[
    Goal("Sell the bike for at least $150."),
    Plan(every=4, horizon="the rest of the negotiation"),
    Impressions(words=40),
    InnerState({"mood": ["calm", "annoyed", "angry"], "trust_in_buyer": (int, 1, 7)}),
])
```

| Component | Prompt section | Updates |
|---|---|---|
| `Goal(text)` | `# Your goal` | never (a fixed, possibly hidden objective) |
| `Plan(every=5, horizon=...)` | `# Your current plan` | the agent rewrites its plan every `every` turns |
| `Impressions(words=40)` | `# Your impressions of others` | after a turn, for each other agent named in what it saw |
| `InnerState(schema, visible=True)` | `# Your current state` | after every turn, a structured update of the schema |

`InnerState(..., visible=False)` measures the state without feeding it back to
the agent, which is useful when the state is an outcome rather than a mechanism.

Components use the agent's own model unless you pass `llm=` (e.g. a cheaper model
for impressions). Probes see components read-only: interviewing an agent never
triggers a new plan.

## Your own component

```python
from simsim import Component

class Reputation(Component):
    heading = "What people say about you"

    def __init__(self):
        self.rumors = []

    def context(self, agent, observation, t):
        return "\n".join(self.rumors[-3:])

    def update(self, agent, turn, recorder=None):
        if "liar" in turn.observation:
            self.rumors.append(f"At {turn.time or turn.t}, someone called you a liar.")

    def dump(self):
        return list(self.rumors)

    def load(self, data):
        self.rumors = list(data)
```

Implement `view(...)` as well if `context` has side effects (as `Plan` does), so
probes can read the component without changing it.
