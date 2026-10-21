# Your own environment

Most studies need a world of their own. An environment implements `observe` (what
an agent sees), `apply` (what its action does), `metrics`, and calls `stop` when
the run should end. Here is a repeated public-goods game, from
`examples/public_goods.py`:

```python
from simsim import Environment

class PublicGoods(Environment):
    def __init__(self, rounds=5, show_others=True):
        self.rounds, self.show_others = rounds, show_others

    def reset(self, agents, rng):
        super().reset(agents, rng)
        self.history, self.pending = [], {}
        self.earnings = {a.name: 0.0 for a in agents}

    def observe(self, agent, t):
        text = f"Round {len(self.history) + 1} of {self.rounds}. Contribute 0-10 tokens ..."
        if self.history and self.show_others:
            text += f" Last round: {self.history[-1]}"
        return text

    def apply(self, turn, t):
        self.pending[turn.agent] = int(turn.private["contribution"])
        if len(self.pending) == len(self.agents):          # everyone moved: settle the round
            share = 1.6 * sum(self.pending.values()) / len(self.agents)
            for name, c in self.pending.items():
                self.earnings[name] += 10 - c + share
            self.history.append(dict(self.pending))
            self.pending = {}
            if len(self.history) == self.rounds:
                self.stop("all rounds played")

    def metrics(self):
        return {"cooperation_by_round": [sum(r.values()) / (len(r) * 10) for r in self.history]}
```

The action comes from a structured reply, so no parsing is needed:

```python
SCHEMA = {"reasoning": (str, "private"), "contribution": (int, 0, 10), "reply": (str, "a short public note")}
agents = [Agent(f"Player{i}", llm, output_schema=SCHEMA, memory=FullHistory()) for i in range(4)]
Simulation(PublicGoods(), agents, scheduler=everyone, simultaneous=True, max_steps=10)
```

`simultaneous=True` makes the moves simultaneous: every player decides on the same
information. With Qwen3.5-4B and two seeds, "selfish" personas contributed almost
nothing, and plain players who could see each other's contributions raised theirs
over the rounds (from 10–23% to 45–53%), while hidden contributions stayed flat.

Tips:

- Put per-agent tools in `tools_for(agent, t)`; let tools close over the
  environment's state.
- Return `None` from `observe` to skip an agent this step.
- Override `add_agent` if your world keeps per-agent state and agents can join mid-run.
- Everything an agent sees is logged, so check the `observation` of a few turns
  in the viewer before running at scale.
