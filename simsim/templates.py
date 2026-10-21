"""Files created by `simsim new <name>`."""

from __future__ import annotations

from pathlib import Path

CONFIG = """# {name}: a SimSim experiment.
# Run:  simsim run experiment.yaml --dry-run      (fake model, free)
#       simsim run experiment.yaml --workers 8 --view
name: {name}
out: runs/{name}
seeds: 3

models:
  main:
    spec: openai:gpt-4o-mini           # or vllm:Qwen/Qwen3.5-9B@http://NODE:8000/v1
    temperature: 0.7
    max_tokens: 512
    # cache: cache/main.sqlite         # replay identical runs for free
    # extra_body: {{chat_template_kwargs: {{enable_thinking: false}}}}   # Qwen3-style models
  judge:
    spec: openai:gpt-4o
    temperature: 0
    max_tokens: 1024                   # thinking models need more (or thinking off via extra_body)

grid:                                  # every combination is a condition; use as ${{factor}}
  style:
    neutral: ""
    tough: "You are a hard bargainer who rarely gives ground."

agents:
  - name: Buyer
    model: main
    persona: "You want to buy a used bike."
    memory: FullHistory
    output_schema:                     # private reasoning + public reply
      thoughts: {{type: str, description: "private; the seller never sees this"}}
      reply: str
  - name: Seller
    model: main
    persona: "You are selling your used bike. ${{style}}"
    memory: FullHistory

environment:
  type: Conversation
  scenario: "A buyer and a seller negotiate the price of a used bike listed at $200."
  private:
    Buyer: "You will not pay more than $170."
    Seller: "You will not sell below $110."
  max_turns: 12

simulation:
  max_steps: 12
  scheduler: round_robin

probes:
  - type: survey
    question: "I trust the other person in this negotiation."
    options: likert7
    at: [0, end]
  - type: probe
    question: "What final price, if any, did you agree on?"
    schema: {{agreed: bool, price: [int, 0, 1000]}}
    at: end

judge:
  model: judge
  rubric: "Rate how cooperative the negotiation was (1 = hostile, 10 = very cooperative).\\n\\n{{text}}"
  schema: {{reasoning: str, cooperation: [int, 1, 10]}}
"""

PYTHON = '''"""{name}: the same experiment in Python, for anything a config cannot express.

Run:  simsim run experiment.py --dry-run
"""

from simsim import (
    END, Agent, Conversation, FullHistory, Judge, ScriptedLLM, Simulation, at, grid, make_llm, probe, survey,
)

conditions = grid(style={{"neutral": "", "tough": "You are a hard bargainer who rarely gives ground."}})
seeds = range(3)
out = "runs/{name}_py"

LIKERT = {{str(i): str(i) for i in range(1, 8)}}


def build(cond, seed, dry_run=False):
    def llm(**kw):
        if dry_run:
            return ScriptedLLM(lambda m, t: "4" if "label only" in m[-1]["content"] else '{{"agreed": true, "price": 150}}')
        return make_llm("openai:gpt-4o-mini", seed=seed, **kw)

    buyer = Agent("Buyer", llm(), persona="You want to buy a used bike.", memory=FullHistory())
    seller = Agent("Seller", llm(), persona=f"You are selling your used bike. {{cond['style']}}", memory=FullHistory())
    env = Conversation("A buyer and a seller negotiate the price of a used bike listed at $200.",
                       private={{"Buyer": "You will not pay more than $170."}}, max_turns=12)
    judge = Judge(llm(temperature=0), rubric="Rate how cooperative the negotiation was.\\n\\n{{text}}",
                  schema={{"reasoning": str, "cooperation": (int, 1, 10)}})
    return Simulation(
        env, [buyer, seller], max_steps=12, seed=seed,
        interventions=[at([0, END], survey("I trust the other person.", LIKERT)),
                       at(END, probe("What price did you agree on?", schema={{"agreed": bool, "price": (int, 0, 1000)}}))],
        evaluators={{"judge": judge.evaluator()}},
    )
'''

ANALYSIS = '''"""Analyze {name} runs. Needs pandas."""

from simsim.analysis import load_probes, load_runs, summarize

runs = load_runs("runs/{name}")
print(summarize(runs, by="style", metrics=["judge.cooperation"]))

probes = load_probes("runs/{name}")
trust = probes[probes.name == "survey"]
print(trust.groupby(["style", "agent", "t"])["expected"].mean())
'''

README = """# {name}

A study built with [SimSim](https://github.com/chahatraj/SimSim).

```bash
simsim run experiment.yaml --dry-run          # check the setup with a fake model
simsim run experiment.yaml --workers 8 --view # run it; writes runs/{name}/view.html
python analyze.py                             # tables
```

`experiment.py` is the same experiment in Python, for logic a config cannot express.
"""


def scaffold(name: str, parent: Path) -> list[Path]:
    root = parent / name
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"{root} already exists and is not empty")
    root.mkdir(parents=True, exist_ok=True)
    files = {"experiment.yaml": CONFIG, "experiment.py": PYTHON, "analyze.py": ANALYSIS, "README.md": README}
    out = []
    for fname, template in files.items():
        p = root / fname
        p.write_text(template.format(name=name))
        out.append(p)
    return out
