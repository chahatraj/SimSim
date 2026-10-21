# SimSim

SimSim is a Python library for running experiments with LLM agents: a single
agent with memory and tools working through tasks, or many agents with personas
interacting in a shared world. You describe the agents, the world, and the
conditions you want to compare; SimSim runs every condition across seeds, logs
every model call, and gives you tables and a viewer to analyze the results.

It is built the way most agent-simulation papers are built: a short, transparent
loop, personas as text, swappable memory, any model behind an OpenAI-compatible
API, and complete logs. The difference is that you do not write that loop again
for every paper, and the features papers keep re-implementing are already here:

- **Structured replies with private reasoning**: agents answer in a schema, with
  fields the other agents never see; malformed replies are repaired automatically.
- **Probes and surveys** that do not change the agent: interview it mid-run, or
  read answer probabilities for Likert items straight from the model.
- **Memory and minds**: full history, windows, summaries, a notes file the agent
  edits itself, Generative Agents-style retrieval with reflection; plans,
  impressions of others, and inner states (emotions, trust, beliefs) as components.
- **Environments**: conversations (negotiation, debate, user simulators with
  tools), task sequences for memory experiments, and a small social network.
- **Experiments**: crossed conditions × seeds in parallel, resumable, with a
  response cache that replays runs for free, LLM judges, and checkpoints.
- **Any model**: OpenAI, Anthropic, Together, or open models served with vLLM,
  SGLang or Ollama, through one interface.

## Install

```bash
pip install "simsim[analysis] @ git+https://github.com/chahatraj/SimSim"
```

Python 3.10+. Pin a release for a paper: `...SimSim@v0.7.0`.

## A first simulation

```python
from simsim import Agent, Conversation, FullHistory, Simulation, make_llm

llm = make_llm("openai:gpt-4o-mini")   # or "vllm:Qwen/Qwen3.5-9B@http://node:8000/v1"
buyer  = Agent("Buyer",  llm, persona="You want a used bike.", memory=FullHistory())
seller = Agent("Seller", llm, persona="You are selling your bike.", memory=FullHistory())

env = Conversation("Negotiate the price of a bike listed at $200.",
                   private={"Buyer": "Your budget is $170."})
result = Simulation(env, [buyer, seller], max_steps=12, seed=0).run()
print(result.transcript())
```

Or start a whole study from the command line:

```bash
simsim new my_study && cd my_study
simsim run experiment.yaml --dry-run      # a fake model checks the setup for free
simsim run experiment.yaml --workers 8 --view
```

## Where to go next

- [Getting started](getting-started.md): your first experiment, step by step.
- [Concepts](concepts/models.md): how the pieces fit together.
- [Recipes](recipes/index.md): complete, runnable studies to copy.
- [Reporting in papers](papers.md): what to report and how SimSim helps.
