# SimSim

A small Python library for building LLM agent simulations for research. You can
run a single agent with memory and tools working through tasks, or many agents
with personas interacting in a shared world. Every model call is logged, and
experiments run across conditions and seeds.

SimSim follows how most recent agent-simulation papers are built: a short custom
loop, personas as text, simple swappable memory, any model behind an
OpenAI-compatible API, and full logs. It gives you that loop ready-made and keeps
every piece replaceable.

## Install

```bash
pip install -e .            # needs Python 3.10+; core dependency is just `openai`
pip install -e ".[dev]"     # + pytest
pytest                      # runs on a fake model, costs nothing
```

## Thirty-second tour

```python
from simsim import Agent, Conversation, FullHistory, Simulation, make_llm

llm = make_llm("openai:gpt-4o-mini")      # or "vllm:Qwen/Qwen2.5-7B-Instruct@http://gpu012:8000/v1"
buyer  = Agent("Buyer",  llm, persona="You want a used bike.", memory=FullHistory())
seller = Agent("Seller", llm, persona="You are selling your bike.", memory=FullHistory())

env = Conversation("Negotiate the price of a bike listed at $200.",
                   private={"Buyer": "Your budget is $170."})
result = Simulation(env, [buyer, seller], max_steps=12, seed=0).run()
print(result.metrics["transcript"])
```

## The pieces

| Piece | What it is | Built in |
|---|---|---|
| **LLM** | Any model. `make_llm("provider:model")` | `openai`, `anthropic`, `together`, `vllm`, `ollama`, `litellm:*`, and `ScriptedLLM` (a fake model for tests) |
| **Agent** | Model + persona + memory + tools. `act(observation)` runs the tool-calling loop | `Persona` (text + attributes, loadable from JSONL) |
| **Memory** | `write(entry)` after each turn, `read(query)` into the prompt, optional `tools()` | `NoMemory`, `FullHistory`, `WindowMemory`, `SummaryMemory`, `MarkdownMemory` (the agent edits its own notes file), `ScoredMemory` (Generative Agents-style recency + importance + relevance) |
| **Tool** | A Python function with `@tool`; the signature and docstring become the schema | — |
| **Environment** | Decides what each agent observes, which tools it gets, and how actions change the world | `Conversation`, `TaskSequence`, `Feed` (a mini social network) |
| **Scheduler** | Who acts at each step | `round_robin`, `everyone`, `everyone_shuffled`, `random_one`, `zipf` |
| **Simulation** | Runs the loop; `simultaneous=True` lets agents act in parallel on the same state | — |
| **Experiment** | Conditions × seeds, parallel workers, JSONL logs, resumes if interrupted | `grid(...)` crosses factors into conditions |

The design rule: **the environment decides what is new for an agent; memory decides
what the agent remembers.** Keep those separate and you can vary one as an
experimental condition without touching the other.

## Running an experiment

```python
from simsim import Experiment, grid, NoMemory, MarkdownMemory

def build(cond, seed):                      # returns a fresh Simulation per run
    llm = make_llm(cond["model"], seed=seed, temperature=0)
    agent = Agent("assistant", llm, memory=cond["memory"]())
    return Simulation(TaskSequence(tasks, tools=make_tools), [agent], seed=seed)

exp = Experiment(build,
                 grid(model=["openai:gpt-4o-mini", "vllm:Qwen/Qwen2.5-7B-Instruct"],
                      memory={"none": NoMemory, "notes": MarkdownMemory}),
                 seeds=range(5), out="runs/memory_v1")
exp.run(workers=8)
```

The experiment writes these files under `runs/memory_v1/`:

```
conditions.json                  what each condition was
results.jsonl                    one row per run: condition, seed, metrics, token usage
model=gpt-4o-mini,memory=none/
    seed0.jsonl                  every LLM call (full prompt + response), tool call, and turn
```

If you rerun the same script, finished runs are skipped. Runs that crashed are
logged with an `error` field and retried next time.

## Writing your own pieces

**A new environment.** Subclass `Environment` and implement `observe`. Implement
`tools_for`, `apply`, `done` and `metrics` as needed. See
`simsim/envs/conversation.py` (about 80 lines).

**A new memory.** Subclass `Memory` and implement `write` and `read`. To let the
agent manage its memory itself, also return tools from `tools()`.

**A different prompt.** Pass `template=` to `Agent`, using any of `{name}`,
`{persona}`, `{instructions}` and `{memory}`. The exact prompt sent each turn is
in the logs.

## Open models on a Slurm cluster

```bash
mkdir -p logs && VLLM=/path/to/vllm CACHE=/path/to/cache sbatch scripts/serve_vllm.slurm Qwen/Qwen2.5-7B-Instruct
grep Serving logs/vllm-*.out      # -> http://gpu0XX:8000/v1
```

Then use `make_llm("vllm:Qwen/Qwen2.5-7B-Instruct@http://gpu0XX:8000/v1")`.
Tool calling needs the right `TOOL_PARSER` for the model family; the script's
header lists them. Adjust the `#SBATCH` lines to your cluster's partitions.

## Status

Early (v0.1). The core is tested against a fake model; testing on real models is
in progress. Interfaces may change.

## License

MIT

## Examples

```bash
python examples/negotiation.py --dry-run     # persona effect on price, 2 agents
python examples/memory_tasks.py --dry-run    # 5 memory types, planted preference
python examples/social_feed.py --dry-run     # 6 personas, feed ranking condition
```

Drop `--dry-run` and add `--model ...` to run them on a real model.

## Notes for papers

- Report the model, its version, the temperature and the number of seeds; all
  four are in `start` events and `conditions.json`.
- Token usage per run is in `results.jsonl`, which you can use to report cost.
- Recent work finds the base model often matters more than any other design
  choice, so include more than one model in your grid.
