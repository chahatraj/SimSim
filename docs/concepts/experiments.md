# Experiments and logs

An `Experiment` runs a fresh simulation for every condition × seed:

```python
from simsim import Experiment, grid, NoMemory, FullHistory

def build(cond, seed):                       # returns a new Simulation
    llm = make_llm(cond["model"], seed=seed)
    agent = Agent("assistant", llm, memory=cond["memory"]())
    return Simulation(TaskSequence(tasks, tools=make_tools), [agent], seed=seed)

exp = Experiment(build,
                 grid(model=["openai:gpt-4o-mini", "vllm:Qwen/Qwen3.5-9B@http://node:8000/v1"],
                      memory={"none": NoMemory, "full": FullHistory}),
                 seeds=range(5), out="runs/memory_v1")
rows = exp.run(workers=8)
```

`grid` crosses factors into named conditions. A list's values name themselves
(the part after the last `/`); a dict maps short names to values. The condition
names (`model=gpt-4o-mini,memory=none`) become one column per factor in the
analysis tables.

Create LLM objects inside `build` so token and cost counts are per run.

## Output files

```
runs/memory_v1/
  conditions.json                 the conditions and their parameters
  results.jsonl                   one row per run: condition, seed, steps, termination,
                                  metrics, probes, usage (tokens, cost, cache hits)
  model=gpt-4o-mini,memory=none/
    seed0.jsonl                   the full log of one run
```

Each line of a run log is one event: `start` (agents, personas, models, memory and
component types, seed), `llm_call` (the exact messages sent and the response,
reasoning, tool calls, usage), `tool_call`, `turn` (observation, public text,
private fields, components, errors), `repair`, `probe`, `survey`,
`intervention`, `evaluation`, and `end` (metrics, usage, memory summary).
Everything a reviewer could ask about a run is in its log.

## Resuming and failures

Running the same experiment again skips runs already in `results.jsonl`, so an
interrupted job continues where it stopped. A run that raises is recorded with an
`error` field and its traceback in the log, and is retried next time; the other
runs are unaffected.

## Parallelism

`workers` runs that many simulations at once in threads (model calls are
network-bound). For a local vLLM server, 8–32 workers keep the GPU busy; for APIs,
set `rpm`/`max_concurrency` on the model as well.

## Costs

Pass `price=(in, out)` (dollars per million tokens) to `make_llm` and runs report
`cost_usd`; or compute it later with `simsim.analysis.cost_report`. Use
`CachedLLM` to rerun analyses without paying again.
