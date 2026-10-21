# Analysis

`simsim.analysis` loads experiment outputs into pandas tables (`pip install pandas`).

```python
from simsim.analysis import load_runs, load_turns, load_probes, summarize, pass_hat_k, cost_report, transcript

runs = load_runs("runs/negotiation")
```

| Function | One row per | Columns |
|---|---|---|
| `load_runs(dir)` | run | condition, seed, one column per grid factor, steps, termination, flattened metrics (`judge.cooperation`), `usage.*` |
| `load_turns(dir)` | agent turn | agent, t, time, text, observation, reasoning, `private.*`, `component.*` (e.g. `component.InnerState.trust`), `usage.*`, error |
| `load_probes(dir)` | probe or survey answer | name, t, agent, question; surveys: `p.<label>`, `top`, `expected`; structured probes: `answer.<field>` |

`load_*` accept an experiment directory, a file, or a list of them (e.g. several
models' experiment folders); the `experiment` column says which.

## Summaries

```python
summarize(runs, by=["model", "style"], metrics=["judge.cooperation", "n_messages"])
```

returns mean, standard error, n and a 95% confidence interval (normal
approximation) per group and metric. For papers, prefer a proper model (mixed
effects with seed or scenario as a random effect) on the raw tables; `summarize`
is for looking.

## Surveys

`expected` is the probability-weighted mean of numeric labels, a smoother measure
than the most likely answer:

```python
probes = load_probes("runs/negotiation")
trust = probes[probes.name == "survey"]
trust.pivot_table(index=["style", "agent"], columns="t", values="expected")   # before vs. after
```

## Other helpers

- `pass_hat_k(df, success="mean_success", k=3, task="task_id")`: tau-bench's
  pass^k, the chance that k trials of a task all succeed.
- `cost_report(runs, prices={"gpt-4o-mini": (0.15, 0.60)}, by="model")`: tokens
  and dollars per group.
- `transcript(log, private=True)`: one run as readable text, with private fields
  and reasoning.

## The viewer

```bash
simsim view runs/negotiation               # runs/negotiation/view.html
simsim view runs/negotiation --prompts     # include every full prompt (large)
```

One self-contained HTML file, viewable offline and easy to share. It shows each
run's agents, every turn (with private fields, tool calls, component states,
reasoning, and what the agent saw), probes, survey distributions, interventions,
judge results and metrics, with a search box across runs.
