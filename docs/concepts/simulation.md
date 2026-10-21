# Simulation

A `Simulation` runs one environment with a set of agents for up to `max_steps`
steps. At each step the scheduler picks who acts; each chosen agent observes,
acts, and the environment applies the action.

```python
from simsim import Simulation, Clock, everyone_shuffled
sim = Simulation(env, agents,
                 scheduler=everyone_shuffled, max_steps=20, seed=0,
                 simultaneous=False, max_concurrency=16,
                 clock=Clock("2024-03-04 08:00", "30min"),
                 interventions=[...], evaluators={...},
                 checkpoint=None, checkpoint_every=10)
result = sim.run(recorder)        # a Recorder writes the log; optional
```

`result` has `turns`, `metrics`, `probes`, `usage`, `steps`, and `termination`
(why it stopped: `max_steps`, `ended by Buyer`, `judge`, `all tasks done`, ...).

## Schedulers

| Scheduler | Who acts each step |
|---|---|
| `round_robin` | one agent, in order (conversations) |
| `everyone` / `everyone_shuffled` | all agents, in fixed / random order |
| `random_one` | one random agent |
| `bernoulli(p)` | each agent independently with probability `p` (or `p(agent)`) |
| `zipf(s=1.0, k=1)` | `k` agents, Zipf-weighted: a few "power users" act most |
| `by_hour({hour: p})` | probability by simulated hour (needs a `Clock`) |
| `NextSpeaker(llm, guidance="")` | an LLM moderator picks who speaks next from the conversation |

A scheduler is any function `(agents, t, rng) -> list[Agent]`; add a `sim`
keyword argument to see the environment or clock.

## Sequential or simultaneous

By default each agent sees the previous agent's action. With `simultaneous=True`
all agents scheduled in a step observe the same state and act in parallel threads
(at most `max_concurrency` at once); their actions are applied afterwards in
order. Use this for large populations (hundreds of agents) and for designs where
moves should be simultaneous. Pair it with `max_concurrency` / `rpm` on the model
to respect rate limits.

## Clock

With `clock=Clock(start, step)` agents see `# Current time` in their prompt,
turns carry `turn.time`, and memories are timestamped. `step` is `"15min"`,
`"1h"`, `"1 day"` or a `timedelta`.

## Interventions

Things that happen to the simulation at chosen steps:

```python
from simsim import at, every, END, whisper, add_agent, probe, survey

interventions = [
    at(0,   survey("I trust the other side.", LIKERT)),          # before anything happens
    at(END, survey("I trust the other side.", LIKERT)),          # after the last step
    every(5, probe("What is your plan now?"), start=5),
    at(10,  whisper("Ana", "You just learned the bike was stolen.")),  # a planted memory
    at(15,  add_agent(newcomer)),
    at(8,   lambda sim, t: sim.env.inject_post("bot", "Fake news!")),  # anything
]
```

Probe and survey answers go to `result.probes` (and `load_probes` in analysis),
tagged with the step and the probe's `name`.

## Evaluators

`evaluators={"judge": judge.evaluator(), "length": lambda r: {"n": len(r.turns)}}`
runs functions of the `Result` at the end and stores their output under
`result.metrics[name]`. See [Judges](judges.md).

## Checkpoints

`checkpoint="runs/town.ckpt"` saves the whole simulation (agents, memories,
environment, RNG, progress) every `checkpoint_every` steps;
`Simulation.resume("runs/town.ckpt")` continues from there after a crash or a
killed job. Install `cloudpickle` if your setup uses lambdas or closures. The
checkpoint is deleted when the run finishes.
