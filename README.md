# SimSim

A small Python library for building LLM agent simulations for research. You can
run a single agent with memory and tools working through tasks, or many agents
with personas interacting in a shared world. Every model call is logged, and
experiments run across conditions and seeds.

SimSim follows how most recent agent-simulation papers are built: a short custom
loop, personas as text, simple swappable memory, any model behind an
OpenAI-compatible API, and full logs. It gives you that loop ready-made, adds the
features those papers keep re-implementing (structured outputs with repair,
private reasoning, surveys, LLM judges, interventions, user simulators), and keeps
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

llm = make_llm("openai:gpt-4o-mini")      # or "vllm:Qwen/Qwen3.5-9B@http://gpu012:8000/v1"
buyer  = Agent("Buyer",  llm, persona="You want a used bike.", memory=FullHistory())
seller = Agent("Seller", llm, persona="You are selling your bike.", memory=FullHistory())

env = Conversation("Negotiate the price of a bike listed at $200.",
                   private={"Buyer": "Your budget is $170."})
result = Simulation(env, [buyer, seller], max_steps=12, seed=0).run()
print(result.transcript(), result.termination)
```

## The pieces

| Piece | What it is | Built in |
|---|---|---|
| **LLM** | Any model, with retries/backoff and token and cost counts. `make_llm("provider:model")` | `openai`, `anthropic`, `together`, `vllm`, `ollama`, `litellm:*`, and `ScriptedLLM` (a fake model for tests) |
| **Agent** | Model + persona + memory + tools. `act(observation)` runs the tool loop | native tool calls or ReAct text; `output_schema` for structured replies; `ScriptedAgent`, `HumanAgent` |
| **Memory** | `write(entry)` after each turn, `read(query)` into the prompt, optional `tools()` | `NoMemory`, `FullHistory`, `WindowMemory`, `SummaryMemory`, `MarkdownMemory` (the agent edits its own notes file), `ScoredMemory` (Generative Agents-style recency + importance + relevance, with embeddings), `ReflectiveMemory` (+ reflection) |
| **Components** | Pluggable parts of an agent's mind: a prompt section each turn, updated after each turn, logged | `Goal`, `Plan`, `Impressions` (of other agents), `InnerState` (emotions, beliefs, trust: any schema) |
| **Tool** | A Python function with `@tool`; the signature and docstring become the schema | — |
| **Environment** | What each agent observes, which tools it gets, how actions change the world, when to stop | `Conversation` (also hosts user simulators), `TaskSequence`, `Feed` (a mini social network) |
| **Scheduler** | Who acts at each step | `round_robin`, `everyone`, `everyone_shuffled`, `random_one`, `bernoulli`, `zipf`, `by_hour` (needs a `Clock`), `NextSpeaker` (an LLM moderator picks) |
| **Simulation** | Runs the loop, interventions and evaluators; records why it stopped; parallel agents; checkpoints | `at`, `every`, `whisper`, `add_agent`, `probe`, `survey`, `Clock` |
| **Infrastructure** | Things every long experiment needs | `CachedLLM` (replay runs for free), `max_concurrency=` / `rpm=` on any LLM, `emulated_tool` (an LLM fakes a tool), `Agent.save`/`restore` |
| **Judge** | An LLM scores transcripts against a rubric, decides when to stop, re-scores saved runs | `Judge`, `rescore` |
| **Personas** | Sample attributes from distributions or survey rows; optional LLM-written bios | `sample_personas`, `personas_from_rows` |
| **Experiment** | Conditions × seeds, parallel workers, JSONL logs, resumes if interrupted | `grid(...)` crosses factors into conditions |

The design rule: **the environment decides what is new for an agent; memory decides
what the agent remembers.** Keep those separate and you can vary one as an
experimental condition without touching the other.

## Structured replies and private reasoning

```python
schema = {
    "thoughts": (str, "private; the others never see this"),
    "reply": str,                                   # the public message
    "offer": (int, 0, 1000),
    "stance": ["accept", "counter", "walk away"],
}
buyer = Agent("Buyer", llm, output_schema=schema, public_field="reply")
turn = buyer.act("Seller: $200, final offer.")
turn.text      # "How about $150?"            -> what other agents see
turn.private   # {"thoughts": ..., "offer": 150, "stance": "counter"}  -> logged, remembered by the buyer only
```

If a reply is not valid JSON or breaks the schema, the error is sent back and the
model tries again (`max_repairs`). Reasoning from thinking models (e.g. Qwen3 with a
vLLM reasoning parser) is captured separately in `turn.reasoning`.

For models or servers without native tool calling, use `action_format="react"`
(Thought / Action / Action Input / Final Answer, parsed with repair).

## Components and richer memory

```python
from simsim import Goal, Plan, Impressions, InnerState, ReflectiveMemory, OpenAIEmbedder

agent = Agent("Ana", llm, persona=...,
              memory=ReflectiveMemory(llm, embedder=OpenAIEmbedder("text-embedding-3-small"),
                                      weights=(0.5, 3, 2)),          # Park et al.'s weights
              components=[Goal("Get elected class president."), Plan(every=5), Impressions(),
                          InnerState({"mood": ["calm", "anxious", "angry"], "trust_in_Bo": (int, 1, 7)})])
```

Each turn's component states are in `turn.components` and the logs, so you can
plot trust or mood over time. Probes see components and memory read-only.

## Time, scale, caching, checkpoints

```python
Simulation(env, agents, clock=Clock("2024-03-04 08:00", "30min"),   # agents see the time
           scheduler=by_hour({h: 0.1 if h < 8 else 0.6 for h in range(24)}),
           simultaneous=True, max_concurrency=32,                   # many agents in parallel
           checkpoint="runs/town.ckpt")                             # resume with Simulation.resume(path)

llm = CachedLLM(make_llm("openai:gpt-4o-mini", seed=0, rpm=500), "cache/gpt4o-mini.sqlite")
```

With `CachedLLM`, rerunning the same code replays every response from disk (the n-th
identical request gets the n-th stored answer), so you can change analysis code or
add an evaluator and rerun at no cost.

## Probes: interviews and surveys

Questions that do not change the agent: nothing is written to its memory, so
probing mid-run does not alter what happens next.

```python
agent.ask("How do you feel about the other negotiator?")                   # free text
agent.ask("Rate your trust.", schema={"trust": (int, 1, 7)})               # validated dict
agent.survey("I trust the seller.", {"1": "Strongly disagree", ..., "7": "Strongly agree"})
# -> {"1": 0.02, ..., "7": 0.31}: next-token probabilities (OpenAI, vLLM)
```

Schedule them inside a run; answers land in `result.probes`:

```python
from simsim import at, every, END, probe, survey, whisper, add_agent

Simulation(env, agents, interventions=[
    at(0,   survey("I trust the seller.", LIKERT)),       # before
    at(END, survey("I trust the seller.", LIKERT)),       # after
    every(5, probe("What is your plan?")),
    at(10,  whisper("Buyer", "You just heard the bike was stolen.")),   # planted memory
    at(15,  add_agent(newcomer)),
])
```

Thinking models must have thinking off for `survey`
(`extra_body={"chat_template_kwargs": {"enable_thinking": False}}`).

## Judges

```python
from simsim import Judge
judge = Judge(make_llm("openai:gpt-4o"),
              rubric="Rate how cooperative the agents were.\n\n{text}",
              schema={"reasoning": str, "cooperation": (int, 1, 10)})

Simulation(env, agents, evaluators={"judge": judge.evaluator()})            # score each run
Conversation(..., judge=judge.stopper("Have they agreed on a price?"))      # end when the judge says so
rescore("runs/exp/results.jsonl", new_judge, render=lambda row: ...)        # re-score saved runs
```

Use a different, ideally stronger, model as the judge, and report it.

## User simulators and task agents

```python
from simsim import user_simulator
user = user_simulator("user", llm, goal="Get a refund for order 7; you have the receipt.")
assistant = Agent("assistant", llm, tools=[lookup_order, refund])
env = Conversation("Customer support chat.", end_token="###STOP###", enders=["user"],
                   tools={"assistant": [lookup_order, refund]},
                   score=lambda transcript, world: {"refunded": float(world.get("refunded", False))})
```

`TaskSequence` runs one or more agents through tasks in order with memory carried
over (the standard memory-experiment setup); each agent gets its own copy of the world.

## Personas

```python
from simsim import sample_personas, personas_from_rows
personas = sample_personas(100, {"age": range(18, 80),
                                 "party": {"Democrat": .33, "Republican": .29, "Independent": .38}},
                           seed=0, describe=llm)          # describe: LLM writes a short bio
personas = personas_from_rows(csv.DictReader(open("anes.csv")), fields=["age", "party", "religion"])
```

## Running an experiment

```python
from simsim import Experiment, grid, NoMemory, MarkdownMemory

def build(cond, seed):                      # returns a fresh Simulation per run
    llm = make_llm(cond["model"], seed=seed, temperature=0)
    agent = Agent("assistant", llm, memory=cond["memory"]())
    return Simulation(TaskSequence(tasks, tools=make_tools), [agent], seed=seed)

exp = Experiment(build,
                 grid(model=["openai:gpt-4o-mini", "vllm:Qwen/Qwen3.5-9B"],
                      memory={"none": NoMemory, "notes": MarkdownMemory}),
                 seeds=range(5), out="runs/memory_v1")
exp.run(workers=8)
```

The experiment writes these files under `runs/memory_v1/`:

```
conditions.json                  what each condition was
results.jsonl                    one row per run: condition, seed, metrics, probes, termination, token usage
model=gpt-4o-mini,memory=none/
    seed0.jsonl                  every LLM call (full prompt + response), tool call, turn, probe, intervention
```

If you rerun the same script, finished runs are skipped. Runs that crashed are
logged with an `error` field and retried next time. Build LLM objects inside
`build`, so token and cost counts are per run.

## Writing your own pieces

**A new environment.** Subclass `Environment` and implement `observe`. Implement
`tools_for`, `apply`, `metrics` and `add_agent` as needed; call `self.stop(reason)`
to end the run. See `simsim/envs/conversation.py`.

**A new memory.** Subclass `Memory` and implement `write` and `read`. To let the
agent manage its memory itself, also return tools from `tools()`.

**A different prompt.** Pass `template=` to `Agent`, using any of `{name}`,
`{persona}`, `{instructions}` and `{memory}`. The exact prompt sent each turn is
in the logs.

**An intervention.** Any function `fn(sim, t)`, scheduled with `at(...)` or `every(...)`.

## Open models on a Slurm cluster

```bash
mkdir -p logs && VLLM=/path/to/vllm CACHE=/path/to/cache TOOL_PARSER=qwen3_coder \
    EXTRA_ARGS="--reasoning-parser qwen3" sbatch scripts/serve_vllm.slurm Qwen/Qwen3.5-9B
grep Serving logs/vllm-*.out                     # -> http://gpu0XX:8000/v1
python scripts/smoke_test.py vllm:Qwen/Qwen3.5-9B@http://gpu0XX:8000/v1 --no-thinking
```

`smoke_test.py` checks chat, native tool calls, ReAct, structured output, surveys and
a short conversation against any endpoint. Tool calling needs the right
`TOOL_PARSER` for the model family; the script's header lists them. Adjust the
`#SBATCH` lines to your cluster. MIG GPU slices work (the script fixes the device id).

## Examples

```bash
python examples/negotiation.py --dry-run        # persona effect on price, 2 agents
python examples/memory_tasks.py --dry-run       # 5 memory types, planted preference
python examples/social_feed.py --dry-run        # 6 personas, feed ranking condition
python examples/identity_conflict.py --dry-run  # rival national identities co-writing history
```

Drop `--dry-run` and add `--model ...` to run them on a real model.

## Notes for papers

- Report the model, its version, the temperature and the number of seeds; all
  four are in `start` events and `conditions.json`.
- Token usage (and cost, if you pass `price=` to the LLM) per run is in `results.jsonl`.
- Recent work finds the base model often matters more than any other design
  choice, so include more than one model in your grid.

## Status

Early (v0.3). Interfaces may change.

## License

MIT
