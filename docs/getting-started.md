# Getting started

This page walks through a complete small study: two agents negotiate the price of
a bike, and we test whether making the seller a tough bargainer changes the
outcome and how much the buyer trusts the seller.

## 1. Install and pick a model

```bash
pip install "simsim[analysis] @ git+https://github.com/chahatraj/SimSim"
export OPENAI_API_KEY=...            # or serve an open model, see "Open models"
simsim smoke openai:gpt-4o-mini      # checks chat, tools, schemas, surveys, ...
```

A model is named by a spec string, `provider:model`, optionally with `@url` for a
local server:

| Spec | Meaning |
|---|---|
| `openai:gpt-4o-mini` | OpenAI API (`OPENAI_API_KEY`) |
| `anthropic:claude-haiku-4-5` | Anthropic's OpenAI-compatible API (`ANTHROPIC_API_KEY`) |
| `together:meta-llama/Llama-3.3-70B-Instruct-Turbo` | Together AI (`TOGETHER_API_KEY`) |
| `vllm:Qwen/Qwen3.5-9B@http://gpu012:8000/v1` | a vLLM server |
| `ollama:llama3.1` | a local Ollama server |
| `litellm:gemini/gemini-2.5-flash` | anything LiteLLM supports (`pip install litellm`) |

## 2. Agents and an environment

```python
from simsim import Agent, Conversation, FullHistory, Simulation, make_llm

def build(style, seed):
    llm = make_llm("openai:gpt-4o-mini", seed=seed)
    buyer = Agent("Buyer", llm, persona="You want to buy a used bike.", memory=FullHistory(),
                  output_schema={"thoughts": (str, "private; the seller never sees this"), "reply": str})
    seller = Agent("Seller", llm, persona=f"You are selling your used bike. {style}", memory=FullHistory())
    env = Conversation("A buyer and a seller negotiate the price of a used bike listed at $200.",
                       private={"Buyer": "You will not pay more than $170."}, max_turns=12)
    return Simulation(env, [buyer, seller], max_steps=12, seed=seed)

result = build("You are a hard bargainer.", seed=0).run()
for turn in result.turns:
    print(turn.agent, ":", turn.text, turn.private)
```

- The **environment** decides what each agent sees each turn (here: the scenario,
  its private information, and the messages since its last turn).
- **Memory** decides what the agent remembers (here: everything).
- `output_schema` makes the buyer answer in JSON; `reply` is what the seller sees,
  and `thoughts` is private, logged for you and remembered by the buyer only.

## 3. Measure something: surveys and a judge

```python
from simsim import END, Judge, at, survey

LIKERT = {"1": "Strongly disagree", "2": "Disagree", "3": "Neutral", "4": "Agree", "5": "Strongly agree"}
trust = survey("I trust the other person in this negotiation.", LIKERT, agents=["Buyer"])
judge = Judge(make_llm("openai:gpt-4o", temperature=0),
              rubric="Rate how cooperative this negotiation was.\n\n{text}",
              schema={"reasoning": str, "cooperation": (int, 1, 10)})

Simulation(env, [buyer, seller], max_steps=12, seed=seed,
           interventions=[at([0, END], trust)],              # before and after
           evaluators={"judge": judge.evaluator()})
```

A survey reads the probability of each answer from the model's next token, so it
is a distribution (`{"1": 0.02, ..., "5": 0.31}`), not a single sampled answer. It
never touches the agent's memory, so measuring does not change the run.

## 4. Run every condition across seeds

```python
from simsim import Experiment, grid

STYLES = {"neutral": "", "tough": "You are a hard bargainer who rarely gives ground."}
exp = Experiment(lambda cond, seed: build(cond["style"], seed),
                 grid(style=STYLES), seeds=range(10), out="runs/negotiation")
exp.run(workers=8)
```

This runs 2 conditions × 10 seeds, 8 at a time. Each run's full log goes to
`runs/negotiation/<condition>/seed<k>.jsonl`, and one summary row per run to
`runs/negotiation/results.jsonl`. Stop it at any point; running it again skips
finished runs.

## 5. Look at the results

```bash
simsim view runs/negotiation          # writes runs/negotiation/view.html
simsim summarize runs/negotiation --by style --metric judge.cooperation
```

```python
from simsim.analysis import load_runs, load_probes, summarize
runs = load_runs("runs/negotiation")
summarize(runs, by="style", metrics="judge.cooperation")      # mean, sem, n, 95% CI
probes = load_probes("runs/negotiation")
probes.groupby(["style", "t"])["expected"].mean()             # trust before vs. after
```

## 6. The same study as a config

Everything above fits in a YAML file that `simsim run` executes; `simsim new`
writes one for you. See the [config reference](config.md).

```bash
simsim new negotiation && cd negotiation
simsim run experiment.yaml --dry-run
simsim run experiment.yaml --workers 8 --view
```

`--dry-run` replaces every model with a fake one that returns valid answers for
every schema, so you can check a whole setup (and its logs and analysis) before
spending anything.
