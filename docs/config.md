# Config reference

`simsim run experiment.yaml` runs an experiment described in YAML (or JSON).
`simsim new NAME` writes a complete, commented example.

```yaml
name: negotiation                   # default output: runs/<name>
out: runs/negotiation
seeds: 5                            # or a list of seeds

models:                             # named models, referenced by agents and the judge
  main:
    spec: openai:gpt-4o-mini        # any make_llm spec
    temperature: 0.7
    max_tokens: 512
    price: [0.15, 0.60]             # optional, for cost_usd
    max_concurrency: 16             # optional
    rpm: 500                        # optional
    cache: cache/main.sqlite        # optional: replay identical runs for free
    extra_body: {chat_template_kwargs: {enable_thinking: false}}
  judge: {spec: openai:gpt-4o, temperature: 0}

grid:                               # crossed into conditions; use values as ${factor}
  style:
    neutral: ""
    tough: "You are a hard bargainer."
  model: [main]

agents:
  - name: Buyer
    model: main                     # a name from models:, or a spec string
    persona: "You want to buy a used bike."        # or {description: ..., age: 40, ...}
    instructions: "..."
    memory: FullHistory             # or {type: ScoredMemory, k: 8, embedder: HashEmbedder}
    components:
      - {type: Goal, text: "Pay at most $150."}
      - {type: InnerState, schema: {trust: [int, 1, 7]}}
    tools: ["mytools:lookup_order"]  # your own code: "module:attribute"
    output_schema:
      thoughts: {type: str, description: "private"}
      reply: str
    remember_observations: 500      # any other Agent argument
  - name: Seller
    model: main
    persona: "You are selling your bike. ${style}"

environment:
  type: Conversation                # Conversation, TaskSequence, Feed, or "module:MyEnv"
  scenario: "..."
  private: {Buyer: "Your budget is $170."}
  max_turns: 12
  tools: {Seller: "mytools:make_tools"}    # per agent: a list, or a factory world -> tools
  score: "mytools:score"

simulation:
  max_steps: 12
  scheduler: round_robin            # or {type: bernoulli, p: 0.3}, {type: zipf, s: 1.0}
  simultaneous: false
  clock: {start: "2024-05-06 08:00", step: "1h"}

probes:
  - {type: survey, question: "I trust the seller.", options: likert7, agents: [Buyer], at: [0, end]}
  - {type: probe, question: "What price did you agree on?", schema: {price: [int, 0, 1000]}, at: end}
  - {type: probe, question: "What is your plan?", every: 4, start: 4}
  - {type: whisper, agent: Buyer, text: "You just lost your job.", at: 6}

judge:
  model: judge
  rubric: "Rate how cooperative it was.\n\n{text}"
  schema: {reasoning: str, cooperation: [int, 1, 10]}
```

## Substitution

`${factor}` anywhere in the config is replaced by the condition's value. A string
that is exactly `${factor}` takes the value as is, so a factor value can be a whole
dict (e.g. a persona or a memory spec).

## Schemas in YAML

| YAML | Means |
|---|---|
| `str`, `int`, `float`, `bool` | a value of that type |
| `[agree, disagree, neutral]` | one of these strings |
| `[int, 1, 7]` or `{type: int, min: 1, max: 7}` | a number in a range |
| `{type: str, description: "..."}` | adds a description for the model |
| `{choices: [a, b]}` | one of these |

Survey `options` can be `likert5`, `likert7`, a list, or a mapping `{label: text}`.

## Populations

Instead of (or as well as) `agents:`, generate agents from attribute
distributions or a file of personas:

```yaml
population:
  count: 50
  attributes:                        # lists are uniform; mappings are weights
    age: [19, 25, 34, 45, 58, 67]
    politics: {liberal: 0.4, conservative: 0.35, moderate: 0.25}
  describe: main                     # optional: an LLM writes a short bio per person
  # file: personas.jsonl             # or real respondents: {"name": ..., "description": ..., ...}
  agent:                             # settings shared by every member
    model: main
    persona: "You are an ordinary user of a social media platform."   # prefix
    memory: {type: WindowMemory, k: 5}
```

## TaskSequence configs

```yaml
environment:
  type: TaskSequence
  tools: "mytools:make_tools"        # world -> tools
  tasks:
    - {prompt: "I always want an aisle seat.", id: plant}
    - {prompt: "Book me a flight to Lisbon.", id: probe, score: "mytools:booked_aisle"}
```

## Your own code

Any `type`, tool, `score` or scheduler can be `"module:attribute"`. Modules are
imported from the config file's folder, so a `mytools.py` next to the config just
works. When a design outgrows the config, write the same experiment in Python
(`experiment.py` from `simsim new`) and run it the same way:
`simsim run experiment.py`.
