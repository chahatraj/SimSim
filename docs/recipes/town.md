# A small town

Five residents with Generative Agents-style minds (reflective memory, a daily
plan, impressions of each other) chat in the town square on a clock. One of them
is told a party is happening (`whisper`); at the end, every resident is asked
whether they know of anything happening this week. Who first hears the news is
the condition.

```bash
python examples/town.py --dry-run
python examples/town.py --model vllm:Qwen/Qwen3.5-9B@http://node:8000/v1 --seeds 5
```

```python
Agent(name, llm, persona=desc,
      memory=ReflectiveMemory(llm, threshold=4.0, embedder=HashEmbedder()),
      components=[Plan(every=6, horizon="today"), Impressions()])
...
Simulation(env, residents, scheduler=bernoulli(0.5), clock=Clock("2024-02-12 09:00", "20min"),
           interventions=[at(0, whisper("Tom", party)), at(END, probe(..., name="knows_party"))])
```

Each turn logs the residents' plans and impressions, for example Klaus's
impression of Isabella after a few exchanges: *"her frantic hospitality stems from
genuine fear of losing her home, rather than just trying to impress."*

Swap `HashEmbedder` for a real embedding model, and pass
`importance=llm_importance(llm)`, for retrieval closer to the original paper.
