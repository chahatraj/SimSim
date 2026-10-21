# Social feed

A sampled population on a minimal social network. The feed's ranking (recent vs.
popular) is the condition; agents act at random times on a clock, and a survey
measures perceived hostility before and after.

```bash
simsim run examples/configs/social_feed.yaml --dry-run
simsim run examples/configs/social_feed.yaml --workers 2 --view
```

```yaml
population:
  count: 30
  attributes:
    age: [19, 25, 34, 45, 58, 67]
    politics: {liberal: 0.4, conservative: 0.35, moderate: 0.25}
  agent: {model: main, memory: {type: WindowMemory, k: 5}, remember_observations: 300}
environment: {type: Feed, ranking: "${ranking}", news: [...]}
simulation:
  scheduler: {type: bernoulli, p: 0.3}
  simultaneous: true
  clock: {start: "2024-05-06 08:00", step: "1h"}
```

With 50 agents acting simultaneously, a 4B model on one 40 GB A100 slice
produced 150 turns (128 posts, 41 likes) in about 40 seconds.

`remember_observations: 300` matters here: the observation is the whole timeline,
so storing it in full every turn would fill memory with repeated posts. Analyze
the network from `metrics.posts` and `metrics.follows` in `results.jsonl`, and
inject content mid-run with an intervention:

```python
at(5, lambda sim, t: sim.env.inject_post("newsbot", "Unverified: the bridge is closing."))
```
