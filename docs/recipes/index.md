# Recipes

Complete studies you can run, copy and adapt. Each is a file in the repository's
`examples/` folder and runs with `--dry-run` (a fake model, no cost) or with a
real model via `--model`.

| Recipe | Shows | File |
|---|---|---|
| [Negotiation](negotiation.md) | two agents, private information and reasoning, surveys before/after, a judge | `examples/configs/negotiation.yaml`, `examples/negotiation.py` |
| [Support agent vs. a simulated user](support-agent.md) | user simulator, tools acting on a per-run world, policy checks | `examples/configs/support_agent.yaml` |
| [Memory experiment](memory-experiment.md) | one agent across sessions, five memory types as conditions | `examples/memory_tasks.py` |
| [Survey panel](survey-panel.md) | persona-conditioned survey answers from token probabilities | `examples/survey_panel.py` |
| [Social feed](social-feed.md) | a sampled population on a mini social network, clock, ranking as a condition | `examples/configs/social_feed.yaml`, `examples/social_feed.py` |
| [A small town](town.md) | Generative Agents-style residents: reflection, plans, impressions, a planted memory | `examples/town.py` |
| [Your own environment](custom-environment.md) | a repeated public-goods game written from scratch | `examples/public_goods.py` |

Real-model outputs quoted on these pages come from small open models (Qwen3.5-4B)
with few seeds; they illustrate what the recipe measures, not findings.
