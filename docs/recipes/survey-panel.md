# Survey panel

"Silicon sampling": condition agents on personas and ask survey items, then
compare answer distributions across groups. No interaction, so it is cheap.

```bash
python examples/survey_panel.py --dry-run
python examples/survey_panel.py --model vllm:Qwen/Qwen3.5-9B@http://node:8000/v1 --n 200
```

```python
people = sample_personas(n, {"age": range(18, 85),
                             "party": {"Democrat": 0.33, "Republican": 0.29, "Independent": 0.38},
                             "region": [...], "education": [...]}, seed=0)
for p in people:
    agent = Agent(p.name, llm, persona=p, instructions="Answer as this person would.")
    probs = agent.survey("Government should do more to address climate change.", LIKERT)
```

`survey` returns the probability of each label as the model's next token, so you
get a full distribution per respondent, not one sample. A run with Qwen3.5-4B
(n=30) gave mean agreement with "Government should do more to address climate
change" of 4.19 for Democrats, 3.70 for Independents and 2.78 for Republicans, the
direction of real polling, with a compressed spread, a known pattern for small
models.

To condition on real respondents, load rows with `personas_from_rows` (e.g. ANES
or GSS) and compare model answers with each respondent's actual answer. Use
`describe=llm` to turn attributes into short biographies if you want richer
personas, and report which you used: it changes the results.
