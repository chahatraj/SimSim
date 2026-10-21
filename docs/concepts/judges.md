# Judges

A `Judge` is an LLM that scores text against a rubric with structured output.

```python
from simsim import Judge
judge = Judge(make_llm("openai:gpt-4o", temperature=0),
              rubric="Rate how cooperative this negotiation was.\n\n{text}",
              schema={"reasoning": (str, "brief justification, before the score"),
                      "cooperation": (int, 1, 10)})
judge.score(transcript_text)            # {"reasoning": ..., "cooperation": 7}
```

The rubric can use `{text}` and any keyword you pass to `score(text, **context)`;
without `{text}`, the text is appended. Put a reasoning field first so the judge
explains before it scores. Malformed answers are repaired like any structured
output; if repair fails the result is `{"error": ...}` instead of an exception.

## Three ways to use one

```python
Simulation(env, agents, evaluators={"judge": judge.evaluator()})        # score every run
Conversation(..., judge=judge.stopper("Have they agreed on a price?"))   # end runs when it says yes
from simsim import rescore
rescore("runs/exp/results.jsonl", new_judge, render=lambda row: ..., out="rescored.jsonl")
```

`evaluator(render=...)` scores `render(result)` (default: the transcript).
`rescore` scores finished runs again, for example with a stronger judge or a new
rubric, without re-running any simulation.

## Good practice

- Use a different, ideally stronger, model family as the judge than as the
  agents, and report it.
- Validate the judge on a sample against human ratings before relying on it.
- Keep the rubric fixed across conditions; it is part of the measurement.
- Prefer probes and surveys of the agents themselves, or programmatic checks of
  the world (`score=`), where they can answer the question.
