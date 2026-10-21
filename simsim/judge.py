"""LLM-as-judge: score transcripts against a rubric, decide when to stop, re-score saved runs.

    judge = Judge(make_llm("openai:gpt-4o"), rubric="How cooperative was {agents}? ...",
                  schema={"reasoning": str, "cooperation": (int, 1, 10)})
    sim = Simulation(env, agents, evaluators={"judge": judge.evaluator()})
    env = Conversation(..., judge=judge.stopper("Have they agreed on a final price?"))

Use a different (ideally stronger) model than the agents, and report it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .llm import LLM
from .record import read_jsonl
from .structured import Schema, structured_chat

DEFAULT_SCHEMA: Schema = {"reasoning": (str, "brief justification, written before the score"), "score": (int, 1, 10)}

JUDGE_SYSTEM = "You are a careful, impartial evaluator. Judge only what is in the text."


class Judge:
    """
    Args:
        llm: the judge model.
        rubric: instructions; may use {text} (the transcript) and any keyword passed to `score`.
            If {text} is absent, the text is appended after the rubric.
        schema: fields to return (see `simsim.structured`). Put a reasoning field first so
            the judge explains before scoring.
    """

    def __init__(self, llm: LLM, rubric: str, schema: Schema | None = None, system: str = JUDGE_SYSTEM,
                 max_repairs: int = 2):
        self.llm, self.rubric, self.system, self.max_repairs = llm, rubric, system, max_repairs
        self.schema = schema or DEFAULT_SCHEMA

    def score(self, text: str, **context: Any) -> dict[str, Any] | None:
        if "{text}" in self.rubric:
            prompt = self.rubric.format(text=text, **context)
        else:  # only format when asked to, so rubrics may contain literal braces
            rubric = self.rubric.format(**context) if context else self.rubric
            prompt = f"{rubric}\n\nText to evaluate:\n{text}"
        messages = [{"role": "system", "content": self.system}, {"role": "user", "content": prompt}]
        data, _, errors = structured_chat(self.llm, messages, self.schema, self.max_repairs)
        return data if data is not None else {"error": "; ".join(errors)}

    def evaluator(self, render: Callable[[Any], str] | None = None, **context: Any) -> Callable[[Any], dict]:
        """For `Simulation(evaluators={...})`: scores `render(result)` (default: the transcript)."""
        return lambda result: self.score((render or (lambda r: r.transcript()))(result), **context)

    def stopper(self, question: str) -> Callable[[str], bool]:
        """For `Conversation(judge=...)`: True when the judge answers yes to `question`."""
        schema = {"answer": ["yes", "no"]}

        def stop(text: str) -> bool:
            messages = [
                {"role": "system", "content": self.system},
                {"role": "user", "content": f"Conversation:\n{text}\n\nQuestion: {question}"},
            ]
            data, _, _ = structured_chat(self.llm, messages, schema, self.max_repairs)
            return bool(data) and data["answer"] == "yes"

        return stop


def rescore(results: str | Path, judge: Judge, render: Callable[[dict], str], name: str = "judge",
            out: str | Path | None = None) -> list[dict[str, Any]]:
    """Score finished runs again with a (new) judge, without re-running the simulation.

    `results` is an experiment's results.jsonl; `render(row)` turns a row (its
    `metrics`, e.g. the transcript) into the text to score. Writes rows with
    `row[name]` set to `out` if given and returns them.
    """
    import json

    rows = [r for r in read_jsonl(results) if not r.get("error")]
    for r in rows:
        r[name] = judge.score(render(r))
    if out:
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")
    return rows
