"""Persona generation: sample attributes from distributions or real survey rows,
then (optionally) have an LLM write a short biography from them.

    personas = sample_personas(
        100,
        {"age": range(18, 80), "party": {"Democrat": 0.33, "Republican": 0.29, "Independent": 0.38},
         "region": ["Northeast", "South", "Midwest", "West"]},
        seed=0, describe=make_llm("openai:gpt-4o-mini"))

    personas = personas_from_rows(csv.DictReader(open("anes_2020.csv")), fields=["age", "party", ...])
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable
from typing import Any

from .agent import Persona
from .llm import LLM

BIO_PROMPT = (
    "Write a short third-person biography (2-3 sentences, under {words} words) of a person with "
    "these attributes. Make them a specific, ordinary individual. Do not add political opinions "
    "beyond what the attributes say. Output only the biography.\n\n{attributes}"
)

Distribution = list | range | dict[Any, float] | Callable[[random.Random], Any]


def _draw(dist: Distribution, rng: random.Random) -> Any:
    if callable(dist):
        return dist(rng)
    if isinstance(dist, dict):
        return rng.choices(list(dist), weights=list(dist.values()), k=1)[0]
    return rng.choice(list(dist))


def describe_persona(p: Persona, llm: LLM, words: int = 60, prompt: str = BIO_PROMPT) -> Persona:
    """Fill `p.description` with an LLM-written biography of its attributes."""
    attrs = "\n".join(f"- {k}: {v}" for k, v in p.attributes.items())
    p.description = llm.complete(prompt.format(words=words, attributes=attrs)).strip()
    return p


def sample_personas(
    n: int,
    attributes: dict[str, Distribution],
    seed: int = 0,
    describe: LLM | None = None,
    name: Callable[[int, dict], str] = lambda i, attrs: f"person_{i}",
    words: int = 60,
) -> list[Persona]:
    """Draw `n` personas, each attribute independently from its distribution.

    A distribution is a list/range (uniform), a dict {value: weight}, or a
    function `rng -> value` (use this for dependent attributes).
    """
    rng = random.Random(seed)
    out = []
    for i in range(n):
        attrs = {k: _draw(d, rng) for k, d in attributes.items()}
        p = Persona(name(i, attrs), "", attrs)
        out.append(describe_persona(p, describe, words) if describe else p)
    return out


def personas_from_rows(
    rows: Iterable[dict[str, Any]],
    fields: list[str] | None = None,
    name_key: str | None = None,
    describe: LLM | None = None,
    sample: int | None = None,
    seed: int = 0,
    words: int = 60,
) -> list[Persona]:
    """Personas from real respondents (e.g. ANES, GSS, Prolific rows), one per row.

    `fields` picks the columns used as attributes (default: all); `sample` draws
    that many rows at random.
    """
    rows = list(rows)
    if sample is not None:
        rows = random.Random(seed).sample(rows, sample)
    out = []
    for i, row in enumerate(rows):
        attrs = {k: row[k] for k in (fields or row) if k != name_key}
        p = Persona(str(row[name_key]) if name_key else f"person_{i}", "", attrs)
        out.append(describe_persona(p, describe, words) if describe else p)
    return out
