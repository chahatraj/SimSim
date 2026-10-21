"""A small social network: does timeline ranking change who interacts with whom?

    python examples/social_feed.py --model vllm:Qwen/Qwen2.5-7B-Instruct --steps 30
    python examples/social_feed.py --dry-run
"""

import argparse
import random
from pathlib import Path

from simsim import Agent, Experiment, Feed, Persona, ScriptedLLM, Simulation, WindowMemory, grid, make_llm, zipf

PERSONAS = Persona.load_jsonl(Path(__file__).parent / "personas.jsonl")
NEWS = [
    "City council approves new bike lanes downtown.",
    "Gas prices rise for the third straight week.",
    "State announces teacher pay raise.",
    "Heat wave expected through the weekend.",
]


def build(cond, seed, model="openai:gpt-4o-mini", steps=30, dry_run=False):
    personas = list(PERSONAS)
    random.Random(seed).shuffle(personas)  # who is a 'power user' varies by seed
    agents = [
        Agent(
            p.name,
            ScriptedLLM(lambda m, t: "I scrolled past.") if dry_run else make_llm(model, seed=seed),
            persona=p,
            memory=WindowMemory(k=6),
            max_tool_steps=3,
        )
        for p in personas
    ]
    env = Feed(news=NEWS, ranking=cond["ranking"], timeline_size=8)
    return Simulation(env, agents, scheduler=zipf(s=1.0, k=1), max_steps=steps, seed=seed)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai:gpt-4o-mini")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="runs/social_feed")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    exp = Experiment(
        lambda c, s: build(c, s, args.model, args.steps, args.dry_run),
        grid(ranking=["recent", "popular"]),
        seeds=range(args.seeds),
        out=args.out,
    )
    exp.run(workers=args.workers)
    for row in exp.results():
        if not row.get("error"):
            m = row["metrics"]
            n_follows = sum(len(v) for v in m["follows"].values())
            print(f"{row['condition']:20s} seed={row['seed']} posts={m['n_posts']} follows={n_follows}")
