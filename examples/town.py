"""A tiny Generative Agents-style town (Park et al., 2023): residents with reflective
memory, plans and impressions of each other, on a clock, talking in a shared square.

One resident is told (via `whisper`) that a party is happening; does the news spread,
and who shows up knowing about it? Information diffusion is measured with probes.

    python examples/town.py --model vllm:Qwen/Qwen3.5-9B@http://node:8000/v1
    python examples/town.py --dry-run
"""

import argparse

from simsim import (
    END,
    Agent,
    Clock,
    Conversation,
    Experiment,
    HashEmbedder,
    Impressions,
    Plan,
    ReflectiveMemory,
    ScriptedLLM,
    Simulation,
    at,
    bernoulli,
    grid,
    make_llm,
    probe,
    whisper,
)

RESIDENTS = {
    "Isabella": "Isabella runs the café on the square. She is warm and loves organizing events.",
    "Klaus": "Klaus is a sociology student writing a thesis on gentrification. He is earnest and a bit shy.",
    "Maria": "Maria is a physics student who streams video games at night. She is curious and talkative.",
    "Tom": "Tom owns the hardware store. He is practical, gruff, and suspicious of newcomers.",
    "Ayesha": "Ayesha is a nurse who just moved to town. She wants to make friends.",
}


def build(cond, seed, model="openai:gpt-4o-mini", dry_run=False):
    def llm():
        if dry_run:
            return ScriptedLLM(lambda m, t: '{"knows": true, "details": "party"}' if "JSON" in m[-1]["content"]
                               else "Lovely day on the square.")
        return make_llm(model, seed=seed, temperature=0.8, max_tokens=300,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}})

    residents = [
        Agent(name, (m := llm()), persona=desc,
              memory=ReflectiveMemory(m, threshold=4.0, embedder=HashEmbedder(), importance=lambda text: 0.5),
              components=[Plan(every=6, horizon="today"), Impressions()], remember_observations=400)
        for name, desc in RESIDENTS.items()
    ]
    party = "Isabella is hosting a Valentine's Day party at the café on Friday at 5pm."
    env = Conversation("You are in the town square, chatting with whoever is around. Keep messages to 1-2 sentences.",
                       end_token=None, show="new")
    knows = probe("Is anything special happening in town this week? If so, what and when?",
                  schema={"knows": bool, "details": str}, name="knows_party")
    return Simulation(
        env, residents, scheduler=bernoulli(0.5), max_steps=cond["steps"], seed=seed,
        clock=Clock("2024-02-12 09:00", "20min", fmt="%A %H:%M"),
        interventions=[at(0, whisper(cond["seed_agent"], party)), at(END, knows)],
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai:gpt-4o-mini")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="runs/town")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    exp = Experiment(lambda c, s: build(c, s, args.model, args.dry_run),
                     grid(seed_agent=["Isabella", "Tom"], steps=[args.steps]), seeds=range(args.seeds), out=args.out)
    exp.run(workers=args.workers)
    for row in exp.results():
        if row.get("error"):
            continue
        heard = [p["agent"] for p in row["probes"] if p["name"] == "knows_party" and (p["answer"] or {}).get("knows")]
        print(f"{row['condition']:35s} seed={row['seed']}  knows about the party: {', '.join(heard) or 'nobody'}")
