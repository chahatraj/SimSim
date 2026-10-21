"""Silicon sampling: ask a panel of persona-conditioned agents survey questions and
compare answer distributions across groups (Argyle et al. 2023-style).

No interaction here: each agent answers each item once, from token probabilities,
so this is cheap and deterministic. Swap `sample_personas` for `personas_from_rows`
to condition on real respondents (e.g. ANES rows).

    python examples/survey_panel.py --model vllm:Qwen/Qwen3.5-9B@http://node:8000/v1 --n 60
    python examples/survey_panel.py --dry-run
"""

import argparse

from simsim import Agent, ScriptedLLM, make_llm, sample_personas

ITEMS = {
    "climate": "Government should do more to address climate change.",
    "guns": "Gun laws in this country should be stricter.",
    "immigration": "Immigration is good for this country overall.",
}
LIKERT = {"1": "Strongly disagree", "2": "Disagree", "3": "Neither agree nor disagree", "4": "Agree",
          "5": "Strongly agree"}
ATTRIBUTES = {
    "age": range(18, 85),
    "party": {"Democrat": 0.33, "Republican": 0.29, "Independent": 0.38},
    "region": ["Northeast", "Midwest", "South", "West"],
    "education": ["high school", "some college", "bachelor's degree", "graduate degree"],
}


def run(model: str, n: int, seed: int = 0, dry_run: bool = False):
    llm = ScriptedLLM(lambda m, t: "3") if dry_run else make_llm(
        model, seed=seed, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    rows = []
    for p in sample_personas(n, ATTRIBUTES, seed=seed):
        agent = Agent(p.name, llm, persona=p,
                      instructions="Answer survey questions as this person would, honestly and in character.")
        for item, question in ITEMS.items():
            probs = agent.survey(question, LIKERT)
            rows.append({**p.attributes, "item": item, **{f"p{k}": v for k, v in probs.items()},
                         "expected": sum(int(k) * v for k, v in probs.items())})
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai:gpt-4o-mini")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=1, help="(unused; for CI symmetry)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = run(args.model, args.n, dry_run=args.dry_run)
    by = {}
    for r in rows:
        by.setdefault((r["item"], r["party"]), []).append(r["expected"])
    print(f"{'item':12s} {'party':12s} mean agreement (1-5)")
    for (item, party), xs in sorted(by.items()):
        print(f"{item:12s} {party:12s} {sum(xs) / len(xs):.2f}  (n={len(xs)})")
