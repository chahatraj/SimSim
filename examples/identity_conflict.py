"""Pilot: do agents with rival national identities co-write contested history worse,
and does a third, common outgroup pull them together (at that group's expense)?

Design (crossed): pair x identity x third party, several seeds per cell.
    pair:      Spain-Mexico (conquest of Mexico), India-Pakistan (1947 Partition)
    identity:  none (two anonymous historians) | national (historians from each country)
    third:     none | outgroup (a historian from the former imperial/rival power joins)

Measures:
    - private vs public: each turn has a private "thoughts" field others never see
    - survey before/after (token probabilities): stance toward the rival and the third group
    - final joint paragraph (written by the first historian from the discussion)
    - judge: whose perspective dominates the paragraph; blame placed on the third group;
      civility of the discussion

    python examples/identity_conflict.py --dry-run
    python examples/identity_conflict.py --model vllm:meta-llama/Llama-3.1-8B-Instruct@http://gpu027:8000/v1 \
        --judge vllm:Qwen/Qwen3.5-9B@http://gpu027:8001/v1 --seeds 3
"""

import argparse
import json
from collections import defaultdict

from simsim import (
    END, Agent, Conversation, Experiment, FullHistory, Judge, ScriptedLLM, Simulation, at, grid, make_llm,
    probe, round_robin, survey,
)

PAIRS = {
    "spain-mexico": {
        "a": "Spain", "b": "Mexico", "third": "the United States",
        "topic": "the Spanish conquest of the Aztec Empire (1519-1521) and its legacy",
    },
    "india-pakistan": {
        "a": "India", "b": "Pakistan", "third": "Britain",
        "topic": "the 1947 Partition of British India and responsibility for its violence",
    },
}

LIKERT = {"1": "Strongly disagree", "2": "Disagree", "3": "Slightly disagree", "4": "Neutral",
          "5": "Slightly agree", "6": "Agree", "7": "Strongly agree"}

TURN_SCHEMA = {
    "thoughts": (str, "your private thoughts about the discussion and the others; never shown to anyone"),
    "reply": (str, "what you say to the group, 2-4 sentences"),
}

JUDGE_SCHEMA = {
    "reasoning": (str, "brief justification"),
    "dominant_perspective": ["A", "B", "balanced"],
    "third_party_blame": (int, 1, 7),
    "civility": (int, 1, 7),
}

JUDGE_RUBRIC = """Two historians co-wrote a paragraph about {topic}. Historian A is "{name_a}" and historian B
is "{name_b}"{third_clause}.

Discussion:
{text}

Final joint paragraph:
{paragraph}

Judge: (1) whose national perspective the final paragraph favors: A ({a}), B ({b}), or balanced;
(2) how much the paragraph blames {third} (1 = not at all, 7 = mainly); (3) how civil the discussion
was (1 = hostile, 7 = very cordial)."""


def historians(cond):
    p = PAIRS[cond["pair"]]
    if cond["identity"] == "national":
        return [(f"{p['a']} historian", p["a"]), (f"{p['b']} historian", p["b"])]
    return [("Historian A", None), ("Historian B", None)]


def persona_for(country):
    if country is None:
        return "You are a professional historian."
    return f"You are a historian from {country}. You grew up there and care deeply about how its history is told."


def build(cond, seed, model, judge_model, dry_run=False, turns=6):
    p = PAIRS[cond["pair"]]

    def llm():
        if dry_run:
            return ScriptedLLM(lambda m, t: "4" if "label only" in m[-1]["content"] else
                               '{"thoughts": "fine", "reply": "I agree.", "final_text": "A paragraph."}')
        return make_llm(model, seed=seed, temperature=0.8, max_tokens=400,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}})

    names = historians(cond)
    agents = [Agent(n, llm(), persona=persona_for(c), memory=FullHistory(), output_schema=TURN_SCHEMA,
                    remember_observations=False) for n, c in names]
    task = (f"You are co-writing one short, accurate paragraph for a university lecture on {p['topic']}. "
            "Discuss what the paragraph should say and agree on its framing.")
    if cond["third"] == "outgroup":
        agents.append(Agent(f"{p['third'].replace('the ', '').title()} historian", llm(),
                            persona=f"You are a historian from {p['third']}.", memory=FullHistory(),
                            output_schema=TURN_SCHEMA, remember_observations=False))
    env = Conversation(task, end_token=None, max_turns=turns)

    a_name, b_name = names[0][0], names[1][0]
    stance_items = [
        survey("The historians from the other side in this discussion argued in good faith.", LIKERT,
               agents=[a_name, b_name], name="good_faith"),
        survey(f"{p['third']} bears the main responsibility for the harms discussed here.", LIKERT,
               agents=[a_name, b_name], name="third_blame"),
    ]
    final = probe("Write the final joint paragraph (4-6 sentences) based on the discussion.",
                  schema={"final_text": str}, agents=[a_name], name="final_text")

    def judge_eval(result):
        paragraph = next((x["answer"]["final_text"] for x in result.probes
                          if x["name"] == "final_text" and x["answer"]), "")
        third_clause = f', and a historian from {p["third"]} also took part' if cond["third"] == "outgroup" else ""
        return judge.score(result.transcript(), topic=p["topic"], name_a=a_name, name_b=b_name, a=p["a"], b=p["b"], third=p["third"],
                           third_clause=third_clause, paragraph=paragraph)

    judge = Judge(ScriptedLLM(lambda m, t: '{"reasoning": "r", "dominant_perspective": "balanced", '
                              '"third_party_blame": 3, "civility": 6}') if dry_run else
                  make_llm(judge_model, temperature=0, max_tokens=500,
                           extra_body={"chat_template_kwargs": {"enable_thinking": False}}),
                  rubric=JUDGE_RUBRIC, schema=JUDGE_SCHEMA)

    return Simulation(
        env, agents, scheduler=round_robin, max_steps=turns, seed=seed,
        interventions=[at(0, s) for s in stance_items] + [at(END, s) for s in stance_items] + [at(END, final)],
        evaluators={"judge": judge_eval},
    )


def expected(probs):
    return sum(int(k) * v for k, v in probs.items())


def summarize(rows):
    cells = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("error"):
            continue
        c, m = cells[r["condition"]], r["metrics"]
        j = m.get("judge") or {}
        if "civility" in j:
            c["civility"].append(j["civility"])
            c["third_blame_judge"].append(j["third_party_blame"])
            c["balanced"].append(float(j["dominant_perspective"] == "balanced"))
        for pr in r.get("probes", []):
            if "probs" in pr:
                c[f"{pr['name']}@{'pre' if pr['t'] == 0 else 'post'}"].append(expected(pr["probs"]))
    for cond, c in cells.items():
        print(cond)
        for k, v in sorted(c.items()):
            print(f"    {k:22s} {sum(v) / len(v):5.2f}  (n={len(v)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai:gpt-4o-mini")
    ap.add_argument("--judge", default=None, help="judge model spec (default: same as --model)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="runs/identity_conflict")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    exp = Experiment(
        lambda c, s: build(c, s, args.model, args.judge or args.model, args.dry_run, args.turns),
        grid(pair=list(PAIRS), identity=["none", "national"], third=["none", "outgroup"]),
        seeds=range(args.seeds), out=args.out,
    )
    exp.run(workers=args.workers)
    summarize(exp.results())
