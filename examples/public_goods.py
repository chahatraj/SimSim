"""A custom environment: the repeated public-goods game.

Each round every agent privately chooses how much of its 10 tokens to contribute
to a common pot; the pot is multiplied by 1.6 and shared equally. Do agents
cooperate, and does it decay over rounds? This file shows how to write your own
`Environment` (about 40 lines) and use structured answers as actions.

    python examples/public_goods.py --model vllm:Qwen/Qwen3.5-9B@http://node:8000/v1
    python examples/public_goods.py --dry-run
"""

import argparse

from simsim import Agent, Environment, Experiment, FullHistory, ScriptedLLM, Simulation, everyone, grid, make_llm

ENDOWMENT, MULTIPLIER = 10, 1.6


class PublicGoods(Environment):
    def __init__(self, rounds: int = 5, show_others: bool = True):
        self.rounds, self.show_others = rounds, show_others

    def reset(self, agents, rng):
        super().reset(agents, rng)
        self.history: list[dict[str, int]] = []   # contributions per round
        self.pending: dict[str, int] = {}
        self.earnings = {a.name: 0.0 for a in agents}

    def observe(self, agent, t):
        lines = [f"Round {len(self.history) + 1} of {self.rounds}. You have {ENDOWMENT} tokens. Contribute any "
                 f"whole number from 0 to {ENDOWMENT} to the group pot; the pot is multiplied by {MULTIPLIER} "
                 f"and split equally among all {len(self.agents)} players. You keep what you do not contribute."]
        if self.history:
            last = self.history[-1]
            mine = last[agent.name]
            others = ", ".join(f"{n}: {c}" for n, c in last.items() if n != agent.name)
            lines.append(f"Last round you gave {mine}" + (f"; the others gave {others}." if self.show_others else "."))
            lines.append(f"Your total earnings so far: {self.earnings[agent.name]:.1f}.")
        return "\n".join(lines)

    def apply(self, turn, t):
        self.pending[turn.agent] = int(turn.private.get("contribution", 0))
        if len(self.pending) == len(self.agents):          # everyone has moved: settle the round
            share = MULTIPLIER * sum(self.pending.values()) / len(self.agents)
            for name, c in self.pending.items():
                self.earnings[name] += ENDOWMENT - c + share
            self.history.append(dict(self.pending))
            self.pending = {}
            if len(self.history) == self.rounds:
                self.stop("all rounds played")

    def metrics(self):
        per_round = [sum(r.values()) / (len(r) * ENDOWMENT) for r in self.history]
        return {"cooperation_by_round": per_round,
                "mean_cooperation": sum(per_round) / len(per_round) if per_round else None,
                "earnings": self.earnings}


SCHEMA = {"reasoning": (str, "private"), "contribution": (int, 0, ENDOWMENT), "reply": (str, "a short public note")}


def build(cond, seed, model="openai:gpt-4o-mini", dry_run=False):
    def llm():
        if dry_run:
            return ScriptedLLM(lambda m, t: '{"reasoning": "r", "contribution": 5, "reply": "ok"}')
        return make_llm(model, seed=seed, temperature=0.8, max_tokens=300,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}})

    agents = [Agent(f"Player{i + 1}", llm(), persona=cond["persona"], memory=FullHistory(),
                    output_schema=SCHEMA, remember_observations=True) for i in range(4)]
    env = PublicGoods(rounds=5, show_others=cond["visibility"] == "visible")
    return Simulation(env, agents, scheduler=everyone, simultaneous=True, max_steps=10, seed=seed)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai:gpt-4o-mini")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="runs/public_goods")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    exp = Experiment(lambda c, s: build(c, s, args.model, args.dry_run),
                     grid(persona={"plain": "You are a participant in an economics experiment.",
                                   "selfish": "You are a participant in an economics experiment. You care mostly "
                                              "about your own earnings."},
                          visibility=["visible", "hidden"]),
                     seeds=range(args.seeds), out=args.out)
    exp.run(workers=args.workers)
    for row in exp.results():
        if not row.get("error"):
            by_round = " ".join(f"{x:.2f}" for x in row["metrics"]["cooperation_by_round"])
            print(f"{row['condition']:35s} seed={row['seed']}  cooperation by round: {by_round}")
