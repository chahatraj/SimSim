"""Two-agent negotiation: does the seller's persona change the final price?

    python examples/negotiation.py --model openai:gpt-4o-mini --seeds 5
    python examples/negotiation.py --model vllm:Qwen/Qwen2.5-7B-Instruct@http://gpu012:8000/v1
    python examples/negotiation.py --dry-run          # fake model, no cost
"""

import argparse
import re

from simsim import Agent, Conversation, Experiment, FullHistory, ScriptedLLM, Simulation, grid, make_llm

SELLER_STYLES = {
    "neutral": "",
    "warm": "You are friendly and value a good relationship with buyers.",
    "tough": "You are a hard bargainer who rarely gives ground.",
}


def build(cond, seed, model="openai:gpt-4o-mini", dry_run=False):
    def llm():
        return ScriptedLLM(lambda m, t: "I offer $120. [END]") if dry_run else make_llm(model, seed=seed)

    buyer = Agent("Buyer", llm(), persona="You want to buy a used bike.", memory=FullHistory())
    seller = Agent(
        "Seller", llm(), persona=f"You are selling your used bike. {SELLER_STYLES[cond['seller']]}",
        memory=FullHistory(),
    )
    env = Conversation(
        "A buyer and a seller negotiate the price of a used bike listed at $200. "
        "When you agree, state the final price as $N and end the conversation.",
        private={"Buyer": "You will not pay more than $170.", "Seller": "You will not sell below $110."},
        max_turns=12,
    )
    return Simulation(env, [buyer, seller], max_steps=12, seed=seed)


def final_price(metrics):
    for msg in reversed(metrics["transcript"]):
        prices = re.findall(r"\$(\d+)", msg["text"])
        if prices:
            return int(prices[-1])
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai:gpt-4o-mini")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="runs/negotiation")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    exp = Experiment(
        lambda c, s: build(c, s, args.model, args.dry_run),
        grid(seller=list(SELLER_STYLES)),
        seeds=range(args.seeds),
        out=args.out,
    )
    exp.run(workers=args.workers)
    for row in exp.results():
        if not row.get("error"):
            print(row["condition"], row["seed"], "price:", final_price(row["metrics"]))
