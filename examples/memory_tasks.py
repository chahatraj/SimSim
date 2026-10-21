"""Single agent with tools and memory: does memory carry a user's preference across sessions?

Early tasks plant facts; later tasks test whether the agent uses them. Memory
type is the manipulated condition.

    python examples/memory_tasks.py --model openai:gpt-4o-mini --seeds 3
    python examples/memory_tasks.py --dry-run
"""

import argparse
import json

from simsim import (
    Agent, Experiment, FullHistory, MarkdownMemory, NoMemory, ScoredMemory, ScriptedLLM, Simulation,
    SummaryMemory, Task, TaskSequence, grid, make_llm, tool,
)


def make_tools(world):
    @tool
    def search_flights(destination: str) -> str:
        """Search flights to a destination. Returns options with seat types."""
        return json.dumps([
            {"id": "F1", "dest": destination, "seat": "aisle", "price": 320},
            {"id": "F2", "dest": destination, "seat": "window", "price": 290},
        ])

    @tool
    def book_flight(flight_id: str) -> str:
        """Book a flight by id."""
        world.setdefault("booked", []).append(flight_id)
        return f"booked {flight_id}"

    return [search_flights, book_flight]


def booked(flight_id):
    return lambda turn, world: {"success": float(world.get("booked", [None])[-1] == flight_id)}


TASKS = [
    Task("Hi! For future trips: I always want an aisle seat, I get claustrophobic by the window.", id="plant"),
    Task("What's the weather usually like in Lisbon in May?", id="distract"),
    Task("Book me a flight to Lisbon.", id="probe", score=booked("F1")),
]

MEMORIES = {
    "none": lambda llm: NoMemory(),
    "full": lambda llm: FullHistory(),
    "summary": lambda llm: SummaryMemory(llm, every=2),
    "scored": lambda llm: ScoredMemory(k=4),
    "notes": lambda llm: MarkdownMemory(),
}


def build(cond, seed, model="openai:gpt-4o-mini", dry_run=False):
    llm = ScriptedLLM(lambda m, t: "Done.") if dry_run else make_llm(model, seed=seed, temperature=0)
    agent = Agent(
        "assistant", llm,
        persona="You are a travel assistant. Each message is a new session with the same user.",
        memory=cond["memory"](llm),
    )
    return Simulation(TaskSequence(TASKS, tools=make_tools), [agent], max_steps=len(TASKS), seed=seed)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai:gpt-4o-mini")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="runs/memory_tasks")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    exp = Experiment(
        lambda c, s: build(c, s, args.model, args.dry_run), grid(memory=MEMORIES), seeds=range(args.seeds), out=args.out
    )
    exp.run(workers=args.workers)
    scores = {}
    for row in exp.results():
        if not row.get("error"):
            scores.setdefault(row["condition"], []).append(row["metrics"].get("mean_success", 0))
    for cond, s in scores.items():
        print(f"{cond:20s} success={sum(s) / len(s):.2f} (n={len(s)})")
