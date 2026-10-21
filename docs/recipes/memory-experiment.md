# Memory experiment

One assistant meets the same user across several sessions. An early session
plants a preference (an aisle seat), a later one tests whether the agent uses it
when booking a flight. Memory type is the condition.

```bash
python examples/memory_tasks.py --dry-run
python examples/memory_tasks.py --model vllm:Qwen/Qwen3.5-9B@http://node:8000/v1 --seeds 5
```

```python
TASKS = [
    Task("Hi! For future trips: I always want an aisle seat.", id="plant"),
    Task("What's the weather usually like in Lisbon in May?", id="distract"),
    Task("Book me a flight to Lisbon.", id="probe", score=booked("F1")),   # F1 is the aisle seat
]
MEMORIES = {
    "none": lambda llm: NoMemory(),
    "full": lambda llm: FullHistory(),
    "summary": lambda llm: SummaryMemory(llm, every=2),
    "scored": lambda llm: ScoredMemory(k=4),
    "notes": lambda llm: MarkdownMemory(),      # the agent decides what to write down
}
exp = Experiment(build, grid(memory=MEMORIES), seeds=range(5), out="runs/memory_tasks")
```

`TaskSequence` keeps each agent's memory across tasks and scores each task with a
function of the turn and the tools' world. Extend it with more distractor tasks to
test memory at longer distances, with conflicting updates ("actually, window seats
now"), or with `whisper` to plant false memories.

```python
from simsim.analysis import load_runs, summarize
summarize(load_runs("runs/memory_tasks"), by="memory", metrics="mean_success")
```
