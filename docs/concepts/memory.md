# Memory

All memories share one interface, so the memory type can itself be an
experimental condition:

| Method | Called | Does |
|---|---|---|
| `write(entry)` | after every turn | store what happened (one `MemoryEntry` per turn) |
| `read(query)` | before every turn | the text placed in the prompt (the query is the current observation) |
| `peek(query)` | by probes | like `read`, but must not change the memory |
| `tools()` | each turn | optional tools the agent uses to manage memory itself |
| `dump()` / `load()` | saving | the full contents as JSON |

The **environment decides what is new** for an agent (e.g. the messages since its
last turn); the **memory decides what is remembered**. Keeping these separate is
what lets you vary memory without touching anything else.

## Built-in memories

| Memory | Remembers | LLM calls |
|---|---|---|
| `NoMemory()` | nothing beyond the current observation | none |
| `FullHistory(max_chars=None)` | every turn, in order | none |
| `WindowMemory(k=10)` | the last `k` turns | none |
| `SummaryMemory(llm, every=5)` | a rolling summary plus the newest turns | one per `every` turns |
| `MarkdownMemory(path=None, initial="")` | a notes file the agent edits itself through `memory_append` / `memory_replace` tools | none (the agent decides) |
| `ScoredMemory(k=8, embedder=..., llm=...)` | the top-k turns by recency, importance and relevance | optional importance rating |
| `ReflectiveMemory(llm, threshold=5.0)` | `ScoredMemory` plus reflections written as new memories | reflection when importance accumulates |

What a turn's memory entry contains is set on the agent:
`remember_observations=True` (the full observation), an int (its first N
characters), or `False` (only what the agent did); `remember_private` for private
fields; `memory_max_chars` caps what goes into the prompt.

## Generative Agents-style retrieval

```python
from simsim import ScoredMemory, ReflectiveMemory, OpenAIEmbedder, llm_importance

memory = ReflectiveMemory(
    llm,                                            # writes reflections
    embedder=OpenAIEmbedder("text-embedding-3-small"),
    importance=llm_importance(llm),                 # LLM rates each memory 1-10
    weights=(0.5, 3, 2),                            # recency, importance, relevance (Park et al.)
    threshold=5.0,                                  # reflect when importance sums past this
)
```

- **Recency** decays with the steps since an entry was last retrieved.
- **Importance** comes from `importance(text)`, an LLM rating, or 0.5 by default.
- **Relevance** is the cosine similarity of embeddings with `embedder`, otherwise
  word overlap.
- Each component is min-max normalized over the candidates (as in the original),
  so the weights are comparable.

Embedders: `OpenAIEmbedder(model, base_url=...)` works with OpenAI and any
OpenAI-compatible `/embeddings` endpoint (e.g. vLLM serving `BAAI/bge-m3` with
`--runner pooling`). `HashEmbedder()` needs no model; it captures word overlap
and is meant for tests and cheap baselines. Vectors are cached per text.

## Planted and persistent memories

- `whisper("Ana", "You heard the mayor is resigning.")` writes a memory at a
  chosen step (see [Simulation](simulation.md)).
- `MarkdownMemory(initial="...")` seeds the notes file; `MarkdownMemory(path=...)`
  keeps it across runs.
- `agent.save(path)` / `agent.restore(path)` carry any memory to a later experiment.

## Your own memory

```python
from simsim import Memory

class KeywordMemory(Memory):
    def __init__(self):
        self.entries = []
    def write(self, entry):
        if "important" in entry.text:
            self.entries.append(entry)
    def read(self, query=""):
        return "\n".join(e.render() for e in self.entries[-5:])
```

If your memory keeps `self.entries`, `dump()`/`load()` work automatically.
