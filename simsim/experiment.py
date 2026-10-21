"""Run a simulation across conditions x seeds, with logs and a results table.

    def build(cond, seed):
        llm = make_llm(cond["model"], seed=seed)
        agent = Agent("assistant", llm, memory=cond["memory"]())
        return Simulation(TaskSequence(tasks), [agent], seed=seed)

    exp = Experiment(build, grid(model=["openai:gpt-4o-mini", "vllm:Qwen/Qwen2.5-7B-Instruct"],
                                 memory={"none": NoMemory, "notes": MarkdownMemory}),
                     seeds=range(5), out="runs/memory_v1")
    rows = exp.run(workers=8)

Each run writes `out/<condition>/seed<k>.jsonl` (every LLM call and tool call)
and one line in `out/results.jsonl`. Re-running skips runs already in
results.jsonl, so an interrupted experiment resumes where it stopped.
"""

from __future__ import annotations

import itertools
import json
import traceback
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .record import Recorder, read_jsonl
from .sim import Simulation


def grid(**factors: list | dict) -> dict[str, dict[str, Any]]:
    """Cross factors into named conditions.

    A list's values name themselves; a dict maps short names to values:
        grid(model=["a", "b"], memory={"none": NoMemory, "md": MarkdownMemory})
        -> {"model=a,memory=none": {"model": "a", "memory": NoMemory}, ...}
    """
    named = {k: (v if isinstance(v, dict) else {str(x).split("/")[-1]: x for x in v}) for k, v in factors.items()}
    out = {}
    for combo in itertools.product(*[[(k, n, v) for n, v in d.items()] for k, d in named.items()]):
        out[",".join(f"{k}={n}" for k, n, _ in combo)] = {k: v for k, _, v in combo}
    return out


class Experiment:
    def __init__(
        self,
        build: Callable[[dict[str, Any], int], Simulation],
        conditions: dict[str, dict[str, Any]],
        seeds: Iterable[int] = (0,),
        out: str | Path = "runs/experiment",
    ):
        self.build = build
        self.conditions = conditions
        self.seeds = list(seeds)
        self.out = Path(out)

    def _done(self) -> set[tuple[str, int]]:
        path = self.out / "results.jsonl"
        if not path.exists():
            return set()
        return {(r["condition"], r["seed"]) for r in read_jsonl(path) if not r.get("error")}

    def run_one(self, name: str, seed: int) -> dict[str, Any]:
        rec = Recorder(self.out / _safe(name) / f"seed{seed}.jsonl", run_id=f"{name}/seed{seed}")
        row: dict[str, Any] = {"condition": name, "seed": seed}
        try:
            sim = self.build(self.conditions[name], seed)
            res = sim.run(rec)
            row.update(steps=res.steps, termination=res.termination, usage=res.usage, metrics=res.metrics,
                       probes=res.probes)
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
            rec.log("error", traceback=traceback.format_exc())
        finally:
            rec.close()
        return row

    def run(self, workers: int = 1, progress: bool = True) -> list[dict[str, Any]]:
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "conditions.json").write_text(json.dumps(self.conditions, indent=2, default=_describe))
        done = self._done()
        todo = [(c, s) for c in self.conditions for s in self.seeds if (c, s) not in done]
        if progress:
            print(f"{len(todo)} runs to do ({len(done)} already done) -> {self.out}")
        rows = []
        with open(self.out / "results.jsonl", "a") as fh, ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self.run_one, c, s): (c, s) for c, s in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                row = fut.result()
                fh.write(json.dumps(row, default=str) + "\n")
                fh.flush()
                rows.append(row)
                if progress:
                    status = row.get("error") or f"{row['steps']} steps"
                    print(f"[{i}/{len(todo)}] {row['condition']} seed={row['seed']}: {status}")
        return rows

    def results(self) -> list[dict[str, Any]]:
        """All finished rows, including runs from earlier invocations."""
        return read_jsonl(self.out / "results.jsonl")


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "=,.-_" else "_" for ch in name)


def _describe(x: Any) -> str:
    return getattr(x, "__name__", None) or repr(x)
