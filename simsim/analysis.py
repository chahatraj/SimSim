"""Load experiment outputs into pandas tables and summarize them (`pip install pandas`).

    from simsim.analysis import load_runs, load_turns, load_probes, summarize

    runs = load_runs("runs/memory_v1")        # one row per run: factors, metrics, tokens, cost
    summarize(runs, by=["memory"], metrics=["mean_success"])
    turns = load_turns("runs/memory_v1")      # one row per agent turn: text, private fields, components
    probes = load_probes("runs/memory_v1")    # one row per probe/survey answer

Condition names made by `grid` ("model=a,memory=none") are split into one
column per factor, so you can group by any factor directly.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .record import read_jsonl


def _pd():
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise ImportError("simsim.analysis needs pandas: pip install pandas") from e
    return pd


def _paths(source: str | Path | Iterable[str | Path], name: str) -> list[Path]:
    sources = [source] if isinstance(source, (str, Path)) else list(source)
    out = []
    for s in map(Path, sources):
        out += sorted(s.rglob(name)) if s.is_dir() else [s]
    return out


def parse_condition(name: str) -> dict[str, str]:
    """"model=a,memory=none" -> {"model": "a", "memory": "none"} (names without '=' give {})."""
    parts = [p.split("=", 1) for p in name.split(",")]
    return {k: v for k, v in parts} if all(len(p) == 2 for p in parts) else {}


def flatten(d: dict[str, Any], prefix: str = "", skip: set[str] = frozenset()) -> dict[str, Any]:
    """Nested dicts -> dotted keys; lists and long strings are dropped (keep tables tidy)."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if k in skip:
            continue
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, (int, float, bool)) or v is None or (isinstance(v, str) and len(v) <= 200):
            out[key] = v
    return out


def _usage_totals(usage: dict[str, Any]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for u in (usage or {}).get("backends", {}).values():
        for k in ("calls", "prompt_tokens", "completion_tokens", "cost_usd", "cache_hits", "retries"):
            if k in u:
                totals[k] = totals.get(k, 0) + u[k]
    return totals


def load_runs(source, include_errors: bool = False):
    """One row per run from results.jsonl files (an experiment dir, a file, or several)."""
    pd = _pd()
    rows = []
    for path in _paths(source, "results.jsonl"):
        for r in read_jsonl(path):
            if r.get("error") and not include_errors:
                continue
            row = {"experiment": path.parent.name, "condition": r["condition"], "seed": r["seed"],
                   **parse_condition(r["condition"]), "steps": r.get("steps"), "termination": r.get("termination"),
                   "error": r.get("error")}
            row.update(flatten(r.get("metrics") or {}, skip={"transcript", "tasks", "posts", "follows"}))
            row.update({f"usage.{k}": v for k, v in _usage_totals(r.get("usage")).items()})
            rows.append(row)
    return pd.DataFrame(rows)


def load_turns(source):
    """One row per agent turn from the per-run logs (seed*.jsonl) of an experiment dir."""
    pd = _pd()
    rows = []
    for path in _paths(source, "seed*.jsonl"):
        for e in read_jsonl(path):
            if e["event"] != "turn":
                continue
            cond, _, seed = e["run"].rpartition("/seed")
            row = {"condition": cond, "seed": int(seed) if seed.isdigit() else seed, **parse_condition(cond),
                   "agent": e["agent"], "t": e["t"], "time": e.get("time"), "text": e["text"],
                   "observation": e.get("observation"), "reasoning": e.get("reasoning") or "",
                   "n_tool_calls": len(e.get("tool_calls") or []), "error": e.get("error"),
                   **{f"usage.{k}": v for k, v in (e.get("usage") or {}).items()}}
            row.update({f"private.{k}": v for k, v in (e.get("private") or {}).items()})
            row.update(flatten(e.get("components") or {}, prefix="component."))
            rows.append(row)
    return pd.DataFrame(rows)


def expected_value(probs: dict[str, float]) -> float | None:
    """Mean of numeric labels weighted by probability (e.g. a Likert item), else None."""
    try:
        return sum(float(k) * v for k, v in probs.items())
    except ValueError:
        return None


def load_probes(source):
    """One row per probe or survey answer. Survey rows get `top` (most likely label),
    `expected` (probability-weighted mean of numeric labels) and `p.<label>` columns;
    structured probe answers are expanded to `answer.<field>` columns."""
    pd = _pd()
    rows = []
    for path in _paths(source, "results.jsonl"):
        for r in read_jsonl(path):
            for p in r.get("probes") or []:
                row = {"condition": r["condition"], "seed": r["seed"], **parse_condition(r["condition"]),
                       "name": p["name"], "t": p["t"], "agent": p["agent"], "question": p["question"]}
                if "probs" in p:
                    probs = p["probs"]
                    row.update({f"p.{k}": v for k, v in probs.items()})
                    row["top"] = max(probs, key=probs.get) if probs else None
                    row["expected"] = expected_value(probs)
                ans = p.get("answer")
                if isinstance(ans, dict):
                    row.update({f"answer.{k}": v for k, v in ans.items()})
                elif ans is not None:
                    row["answer"] = ans
                rows.append(row)
    return pd.DataFrame(rows)


def summarize(df, by: str | list[str], metrics: str | list[str] | None = None, ci: float = 0.95):
    """Mean, standard error, n, and a normal-approximation confidence interval per group."""
    pd = _pd()
    by = [by] if isinstance(by, str) else list(by)
    if metrics is None:
        metrics = [c for c in df.select_dtypes("number").columns if c not in by and c != "seed"]
    metrics = [metrics] if isinstance(metrics, str) else list(metrics)
    z = {0.9: 1.645, 0.95: 1.96, 0.99: 2.576}.get(ci, 1.96)
    rows = []
    for key, g in df.groupby(by, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        for m in metrics:
            x = pd.to_numeric(g[m], errors="coerce").dropna()
            n = len(x)
            mean = x.mean() if n else math.nan
            sem = x.std(ddof=1) / math.sqrt(n) if n > 1 else math.nan
            rows.append({**dict(zip(by, key, strict=True)), "metric": m, "mean": mean, "sem": sem, "n": n,
                         "ci_low": mean - z * sem if n > 1 else math.nan,
                         "ci_high": mean + z * sem if n > 1 else math.nan})
    return pd.DataFrame(rows)


def pass_hat_k(df, success: str, k: int, task: str = "condition", threshold: float = 1.0) -> float:
    """tau-bench's pass^k: the chance that k random trials (seeds) of a task all succeed,
    averaged over tasks. Each row is one trial; `task` groups trials of the same task."""
    total, n_tasks = 0.0, 0
    for _, g in df.groupby(task):
        n, c = len(g), int((g[success] >= threshold).sum())
        if n < k:
            continue
        total += math.comb(c, k) / math.comb(n, k)
        n_tasks += 1
    return total / n_tasks if n_tasks else math.nan


def cost_report(df, prices: dict[str, tuple[float, float]] | None = None, by: str | list[str] = "condition"):
    """Tokens (and dollars) per group. With `prices` {model substring: ($/1M in, $/1M out)}
    applied to the `model` factor column; otherwise uses `usage.cost_usd` if the LLMs had `price=`."""
    by = [by] if isinstance(by, str) else list(by)
    cols = [c for c in ("usage.calls", "usage.prompt_tokens", "usage.completion_tokens", "usage.cost_usd")
            if c in df.columns]
    out = df.groupby(by)[cols].sum().reset_index()
    if prices and "model" in df.columns and "model" in by:
        def dollars(row):
            for sub, (pin, pout) in prices.items():
                if sub in str(row["model"]):
                    return (row["usage.prompt_tokens"] * pin + row["usage.completion_tokens"] * pout) / 1e6
            return math.nan
        out["cost_usd"] = out.apply(dollars, axis=1)
    return out


def load_log(path: str | Path) -> list[dict[str, Any]]:
    """All events of one run log, in order (for custom analysis)."""
    return read_jsonl(path)


def transcript(path: str | Path, private: bool = False) -> str:
    """A readable transcript of one run log; `private=True` adds private fields and reasoning."""
    lines = []
    for e in read_jsonl(path):
        if e["event"] == "turn":
            stamp = f"[{e['time']}] " if e.get("time") else f"[t={e['t']}] "
            lines.append(f"{stamp}{e['agent']}: {e['text']}")
            for c in e.get("tool_calls") or []:
                lines.append(f"    -> {c['name']}({json.dumps(c['arguments'])}) = {str(c['result'])[:200]}")
            if private and e.get("private"):
                lines.append(f"    (private) {json.dumps(e['private'])}")
            if private and e.get("reasoning"):
                lines.append(f"    (reasoning) {e['reasoning'].strip()[:500]}")
        elif e["event"] == "intervention":
            lines.append(f"    * intervention at t={e['t']}: {e['fn']}")
    return "\n".join(lines)
