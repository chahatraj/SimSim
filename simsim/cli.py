"""The `simsim` command.

    simsim new my_study                       # scaffold a project folder for a new paper
    simsim run experiment.yaml --workers 8    # run a config (or a Python file with `build`)
    simsim run experiment.yaml --dry-run      # fake model: check the setup for free
    simsim view runs/my_study                 # write an offline HTML viewer of all runs
    simsim summarize runs/my_study --by model memory --metric mean_success
    simsim transcript runs/my_study/model=x/seed0.jsonl --private
    simsim smoke vllm:Qwen/Qwen3.5-9B@http://node:8000/v1 --no-thinking
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def _run(args) -> int:
    path = Path(args.experiment)
    if path.suffix in (".yaml", ".yml", ".json"):
        from .config import run_config

        exp, rows = run_config(path, workers=args.workers, dry_run=args.dry_run, out=args.out, seeds=args.seeds)
    elif path.suffix == ".py":
        exp = _python_experiment(path, args)
        rows = exp.run(workers=args.workers)
    else:
        print(f"Don't know how to run {path} (use .yaml, .json or .py)", file=sys.stderr)
        return 2
    errors = [r for r in rows if r.get("error")]
    print(f"\n{len(rows) - len(errors)} runs finished, {len(errors)} failed -> {exp.out}")
    for r in errors[:5]:
        print(f"  {r['condition']} seed={r['seed']}: {r['error']}")
    if args.view:
        from .viewer import write_viewer

        print(f"viewer: {write_viewer(exp.out)}")
    return 1 if errors else 0


def _python_experiment(path: Path, args):
    """A Python experiment defines `experiment` (an Experiment), or `build` plus `conditions`
    (and optionally `seeds` and `out`). With --dry-run, `build` gets `dry_run=True` if it accepts it."""
    import inspect

    from .experiment import Experiment

    sys.path.insert(0, str(path.parent.resolve()))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "experiment"):
        exp = mod.experiment
        if args.out:
            exp.out = Path(args.out)
        return exp
    if not hasattr(mod, "build"):
        raise SystemExit(f"{path} must define `experiment` or `build(cond, seed)` and `conditions`")
    build = mod.build
    if args.dry_run and "dry_run" in inspect.signature(build).parameters:
        build = lambda cond, seed, _b=mod.build: _b(cond, seed, dry_run=True)  # noqa: E731
    seeds = range(args.seeds) if args.seeds is not None else getattr(mod, "seeds", range(1))
    out = args.out or getattr(mod, "out", f"runs/{path.stem}")
    return Experiment(build, getattr(mod, "conditions", {"default": {}}), seeds=seeds, out=out)


def _view(args) -> int:
    from .viewer import write_viewer

    print(write_viewer(args.runs, args.output, include_prompts=args.prompts))
    return 0


def _summarize(args) -> int:
    from .analysis import load_runs, summarize

    runs = load_runs(args.runs)
    if runs.empty:
        print("No finished runs found.")
        return 1
    for col in [c for c in runs.columns if c.endswith(".error") and runs[c].notna().any()]:
        bad = runs[col].dropna()
        print(f"warning: {len(bad)} of {len(runs)} runs have {col}: {str(bad.iloc[0])[:200]}", file=sys.stderr)
    by = args.by or ["condition"]
    try:
        table = summarize(runs, by=by, metrics=args.metric)
    except KeyError as e:
        print(f"error: {e.args[0]}", file=sys.stderr)
        return 1
    import pandas as pd

    with pd.option_context("display.max_rows", 500, "display.width", 200):
        print(table.round(3).to_string(index=False))
    return 0


def _transcript(args) -> int:
    from .analysis import transcript

    print(transcript(args.log, private=args.private))
    return 0


def _smoke(args) -> int:
    from .smoke import main

    return main(args)


def _new(args) -> int:
    from .templates import scaffold

    for p in scaffold(args.name, Path(args.dir)):
        print(f"created {p}")
    print(f"\nNext: cd {Path(args.dir) / args.name} && simsim run experiment.yaml --dry-run")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="simsim", description="Build, run, and inspect LLM agent simulations.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="run an experiment (.yaml/.json config or .py with build)")
    p.add_argument("experiment")
    p.add_argument("--workers", type=int, default=4, help="runs in parallel")
    p.add_argument("--seeds", type=int, help="override the number of seeds")
    p.add_argument("--out", help="output directory")
    p.add_argument("--dry-run", action="store_true", help="use a fake model (no cost) to check the setup")
    p.add_argument("--view", action="store_true", help="write view.html when done")
    p.set_defaults(fn=_run)

    p = sub.add_parser("view", help="write an offline HTML viewer for runs")
    p.add_argument("runs", help="experiment directory or one run log")
    p.add_argument("-o", "--output")
    p.add_argument("--prompts", action="store_true", help="include full prompts of every LLM call (large)")
    p.set_defaults(fn=_view)

    p = sub.add_parser("summarize", help="mean, sem and CI of metrics by factor (needs pandas)")
    p.add_argument("runs")
    p.add_argument("--by", nargs="*")
    p.add_argument("--metric", nargs="*")
    p.set_defaults(fn=_summarize)

    p = sub.add_parser("transcript", help="print a readable transcript of one run log")
    p.add_argument("log")
    p.add_argument("--private", action="store_true", help="include private fields and reasoning")
    p.set_defaults(fn=_transcript)

    p = sub.add_parser("smoke", help="check that a model endpoint supports SimSim's features")
    from .smoke import add_arguments

    add_arguments(p)
    p.set_defaults(fn=_smoke)

    p = sub.add_parser("new", help="create a project folder for a new study")
    p.add_argument("name")
    p.add_argument("--dir", default=".")
    p.set_defaults(fn=_new)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
