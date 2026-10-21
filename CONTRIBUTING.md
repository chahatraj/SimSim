# Contributing

Thanks for helping. Bug reports, new environments, memories, components and
recipes are all welcome.

## Setup

```bash
git clone https://github.com/chahatraj/SimSim && cd SimSim
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,analysis,checkpoint,docs]"
pytest                      # uses a fake model; no keys or GPUs needed
ruff check simsim tests examples
mkdocs serve                # docs at http://127.0.0.1:8000
```

## What a change needs

- **Tests** with `ScriptedLLM` (see `tests/`). If the change touches model calls,
  also run `simsim smoke <model>` against a real endpoint and say which in the PR.
- **Docs**: update the relevant page in `docs/` and docstrings (they become the
  API reference). New user-facing features get a line in `CHANGELOG.md`.
- **Small, readable code** that matches the surrounding style. SimSim's value is
  that a researcher can read the whole loop; avoid hidden prompts and magic.

## Adding things

- **An environment**: subclass `Environment` (see `docs/recipes/custom-environment.md`).
  Built-in environments should be general (a mechanism many studies need), not
  one study's scenario; scenario-specific worlds belong in `examples/`.
- **A memory or component**: subclass `Memory` / `Component`, implement
  `dump`/`load`, and keep `peek`/`view` side-effect free.
- **A recipe**: a runnable file in `examples/` that supports `--dry-run`, plus a
  page in `docs/recipes/`. CI dry-runs every example.

## Prompts are part of the method

Built-in prompts (memory summaries, reflections, user simulators, judges) affect
results. Changing one changes what experiments measure, so such changes need a
clear reason in the PR and a changelog entry.

## Releasing (maintainers)

Bump `version` in `pyproject.toml` and `simsim/__init__.py`, update
`CHANGELOG.md`, then `git tag -a vX.Y.Z -m vX.Y.Z && git push --tags`.
