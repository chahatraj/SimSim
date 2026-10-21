# Stability and versions

SimSim is pre-1.0: interfaces may still change between minor versions (0.6 →
0.7), and every change is listed in the
[changelog](https://github.com/chahatraj/SimSim/blob/main/CHANGELOG.md). Pin
the version a paper used:

```bash
pip install "simsim @ git+https://github.com/chahatraj/SimSim@v0.7.0"
```

## What counts as the public API

Everything in `simsim.__all__` (what `from simsim import ...` gives you), the
`simsim.analysis` and `simsim.viewer` functions, the `simsim` command, the
config format, and the log event format. These change only with a changelog
entry. Anything else (underscore names, internal modules) may change at any time.

## Prompts

Built-in prompts (summaries, reflection, importance ratings, user simulators,
judges, the conversation instructions) are part of the method: changing them can
change results. They change only in minor versions, always with a changelog
entry, and every prompt actually sent is in your run logs. For full control,
override them: most are class attributes (`SummaryMemory.PROMPT`,
`ReflectiveMemory.QUESTIONS_PROMPT`, `Plan.PROMPT`, ...) or arguments.
