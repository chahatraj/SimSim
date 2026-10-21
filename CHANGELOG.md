# Changelog

SimSim follows [semantic versioning](https://semver.org/) from 1.0. Before 1.0,
minor versions (0.x) may change interfaces; every change is listed here.

## 0.7.1

Found by installing from GitHub into a fresh environment and running the
quickstart on a real model:

- Structured calls (agents' schemas, probes, judges) now report "response cut
  off at max_tokens (thinking used the budget...)" instead of "no JSON object
  found", and stop retrying an answer that cannot fit.
- `simsim summarize` warns when runs contain judge or evaluator errors, and a
  missing metric lists the available columns instead of a traceback.
- The `simsim new` template notes the token budget thinking models need.

## 0.7.0

- Repository readiness: `__all__` defines the public API, a stability policy,
  ruff linting and coverage in CI, contributor guide, issue and PR templates,
  `CITATION.cff`, changelog.
- Tests for the OpenAI-compatible backend against a local fake server (tool
  calls, reasoning, truncation, log-probs, retries, embeddings).

## 0.6.0

- Documentation site (getting started, concepts, recipes, config reference,
  open models, reporting in papers, API reference).
- Examples: survey panel, Generative Agents-style town, public-goods game.
- Fix: each request now gets its own seed derived from the run seed and the
  prompt; previously every call reused one seed, correlating samples.

## 0.5.0

- `simsim` command line: `new`, `run`, `view`, `summarize`, `transcript`, `smoke`.
- YAML/JSON experiment configs with `${factor}` substitution, populations,
  probes, judges and your own code via `module:attribute`; `--dry-run`.
- `Conversation` accepts per-run tool factories; `make_llm` fails fast on a
  missing API key.

## 0.4.0

- `simsim.analysis`: runs, turns and probes as pandas tables; summaries with
  CIs; pass^k; cost reports; transcripts.
- `simsim.viewer`: a self-contained HTML viewer for experiments.
- Log events use `ts` for wall-clock time.

## 0.3.0

- Components: `Goal`, `Plan`, `Impressions`, `InnerState`.
- Embedding retrieval, Generative Agents-style scoring and `ReflectiveMemory`;
  `dump`/`load` on memories; `Agent.save`/`restore`.
- `CachedLLM`, concurrency and rate limits, clock, `bernoulli`/`by_hour`/
  `NextSpeaker` schedulers, checkpoints, `emulated_tool`.

## 0.2.0

- Retries, structured outputs with repair, ReAct, private reasoning, probes and
  surveys, judges, interventions, persona sampling, user simulators, scripted
  and human agents, termination reasons. Tested on real models.

## 0.1.0

- Agents, memories, environments, schedulers, simulation loop, experiments, logs.
