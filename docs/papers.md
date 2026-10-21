# Reporting in papers

LLM simulations are sensitive to small choices (recent work finds the base model
often matters more than any other design decision), and reviewers increasingly
ask for the details. SimSim records all of them; this page lists what to report
and where to find it.

## Checklist

| Report | Where SimSim keeps it |
|---|---|
| Models, exact versions or checkpoints, and how they were served | `start` event (`llm` per agent), `conditions.json` |
| Sampling: temperature, max tokens, seeds | `start` event; seeds in `results.jsonl` |
| Number of runs per condition | `results.jsonl` (one row per run) |
| Agent prompts: persona, instructions, the template | `llm_call` events contain the exact messages |
| Memory and components used | `start` event (`memory`, `components`) |
| Environment rules, turn order, stopping rule | your code/config; `termination` per run |
| Measures: probe and survey wording, judge model and rubric | `probe`/`survey` events, the config |
| Failure rates: schema repairs, unparsed replies, crashed runs | `repair` events, `turn.error`, `error` rows |
| Token use and cost | `usage` in `results.jsonl`; `cost_report` |
| Validation of judges or simulated behavior against humans | your own study |

## Reproducibility

- **Pin SimSim**: `pip install "simsim @ git+https://github.com/chahatraj/SimSim@v0.6.0"`.
- **Pin models**: dated API snapshots (`gpt-4o-mini-2024-07-18`) or a specific
  open checkpoint revision.
- **Seeds**: with `seed` set, each request gets a deterministic seed derived from
  the run seed and the prompt. API providers do not guarantee determinism even
  then, and vLLM batching can change results slightly.
- **Cache**: wrap models in `CachedLLM` and share the cache file; anyone rerunning
  your code with the same cache replays your exact runs.
- **Release logs**: `results.jsonl` plus the run logs are the full record; the
  viewer (`simsim view`) is an easy supplementary artifact.

## Validity

Simulation results describe the models, not people, unless validated. Common
practice:

- vary the base model (more than one family) and report all of them;
- vary prompts and persona wording, and check that effects survive;
- compare against human data where a ground truth exists (survey marginals,
  behavioral experiments);
- check that measures are not artifacts: e.g. surveys with probabilities rather
  than one sampled answer, judges validated on a human-rated sample.

## Citing SimSim

The repository has a `CITATION.cff`; GitHub's "Cite this repository" button
gives BibTeX.
