# Open models

Any server that speaks the OpenAI chat API works: vLLM, SGLang, Ollama, LM Studio,
llama.cpp's server. vLLM is the usual choice on a GPU cluster.

## Serving with vLLM

```bash
vllm serve Qwen/Qwen3.5-9B --port 8000 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 --max-model-len 32768
```

```python
llm = make_llm("vllm:Qwen/Qwen3.5-9B@http://gpu012:8000/v1",
               extra_body={"chat_template_kwargs": {"enable_thinking": False}})
```

Then check everything works with `simsim smoke vllm:...@http://gpu012:8000/v1 --no-thinking`.

### Tool-call parsers

Native tool calling needs the parser that matches the model family's chat
template:

| Model family | `--tool-call-parser` |
|---|---|
| Qwen3.5 / Qwen3-Coder (XML `<function=...>` calls) | `qwen3_coder` |
| Qwen2.5, Qwen3 (`<tool_call>` JSON) | `hermes` |
| Llama 3.1–3.3 | `llama3_json` |
| Mistral / Ministral | `mistral` |

If a model or server has no tool calling, use `Agent(..., action_format="react")`.

### Thinking models

`--reasoning-parser qwen3` (or the parser for your model) makes the reasoning
arrive separately (`turn.reasoning`). Either give thinking enough `max_tokens`, or
turn it off per request with
`extra_body={"chat_template_kwargs": {"enable_thinking": False}}`. Surveys need it
off.

### Embedding models

```bash
vllm serve BAAI/bge-small-en-v1.5 --port 8002 --runner pooling
```

```python
from simsim import OpenAIEmbedder
embedder = OpenAIEmbedder("BAAI/bge-small-en-v1.5", base_url="http://gpu012:8002/v1")
```

## On a Slurm cluster

`scripts/serve_vllm.slurm` in the repository serves a model as a batch job:

```bash
mkdir -p logs
VLLM=/path/to/venv/bin/vllm CACHE=/scratch/$USER/cache TOOL_PARSER=qwen3_coder \
  EXTRA_ARGS="--reasoning-parser qwen3" sbatch scripts/serve_vllm.slurm Qwen/Qwen3.5-9B
grep Serving logs/vllm-*.out        # -> http://gpuXXX:8000/v1
```

Adjust the `#SBATCH` lines to your cluster. Things that commonly go wrong:

- **MIG GPU slices**: Slurm sets `CUDA_VISIBLE_DEVICES` to a MIG UUID, which
  vLLM cannot parse. The script remaps it to `0` (the job only sees its slice).
- **Caches in your home directory** fill small quotas; the script points Hugging
  Face, vLLM, Triton and Torch caches at `CACHE`.
- **Offline compute nodes**: download weights on a login node first, then set
  `HF_HUB_OFFLINE=1`. Vision-language checkpoints (such as Qwen3.5) also need
  their `preprocessor_config.json` files even for text-only use.
- **Rate-limited downloads** from shared cluster IPs: set `HF_TOKEN`.
- **Newer architectures** may need a newer `transformers` than your vLLM
  environment has (the error names an unknown `model_type`).

## Concurrency

vLLM batches concurrent requests, so throughput grows with the number of agents
acting at once. Use `Simulation(simultaneous=True, max_concurrency=32)` for
populations and `Experiment.run(workers=8..32)` across runs. On one 40 GB A100
slice, a 4B model ran 50 simultaneous agents for 150 turns in about 40 seconds.
