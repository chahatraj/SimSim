#!/bin/bash
# Serve an open model with vLLM on a GPU node, behind an OpenAI-compatible API.
#
#   sbatch scripts/serve_vllm.slurm Qwen/Qwen2.5-7B-Instruct
#   # once running, the node and port are in logs/vllm-<jobid>.out; then:
#   make_llm("vllm:Qwen/Qwen2.5-7B-Instruct@http://<node>:8000/v1")
#
# Tool calling needs a parser matched to the model family (hermes for Qwen2.5/Qwen3,
# qwen3_coder for Qwen3.5, llama3_json for Llama 3.x, mistral for Mistral). Override
# with TOOL_PARSER=... Thinking models also want EXTRA_ARGS="--reasoning-parser qwen3"
# so their reasoning arrives separately from the answer.
#
# Adjust the #SBATCH lines to your cluster. Set VLLM to your vllm binary and
# CACHE to a roomy directory (model weights and compile caches go there).
#SBATCH --job-name=vllm
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/vllm-%j.out

MODEL=${1:-Qwen/Qwen2.5-7B-Instruct}
PORT=${PORT:-8000}
TOOL_PARSER=${TOOL_PARSER:-hermes}
VLLM=${VLLM:-vllm}
CACHE=${CACHE:-$HOME/.cache}

export HF_HOME=${HF_HOME:-$CACHE/huggingface}
export XDG_CACHE_HOME=$CACHE
export VLLM_CACHE_ROOT=$CACHE/vllm
export TRITON_CACHE_DIR=$CACHE/triton
export TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor

# On MIG slices Slurm sets CUDA_VISIBLE_DEVICES to a MIG UUID, which vLLM cannot parse.
# The job only sees its own slice, so index 0 refers to it.
if [[ "$CUDA_VISIBLE_DEVICES" == MIG-* ]]; then export CUDA_VISIBLE_DEVICES=0; fi

echo "Serving $MODEL at http://$(hostname):$PORT/v1"
"$VLLM" serve "$MODEL" \
    --port "$PORT" \
    --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER" \
    --max-model-len "${MAX_LEN:-32768}" \
    $EXTRA_ARGS
