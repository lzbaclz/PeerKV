#!/usr/bin/env bash
# S12 runner: one vLLM server + the sensitivity A/B grid.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
RES="$ROOT/experiments/results"; mkdir -p "$RES"
PEERKV_PY="${PEERKV_PY:-/home/lzq/miniconda3/envs/peerkv/bin/python}"
export PATH="$(dirname "$PEERKV_PY"):$PATH"   # ninja for native-pacer JIT
MODEL="${MODEL:-/public/model_zoo/Llama-3.1-8B-Instruct}"
PORT="${PORT:-8000}"

echo "clocks: $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits -i 0 | head -1)MHz"
PATH="$ROOT/.venv_vllm/bin:$PATH" PYTHONPATH="" \
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
  "$ROOT/.venv_vllm/bin/vllm" serve "$MODEL" --port "$PORT" \
  --gpu-memory-utilization 0.75 --max-model-len 8192 \
  --max-num-seqs 32 --enforce-eager > "$RES/s12_server.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for i in $(seq 1 360); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "ready ${i}s"; break; }
  kill -0 $SRV 2>/dev/null || { echo "server died"; tail -20 "$RES/s12_server.log"; exit 1; }
  sleep 1
done

PYTHONPATH="$ROOT" PEERKV_SKIP_IDLE_PROBE=1 \
  "$PEERKV_PY" experiments/s12_sensitivity_ab.py \
  --base "http://127.0.0.1:$PORT" --model "$MODEL" --client-py "$PEERKV_PY"
RC=$?
exit $RC
