#!/usr/bin/env bash
# S11 runner: vLLM server + constant background client load + the
# engine-in-the-loop script (census -> validate -> fb feed demo).
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
RES="$ROOT/experiments/results"; mkdir -p "$RES"
PEERKV_PY="${PEERKV_PY:-/home/lzq/miniconda3/envs/peerkv/bin/python}"
export PATH="$(dirname "$PEERKV_PY"):$PATH"   # ninja for native-pacer JIT
export PATH="$(dirname "$PEERKV_PY"):$PATH"   # ninja for the native-pacer JIT
MODEL="${MODEL:-/public/model_zoo/Llama-3.1-8B-Instruct}"
PORT="${PORT:-8000}"
CONC="${CONC:-32}"
GPU_UTIL="${GPU_UTIL:-0.75}"
# load window must outlast all phases: ~10+6*16+10+25+45 + margin ~ 300s
LOAD_SECS="${LOAD_SECS:-330}"

echo "clocks: $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits -i 0 | head -1)MHz"
echo "=== start vLLM on physical GPU1 ==="
PATH="$ROOT/.venv_vllm/bin:$PATH" PYTHONPATH="" \
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
  "$ROOT/.venv_vllm/bin/vllm" serve "$MODEL" --port "$PORT" \
  --gpu-memory-utilization "$GPU_UTIL" --max-model-len 8192 \
  --max-num-seqs "$CONC" --enforce-eager \
  > "$RES/s11_server.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for i in $(seq 1 360); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "ready ${i}s"; break; }
  kill -0 $SRV 2>/dev/null || { echo "server died"; tail -20 "$RES/s11_server.log"; exit 1; }
  sleep 1
done

echo "=== background client load (looped 60s windows, conc=$CONC, unique-prefix) ==="
ALIVE="/tmp/s11_load_alive.$$"; touch "$ALIVE"
(
  i=0
  while [ -f "$ALIVE" ] && kill -0 $SRV 2>/dev/null; do
    i=$((i+1))
    echo "--- client window $i $(date +%T) ---" >> "$RES/s11_client.log"
    "$PEERKV_PY" experiments/e2e_vllm_client.py --model "$MODEL" \
      --concurrency "$CONC" --secs 60 --max-tokens 256 --unique-prefix \
      --label "s11_load_$i" >> "$RES/s11_client.log" 2>&1
  done
  echo "--- client loop exited $(date +%T) ---" >> "$RES/s11_client.log"
) &
CLIENT_LOOP=$!
trap 'rm -f "$ALIVE"; kill $SRV $CLIENT_LOOP 2>/dev/null' EXIT
sleep 20   # past client ramp/prefill burst before any baseline window

echo "=== engine-in-the-loop phases ==="
PYTHONPATH="$ROOT" PEERKV_SKIP_IDLE_PROBE=1 \
  "$PEERKV_PY" experiments/s11_engine_loop.py --base "http://127.0.0.1:$PORT"
RC=$?
rm -f "$ALIVE"
exit $RC
