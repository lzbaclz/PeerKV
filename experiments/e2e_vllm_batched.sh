#!/usr/bin/env bash
# Memory-bound e2e: batched continuous handoff vs real vLLM decoder (M3).
#
# Uses a larger cached model (OPT-1.3B) with long context and high concurrency
# so the holder is memory/throughput stressed.  The stressor runs back-to-back
# 512MB peer handoffs for the full measurement window (not a single transient).
#
# Requires: .venv_vllm, peerkv conda env, sudo for clock lock, cached model weights.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
RES="$ROOT/experiments/results"; mkdir -p "$RES"
PEERKV_PY="${PEERKV_PY:-/home/lzq/miniconda3/envs/peerkv/bin/python}"
MODEL="${MODEL:-facebook/opt-1.3b}"
MAXLEN="${MAXLEN:-8192}"
PORT="${PORT:-8000}"
CONC="${CONC:-32}"
SECS="${SECS:-45}"
GPU_UTIL="${GPU_UTIL:-0.75}"

echo "=== lock clocks ==="
sudo nvidia-smi -pm 1 >/dev/null
sudo nvidia-smi -lgc 1410,1410 >/dev/null

echo "=== start vLLM on physical GPU1 (model=$MODEL maxlen=$MAXLEN util=$GPU_UTIL) ==="
PATH="$ROOT/.venv_vllm/bin:$PATH" PYTHONPATH="" \
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
  "$ROOT/.venv_vllm/bin/vllm" serve "$MODEL" --port "$PORT" \
  --gpu-memory-utilization "$GPU_UTIL" --max-model-len "$MAXLEN" \
  --max-num-seqs "$CONC" --enforce-eager \
  > "$RES/e2e_batched_server.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null; pkill -f e2e_handoff_stressor 2>/dev/null; sudo nvidia-smi -rgc >/dev/null 2>&1' EXIT

echo "=== wait for /health (up to 360s) ==="
for i in $(seq 1 360); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then echo "ready in ${i}s"; break; fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "server died"; tail -30 "$RES/e2e_batched_server.log"; exit 1
  fi
  sleep 1
done

echo "=== warmup (${CONC} concurrent, ${SECS}s) ==="
"$PEERKV_PY" experiments/e2e_vllm_client.py --model "$MODEL" --concurrency "$CONC" \
  --secs 12 --max-tokens 256 --label warmup >/dev/null 2>&1 || true

for COND in idle push pull host; do
  echo "=== condition: $COND (continuous handoff) ==="
  if [ "$COND" != "idle" ]; then
    "$PEERKV_PY" experiments/e2e_handoff_stressor.py --dir "$COND" --secs $((SECS + 20)) \
      > "$RES/e2e_batched_stressor_$COND.log" 2>&1 &
    STRESS_PID=$!
    sleep 5
  fi
  "$PEERKV_PY" experiments/e2e_vllm_client.py --model "$MODEL" --concurrency "$CONC" \
    --secs "$SECS" --max-tokens 256 --label "$COND" \
    --out "$RES/e2e_batched_cond_$COND.json"
  if [ "$COND" != "idle" ]; then
    kill "$STRESS_PID" 2>/dev/null; wait "$STRESS_PID" 2>/dev/null
    grep -h "stressor:" "$RES/e2e_batched_stressor_$COND.log" || true
  fi
  sleep 3
done

echo "=== aggregate ==="
"$PEERKV_PY" - "$RES" <<'PY'
import json, sys, glob, os
res = sys.argv[1]
conds = {}
for f in sorted(glob.glob(os.path.join(res, "e2e_batched_cond_*.json"))):
    d = json.load(open(f)); conds[d["label"]] = d
base = conds.get("idle", {}).get("tpot_ms_p99")
for c in ("idle", "push", "pull", "host"):
    if c in conds and base:
        conds[c]["p99_inflation_pct_vs_idle"] = round((conds[c]["tpot_ms_p99"] / base - 1) * 100, 2)
out = {
    "_experiment": "e2e_vllm_batched",
    "_is_measured": True,
    "model": conds.get("idle", {}).get("model", "?"),
    "conditions": conds,
    "note": "OPT-1.3B long-context batched vLLM; sustained 512MB handoff stressor",
}
json.dump(out, open(os.path.join(res, "e2e_vllm_batched.json"), "w"), indent=2)
print(json.dumps({c: {k: conds[c].get(k) for k in ("tpot_ms_p50", "tpot_ms_p99", "throughput_tok_s", "p99_inflation_pct_vs_idle")} for c in conds}, indent=2))
PY

echo "DONE -> $RES/e2e_vllm_batched.json"
