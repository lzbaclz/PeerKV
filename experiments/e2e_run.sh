#!/usr/bin/env bash
# Real-engine e2e: does a concurrent cross-GPU KV handoff harm a real vLLM
# continuous-batching decoder, and does transfer direction matter?
#
# Holder = physical GPU1 runs `vllm serve` (Llama-2-7B). A separate stressor process
# runs a sustained 512MB handoff {push,pull,host} contending on GPU1's HBM. A
# streaming client measures decode TPOT (P50/P95/P99) per condition.
#
# vLLM lives in an isolated venv (.venv_vllm); the stressor uses the peerkv env's
# torch (sees both physical GPUs). Clocks locked for the run.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
RES="$ROOT/experiments/results"; mkdir -p "$RES"
VLLM_PY="$ROOT/.venv_vllm/bin/python"
PEERKV_PY="/home/lzq/miniconda3/envs/peerkv/bin/python"
# Default to a real memory-bound 7B/8B GQA model (same geometry as the g1-g4
# synthetic holder: D=4096,H=32,HKV=8,HD=128). Override with MODEL=...
MODEL="${MODEL:-/public/model_zoo/Llama-3.1-8B-Instruct}"
# Memory-bound holder: a long resident KV per request makes the decode step
# HBM-read-bound (the regime the cost model is about), so a concurrent peer copy
# actually contends -- unlike the earlier compute-light GPT-2 short-context run.
MAXLEN="${MAXLEN:-20480}"
PROMPT_WORDS="${PROMPT_WORDS:-11000}"   # ~14-15K prompt tokens of resident KV per request
MAXTOK="${MAXTOK:-256}"
GMU="${GMU:-0.8}"                       # 0.8*80GB; leaves headroom on GPU1 for the stressor
PORT="${PORT:-8000}"
CONC="${CONC:-12}"
SECS="${SECS:-45}"

echo "=== lock clocks (idempotent; box is already locked at 1410) ==="
sudo nvidia-smi -pm 1 >/dev/null; sudo nvidia-smi -lgc 1410,1410 >/dev/null

echo "=== start vLLM serve on physical GPU1 (model=$MODEL, maxlen=$MAXLEN, gmu=$GMU) ==="
# --no-enable-prefix-caching: with a unique per-request prefix the prompts differ, but
# we also disable prefix caching so each request's full KV is materialised and read.
PATH="$ROOT/.venv_vllm/bin:$PATH" PYTHONPATH="" \
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
  "$ROOT/.venv_vllm/bin/vllm" serve "$MODEL" --port "$PORT" \
  --gpu-memory-utilization "$GMU" --max-model-len "$MAXLEN" \
  --no-enable-prefix-caching --enforce-eager \
  > "$RES/e2e_vllm_server.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null; pkill -f e2e_handoff_stressor 2>/dev/null' EXIT

echo "=== wait for server /health (up to 300s) ==="
for i in $(seq 1 300); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then echo "ready in ${i}s"; break; fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then echo "server died; see e2e_vllm_server.log"; tail -20 "$RES/e2e_vllm_server.log"; exit 1; fi
  sleep 1
done

CLIENT_ARGS=(--model "$MODEL" --concurrency "$CONC" --prompt-words "$PROMPT_WORDS" \
             --max-tokens "$MAXTOK" --unique-prefix)

echo "=== warmup ==="
"$PEERKV_PY" experiments/e2e_vllm_client.py "${CLIENT_ARGS[@]}" --secs 12 --label warmup >/dev/null 2>&1 || true

for COND in idle push pull host local; do
  echo "=== condition: $COND ==="
  if [ "$COND" != "idle" ]; then
    "$PEERKV_PY" experiments/e2e_handoff_stressor.py --dir "$COND" --secs $((SECS+10)) \
      > "$RES/e2e_stressor_$COND.log" 2>&1 &
    STRESS_PID=$!
    sleep 3   # let the handoff reach steady state
  fi
  "$PEERKV_PY" experiments/e2e_vllm_client.py "${CLIENT_ARGS[@]}" \
    --secs "$SECS" --label "$COND" --out "$RES/e2e_cond_$COND.json"
  if [ "$COND" != "idle" ]; then
    kill "$STRESS_PID" 2>/dev/null; wait "$STRESS_PID" 2>/dev/null
    grep -h "stressor:" "$RES/e2e_stressor_$COND.log" || true
  fi
  sleep 2
done

echo "=== aggregate ==="
"$PEERKV_PY" - "$RES" <<'PY'
import json, sys, glob, os
res = sys.argv[1]
conds = {}
for f in sorted(glob.glob(os.path.join(res, "e2e_cond_*.json"))):
    d = json.load(open(f)); conds[d["label"]] = d
base = conds.get("idle", {}).get("tpot_ms_p99")
for c in ("idle","push","pull","host","local"):
    if c in conds and base:
        conds[c]["p99_inflation_pct_vs_idle"] = round((conds[c]["tpot_ms_p99"]/base-1)*100,2)
out = {"_experiment":"e2e_vllm_handoff","_is_measured":True,
       "model": conds.get("idle",{}).get("model","?"),"conditions":conds,
       "note":"holder=real vLLM continuous-batching decoder on GPU1; stressor=512MB cross-GPU handoff on GPU1 HBM"}
json.dump(out, open(os.path.join(res,"e2e_vllm.json"),"w"), indent=2)
print(json.dumps({c:{k:conds[c][k] for k in ("tpot_ms_p50","tpot_ms_p99","throughput_tok_s") if k in conds[c]} for c in conds}, indent=2))
PY
# NOTE: we intentionally do NOT run `nvidia-smi -rgc` here -- the box is kept clock-locked
# at 1410 MHz so the g1-g4 measurement protocol stays valid for subsequent runs.
echo "DONE -> $RES/e2e_vllm.json"
