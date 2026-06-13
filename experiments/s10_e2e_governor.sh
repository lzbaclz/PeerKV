#!/usr/bin/env bash
# S10 -- vLLM e2e anchor for the governor: receiver-ingress against a REAL
# engine.  vLLM serves on physical GPU1; the stressor WRITES GPU1's HBM
# (GPU0->GPU1, the s2 headline regime).  Conditions:
#   idle      no ingress (baseline)
#   unpaced   saturating 512MB ingress (bandwidth-greedy engine analog)
#   governed  Governor(ff, worst-bucket, native pacer), eps=5%
# Client-side TPOT percentiles per condition; the contract under test: the
# governed arm's TPOT inflation stays near eps even though the calibration
# victim was a synthetic decode layer, not this engine.
#
# Assumes clocks already locked (persistence on, -lgc 1410 -- the box default
# in this project).  vLLM env: .venv_vllm; stressor env: peerkv conda.
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

CLK=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits -i 0 | head -1)
echo "clocks: ${CLK}MHz (expect 1410 locked)"

echo "=== start vLLM on physical GPU1 ==="
PATH="$ROOT/.venv_vllm/bin:$PATH" PYTHONPATH="" \
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
  "$ROOT/.venv_vllm/bin/vllm" serve "$MODEL" --port "$PORT" \
  --gpu-memory-utilization "$GPU_UTIL" --max-model-len "$MAXLEN" \
  --max-num-seqs "$CONC" --enforce-eager \
  > "$RES/s10_server.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null; pkill -f s10_governor_stressor 2>/dev/null' EXIT

echo "=== wait for /health (up to 360s) ==="
for i in $(seq 1 360); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "ready in ${i}s"; break; }
  kill -0 $SERVER_PID 2>/dev/null || { echo "server died"; tail -30 "$RES/s10_server.log"; exit 1; }
  sleep 1
done

echo "=== warmup ==="
"$PEERKV_PY" experiments/e2e_vllm_client.py --model "$MODEL" --concurrency "$CONC" \
  --secs 12 --max-tokens 256 --label warmup >/dev/null 2>&1 || true

for COND in idle unpaced governed; do
  echo "=== condition: $COND ==="
  if [ "$COND" != "idle" ]; then
    PYTHONPATH="$ROOT" PEERKV_SKIP_IDLE_PROBE=1 \
      "$PEERKV_PY" experiments/s10_governor_stressor.py --arm "$COND" \
      --secs $((SECS + 25)) > "$RES/s10_stressor_$COND.log" 2>&1 &
    STRESS_PID=$!
    sleep 8   # native pacer JIT is cached after first import; warm ingress
  fi
  "$PEERKV_PY" experiments/e2e_vllm_client.py --model "$MODEL" --concurrency "$CONC" \
    --secs "$SECS" --max-tokens 256 --label "$COND" \
    --out "$RES/s10_cond_$COND.json"
  if [ "$COND" != "idle" ]; then
    kill $STRESS_PID 2>/dev/null; wait $STRESS_PID 2>/dev/null
    cat "$RES/s10_stressor_$COND.log" | grep -v Warning | tail -2
  fi
done

echo "=== summarize ==="
"$PEERKV_PY" - <<'EOF'
import json, datetime
from pathlib import Path
res = Path("experiments/results")
conds = {c: json.loads((res / f"s10_cond_{c}.json").read_text())
         for c in ("idle", "unpaced", "governed")}
base = conds["idle"]
out = {"_experiment": "s10_e2e_governor", "_is_measured": True,
       "victim": "vLLM " + conds["idle"].get("model", "(see cond files)") + " decode on physical GPU1 (receiver side)",
       "_timing_method": "client-side TPOT; stressor in separate process",
       "conditions": {}}
for c, d in conds.items():
    out["conditions"][c] = {
        "tpot_ms_p50": d["tpot_ms_p50"], "tpot_ms_p99": d["tpot_ms_p99"],
        "tpot_ms_mean": d["tpot_ms_mean"], "n_tokens": d["n_tokens"],
        "throughput_tok_s": d["throughput_tok_s"],
        "p50_inflation_pct": round((d["tpot_ms_p50"]/base["tpot_ms_p50"]-1)*100, 2),
        "mean_inflation_pct": round((d["tpot_ms_mean"]/base["tpot_ms_mean"]-1)*100, 2),
    }
for c in ("unpaced", "governed"):
    log = (res / f"s10_stressor_{c}.log").read_text()
    for line in log.splitlines():
        if "GB/s sustained" in line:
            out["conditions"][c]["stressor"] = line.strip()
out["_generated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
(res / "s10_e2e_governor.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out["conditions"], indent=1))
EOF
echo "-> $RES/s10_e2e_governor.json"
