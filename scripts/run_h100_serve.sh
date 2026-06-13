#!/usr/bin/env bash
# run_h100_serve.sh -- vLLM serving suite for a DEDICATED box (closes W3/Q-i).
#
# WHY a dedicated box: vLLM's mandatory memory profiling + real-model weight load
# are unreliable on the shared A100 box (co-tenant churns GPU mem/IO -> the 7B run
# stalled 3x there; only tiny models slipped through). On a dedicated box these run
# cleanly. This script measures, with a real 7B:
#   * M1  single-GPU KV ceiling (num_gpu_blocks) + offline throughput, util sweep
#   * TP-2 native tensor-parallel throughput (the honest "use both GPUs" baseline
#     the paper's related-work TP section now needs -- MEASURED, not asserted)
#   * (online TTFT/TPOT/P50/P99 via `vllm serve` + `vllm bench serve`)
#
# The PeerKV peer-tier serve (M2/M3: peer_kv_alloc.patch_peer_kv_allocation +
# peerkv_register.register_peerkv) is intentionally NOT run unattended -- its one
# remaining unknown is CUDA-graph-replay behaviour; run it interactively with
# --enforce-eager first (see experiments/serve/INTEGRATION_PLAN.md).
#
# USAGE (dedicated box, repo present, vLLM 0.8.5 env active or VLLM_PY set):
#   bash scripts/run_h100_serve.sh
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=/tmp/h100_serve.log; : > "$LOG"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

VLLM_PY="${VLLM_PY:-$(command -v vllm || true)}"
[ -n "$VLLM_PY" ] || { say "vLLM CLI not found; set VLLM_PY=/path/to/vllm (env: vllm==0.8.5, torch 2.6 cu124)"; exit 1; }
MODEL="${MODEL:-NousResearch/Llama-2-7b-hf}"
NGPU=$(nvidia-smi -L | wc -l)
say "vllm=$VLLM_PY model=$MODEL ngpu=$NGPU"

bench(){ say ">>> $*"; timeout 1200 $VLLM_PY bench throughput --model "$MODEL" \
  --input-len 1024 --output-len 128 --num-prompts 200 --enforce-eager "$@" 2>&1 | tee -a "$LOG"; }

# M1: single-GPU KV ceiling + throughput across util (num_gpu_blocks logged at init)
for U in 0.5 0.9; do
  CUDA_VISIBLE_DEVICES=0 bench --gpu-memory-utilization "$U" --max-model-len 4096
done

# TP-2: native tensor parallelism across 2 GPUs (the honest TP baseline)
if [ "$NGPU" -ge 2 ]; then
  bench --gpu-memory-utilization 0.9 --max-model-len 4096 --tensor-parallel-size 2
fi

# Online P50/P99 (TTFT/TPOT/ITL): start a server in its OWN process group, hit it,
# then KILL THE WHOLE GROUP (orphan-safe -- a plain `kill $SRV` leaks the vLLM
# EngineCore worker, which on a shared box holds GPU mem + cache locks).
say ">>> online serve (P50/P99) -- single GPU"
setsid env CUDA_VISIBLE_DEVICES=0 $VLLM_PY serve "$MODEL" --served-model-name l2 \
  --gpu-memory-utilization 0.9 --max-model-len 4096 --port 8011 >"$LOG.server" 2>&1 &
SRV=$!; sleep 2; PGID=$(ps -o pgid= -p "$SRV" 2>/dev/null | tr -d ' ')
ok=0; for i in $(seq 1 60); do grep -q "Application startup complete" "$LOG.server" 2>/dev/null && ok=1 && break; sleep 5; done
if [ "$ok" = 1 ]; then
  timeout 600 $VLLM_PY bench serve --model l2 --base-url http://127.0.0.1:8011 \
    --dataset-name random --num-prompts 200 --request-rate 8 2>&1 | tee -a "$LOG" || say "online bench failed"
else
  say "server did not start in 300s; tail:"; tail -8 "$LOG.server" | tee -a "$LOG"
fi
# orphan-safe teardown: terminate the whole server process group, then verify
if [ -n "$PGID" ]; then kill -TERM -- -"$PGID" 2>/dev/null; sleep 4; kill -KILL -- -"$PGID" 2>/dev/null; fi
say "leftover serve procs (should be 0): $(pgrep -fc 'vllm serve' 2>/dev/null || echo 0)"
say "DONE. Send /tmp/h100_serve.log back (grep 'GPU KV cache|Throughput|P99|TPOT|TTFT')."
