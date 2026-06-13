#!/usr/bin/env bash
# One-command GH200 evaluation battery for the CCF-A submission.
#
# Runs every stage that needs the box, in order, into a timestamped results
# dir with per-stage logs. Designed to be launched unattended so you do not
# pay GH200 rates while reading output. Re-runnable; each run gets its own dir.
#
# Prereqs (do the cheap parts on a cheap GPU FIRST -- see CCFA_runbook.md):
#   UMA_BUILD_CUDA=1 pip install -e .[vllm]    # build umallm._uma_native (sm_90)
#
# Usage:
#   bash scripts/run_gh200_ccfa.sh                       # 8B, default sweep
#   MODEL_BIG=meta-llama/Llama-3.1-70B bash scripts/run_gh200_ccfa.sh
#   SKIP_BIG=1 bash scripts/run_gh200_ccfa.sh            # skip the 70B headline
set -euo pipefail

MODEL_SMALL="${MODEL_SMALL:-meta-llama/Llama-3.1-8B}"
MODEL_BIG="${MODEL_BIG:-meta-llama/Llama-3.1-70B}"
CTX_SWEEP="${CTX_SWEEP:-4096,16384,65536,131072}"
CTX_SWEEP_BIG="${CTX_SWEEP_BIG:-8192,32768,131072}"
HBM_BUDGET="${HBM_BUDGET:-40}"
CONC="${CONC:-16}"
DEADLINE_MS="${DEADLINE_MS:-50}"
SKIP_BIG="${SKIP_BIG:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT="experiments/results/gh200_${TS}"
mkdir -p "$OUT"
echo "results -> $OUT"

log() { echo "=== $* ==="; }
run() { echo "+ $*"; "$@"; }

# 0. environment provenance (so a result is reproducible/auditable)
log "stage 0: environment"
{
  date
  nvidia-smi || true
  python -c "import torch,sys;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
  python -c "import vllm;print('vllm',vllm.__version__)" 2>/dev/null || echo "vllm: not importable"
  python -c "import umallm;print('umallm native_available=',umallm.native_available())" 2>/dev/null || true
  git -C . rev-parse HEAD 2>/dev/null || true
} >"$OUT/00_env.txt" 2>&1
cat "$OUT/00_env.txt"

# 1. confirm the managed allocator catches vLLM's KV alloc (cheap, fail fast)
log "stage 1: verify MemPool catches KV"
run python scripts/verify_mempool_catches_kv.py --model "$MODEL_SMALL" \
  >"$OUT/01_mempool.txt" 2>&1 || { echo "MemPool check FAILED; see $OUT/01_mempool.txt"; exit 1; }
tail -n 5 "$OUT/01_mempool.txt"

# 2. worker-path correctness + cost-model fidelity + coherence probe
log "stage 2: worker validation"
run python scripts/validate_gh200_worker.py --blocks 512 \
  --json "$OUT/02_worker.json" >"$OUT/02_worker.txt" 2>&1 \
  || { echo "worker validation FAILED; see $OUT/02_worker.txt"; exit 1; }
tail -n 8 "$OUT/02_worker.txt"

# 3. calibrate the GH200 cost model (feeds e1 <10% prediction-error claim)
log "stage 3: calibration"
run python -c "from umallm.calibration import write_calibration; \
import json; print(json.dumps(write_calibration('$OUT/03_calibration.json'), indent=2))" \
  >"$OUT/03_calibration.txt" 2>&1 || echo "calibration stage warned; see $OUT/03_calibration.txt"

# 4. THE headline: Route B vs passive-UVM vs vLLM-offload vs HBM-only (8B)
log "stage 4: routeb benchmark (8B)"
run python experiments/routeb_benchmark.py --backend vllm --model "$MODEL_SMALL" \
  --context-sweep "$CTX_SWEEP" --hbm-budget-gib "$HBM_BUDGET" \
  --concurrency "$CONC" --deadline-ms "$DEADLINE_MS" \
  --out "$OUT/04_routeb_8b.json" >"$OUT/04_routeb_8b.txt" 2>&1 \
  || { echo "8B benchmark FAILED; see $OUT/04_routeb_8b.txt"; exit 1; }
tail -n 16 "$OUT/04_routeb_8b.txt"
run python experiments/analyze_routeb.py "$OUT/04_routeb_8b.json" \
  --outdir "$OUT/04_routeb_8b_figs" >>"$OUT/04_routeb_8b.txt" 2>&1 || true

# 5. headline capacity story on the big model (single GH200, KV via Grace)
if [ "$SKIP_BIG" != "1" ]; then
  log "stage 5: routeb benchmark (70B headline)"
  run python experiments/routeb_benchmark.py --backend vllm --model "$MODEL_BIG" \
    --context-sweep "$CTX_SWEEP_BIG" --hbm-budget-gib "$HBM_BUDGET" \
    --concurrency "$CONC" --deadline-ms "$DEADLINE_MS" \
    --n-layers 80 --n-kv-heads 8 --head-dim 128 \
    --out "$OUT/05_routeb_70b.json" >"$OUT/05_routeb_70b.txt" 2>&1 \
    || { echo "70B benchmark warned; see $OUT/05_routeb_70b.txt"; }
  run python experiments/analyze_routeb.py "$OUT/05_routeb_70b.json" \
    --outdir "$OUT/05_routeb_70b_figs" >>"$OUT/05_routeb_70b.txt" 2>&1 || true
else
  log "stage 5: skipped (SKIP_BIG=1)"
fi

log "DONE -> $OUT"
echo "Tables + figures: $OUT/*_figs/  |  raw JSON: $OUT/*.json"
echo "Remember: routeb_benchmark with --backend vllm is MEASURED; the mock is not."
