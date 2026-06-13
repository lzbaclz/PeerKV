#!/usr/bin/env bash
# Reproduce the PeerKV/UMA tiering evaluation scripts (e15/e18/e21...).
#
# Reproduce every measured number in the NVTier/PeerKV evaluation on a dual-A100
# (or any >=2-GPU NVLink) box. Frozen env: torch 2.6.0+cu124, driver 535.309.01,
# CUDA 12.4, conda env `peerkv`. All timings use CUDA events; >=30 trials/3 seeds.
set -uo pipefail
cd "$(dirname "$0")/.."
source /home/lzq/miniconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate peerkv 2>/dev/null || true

say(){ printf '\n===== %s =====\n' "$*"; }

say "0. Prerequisite: NVLink active (NV12) and MIG off. If topo shows NODE, run:"
echo "   nvidia-smi --query-gpu=index,mig.mode.current --format=csv   # check MIG"
echo "   sudo bash scripts/activate_nvlink.sh                          # disables MIG, retrains links"
nvidia-smi topo -m 2>/dev/null | sed -n '2,3p'

say "1. Unit tests (cost model, sizing, placement, exact streaming merge, C*)"
python -m pytest -q

say "2. RQ1 -- per-tier bandwidth + setup calibration (e15, e20)"
python experiments/e15_multigpu_kv_tiers.py --mb 512 --trials 30
python scripts/_e20_calib.py --trials 40

say "3. RQ2 -- coalesced Pareto, overlapped prefetch, decode-loop throughput"
python scripts/_e18_chunked.py --n-blocks 2048 --local-blocks 128 --block-tokens 256 --chunks 1,8,32,128,512 --trials 8
python scripts/_e19_prefetch.py --n-blocks 2048 --local-blocks 128 --chunk-blocks 128 --trials 30 --seeds 3
python scripts/_e21_decode_loop.py --model llama3-8b --ctx-tokens 65536 --overflow-frac 0.5 --block-sizes 16,64,256 --chunk-kb 8192 --steps 20
python experiments/e18_p2p_flash.py --n-blocks 2048 --local-blocks 128 --trials 30 --chunk-blocks 1   # per-block (loses)
python experiments/e10_route_b_residency.py --mb 256 --trials 20                                       # discrete-migration endpoint

say "4. RQ3/RQ4 -- cost-model prediction, sizing@binding-SLO, generality, contention"
python experiments/e16_placement_tpot.py --calib experiments/results/multigpu_kv_tiers.json --block-kb 8192
python scripts/_e22_sizing.py
python scripts/_e24_generality.py
python scripts/_e23_contention.py --mb 256 --trials 30

say "DONE. Results in experiments/results/*.json (ignored by git)."
