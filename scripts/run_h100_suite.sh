#!/usr/bin/env bash
# H100 (or any 2-GPU NVLink box) cross-hardware replication of the cross-GPU
# KV-transfer measurement suite (g1-g4). Verifies the A100 findings generalize:
# (1) direction is a non-effect, (2) victim cost scales with HBM-read footprint.
#
# Self-contained: queries each GPU's max graphics clock and locks to it (do NOT
# hardcode 1410, that is A100-specific), runs g1-g4 + nsys, tags outputs into
# experiments/results_h100/, and prints a summary to compare against the A100 JSONs.
#
# Usage on the H100 machine:
#   1) clone the repo, create the same conda/venv with torch+CUDA, `pip install -e .`
#      (only needs: torch, matplotlib, nvidia-ml-py; DCGM `dcgmi` + Nsight `nsys` for
#      counters/traces -- optional but recommended).
#   2) bash scripts/run_h100_suite.sh
#   3) send back experiments/results_h100/*.json (and the printed summary).
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PY="${PY:-python}"               # set PY=/path/to/python if not on PATH; exported so g1_run.sh inherits it
TAG_DIR="experiments/results_h100"
mkdir -p "$TAG_DIR"

echo "=== environment ==="
$PY - <<'PY'
import torch
assert torch.cuda.device_count() >= 2, "need >=2 GPUs"
for i in range(2):
    print(f"GPU{i}: {torch.cuda.get_device_name(i)}")
print("torch", torch.__version__, "cuda", torch.version.cuda)
PY
echo "--- topology (expect NV# between GPU0 and GPU1; NVSwitch is fine too) ---"
nvidia-smi topo -m | head -5
echo "--- MIG must be Disabled ---"
nvidia-smi --query-gpu=index,mig.mode.current --format=csv

echo "=== lock clocks to each GPU's max (generic; H100 != 1410 MHz) ==="
sudo nvidia-smi -pm 1 >/dev/null || true
MAXCLK=$(nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo "max graphics clock = ${MAXCLK} MHz"
sudo nvidia-smi -lgc "${MAXCLK},${MAXCLK}" || echo "WARN: could not lock clocks (need sudo); results may have DVFS noise"

echo "=== DCGM host engine (for g1 counters; optional) ==="
if command -v dcgmi >/dev/null 2>&1; then
  pgrep -x nv-hostengine >/dev/null || sudo nv-hostengine || true
fi

run() { echo; echo ">>> $*"; "$@"; }

echo "=== G1: attribution + counters ==="
if command -v dcgmi >/dev/null 2>&1; then
  bash experiments/g1_run.sh --iters 300 --warmup 30 --counter-secs 6 || true
else
  run $PY experiments/g1_victim_attribution.py --iters 300 --warmup 30 --counter-secs 0 || true
fi
cp -f experiments/results/g1_attribution.json "$TAG_DIR/" 2>/dev/null || \
  cp -f experiments/results/g1_timing.json "$TAG_DIR/" 2>/dev/null || true

echo "=== G2: realistic handoff window ==="
run $PY experiments/g2_handoff_window.py --seeds 3
cp -f experiments/results/g2_handoff_window.json "$TAG_DIR/"

echo "=== G3: placement cost model ==="
run $PY experiments/g3_placement.py --seeds 3
cp -f experiments/results/g3_placement.json "$TAG_DIR/"

echo "=== G4: measurement trap (locked then unlocked) ==="
run $PY experiments/g4_measurement_trap.py --tag locked --seeds 3
cp -f experiments/results/g4_trap_locked.json "$TAG_DIR/"
sudo nvidia-smi -rgc >/dev/null 2>&1 && sleep 2 && \
  run $PY experiments/g4_measurement_trap.py --tag unlocked --seeds 3 && \
  cp -f experiments/results/g4_trap_unlocked.json "$TAG_DIR/" && \
  sudo nvidia-smi -lgc "${MAXCLK},${MAXCLK}" >/dev/null 2>&1 || \
  echo "skip unlocked cell (no sudo)"

echo "=== nsys overlap (optional) ==="
if command -v nsys >/dev/null 2>&1; then
  mkdir -p "$TAG_DIR/nsys"
  nsys profile --force-overwrite true -o "$TAG_DIR/nsys/push_overlap" \
    --trace=cuda --capture-range=cudaProfilerApi --capture-range-end=stop \
    $PY experiments/prof_target.py 2>/dev/null || echo "nsys step skipped"
fi

echo; echo "=== SUMMARY (compare to A100) ==="
$PY - "$TAG_DIR" <<'PY'
import json, sys, os
d = sys.argv[1]
def load(n):
    p = os.path.join(d, n)
    return json.load(open(p)) if os.path.exists(p) else None
g3 = load("g3_placement.json")
g2 = load("g2_handoff_window.json")
g4l = load("g4_trap_locked.json")
if g3:
    print("G3 placement (victim vs HBM-read BW):")
    for k, v in g3["destinations"].items():
        print(f"  {k:11s} {v['copy_bw_gbs']:7.1f} GB/s  victim +{v['victim_slowdown_pct']:.1f}%")
if g2:
    c = g2["configs"]
    print("G2 direction non-effect (512MB): push +%.1f%% vs pull +%.1f%%" % (
        c["push_x1"]["victim_slowdown_pct"], c["pull_x1"]["victim_slowdown_pct"]))
if g4l:
    r = g4l["results"]
    print("G4 trap (locked): naive push +%.0f%% pull +%.0f%%  |  clean push +%.1f%% pull +%.1f%%" % (
        r["push"]["naive"]["victim_pct"], r["pull"]["naive"]["victim_pct"],
        r["push"]["clean"]["victim_pct"], r["pull"]["clean"]["victim_pct"]))
print("\nExpected (A100): direction non-effect (push~=pull); victim monotone in HBM-read BW;")
print("naive manufactures a large phantom push/pull gap that the clean protocol removes.")
PY
sudo nvidia-smi -rgc >/dev/null 2>&1 || true
echo "DONE. Send back: $TAG_DIR/*.json"
