#!/usr/bin/env bash
# HGX (NVSwitch, >=3 GPUs; ideally 8x H100/A100 SXM) extension of the PeerKV suite.
#
# What it adds over run_h100_suite.sh (which covers 1-to-1 on a 2-GPU slice):
#   0) topology forensics: prove the fabric is NVSwitch-routed (topo -m, nvlink -s,
#      lspci NVSwitch enumeration) -- this is recorded into the g8 JSON too.
#   1) 1-to-1 replication through the switch on the (0,1) pair: g2/g3/g4.
#   2) g8 fan-out (1 busy holder -> K consumers, push vs pull, K = 1,2,4,N-1)
#      and fan-in (K busy holders -> 1 consumer): the open reviewer question.
#
# Usage on the rented HGX box:
#   1) unzip bundle (or clone repo); python with torch+CUDA on PATH (PY=... to override)
#   2) bash scripts/run_hgx_suite.sh
#   3) send back experiments/results_hgx/*.json and the console log.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PY="${PY:-python}"
TAG_DIR="experiments/results_hgx"
mkdir -p "$TAG_DIR"

echo "=== environment ==="
$PY - <<'PY'
import torch
n = torch.cuda.device_count()
assert n >= 3, f"HGX suite expects >=3 GPUs (got {n}); use run_h100_suite.sh for 2"
for i in range(n):
    print(f"GPU{i}: {torch.cuda.get_device_name(i)}")
print("torch", torch.__version__, "cuda", torch.version.cuda)
PY

echo "=== topology forensics (keep this output!) ==="
nvidia-smi topo -m || true
echo "--- NVLink status GPU0 ---"
nvidia-smi nvlink -s -i 0 | head -25 || true
echo "--- PCIe NVSwitch enumeration (NVSwitch3 = 10de:22a3) ---"
lspci -d 10de: 2>/dev/null | grep -i -E "bridge|switch" || echo "(lspci empty/unavailable)"
echo "--- MIG must be Disabled ---"
nvidia-smi --query-gpu=index,mig.mode.current --format=csv

echo "=== lock clocks to each GPU's max (best effort) ==="
sudo nvidia-smi -pm 1 >/dev/null 2>&1 || true
MAXCLK=$(nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo "max graphics clock = ${MAXCLK} MHz"
sudo nvidia-smi -lgc "${MAXCLK},${MAXCLK}" 2>/dev/null || \
  echo "WARN: could not lock clocks (no sudo); C2 ratio metric keeps results honest"

run() { echo; echo ">>> $*"; "$@"; }

echo "=== 1-to-1 through the switch, pair (0,1): G3 placement ==="
run $PY experiments/g3_placement.py --seeds 3
cp -f experiments/results/g3_placement.json "$TAG_DIR/"

echo "=== 1-to-1 through the switch: G2 handoff window (chunking) ==="
run $PY experiments/g2_handoff_window.py --seeds 5
cp -f experiments/results/g2_handoff_window.json "$TAG_DIR/"

echo "=== 1-to-1 through the switch: G4 measurement trap ==="
run $PY experiments/g4_measurement_trap.py --tag locked --seeds 3
cp -f experiments/results/g4_trap_locked.json "$TAG_DIR/"

echo "=== G8: NVSwitch fan-out / fan-in (the reviewer question) ==="
run $PY experiments/g8_nvswitch_fan.py --seeds 5
cp -f experiments/results/g8_nvswitch_fan.json "$TAG_DIR/"

echo "=== G9: stream-priority sweep on the switched pair (optional cross-check) ==="
run $PY experiments/g9_stream_priority.py --seeds 5 || true
cp -f experiments/results/g9_stream_priority.json "$TAG_DIR/" 2>/dev/null || true

echo; echo "=== SUMMARY ==="
$PY - "$TAG_DIR" <<'PY'
import json, sys, os
d = sys.argv[1]
def load(n):
    p = os.path.join(d, n)
    return json.load(open(p)) if os.path.exists(p) else None
g8 = load("g8_nvswitch_fan.json")
if g8:
    print("G8 fan-out (1 holder -> K consumers): victim should track aggregate BW; push==pull at every K")
    for k, v in g8["results"]["fan_out"].items():
        print(f"  {k:10s} agg={v['aggregate_copy_bw_gbs']:7.1f} GB/s  victim +{v['victim_slowdown_pct']:.1f}% (+-{v['victim_slowdown_std']})")
    print("G8 fan-in (K holders -> 1 consumer): per-copy BW drops with K; holder0 cost should drop with it")
    for k, v in g8["results"]["fan_in"].items():
        print(f"  {k:10s} agg={v['aggregate_copy_bw_gbs']:7.1f} GB/s  holder0 +{v['holder0_victim_slowdown_pct']:.1f}% (+-{v['holder0_victim_slowdown_std']})")
print("\nDecision rules:")
print("  push==pull at every K        -> direction null extends to switched fan (paper upgrade)")
print("  victim tracks aggregate BW   -> cost law extends; chunk/stagger fans accordingly")
print("  pull >> push at K>1          -> issuer matters under fan: report as new switched-fabric finding")
PY
sudo nvidia-smi -rgc >/dev/null 2>&1 || true
echo "DONE. Send back: $TAG_DIR/*.json + this console log."
