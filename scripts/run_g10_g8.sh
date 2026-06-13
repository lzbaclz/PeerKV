#!/usr/bin/env bash
# One-shot runner for g10 (paced-footprint sweep) and g8 (NVSwitch fan).
#
# Behavior:
#   >=2 GPUs: runs g10 (A100 auto-detected or override with --hbm-peak-gbs)
#   >=3 GPUs + NVSwitch: also runs g8 fan-in/fan-out
#
# Usage:
#   bash scripts/run_g10_g8.sh               # auto-detect everything
#   bash scripts/run_g10_g8.sh --h100        # force H100 peak (3350 GB/s)
#   PY=/path/to/python bash scripts/run_g10_g8.sh  # custom python
#
# Time estimate:  g10 alone ~5 min;  g10 + g8 (8 GPUs) ~25 min total.
# Output:  experiments/results/g10_paced_sweep.json
#          experiments/results/g8_nvswitch_fan.json  (if >=3 GPUs)
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
PY="${PY:-python}"

echo "============================================="
echo "  PeerKV: g10 + g8 one-shot runner"
echo "============================================="
echo "python: $PY"
echo "time:   $(date)"
echo

# --- Environment check ---
NGPU=$($PY -c "import torch; print(torch.cuda.device_count())")
echo "GPUs detected: $NGPU"
if [ "$NGPU" -lt 2 ]; then
  echo "ERROR: need >=2 GPUs. Exiting."
  exit 1
fi
$PY -c "
import torch
for i in range(min(int('$NGPU'), 4)):
    print(f'  GPU{i}: {torch.cuda.get_device_name(i)}')
print(f'  torch {torch.__version__}  cuda {torch.version.cuda}')
"
echo

# --- Lock clocks (best effort) ---
echo ">>> Locking clocks (best effort)..."
sudo nvidia-smi -pm 1 >/dev/null 2>&1 || true
MAXCLK=$(nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
if [ -n "$MAXCLK" ]; then
  echo "    max graphics clock = ${MAXCLK} MHz"
  sudo nvidia-smi -lgc "${MAXCLK},${MAXCLK}" 2>/dev/null && echo "    clocks locked." || \
    echo "    WARN: cannot lock (no sudo); C2 ratio metric still valid."
fi
echo

# --- Parse args ---
G10_EXTRA=""
if [[ "${1:-}" == "--h100" ]]; then
  G10_EXTRA="--hbm-peak-gbs 3350"
  echo ">>> H100 mode: HBM peak forced to 3350 GB/s"
fi

# --- g10: paced-footprint sweep ---
echo "============================================="
echo "  G10: paced-footprint sweep (~5 min)"
echo "============================================="
echo ">>> $PY experiments/g10_paced_footprint_sweep.py --seeds 3 $G10_EXTRA"
$PY experiments/g10_paced_footprint_sweep.py --seeds 3 $G10_EXTRA
G10_RC=$?
echo
if [ $G10_RC -eq 0 ]; then
  echo "  g10 OK -> experiments/results/g10_paced_sweep.json"
else
  echo "  g10 FAILED (rc=$G10_RC)"
fi
echo

# --- g8: fan-in/fan-out (only if >=3 GPUs) ---
if [ "$NGPU" -ge 3 ]; then
  echo "============================================="
  echo "  G8: NVSwitch fan-in/fan-out (~20 min)"
  echo "============================================="
  echo ">>> topology forensics"
  nvidia-smi topo -m 2>/dev/null || true
  echo
  echo ">>> $PY experiments/g8_nvswitch_fan.py --seeds 5"
  $PY experiments/g8_nvswitch_fan.py --seeds 5
  G8_RC=$?
  echo
  if [ $G8_RC -eq 0 ]; then
    echo "  g8 OK -> experiments/results/g8_nvswitch_fan.json"
  else
    echo "  g8 FAILED (rc=$G8_RC)"
  fi
else
  echo ">>> Skipping g8 (need >=3 GPUs for fan experiments; have $NGPU)"
fi
echo

# --- Unlock clocks ---
sudo nvidia-smi -rgc >/dev/null 2>&1 || true

# --- Summary ---
echo "============================================="
echo "  DONE  $(date)"
echo "============================================="
echo "Results:"
ls -lh experiments/results/g10_paced_sweep.json 2>/dev/null || echo "  (g10 not found)"
ls -lh experiments/results/g8_nvswitch_fan.json 2>/dev/null || echo "  (g8 not found or skipped)"
echo
echo "Next steps:"
echo "  1. Send back the JSON(s) + this console log"
echo "  2. g10 data -> fit the normalized tiering rule into a continuous cost curve"
echo "  3. g8 data -> decide whether direction null extends to fan (paper upgrade)"
