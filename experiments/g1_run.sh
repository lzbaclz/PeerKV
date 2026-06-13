#!/usr/bin/env bash
# G1: run the victim-attribution driver with DCGM profiling counters captured in
# parallel, then align counters to per-condition windows. Clocks must be locked.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
PY="${PY:-python}"   # override with PY=/path/to/python on other machines
# Best-effort conda activation; harmless/no-op if conda is absent or PY is already set.
# Override the env with PEERKV_CONDA_ENV; point at any conda via $CONDA_PREFIX or PATH.
if [ -z "${PY_OVERRIDDEN:-}" ] && command -v conda >/dev/null 2>&1; then
  _conda_base="$(conda info --base 2>/dev/null || true)"
  [ -n "${_conda_base}" ] && source "${_conda_base}/etc/profile.d/conda.sh" 2>/dev/null && \
    conda activate "${PEERKV_CONDA_ENV:-peerkv}" 2>/dev/null || true
fi

RES=experiments/results
DCGM_LOG=$RES/g1_dcgm.log
mkdir -p "$RES"

# Fields: 1001 GR_ACTIVE, 1002 SM_ACTIVE, 1003 SM_OCCUPANCY, 1005 DRAM_ACTIVE,
#         1011 NVLINK_TX_BYTES, 1012 NVLINK_RX_BYTES
FIELDS=1001,1002,1003,1005,1011,1012

echo "=== ensure host engine ==="
pgrep -x nv-hostengine >/dev/null || sudo nv-hostengine
sleep 1

echo "=== start DCGM dmon (100ms) with epoch timestamps ==="
: > "$DCGM_LOG"
stdbuf -oL dcgmi dmon -e $FIELDS -d 100 2>/dev/null | while IFS= read -r line; do
  printf '%s %s\n' "$(date +%s.%N)" "$line"
done >> "$DCGM_LOG" &
DMON_PID=$!
sleep 1

echo "=== run G1 driver (seeds via repeats handled in python; here single rich run) ==="
"$PY" experiments/g1_victim_attribution.py "$@"

sleep 1
echo "=== stop dmon (pid $DMON_PID and child) ==="
pkill -P "$DMON_PID" 2>/dev/null || true
kill "$DMON_PID" 2>/dev/null || true

echo "=== align counters to condition windows ==="
"$PY" experiments/g1_analyze.py

echo "=== DONE: $RES/g1_attribution.json ==="
