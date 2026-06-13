#!/usr/bin/env bash
# G12 DVFS forensics on the A800 HGX partition.  No sudo, no network needed.
#
# Usage:
#   bash scripts/run_g13_a800.sh 2>&1 | tee g13_a800_console.log
#   PY=/path/to/python bash scripts/run_g13_a800.sh   # custom python
#
# Time estimate: ~5 min.
# BRING BACK:  experiments/results/g13_dvfs_a800.json  +  the console log
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python}"
R=experiments/results

echo "============================================="
echo "  PeerKV g13: DVFS forensics (A800)"
echo "============================================="
echo "time: $(date)"
$PY -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| ngpu', torch.cuda.device_count())"
nvidia-smi --query-gpu=index,name,driver_version,persistence_mode,clocks.sm,clocks.max.sm --format=csv
echo

$PY experiments/g13_dvfs_forensics.py
RC=$?
if [ $RC -ne 0 ]; then
  echo "g13 FAILED (rc=$RC)"
  exit $RC
fi
mv "$R/g13_dvfs_forensics.json" "$R/g13_dvfs_a800.json"

echo
echo "============================================="
echo "  DONE  $(date)"
echo "============================================="
ls -lh "$R/g13_dvfs_a800.json"
echo "BRING BACK: experiments/results/g13_dvfs_a800.json + this console log"
