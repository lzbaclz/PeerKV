#!/usr/bin/env bash
# DVFS counterfactual on the local dual-A100 box.  Needs sudo (clock lock).
#
# Tests: does unlocking DVFS deflate the g10 victim curve toward the A800
# numbers?  (torch version already ruled out, 2026-06-10.)
#
# Runs, in order:
#   1. g13 forensics, clocks LOCKED (control trace; clocks must be at 1410)
#   2. sudo nvidia-smi -rgc          (unlock DVFS)
#   3. g13 forensics, UNLOCKED
#   4. full g10 paced sweep, UNLOCKED (torch 2.7 venv, same as 6/10 rerun)
#   5. relock 1410 + restore the paper's g10 json (trap-guaranteed)
#
# Usage:
#   bash scripts/run_dvfs_counterfactual_local.sh 2>&1 | tee /tmp/dvfs_cf.log
#
# Time estimate: ~15 min total.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv_torch27/bin/python}"
R=experiments/results
STAMP=$(date +%Y%m%d_%H%M%S)
BK="/tmp/g10_paper_backup_${STAMP}.json"

echo "== preflight =="
sudo -n true || { echo "need sudo (NOPASSWD or cached credentials)"; exit 1; }
$PY -c "import torch; print('torch', torch.__version__, '| ngpu', torch.cuda.device_count())"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,clocks.sm --format=csv,noheader
echo "GPUs must be IDLE and clocks at 1410 MHz before continuing."
read -r -p "Continue? [y/N] " ok
[[ "$ok" == "y" ]] || exit 1

cp -v "$R/g10_paced_sweep.json" "$BK"
md5sum "$R/g10_paced_sweep.json" "$BK"

relock_and_restore() {
  echo "== cleanup: relock 1410 + restore paper g10 json =="
  sudo nvidia-smi -lgc 1410,1410 \
    || echo "WARN: relock FAILED -- run manually: sudo nvidia-smi -lgc 1410,1410"
  [ -f "$BK" ] && cp -v "$BK" "$R/g10_paced_sweep.json"
  nvidia-smi --query-gpu=index,clocks.sm --format=csv,noheader
}
trap relock_and_restore EXIT

echo "== 1/4 g13 LOCKED =="
$PY experiments/g13_dvfs_forensics.py || exit 1
mv "$R/g13_dvfs_forensics.json" "$R/g13_dvfs_a100_locked.json"

echo "== 2/4 unlock DVFS =="
sudo nvidia-smi -rgc
sleep 3
nvidia-smi --query-gpu=index,clocks.sm --format=csv,noheader

echo "== 3/4 g13 UNLOCKED =="
$PY experiments/g13_dvfs_forensics.py || exit 1
mv "$R/g13_dvfs_forensics.json" "$R/g13_dvfs_a100_unlocked.json"

echo "== 4/4 g10 full sweep UNLOCKED (~5 min) =="
$PY experiments/g10_paced_footprint_sweep.py --seeds 3 || exit 1
mv "$R/g10_paced_sweep.json" "$R/g10_paced_sweep_a100_torch27_unlocked.json"

# explicit cleanup now (trap stays as safety net; cp is idempotent)
relock_and_restore
trap - EXIT

echo "== outputs =="
ls -lh "$R/g13_dvfs_a100_locked.json" "$R/g13_dvfs_a100_unlocked.json" \
       "$R/g10_paced_sweep_a100_torch27_unlocked.json"
echo "paper json restored; md5 must match the backup:"
md5sum "$BK" "$R/g10_paced_sweep.json"
