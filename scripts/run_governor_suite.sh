#!/usr/bin/env bash
# Governor suite runner: s1 census -> s2 -> s3 -> s5 fan-in -> s6 -> s8 -> s9.
# Designed to run unchanged on the dual-A100 box (root, locked clocks), the
# 8xGPU HGX containers, and SLURM compute nodes (no sudo: clock locking is
# skipped, boost-stable protocol; fan-in gets K up to device_count-1).
#
#   bash scripts/run_governor_suite.sh            # full suite
#   ONLY=s5 bash scripts/run_governor_suite.sh    # one experiment
#   PY=/path/to/python bash scripts/run_governor_suite.sh
#
# Per-experiment failures are non-fatal (WARN + continue): one broken
# experiment must not cost a whole batch-queue allocation its remaining work.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
PY="${PY:-python}"
if [ -z "${PY_OVERRIDDEN:-}" ] && command -v conda >/dev/null 2>&1; then
  _conda_base="$(conda info --base 2>/dev/null || true)"
  [ -n "${_conda_base}" ] && source "${_conda_base}/etc/profile.d/conda.sh" 2>/dev/null && \
    conda activate "${PEERKV_CONDA_ENV:-peerkv}" 2>/dev/null || true
fi
ONLY="${ONLY:-all}"

echo "python: $PY"
$PY -c "import torch; print('devices:', torch.cuda.device_count(), torch.cuda.get_device_name(0))"

# best-effort clock locking (root only; managed containers: tolerated failure)
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -pm 1 >/dev/null 2>&1 || echo "NOTE: persistence mode not settable (no sudo?) -- boost-stable protocol"
  MAXCLK=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits -i 0 | head -1)
  nvidia-smi -lgc "$MAXCLK" >/dev/null 2>&1 && echo "clocks locked at ${MAXCLK}MHz" \
    || echo "NOTE: -lgc unavailable -- record DVFS caveat with results"
fi

run() { echo; echo "=== $1 ==="; shift; "$@"; }
gate() { [ "$ONLY" = all ] || [ "$ONLY" = "$1" ]; }

gate s1 && { run "s1 census"        $PY experiments/s1_governor_calib.py     || echo "WARN: s1 failed rc=$?"; }
gate s2 && { run "s2 receiver ingress" $PY experiments/s2_receiver_ingress.py || echo "WARN: s2 failed rc=$?"; }
gate s3 && { run "s3 quote accuracy" $PY experiments/s3_quote_accuracy.py    || echo "WARN: s3 failed rc=$?"; }
gate s5 && { run "s5 fan-in"        $PY experiments/s5_fanin_admission.py    || echo "WARN: s5 failed rc=$?"; }
gate s6 && { run "s6 mixed-route"   $PY experiments/s6_mixed_route_ledger.py || echo "WARN: s6 failed rc=$?"; }
gate s8 && { run "s8 stale-calib"   $PY experiments/s8_stale_calib_recovery.py || echo "WARN: s8 failed rc=$?"; }
gate s9 && { run "s9 wrong-hint"    $PY experiments/s9_wrong_hint.py         || echo "WARN: s9 failed rc=$?"; }

echo
echo "results in experiments/results/"
exit 0
