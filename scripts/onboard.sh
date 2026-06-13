#!/usr/bin/env bash
# PeerKV onboarding self-check (risk 9.C.3). Run once after cloning:
#   bash scripts/onboard.sh
# Verifies env + CPU tests + core imports + (optional) GPU topology. Read-only
# except for `pip install -e .`. Pairs with AGENTS.md and collaboration_plan/README.md SS6.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
echo "=== PeerKV onboard @ $ROOT ==="

echo "--- [1/5] editable install (numpy+pyyaml; torch/vllm are extras) ---"
pip install -e . || echo "  (pip install failed; fix before continuing)"

echo "--- [2/5] pytest collect (expect 0 import errors) ---"
python -m pytest --collect-only -q 2>&1 | tail -2

echo "--- [3/5] CPU unit tests (torch/gpu tests skip if absent) ---"
python -m pytest tests/ -m "not gpu" -q 2>&1 | tail -8 || true

echo "--- [4/5] core mainline imports ---"
python -c "from umallm.elastic_policy import select_point, admissible_points; print('  elastic_policy OK (offline selector)')" || true
python -c "from umallm.observability.box_probe import box_idle; print('  box_probe OK (co-tenant gate)')" || true
python -c "from umallm.peer_parallel_attn import merge_partial; print('  peer_parallel_attn OK (CFK math)')" || true

echo "--- [5/5] GPU box check (optional; needs nvidia-smi + 2 GPUs) ---"
python -m umallm.observability.topology_probe 2>/dev/null || echo "  (no nvidia-smi / not a GPU box -- fine for CPU dev)"

echo
echo "=== next steps ==="
echo "  - read AGENTS.md (D1/D2/D3 + three do-no-harm red lines)"
echo "  - read collaboration_plan/README.md (single entry point) then SS6 first-week plan"
echo "  - on the dual-A100 box: sudo bash scripts/activate_nvlink.sh  (MIG off -> NV12)"
