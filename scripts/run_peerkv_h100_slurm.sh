#!/usr/bin/env bash
# Slurm job script for PeerKV g1-g4 cross-hardware replication on 并行科技 (or any
# Slurm cluster). Requests 2 GPUs on one node, runs scripts/run_h100_suite.sh, and
# packs results for download on the login node.
#
# Prerequisites (login node only -- compute nodes have no internet):
#   Option A -- conda (if available):
#     conda create -n peerkv-h100 python=3.11 -y && conda activate peerkv-h100
#   Option B -- venv (no conda on 并行科技 login nodes):
#     python3 -m venv ~/workspace/.venv_peerkv
#     source ~/workspace/.venv_peerkv/bin/activate
#   Then (either option):
#     pip install --index-url https://download.pytorch.org/whl/cu124 torch
#     pip install nvidia-ml-py matplotlib
#   Submit with venv:
#     REPO_DIR=~/workspace/PeerKV VENV_DIR=~/workspace/.venv_peerkv sbatch ...
#
# Submit from repo root:
#   mkdir -p experiments/results_h100
#   sbatch scripts/run_peerkv_h100_slurm.sh
#   # or override partition: sbatch -p gpu_h200 scripts/run_peerkv_h100_slurm.sh
#
# After the job finishes, fetch:
#   experiments/results_h100/h100_console_<JOBID>.log   (primary log; see note below)
#   h100_results_<JOBID>.tgz
#
# NOTE: stdout/stderr are tee'd to h100_console_*.log, so slurm-*.out/.err may be
# nearly empty. Always read h100_console_<JOBID>.log for the full transcript.
#
#SBATCH -J peerkv-h100
#SBATCH -p gpu_h200
#SBATCH --gpus=2
#SBATCH --cpus-per-task=24
#SBATCH -o experiments/results_h100/slurm-%x-%j.out
#SBATCH -e experiments/results_h100/slurm-%x-%j.err

set -Eeuo pipefail

############################################
# User config (override via env if needed)
############################################

CONDA_ENV_NAME="${CONDA_ENV_NAME:-peerkv-h100}"
VENV_DIR="${VENV_DIR:-}"          # e.g. ~/workspace/.venv_peerkv when conda is absent
PY="${PY:-}"                      # optional absolute python path override
REPO_DIR="${REPO_DIR:-${HOME}/PeerKV}"
BRANCH_NAME="${BRANCH_NAME:-peerkv-parallel}"

RESULT_DIR="${REPO_DIR}/experiments/results_h100"
JOB_TAG="${SLURM_JOB_ID:-manual}"
CONSOLE_LOG="${RESULT_DIR}/h100_console_${JOB_TAG}.log"
TOPO_FILE="${RESULT_DIR}/nvidia_topo_${JOB_TAG}.txt"
TARBALL="${REPO_DIR}/h100_results_${JOB_TAG}.tgz"

############################################
# Helpers
############################################

section() {
    echo
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

warn() {
    echo "[WARN] $*" >&2
}

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

############################################
# Main (tee everything to CONSOLE_LOG)
############################################

mkdir -p "${RESULT_DIR}"

{
section "Job information"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Job name: ${SLURM_JOB_NAME:-N/A}"
echo "Partition: ${SLURM_JOB_PARTITION:-N/A}"
echo "Node: $(hostname)"
echo "User: ${USER}"
echo "Date: $(date)"
echo "Submission PWD: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

section "Activate Python environment"
if [ -n "${PY}" ]; then
    echo "Using PY override: ${PY}"
    command -v "${PY}" >/dev/null 2>&1 || die "PY not found: ${PY}"
elif [ -n "${VENV_DIR}" ]; then
    echo "Activating venv: ${VENV_DIR}"
    # shellcheck source=/dev/null
    source "${VENV_DIR}/bin/activate" || die "Failed to activate venv: ${VENV_DIR}"
    export PY="$(which python)"
elif CONDA_BASE="$(conda info --base 2>/dev/null || true)" && \
     [ -n "${CONDA_BASE}" ] && [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    echo "Activating conda env: ${CONDA_ENV_NAME}"
    # shellcheck source=/dev/null
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}" || die "Failed to activate conda env: ${CONDA_ENV_NAME}"
    export PY="$(which python)"
else
    die "No Python env found. Set VENV_DIR=~/workspace/.venv_peerkv (or install conda)."
fi

echo "Python: ${PY}"
"${PY}" --version

section "Enter repository"
cd "${REPO_DIR}" || die "Repo not found: ${REPO_DIR}"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Git branch: $(git branch --show-current 2>/dev/null || echo unknown)"
    git checkout "${BRANCH_NAME}" 2>/dev/null || warn "git checkout ${BRANCH_NAME} skipped"
    echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
else
    warn "Not a git repo (zip upload is fine). Skipping git checkout."
fi

section "System information"
uname -a
lscpu | head -n 40 || true
free -h || true
df -h . || true

section "NVIDIA information"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found"
nvidia-smi
echo
nvidia-smi -L
echo
nvidia-smi topo -m | tee "${TOPO_FILE}"
echo
nvidia-smi --query-gpu=index,name,uuid,mig.mode.current,driver_version,memory.total,memory.free,utilization.gpu,temperature.gpu,power.draw,power.limit,compute_cap --format=csv || true

section "NVLink topology check"
TOPO_CELL="$(awk '/^GPU0/ {print $2; exit}' "${TOPO_FILE}")"
echo "GPU0-GPU1 topo cell: ${TOPO_CELL}"
if [[ "${TOPO_CELL}" =~ ^NV ]]; then
    echo "NVLink between GPU0 and GPU1: OK (${TOPO_CELL})"
else
    warn "GPU0-GPU1 = '${TOPO_CELL}' (expected NV#). SYS/PHB/PIX = PCIe only."
    warn "This node may not be suitable for the NVLink cross-GPU experiment."
fi

section "MIG status"
MIG_STATUS="$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader 2>/dev/null || true)"
echo "${MIG_STATUS}"
if echo "${MIG_STATUS}" | grep -qi "Enabled"; then
    die "MIG is Enabled on at least one GPU. PeerKV suite requires MIG Disabled."
fi

section "PyTorch / P2P sanity check"
GPU_COUNT="$("${PY}" - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
echo "Visible GPU count: ${GPU_COUNT}"
if [ "${GPU_COUNT}" -lt 2 ]; then
    die "Need >=2 visible GPUs (got ${GPU_COUNT}). Check #SBATCH --gpus=2."
fi

"${PY}" - <<'PY'
import sys
import torch

print("Python:", sys.version)
print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available in PyTorch (try module load cuda on login node?)")

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    props = torch.cuda.get_device_properties(i)
    print("  memory GB:", round(props.total_memory / 1024**3, 2))
    print("  capability:", torch.cuda.get_device_capability(i))

if torch.cuda.device_count() < 2:
    raise RuntimeError("Need >=2 GPUs")

x = torch.randn(1024, device="cuda:1")
y = x.to("cuda:0")
torch.cuda.synchronize()
print("P2P copy sanity check: OK")
PY

section "Optional tools"
if command -v nsys >/dev/null 2>&1; then
    echo "nsys: $(command -v nsys)"
    nsys --version 2>/dev/null || true
else
    warn "nsys not found; timeline profiling will be skipped."
fi

if command -v dcgmi >/dev/null 2>&1; then
    echo "dcgmi: $(command -v dcgmi)"
    dcgmi --version 2>/dev/null || true
else
    warn "dcgmi not found; G1 will fall back to g1_timing.json only."
fi

if sudo -n true >/dev/null 2>&1; then
    echo "sudo (non-interactive): available"
else
    warn "sudo unavailable; clock lock / g4_trap_unlocked / DCGM hostengine likely skipped."
fi

section "Run H100/H200 suite"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
mkdir -p experiments/results experiments/results_h100
bash scripts/run_h100_suite.sh

section "Result files"
ls -lah experiments/results_h100/ || true

section "Create tarball"
cd "${REPO_DIR}"
tar czvf "${TARBALL}" \
    experiments/results_h100/ \
    "${CONSOLE_LOG}" \
    "${TOPO_FILE}"
ls -lh "${TARBALL}"

section "Done"
echo "Finished at: $(date)"
echo
echo "Send back:"
echo "  1. ${TARBALL}"
echo "  2. ${CONSOLE_LOG}"
echo "  3. ${TOPO_FILE}"
echo
echo "Also answer: GPU model (H100/H200?), NVSwitch?, lock freq OK? (see SUMMARY in log)"

} 2>&1 | tee "${CONSOLE_LOG}"
