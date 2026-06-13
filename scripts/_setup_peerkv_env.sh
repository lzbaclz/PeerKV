#!/usr/bin/env bash
# Create a clean conda env for the PeerKV/NVTier A100 experiments.
set -euo pipefail
source /home/lzq/miniconda3/etc/profile.d/conda.sh

ENV=peerkv
echo "[setup] creating conda env $ENV (python 3.11)"
conda create -y -n "$ENV" python=3.11

PY=/home/lzq/miniconda3/envs/$ENV/bin/python
PIP=/home/lzq/miniconda3/envs/$ENV/bin/pip

echo "[setup] installing torch 2.6.0 (cu124) + deps"
$PIP install --no-input "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
$PIP install --no-input numpy pyyaml

echo "[setup] torch sanity"
$PY -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available(),'ndev',torch.cuda.device_count())"
echo "[setup] DONE"
