#!/usr/bin/env bash
#SBATCH -J g13_dvfs
#SBATCH -p gpu_a800
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH -o g13_a800_%j.log
#
# G13 DVFS forensics on the A800 partition, batch version.
# Submit from the directory where the tarball was extracted:
#
#   sbatch scripts/sbatch_g13_a800.slurm
#   squeue -u $USER          # watch the queue
#
# 4 GPUs requested to reproduce the original g10/g8 node environment
# (g13 runs on cuda:1, same position as the original run; idle neighbors
# matter for the HBM-thermal hypothesis).  If the 4-GPU queue is hopeless,
# edit --gres=gpu:1 (g13 then falls back to cuda:0) -- note it in the log.
#
# BRING BACK:  experiments/results/g13_dvfs_a800.json  +  g13_a800_<jobid>.log
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# ==== POINT PY AT THE g10/g8 PYTHON (torch 2.7.0+cu128) =====================
# RECOMMENDED (simplest, no activation needed): set the ABSOLUTE path of the
# python binary inside the env you used for the g10/g8 run, e.g.:
#
#   export PY=$HOME/miniconda3/envs/torch27/bin/python
#
# To find it on the login node:    conda env list
# then verify (must print 2.7.0+cu128 / 12.8):
#   <that_path>/bin/python -c "import torch; print(torch.__version__, torch.version.cuda)"
#
# Alternative (if you prefer activation):
#   source ~/miniconda3/etc/profile.d/conda.sh && conda activate <env名>
#   module load anaconda3 && source activate <env名>
# ============================================================================
# export PY=$HOME/miniconda3/envs/CHANGE_ME/bin/python   # <-- EDIT THIS LINE

PY="${PY:-python}"
export PY

echo "== job $SLURM_JOB_ID on $(hostname)  $(date) =="
echo "== GPUs allocated =="
nvidia-smi --query-gpu=index,name,driver_version,persistence_mode,clocks.sm,clocks.max.sm --format=csv
$PY -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| ngpu', torch.cuda.device_count())" \
  || { echo "ERROR: python env not activated correctly -- edit the env block in this script"; exit 1; }
echo

bash scripts/run_g13_a800.sh
RC=$?

echo
echo "== job done (rc=$RC)  $(date) =="
exit $RC
