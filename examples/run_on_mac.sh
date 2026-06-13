#!/usr/bin/env bash
# Turnkey UMA-LLM run on Apple Silicon. RUN THIS ON YOUR MAC.
#   cd ~/codes/papers/next3 && bash examples/run_on_mac.sh
#
# It installs deps, then produces REAL Apple-Silicon numbers:
#   1) the CPU-verifiable test + experiment suite (matches off-Mac)
#   2) UMA-LLM's tiered KV cache inside mlx-lm: stock fp16 KVCache vs
#      UMALayerCache on the SAME model -- resident-KV compression, logit
#      fidelity, and (honest) decode TPOT/peak (examples/mlx_lm_uma_demo.py)
# Results land in experiments/results/mac_mlx_lm_demo.json.
set -e
cd "$(dirname "$0")/.."

echo "== sanity: this must be a Mac =="
uname -a
python3 -c "import platform,sys; sys.exit(0 if platform.system()=='Darwin' else 1)" \
  || { echo "Not macOS -> MLX won't run. Run this on your Apple Silicon Mac."; exit 1; }

echo "== install =="
pip install -e . >/dev/null
pip install "mlx-lm>=0.18" >/dev/null
# Optional, for the e8 llama.cpp baseline:  brew install llama.cpp

echo "== CPU-verifiable suite (should match my sandbox) =="
python3 -m pytest tests/ -q || true
python3 experiments/run_all.py || true

echo "== REAL Apple-Silicon run: UMA-LLM tiered KV cache inside mlx-lm =="
python3 examples/mlx_lm_uma_demo.py \
  --model mlx-community/Llama-3.2-1B-Instruct-4bit \
  --prompt-tokens 4096 --n-gen 8 --block-size 128 --n-active 4
# Bigger model = more RAM (8B needs ~6 GB; the 70B headline needs a 96 GB box):
# python3 examples/mlx_lm_uma_demo.py --model mlx-community/Meta-Llama-3-8B-Instruct-4bit --prompt-tokens 8192

echo "== done. results in experiments/results/ =="
ls -1 experiments/results/*.json
