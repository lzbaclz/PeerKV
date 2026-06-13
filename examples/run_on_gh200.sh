#!/usr/bin/env bash
# Run UMA-LLM's KV residency tiering inside vLLM on a Grace-Hopper (GH200) box.
# RUN THIS ON A CUDA + GH200 MACHINE (my Linux sandbox has no GPU/vLLM).
#
#   cd ~/codes/papers/next3 && bash examples/run_on_gh200.sh
#
# What it does:
#   1) CPU-side sanity: the placement brain is unit-tested (no GPU needed).
#   2) Launches vLLM with the UMAGraceHopperConnector registered, which
#      demotes cold KV blocks to the coherent Grace LPDDR tier and/or
#      KIVI-compresses them, sized from an operator (deadline, miss) SLO.
set -e
cd "$(dirname "$0")/.."

echo "== CPU sanity: placement logic (works anywhere) =="
python3 -m pytest tests/test_vllm_placement.py -q || true

command -v nvidia-smi >/dev/null 2>&1 || { echo "No CUDA GPU -> stop. Run on GH200."; exit 1; }
python3 -c "import vllm" 2>/dev/null || { echo "Installing vLLM..."; pip install vllm; }

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"   # 70B for the headline on a 96GB+ box
echo "== launch vLLM + UMAGraceHopperConnector ($MODEL) =="
vllm serve "$MODEL" \
  --kv-transfer-config '{
    "kv_connector": "UMAGraceHopperConnector",
    "kv_connector_module_path": "umallm.vllm_integration.gh200_connector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "deadline_ms": 50,
      "miss_target": 0.01,
      "cold_bits": 4,
      "coherent_read": true,
      "tokens_per_block": 16
    }
  }'
# Then benchmark, e.g.:
#   vllm bench serve --model "$MODEL" --dataset-name sharegpt --num-prompts 200
# Compare P99 TPOT + peak HBM vs. a run without --kv-transfer-config.
