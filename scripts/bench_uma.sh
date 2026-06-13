#!/usr/bin/env bash
# Run UMA-LLM benchmarks on a local M-series Mac.
# No GPU required (the GPU is the Mac itself!).
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$ROOT/experiments/results"

PROFILE=${1:-M2_Max}  # M2_Max | M3_Ultra | M4_Pro

# Microbenchmarks (no model)
python -m umallm.benchmark > "$ROOT/experiments/results/${PROFILE}_micro.json"

# Per-model
for MODEL in \
  "mlx-community/Meta-Llama-3-8B-Instruct-4bit" \
  "mlx-community/Mistral-7B-Instruct-v0.3-4bit" \
  "mlx-community/Meta-Llama-3-70B-Instruct-4bit"; do
  for CTX in 4096 16384 32768; do
    SHORT=$(basename "$MODEL")
    OUT="$ROOT/experiments/results/${PROFILE}_${SHORT}_ctx${CTX}.json"
    python -m umallm.run_mlx \
      --model "$MODEL" \
      --ctx-len "$CTX" \
      --out "$OUT" \
      --policy uma --enable-pressure-listener \
      || echo "[skip] $MODEL ctx=$CTX failed (OOM expected for some)"
  done
done

echo "Done. Results in $ROOT/experiments/results/"
