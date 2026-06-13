#!/usr/bin/env bash
# Collect attention traces on Mac for predictor fitting.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$ROOT/experiments/traces"

python -m umallm.trace_mlx \
  --model "mlx-community/Meta-Llama-3-8B-Instruct-4bit" \
  --n-traces 100 \
  --ctx 4096 \
  --out "$ROOT/experiments/traces/llama3_8b_4bit.jsonl"
