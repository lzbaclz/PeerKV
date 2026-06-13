"""Serving-benchmark support for the GH200 Route B evaluation.

Pure-Python building blocks (no torch/vLLM/CUDA), so the workload generation
and metric math are unit-testable anywhere. The benchmark harness
(``experiments/routeb_benchmark.py``) consumes these and drives a backend
(MockBackend on CPU for orchestration tests; VLLMBackend on a GH200).
"""
from .workloads import (
    RequestSpec,
    make_long_context,
    make_multi_turn,
    synth_text,
)
from .serving_metrics import (
    RequestRecord,
    deadline_miss_ratio,
    goodput_tok_s,
    summarize_run,
    throughput_tok_s,
    tpot_percentiles,
)

__all__ = [
    "RequestSpec",
    "make_long_context",
    "make_multi_turn",
    "synth_text",
    "RequestRecord",
    "tpot_percentiles",
    "throughput_tok_s",
    "deadline_miss_ratio",
    "goodput_tok_s",
    "summarize_run",
]
