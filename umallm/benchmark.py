"""Benchmark utilities for UMA-LLM."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from .compression import dequantize_block, quantize_block, compression_error


def benchmark_quantize(n_blocks: int = 100, block_shape: tuple = (32, 128)) -> dict:
    """Measure quantize/dequantize throughput."""
    rng = np.random.default_rng(0)
    Ks = [rng.normal(size=block_shape).astype(np.float32) for _ in range(n_blocks)]

    # warmup
    for K in Ks[:5]:
        c = quantize_block(K)
        _ = dequantize_block(c)

    t0 = time.perf_counter_ns()
    compressed = [quantize_block(K) for K in Ks]
    quant_us = (time.perf_counter_ns() - t0) / 1e3 / n_blocks

    t0 = time.perf_counter_ns()
    _ = [dequantize_block(c) for c in compressed]
    dequant_us = (time.perf_counter_ns() - t0) / 1e3 / n_blocks

    # error
    errors = [compression_error(Ks[i], compressed[i]) for i in range(min(10, n_blocks))]
    avg_rel_rmse = float(np.mean([e["relative_rmse"] for e in errors]))

    # compression ratio
    bytes_orig = sum(K.astype(np.float16).nbytes for K in Ks)
    bytes_compressed = sum(c.nbytes() for c in compressed)
    ratio = bytes_orig / bytes_compressed

    return dict(
        quant_us_per_block=quant_us,
        dequant_us_per_block=dequant_us,
        compression_ratio=ratio,
        relative_rmse=avg_rel_rmse,
        n_blocks=n_blocks,
        block_shape=list(block_shape),
    )
