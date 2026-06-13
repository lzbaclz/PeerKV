"""[PARKED — Track D: Apple-Silicon/MLX. NOT the CUDA mainline. The mainline
online-calibration stub is ``umallm.runtime.calibration``; the measured A100
constants live in ``umallm.elastic_policy`` (from experiments e15/e20). Do not
extend this file for the A100/H100 runtime — see AGENTS.md D2 / risk 9.A.5.]

Calibration probes — addresses round-1 M1.

These functions are designed to run on an M-series Mac and emit a
JSON calibration file that overrides UMACostModel defaults.

Probes:
  - probe_soc_bandwidth: streaming read across a large buffer
  - probe_l2_miss_latency: cold-line read on each cache line
  - probe_kivi_kernel: actual MLX 4-bit quantize/dequant timing

In the sandbox we have NumPy as a stand-in to verify the probe logic;
real measurement requires an M-series Mac.
"""
from __future__ import annotations

import time

import numpy as np


def probe_soc_bandwidth(buf_mb: int = 256, n_iters: int = 32) -> dict:
    """Streaming-read bandwidth probe.

    Reads a buf_mb buffer end-to-end. On a real Mac we'd use Metal/Accelerate;
    here NumPy serves as a sandbox stand-in.
    """
    buf = np.random.default_rng(0).bytes(buf_mb * 1024 * 1024)
    arr = np.frombuffer(buf, dtype=np.uint8)
    # warmup
    _ = arr.sum()
    timings_s = []
    for _ in range(n_iters):
        t0 = time.perf_counter_ns()
        s = arr.sum()
        timings_s.append((time.perf_counter_ns() - t0) / 1e9)
    arr_bytes = arr.nbytes
    bw_gbps = arr_bytes / (np.median(timings_s) * 1e9)
    return dict(buf_mb=buf_mb, median_s=float(np.median(timings_s)),
                bandwidth_gbps=float(bw_gbps))


def probe_l2_miss_latency(stride_bytes: int = 128,
                          n_lines: int = 4096) -> dict:
    """Pointer-chasing probe for L2/L3 miss latency.

    Real implementation: a Metal compute kernel that does N random reads
    in a buffer larger than L2; report per-read latency.

    NumPy stand-in below approximates the cost of cold-line strided
    reads.
    """
    buf = np.zeros(n_lines * stride_bytes, dtype=np.uint8)
    # shuffle access order
    rng = np.random.default_rng(0)
    indices = rng.permutation(n_lines)[: n_lines // 4] * stride_bytes
    # warmup
    _ = buf[indices].sum()
    t0 = time.perf_counter_ns()
    s = 0
    for i in indices:
        s += int(buf[i])
    dt = (time.perf_counter_ns() - t0) / 1e9
    per_line_ns = dt / len(indices) * 1e9
    return dict(n_reads=int(len(indices)), per_line_ns=float(per_line_ns))


def probe_kivi_kernel(n_blocks: int = 64) -> dict:
    """KIVI 4-bit quantize/dequant timing.

    Real implementation: MLX-backed kernel. Sandbox stand-in uses the
    NumPy implementation in `compression.py`.
    """
    from .compression import quantize_block, dequantize_block
    rng = np.random.default_rng(0)
    K = rng.normal(size=(32, 128)).astype(np.float32)
    # warmup
    for _ in range(8):
        _ = dequantize_block(quantize_block(K))
    timings_us = []
    for _ in range(n_blocks):
        t0 = time.perf_counter_ns()
        c = quantize_block(K)
        _ = dequantize_block(c)
        timings_us.append((time.perf_counter_ns() - t0) / 1e3)
    arr = np.asarray(timings_us)
    block_kb = K.astype(np.float16).nbytes / 1024
    return dict(
        median_us=float(np.median(arr)),
        p99_us=float(np.percentile(arr, 99)),
        block_kb=float(block_kb),
        us_per_kb=float(np.median(arr) / block_kb),
    )


def write_calibration(path: str) -> dict:
    """Run all probes and emit JSON the model can load."""
    import json
    out = dict(
        bandwidth=probe_soc_bandwidth(),
        l2_miss=probe_l2_miss_latency(),
        kivi=probe_kivi_kernel(),
    )
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    return out
