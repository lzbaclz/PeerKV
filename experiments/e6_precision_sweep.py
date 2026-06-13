"""e6 -- Precision sweep (Claim C3: the T2 precision choice).

Fully synthetic (real numbers in-sandbox). Quantizes synthetic Gaussian KV
blocks with KIVI 4-bit and the 2-bit variant and reports, against the fp16
reference, the ``relative_rmse`` (quality proxy) and the compressed byte
count (size). This tabulates the size/quality trade-off that motivates
using 4-bit as the default T2 tier and 2-bit only for the coldest blocks.

The real LongBench/RULER quality numbers are a P2 item that needs the model
weights; here we report the *intrinsic* quantization error, which is the
sandbox-computable part of that story, and leave a clearly-marked slot for
the downstream-task accuracy to be filled on hardware.
"""
from __future__ import annotations

if __name__ == "__main__" and __package__ is None:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "experiments"

import numpy as np  # noqa: E402

from . import add_repo_to_path, save_result  # noqa: E402

add_repo_to_path()

from umallm.compression import (  # noqa: E402
    compression_error,
    quantize_block,
    quantize_block_2bit,
)


def _fp16_bytes(K: np.ndarray) -> int:
    return K.astype(np.float16).nbytes


def run(n_tokens: int = 512, d_head: int = 128, n_blocks: int = 8, seed: int = 0) -> dict:
    """Average quantization error/size over ``n_blocks`` random Gaussian KV."""
    rng = np.random.default_rng(seed)
    methods = {
        "kivi_4bit": quantize_block,
        "kivi_2bit": quantize_block_2bit,
    }

    agg = {m: {"rel_rmse": [], "comp_bytes": [], "fp16_bytes": []} for m in methods}
    for _ in range(n_blocks):
        K = rng.normal(size=(n_tokens, d_head)).astype(np.float32)
        fp16 = _fp16_bytes(K)
        for name, fn in methods.items():
            c = fn(K)
            err = compression_error(K, c)
            agg[name]["rel_rmse"].append(err["relative_rmse"])
            agg[name]["comp_bytes"].append(c.nbytes())
            agg[name]["fp16_bytes"].append(fp16)

    methods_out = {}
    for name, a in agg.items():
        comp = float(np.mean(a["comp_bytes"]))
        fp16 = float(np.mean(a["fp16_bytes"]))
        methods_out[name] = {
            "relative_rmse": float(np.mean(a["rel_rmse"])),
            "compressed_bytes": comp,
            "fp16_bytes": fp16,
            "compression_ratio_vs_fp16": fp16 / comp if comp else float("nan"),
            "bytes_per_value": comp / (n_tokens * d_head),
            "longbench_acc": None,  # P2: needs model weights on hardware
        }

    rmse_ratio = (
        methods_out["kivi_2bit"]["relative_rmse"]
        / max(methods_out["kivi_4bit"]["relative_rmse"], 1e-12)
    )
    # KVQuant/KIVI 2-bit should be roughly ~2x the 4-bit relative RMSE (see
    # compression.quantize_block_2bit docstring). A ratio that large means
    # the round trip is broken, not that 2-bit is that lossy -- flag it so the
    # table is not misread as a legitimate quality cliff. (Known cause: the
    # 2-bit pack layout is not matched by dequantize_block's 4-bit unpacker.)
    # Legitimate 2-bit (4 levels) is typically 2-8x the 4-bit (16 levels)
    # RMSE; the old pack/unpack layout mismatch produced ~88x. Flag only the
    # pathological regime so a correct 2-bit cliff is not misreported.
    expected_2bit_ratio_max = 15.0
    rmse_2bit_plausible = rmse_ratio <= expected_2bit_ratio_max

    payload = {
        "_is_measured": True,  # intrinsic quant error is exact and real now
        "n_tokens": n_tokens,
        "d_head": d_head,
        "n_blocks": n_blocks,
        "distribution": "gaussian",
        "methods": methods_out,
        "rmse_2bit_over_4bit_x": rmse_ratio,
        "rmse_2bit_plausible": rmse_2bit_plausible,
        "anomaly": (
            None if rmse_2bit_plausible else
            "2-bit relative_rmse implausibly high vs 4-bit (>15x): indicates a "
            "quantize/dequantize layout mismatch in compression.py. 4-bit "
            "numbers are unaffected; investigate the 2-bit pack/unpack path."
        ),
    }
    return payload


def main() -> dict:
    payload = run()
    print("=== e6 precision sweep (synthetic Gaussian, real numbers) ===")
    print("  method      rel_rmse   comp_bytes   fp16_bytes   ratio   bpv")
    for name, m in payload["methods"].items():
        print(
            f"  {name:10s} {m['relative_rmse']:.4f}   {m['compressed_bytes']:10.0f}   "
            f"{m['fp16_bytes']:10.0f}   {m['compression_ratio_vs_fp16']:4.1f}x  "
            f"{m['bytes_per_value']:.3f}"
        )
    print(f"  2-bit RMSE is {payload['rmse_2bit_over_4bit_x']:.2f}x the 4-bit RMSE "
          f"(quality/size trade-off)")
    if payload["anomaly"]:
        print(f"  [ANOMALY] {payload['anomaly']}")
    path = save_result("e6_precision_sweep", payload)
    print(f"  -> {path}")
    return payload


if __name__ == "__main__":
    main()
