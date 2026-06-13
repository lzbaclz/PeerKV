"""e1 -- Cost-model validation (Claim C1).

Run the three calibration probes (SoC bandwidth, L2-miss latency, KIVI
4-bit kernel) and compare the *measured* per-transition latency against the
:class:`~umallm.uma_model.UMACostModel` *prediction* for the three cost
terms that the UMA model is built on:

* **T0 <-> T1 cache-warm**: predicted from ``l2_miss_ns``; the probe
  measures the per-line miss latency directly.
* **T2 dequant**: predicted from ``decompress_us_per_kb``; the probe
  measures the KIVI quantize/dequant ``us_per_kb``.
* **T3 swap-in**: predicted from ``swap_in_us_per_kb``; there is no
  in-sandbox swap probe (NVMe page-in needs a real macOS host), so this
  term is reported as model-only and flagged for a real run.

Target (ICCD plan, C1): predicted vs measured relative error < 10% on a
real M-series Mac. In the sandbox the probes are NumPy stand-ins, so the
absolute numbers are *not* hardware-truthful (``_is_measured=False``) --
the value here is exercising the probe->model comparison pipeline so the
Mac run drops straight in.
"""
from __future__ import annotations

if __name__ == "__main__" and __package__ is None:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "experiments"

from . import add_repo_to_path, save_result

add_repo_to_path()

from umallm.calibration import (  # noqa: E402
    probe_kivi_kernel,
    probe_l2_miss_latency,
    probe_soc_bandwidth,
)
from umallm.uma_model import ResidencyTier, UMACostModel  # noqa: E402


def _rel_err(measured: float, predicted: float) -> float:
    denom = abs(measured) if abs(measured) > 1e-12 else 1e-12
    return abs(measured - predicted) / denom


def run(buf_mb: int = 32, n_lines: int = 1024, n_blocks: int = 32) -> dict:
    """Compare probe measurements to model predictions per transition.

    Args mirror the calibration probe knobs and are shrunk by the smoke
    test; defaults are still cheap enough for a normal sandbox run.
    """
    # 1. Run the probes (NumPy stand-ins off-hardware).
    bw = probe_soc_bandwidth(buf_mb=buf_mb, n_iters=8)
    l2 = probe_l2_miss_latency(stride_bytes=128, n_lines=n_lines)
    kivi = probe_kivi_kernel(n_blocks=n_blocks)

    # 2. Build a model seeded from the *measured* probe values, then ask the
    #    model for the same per-transition costs. On a real Mac the seeded
    #    model and an independent default model would differ only by the
    #    calibration delta; here we report both predictions and the implied
    #    relative error of the *default* model vs the measured probe.
    measured_model = UMACostModel(
        soc_bw_gbps=bw["bandwidth_gbps"],
        l2_miss_ns=l2["per_line_ns"],
        compress_us_per_kb=kivi["us_per_kb"],
        decompress_us_per_kb=kivi["us_per_kb"],
    )
    default_model = UMACostModel()
    block_bytes = default_model.block_bytes
    kb = block_bytes / 1024.0
    n_cache_lines = block_bytes // default_model.l2_line_bytes

    transitions = {}

    # --- T0 <-> T1 cache-warm (from l2_miss_ns) --------------------------- #
    measured_t0t1 = n_cache_lines * l2["per_line_ns"] / 1e3  # us
    predicted_t0t1 = default_model.cost(
        ResidencyTier.T0_GPU_ACTIVE, ResidencyTier.T1_CPU_ACTIVE
    )
    transitions["T0_T1_cache_warm"] = {
        "measured_us": measured_t0t1,
        "predicted_us": predicted_t0t1,
        "rel_error": _rel_err(measured_t0t1, predicted_t0t1),
        "source_probe": "l2_miss.per_line_ns",
    }

    # --- T2 dequant (from KIVI us_per_kb) --------------------------------- #
    measured_t2 = kb * kivi["us_per_kb"]
    predicted_t2 = default_model.cost(
        ResidencyTier.T2_COMPRESSED, ResidencyTier.T0_GPU_ACTIVE
    )
    transitions["T2_dequant"] = {
        "measured_us": measured_t2,
        "predicted_us": predicted_t2,
        "rel_error": _rel_err(measured_t2, predicted_t2),
        "source_probe": "kivi.us_per_kb",
    }

    # --- T3 swap-in (no in-sandbox probe) --------------------------------- #
    predicted_t3 = default_model.cost(
        ResidencyTier.T3_SWAPPED, ResidencyTier.T0_GPU_ACTIVE
    )
    transitions["T3_swap_in"] = {
        "measured_us": None,  # needs a real macOS NVMe page-in probe
        "predicted_us": predicted_t3,
        "rel_error": None,
        "source_probe": "swap_in_us_per_kb (model-only; no sandbox probe)",
        "note": "fill with macos_swap_probe on real hardware",
    }

    rel_errors = [
        t["rel_error"] for t in transitions.values() if t["rel_error"] is not None
    ]
    payload = {
        "_is_measured": False,  # NumPy stand-in probes, not real hardware
        "target_rel_error": 0.10,
        "probes": {"bandwidth": bw, "l2_miss": l2, "kivi": kivi},
        "measured_model_params": {
            "soc_bw_gbps": measured_model.soc_bw_gbps,
            "l2_miss_ns": measured_model.l2_miss_ns,
            "decompress_us_per_kb": measured_model.decompress_us_per_kb,
        },
        "block_bytes": block_bytes,
        "transitions": transitions,
        "max_rel_error": max(rel_errors) if rel_errors else None,
        "all_within_target": (
            all(e <= 0.10 for e in rel_errors) if rel_errors else None
        ),
    }
    return payload


def main() -> dict:
    payload = run()
    path = save_result("e1_cost_model_validation", payload)
    print("=== e1 cost-model validation ===")
    print(f"  _is_measured = {payload['_is_measured']} (sandbox NumPy probes)")
    for name, t in payload["transitions"].items():
        if t["rel_error"] is None:
            print(f"  {name:18s} predicted={t['predicted_us']:.3f}us (no sandbox probe)")
        else:
            print(
                f"  {name:18s} measured={t['measured_us']:.3f}us "
                f"predicted={t['predicted_us']:.3f}us rel_err={t['rel_error']*100:.1f}%"
            )
    mx = payload["max_rel_error"]
    if mx is not None:
        print(f"  max rel error = {mx*100:.1f}% (target < 10% on real Mac)")
    print(f"  -> {path}")
    return payload


if __name__ == "__main__":
    main()
