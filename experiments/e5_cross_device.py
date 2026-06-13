"""e5 -- Cross-device cost-model predictions (Claim C1: UMA generality).

Loops over every entry in :data:`umallm.uma_model.KNOWN_DEVICES`
(M2_Max, M2_Ultra, M3_Max, M3_Ultra, M4_Max, M4_Pro, GH200) and reports the
cost-model predictions that distinguish coherent from non-coherent UMA:

* ``access_cost`` for each residency tier (T0/T1/T2/T3),
* ``slow_tier_penalty`` (ell_bar = T2 read - T0 read), the quantity the
  schedulability inversion substitutes for SEER's PCIe latency, and
* a sample sizing result at a fixed (deadline, miss-target).

GH200 uses :class:`umallm.grace_hopper.GraceHopperCostModel`, where T1
(CPU-active) is as fast as T0 thanks to NVLink-C2C coherence -- contrast
that with the M-series, where T0 != T1 because the CPU/GPU caches are not
coherent and a cache-warm cost applies. That contrast is itself the
computer-design insight the ICCD framing leans on.

Every row is **model-predicted** (``_is_measured=False``): real per-SoC
probe numbers (and a real GH200 run) overwrite these later. Off-Mac the
M-series rows are seeded from the device class' SoC bandwidth via
``UMACostModel.from_device``.
"""
from __future__ import annotations

if __name__ == "__main__" and __package__ is None:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "experiments"

from . import add_repo_to_path, save_result  # noqa: E402

add_repo_to_path()

from umallm.grace_hopper import GraceHopperCostModel  # noqa: E402
from umallm.uma_model import KNOWN_DEVICES, ResidencyTier, UMACostModel  # noqa: E402

TIERS = [
    ResidencyTier.T0_GPU_ACTIVE,
    ResidencyTier.T1_CPU_ACTIVE,
    ResidencyTier.T2_COMPRESSED,
    ResidencyTier.T3_SWAPPED,
]


def _model_for(name: str) -> UMACostModel:
    """GH200 gets the coherent-UMA override; others seed from device class."""
    if name == "GH200":
        return GraceHopperCostModel()
    return UMACostModel.from_device(name)


def run(
    deadline_us: float = 50_000.0,
    ema_attention_lat_us: float = 10_000.0,
    ema_compute_lat_us: float = 15_000.0,
    n_blocks: int = 512,
    miss_target: float = 1e-2,
) -> dict:
    devices = {}
    for name, dc in KNOWN_DEVICES.items():
        model = _model_for(name)
        access = {f"T{int(t)}": model.access_cost(t) for t in TIERS}
        sizing = model.min_active_blocks_for_slo(
            deadline_us=deadline_us,
            ema_attention_lat=ema_attention_lat_us,
            ema_compute_lat=ema_compute_lat_us,
            n_blocks=n_blocks,
            miss_target=miss_target,
        )
        devices[name] = {
            "soc_bw_gbps": dc.soc_bw_gbps,
            "has_swap": dc.has_swap,
            "coherent_cpu_gpu": name == "GH200",
            "access_cost_us": access,
            "slow_tier_penalty_us": model.slow_tier_penalty(),
            "t0_eq_t1": abs(access["T0"] - access["T1"]) < 1e-9,
            "sample_sizing": {
                "deadline_ms": deadline_us / 1000.0,
                "miss_target": miss_target,
                "feasible": bool(sizing.feasible),
                "min_active_blocks": int(sizing.min_active_blocks),
                "max_slow_fraction": float(sizing.max_slow_fraction),
                "bound_at_solution": float(sizing.bound_at_solution),
            },
        }

    payload = {
        "_is_measured": False,  # model-predicted across SoCs; no real probes
        "deadline_us": deadline_us,
        "miss_target": miss_target,
        "n_blocks": n_blocks,
        "devices": devices,
        "n_devices": len(devices),
    }
    return payload


def main() -> dict:
    payload = run()
    print("=== e5 cross-device (model-predicted) ===")
    print("  device     BW(GB/s)  T0(us)  T1(us)  T2(us)  ell_bar(us)  T0=T1  min_active")
    for name, d in payload["devices"].items():
        a = d["access_cost_us"]
        print(
            f"  {name:9s} {d['soc_bw_gbps']:8.0f}  {a['T0']:6.3f}  {a['T1']:6.3f}  "
            f"{a['T2']:6.2f}  {d['slow_tier_penalty_us']:11.2f}  "
            f"{str(d['t0_eq_t1']):5s}  {d['sample_sizing']['min_active_blocks']:6d}"
        )
    print("  (GH200: T0==T1 via NVLink-C2C coherence; M-series: T0<T1 cache-warm)")
    path = save_result("e5_cross_device", payload)
    print(f"  -> {path}")
    return payload


if __name__ == "__main__":
    main()
