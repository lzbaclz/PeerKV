"""e10 -- Route B residency microbenchmark (RUN ON THE CUDA BOX).

Measures, on real hardware, what a UMA-LLM "tier demotion" actually costs at
the allocator layer: we allocate a KV-sized buffer through the managed
``CUDAPluggableAllocator`` (so its pages can live on HBM or the host/Grace
node), then time a GPU read of that buffer when it is HBM-resident versus
right after demoting it with ``cudaMemAdvise(PreferredLocation)`` +
``cudaMemPrefetchAsync``.

The point is the **regime**, reported automatically:

* ``coherent_uma`` (GH200, NVLink-C2C): the demoted read is served coherently
  with no migration -- the headline "zero-copy tier" regime.
* ``discrete_migration`` (e.g. A100 over PCIe): the SAME residency hint forces
  a real page migration, so the demoted read pays a PCIe round-trip. This run
  is the paper's **discrete / non-coherent reference**, NOT coherent UMA.

This file was authored on a Mac with no CUDA; it could not be run there. On
the CUDA box:

    UMA_BUILD_CUDA=1 UMA_CUDA_ARCH=80 pip install -e .[vllm]   # A100 (sm_80)
    python experiments/e10_route_b_residency.py --mb 256 --trials 20

If the native extension is not built (or no managed-access GPU), it writes an
``_is_measured=False`` placeholder with the exact build command.
"""
from __future__ import annotations

import argparse
import json
import statistics as stats
from datetime import datetime, timezone
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "cuda_route_b.json"
BUILD_HINT = ("UMA_BUILD_CUDA=1 UMA_CUDA_ARCH=80 pip install -e .[vllm]  "
              "# A100; use 90 for GH200, 80;90 for both")


def _write(res: dict) -> None:
    res["_experiment"] = "e10_route_b_residency"
    res["_generated_at"] = datetime.now(timezone.utc).isoformat()
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"  -> wrote {OUT}")


def _placeholder(reason: str, **extra) -> dict:
    return {"_is_measured": False, "note": reason, "build_hint": BUILD_HINT,
            **extra}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=int, default=256,
                    help="size of the managed KV-like buffer in MiB")
    ap.add_argument("--trials", type=int, default=20)
    args = ap.parse_args()

    try:
        import torch
    except Exception as e:  # noqa: BLE001
        _write(_placeholder(f"torch not importable: {e}"))
        return

    from umallm.uma_alloc import (UMAResidencyController, device_name,
                                  native_available, regime)
    from umallm.vllm_integration.uma_backend import kv_alloc_scope, uma_mem_pool

    reg, dev = regime(), device_name()
    print(f"[e10] regime={reg} device={dev}")

    if not native_available():
        _write(_placeholder(
            "umallm._uma_native not built, or no GPU with concurrentManagedAccess. "
            "Build it on the CUDA host, then re-run.", regime=reg, device=dev))
        print("  native managed allocator unavailable -- wrote placeholder.")
        return
    if not torch.cuda.is_available():
        _write(_placeholder("torch.cuda not available on this host.",
                            regime=reg, device=dev))
        return

    n = args.mb * 1024 * 1024 // 2  # float16 element count
    if uma_mem_pool() is None:
        _write(_placeholder("managed MemPool unavailable.", regime=reg, device=dev))
        return

    # Allocate the KV-like buffer ON managed memory (Route B allocation site).
    with kv_alloc_scope():
        buf = torch.empty(n, dtype=torch.float16, device="cuda")
        buf.fill_(1.0)
    torch.cuda.synchronize()

    ctrl = UMAResidencyController(block_dim=0)
    ctrl.register_kv_caches({"kv": buf.view(1, n)})  # whole buffer = one block

    def timed_read_ms() -> float:
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        y = buf.sum()              # touches every page
        e.record()
        torch.cuda.synchronize()
        _ = y.item()
        return s.elapsed_time(e)   # milliseconds

    for _ in range(3):             # warmup
        timed_read_ms()

    hbm_ms, grace_ms = [], []
    for _ in range(args.trials):
        ctrl.to_hbm("kv", [0]); torch.cuda.synchronize()
        hbm_ms.append(timed_read_ms())            # resident baseline (T0)
        ctrl.to_grace("kv", [0]); torch.cuda.synchronize()
        grace_ms.append(timed_read_ms())          # first read after demotion

    hbm = stats.median(hbm_ms)
    grace = stats.median(grace_ms)
    res = {
        "_is_measured": True,
        "regime": reg,
        "device": dev,
        "buffer_mb": args.mb,
        "dtype": "float16",
        "trials": args.trials,
        "hbm_read_ms_median": hbm,
        "grace_demoted_read_ms_median": grace,
        "slow_tier_penalty_ms": grace - hbm,
        "actual_node_after_grace": ctrl.actual_node("kv", 0),  # -1 == host/Grace
        "footprint": ctrl.footprint(),
        "interpretation": (
            "coherent_uma -> penalty ~ coherent C2C access (no migration); "
            "discrete_migration (A100) -> penalty ~ a real PCIe page migration, "
            "i.e. exactly the cost the paper claims unified memory removes. "
            "This run is the discrete/non-coherent reference, not coherent UMA."),
    }
    _write(res)
    print(f"  HBM-resident read : {hbm:.3f} ms")
    print(f"  demoted read      : {grace:.3f} ms  (penalty {grace - hbm:.3f} ms)")
    print(f"  last node after demote: {res['actual_node_after_grace']} "
          f"(-1 == host/Grace)")


if __name__ == "__main__":
    main()
