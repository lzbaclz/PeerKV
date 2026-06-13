"""e15 -- multi-GPU KV-tier bandwidth microbenchmark (RUN ON THE DUAL-A100 BOX).

Core evidence for the NVLink-tiered KV idea: measure the effective bandwidth of
*fetching a KV-sized block to the compute GPU (cuda:0)* from each tier --
  T0  local HBM            (baseline)
  T1  peer-GPU HBM         over NVLink   (the proposed near tier)
  T2  host DRAM (pinned)   over PCIe     (what FlexGen / vLLM-CPU offload use)
and report the NVLink/PCIe ratio. The whole idea rests on T1 being ~10x T2; this
script confirms it on real hardware in seconds.

Authored on a Mac with no CUDA -- could not be run here; the no-GPU path writes a
placeholder. On the dual-A100 box:

    pip install "torch>=2.4"
    python experiments/e15_multigpu_kv_tiers.py --mb 512 --trials 30

Needs >=2 CUDA GPUs with peer access (NVLink). Writes
experiments/results/multigpu_kv_tiers.json. Send it back.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent / "results" / "multigpu_kv_tiers.json"


def _write(res: dict) -> None:
    res["_experiment"] = "e15_multigpu_kv_tiers"
    res["_generated_at"] = datetime.now(timezone.utc).isoformat()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"  -> wrote {OUT}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=int, default=512, help="KV block size, MiB")
    ap.add_argument("--trials", type=int, default=30)
    args = ap.parse_args()

    res = {"mb": args.mb, "trials": args.trials}
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        res.update({"_is_measured": False, "note": f"torch not importable: {e}"})
        _write(res)
        return

    ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if ngpu < 2:
        res.update({"_is_measured": False,
                    "note": f"need >=2 CUDA GPUs (have {ngpu}); run on the dual-A100 box"})
        _write(res)
        print(f"  only {ngpu} GPU(s) -- wrote placeholder.")
        return

    n = args.mb * 1024 * 1024 // 2          # fp16 element count
    nbytes = n * 2
    torch.cuda.set_device(0)
    p2p = bool(torch.cuda.can_device_access_peer(0, 1))

    src_local = torch.ones(n, dtype=torch.float16, device="cuda:0")
    src_peer = torch.ones(n, dtype=torch.float16, device="cuda:1")   # over NVLink
    src_host = torch.ones(n, dtype=torch.float16, device="cpu").pin_memory()
    dst = torch.empty(n, dtype=torch.float16, device="cuda:0")

    def bw_gbps(copy_fn) -> float:
        for _ in range(3):                  # warmup
            copy_fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(args.trials):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            copy_fn()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        return nbytes / statistics.median(ts) / 1e9

    t0_bw = bw_gbps(lambda: dst.copy_(src_local))           # local HBM
    t1_bw = bw_gbps(lambda: dst.copy_(src_peer))            # peer GPU over NVLink
    t2_bw = bw_gbps(lambda: dst.copy_(src_host, non_blocking=True))  # host over PCIe

    res.update({
        "_is_measured": True,
        "device": torch.cuda.get_device_name(0),
        "n_gpus": ngpu,
        "peer_access_enabled": p2p,
        "T0_local_hbm_gbps": t0_bw,
        "T1_peer_nvlink_gbps": t1_bw,
        "T2_host_pcie_gbps": t2_bw,
        "nvlink_over_pcie": (t1_bw / t2_bw) if t2_bw else None,
        "note": ("If peer_access_enabled is False, T1 fell back to a host bounce "
                 "(not NVLink) -- enable P2P / check the NVLink topology."),
    })
    _write(res)
    print(json.dumps({k: res[k] for k in
                      ("device", "peer_access_enabled", "T0_local_hbm_gbps",
                       "T1_peer_nvlink_gbps", "T2_host_pcie_gbps",
                       "nvlink_over_pcie")}, indent=2))


if __name__ == "__main__":
    main()
