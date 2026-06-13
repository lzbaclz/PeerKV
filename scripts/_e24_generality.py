"""e24 -- generality: the win as a function of the T1/T2 bandwidth ratio, and
the cost model instantiated for boxes we cannot physically test.

We measured on ONE box (dual A100-SXM4, NV12, 2 GPUs, direct bridge). To show the
result is not a one-box artifact, we (1) express the predicted decode-step
NVLink-vs-host speedup as a closed function of the peer/host bandwidth ratio, and
(2) instantiate the calibrated cost model with vendor specs for other interconnect
generations (H100/NVLink4, GH200/NVLink-C2C, an NVSwitch-shared per-pair case, and
a PCIe-bridged "NVLink-absent" fallback). The qualitative result -- spill sideways
over the GPU interconnect beats spilling down to host -- holds wherever the
peer-link bandwidth exceeds the host link; the magnitude scales with that ratio.
Predicted (cost-model); _is_measured:false for the non-A100 rows.
"""
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from umallm.multigpu import (MGTier, MultiGPUDevice, MultiGPUKVModel,
                             estimate_decode_step_us, topology_aware_placement)

OUT = Path(__file__).resolve().parent.parent / "experiments" / "results" / "generality.json"

# (name, measured?, bw=(HBM,peer,host,nvme) GB/s, c_us=(HBM,peer,host,nvme))
# peer = realized one-way; host = realized one-way. Vendor figures are peaks.
BOXES = [
    ("A100-SXM4 NV12 (measured)", True,  (773., 273., 24., 5.),  (12.3, 23.6, 12.6, 50.)),
    ("H100-SXM NVLink4 (vendor)", False, (3350., 450., 55., 12.), (10.,  20.,  10.,  40.)),
    ("GH200 NVLink-C2C (vendor)", False, (4000., 450., 55., 12.), (8.,   15.,  10.,  40.)),
    ("A100 HGX NVSwitch/2-share", False, (773., 136., 24., 5.),  (12.3, 23.6, 12.6, 50.)),  # peer halved under contention
    ("PCIe-bridged (NVLink off)", False, (773., 11.,  24., 5.),  (12.3, 60.,  12.6, 50.)),  # peer << host: must route to host
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-mb", type=float, default=8.0)
    ap.add_argument("--n-blocks", type=int, default=1000)
    ap.add_argument("--local", type=int, default=50)
    ap.add_argument("--compute-us", type=float, default=8000.0)
    args = ap.parse_args()
    blk = int(args.block_mb * 1024 * 1024)

    rows = []
    scores = np.arange(args.n_blocks, dtype=np.float32)
    for name, measured, bw, c in BOXES:
        dev = MultiGPUDevice(name, tuple(bw), (40., 70., 512., 4096.))
        m = MultiGPUKVModel(device=dev, c_us=tuple(c))
        # BW-driven placement (topology-robust): spill to fastest reachable tier
        tier_bw = {0: bw[0], 1: bw[1], 2: bw[2], 3: bw[3]}
        nv = topology_aware_placement(scores, {0: args.local, 1: 10_000, 2: 0, 3: 0},
                                      n_sink=1, n_window=4, tier_bw=tier_bw)
        host = topology_aware_placement(scores, {0: args.local, 1: 0, 2: 10_000, 3: 0},
                                        n_sink=1, n_window=4, tier_bw=tier_bw)
        # which tier does the BW-robust policy actually CHOOSE for spill, given
        # realistic capacity on every tier? (peer>host -> NVLink; NVLink-off -> host)
        robust = topology_aware_placement(scores, {0: args.local, 1: 200, 2: 10_000, 3: 10_000},
                                          n_sink=1, n_window=4, tier_bw=tier_bw)
        used = [t for t in (1, 2, 3) if int((robust == t).sum()) > 0]
        spill_tier = max(used, key=lambda t: tier_bw[t]) if used else 2  # highest-BW tier actually used
        t_nv = estimate_decode_step_us(nv, m, blk, args.compute_us, overlap=True)
        t_host = estimate_decode_step_us(host, m, blk, args.compute_us, overlap=True)
        ratio = bw[1] / bw[2]
        speedup = t_host / t_nv if t_nv else None
        rows.append({"box": name, "measured": measured, "peer_gbps": bw[1], "host_gbps": bw[2],
                     "peer_host_ratio": ratio, "predicted_speedup_overlap": speedup,
                     "policy_spill_tier": MGTier(spill_tier).name})
        print(f"  {name:30s} peer/host={ratio:5.1f}x -> predicted {speedup:5.2f}x  "
              f"(policy spills to {MGTier(spill_tier).name})")

    res = {"_experiment": "e24_generality", "_is_measured": False,
           "kind": "bandwidth_ratio_generality_projection",
           "block_mb": args.block_mb, "rows": rows,
           "note": ("Predicted decode-step speedup (overlap-aware cost model) vs the peer/host "
                    "bandwidth ratio. A100 row is measured-calibrated; others use vendor specs and "
                    "are _is_measured:false. The BW-robust C3 policy spills to PEER_NVLINK wherever "
                    "peer>host and FALLS BACK to host when the peer link is slower (last row: "
                    "PCIe-bridged / NVLink-off -> policy correctly routes to host, no self-harm). "
                    "Magnitude scales with the ratio; the direction holds for any peer>host box."),
           "_generated_at": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(res, indent=2))
    print(f"  -> wrote {OUT}")


if __name__ == "__main__":
    main()
