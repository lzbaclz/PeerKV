"""e16 -- cost-model PREDICTION of NVLink-tiered vs host-offload decode cost (RQ2).

CPU/offline. Uses umallm.multigpu with per-tier bandwidths (vendor-spec defaults,
or calibrated from an e15 JSON via --calib) to predict, as context (block count)
grows past local HBM, the per-decode-step KV-read time for:
  * all-local      (no overflow; reference)
  * NVLink-tiered  (spill -> peer GPU over NVLink)   [ours]
  * host-offload   (spill -> host DRAM over PCIe)    [FlexGen/OrchKvCache line]
and the SLO sizing (min fast-resident) for NVLink vs PCIe spill.

These are COST-MODEL PREDICTIONS (`_is_measured: false`). The measured
end-to-end TPOT comes from the dual-A100 run; calibrate bandwidths with e15
(`--calib experiments/results/multigpu_kv_tiers.json`) to sharpen the prediction.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from umallm.multigpu import (MGTier, MultiGPUKVModel, estimate_decode_step_us,
                             topology_aware_placement)

OUT = Path(__file__).resolve().parent / "results" / "placement_tpot_predicted.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default=None,
                    help="e15 multigpu_kv_tiers.json to calibrate per-tier BW")
    ap.add_argument("--block-kb", type=int, default=256)
    ap.add_argument("--contexts-blocks", default="256,512,1024,2048,4096")
    ap.add_argument("--local-blocks", type=int, default=256,
                    help="#KV blocks that fit local HBM (rest must spill)")
    ap.add_argument("--compute-us", type=float, default=1000.0)
    args = ap.parse_args()

    blk = args.block_kb * 1024
    model = (MultiGPUKVModel.from_calibration(args.calib) if args.calib
             else MultiGPUKVModel())

    rows = []
    for N in [int(x) for x in args.contexts_blocks.split(",") if x.strip()]:
        scores = np.arange(N, dtype=np.float32)            # recency hotness
        nv = topology_aware_placement(scores, {0: args.local_blocks, 1: 10_000,
                                               2: 0, 3: 0}, n_sink=1, n_window=4)
        host = topology_aware_placement(scores, {0: args.local_blocks, 1: 0,
                                                 2: 10_000, 3: 0}, n_sink=1, n_window=4)
        loc = topology_aware_placement(scores, {0: 10_000, 1: 0, 2: 0, 3: 0},
                                       n_sink=1, n_window=4)
        t_nv = estimate_decode_step_us(nv, model, blk, args.compute_us)
        t_host = estimate_decode_step_us(host, model, blk, args.compute_us)
        t_loc = estimate_decode_step_us(loc, model, blk, args.compute_us)
        rows.append({"ctx_blocks": N, "all_local_us": t_loc,
                     "nvlink_tiered_us": t_nv, "host_offload_us": t_host,
                     "nvlink_speedup_vs_host": (t_host / t_nv) if t_nv else None})
        print(f"N={N:>5}  local={t_loc:9.1f}us  nvlink={t_nv:9.1f}us  "
              f"host={t_host:9.1f}us  speedup={t_host/t_nv:.2f}x")

    common = dict(deadline_us=50_000.0, n_blocks=2048, block_bytes=blk,
                  compute_us=15_000.0, attn_us=2_000.0)
    res = {"_experiment": "e16_placement_tpot", "_is_measured": False,
           "kind": "cost_model_prediction",
           "note": ("PREDICTED per-step KV-read time from the cross-tier cost "
                    "model; measured end-to-end TPOT pending dual-A100. "
                    "Bandwidths: " + (f"calibrated from {args.calib}" if args.calib
                                      else "vendor-spec defaults (run e15 to calibrate).")),
           "bandwidths_gbps": {"T0": model.device.bw_gbps[0],
                               "T1": model.device.bw_gbps[1],
                               "T2": model.device.bw_gbps[2]},
           "block_kb": args.block_kb, "local_blocks": args.local_blocks,
           "compute_us": args.compute_us, "tpot_sweep": rows,
           "sizing_nvlink_spill": model.min_fast_resident_for_slo(
               spill_tier=MGTier.PEER_NVLINK, **common),
           "sizing_pcie_spill": model.min_fast_resident_for_slo(
               spill_tier=MGTier.HOST_PCIE, **common),
           "_generated_at": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"  -> wrote {OUT}  (cost-model prediction; measured pending dual-A100)")


if __name__ == "__main__":
    main()
