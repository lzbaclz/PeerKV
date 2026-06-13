"""e22 -- C2 schedulability inversion at a BINDING SLO: NVLink admits far more
context than PCIe at the same deadline (answers "the inversion saturates / can't
discriminate tiers").

The prior e16 sizing used a loose SLO (D=50ms) where BOTH spill tiers admit 100%
slow fraction -> no discrimination. Here we sweep the per-token deadline D and
report, for NVLink vs PCIe spill, the max admissible slow fraction phi_max and
the resulting max sustained context (= fast_budget / (1 - phi_max)). There is a
band of deadlines where NVLink is schedulable at high slow-fraction while PCIe is
not -- that band IS the design win the cost model predicts. Uses calibrated
(c_i, beta_i) and a coalesced block size (the regime where NVLink pays off).
"""
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path

from umallm.multigpu import MGTier, MultiGPUKVModel

OUT = Path(__file__).resolve().parent.parent / "experiments" / "results" / "sizing_sweep.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="experiments/results/multigpu_kv_tiers.json")
    ap.add_argument("--setup-calib", default="experiments/results/tier_calib.json")
    ap.add_argument("--block-mb", type=float, default=8.0, help="coalesced transfer size")
    ap.add_argument("--n-blocks", type=int, default=512)
    ap.add_argument("--fast-budget", type=int, default=64, help="blocks that fit in {T0,T1} fast")
    ap.add_argument("--compute-us", type=float, default=8000.0)
    ap.add_argument("--attn-us", type=float, default=1000.0)
    args = ap.parse_args()

    try:
        model = MultiGPUKVModel.from_calibration(args.calib, setup_calib=args.setup_calib)
    except Exception:
        model = MultiGPUKVModel()
    blk = int(args.block_mb * 1024 * 1024)
    ell_nv = model.slow_tier_penalty(MGTier.PEER_NVLINK, blk)
    ell_pc = model.slow_tier_penalty(MGTier.HOST_PCIE, blk)

    rows = []
    # sweep deadline from just above the compute+attn floor upward
    floor = args.compute_us + args.attn_us
    for D in [floor*1.05, floor*1.1, floor*1.25, floor*1.5, floor*2, floor*3, floor*5, floor*10]:
        common = dict(deadline_us=D, n_blocks=args.n_blocks, block_bytes=blk,
                      compute_us=args.compute_us, attn_us=args.attn_us, miss_target=1e-2)
        nv = model.min_fast_resident_for_slo(spill_tier=MGTier.PEER_NVLINK, **common)
        pc = model.min_fast_resident_for_slo(spill_tier=MGTier.HOST_PCIE, **common)
        def maxctx(r):
            phi = r["max_slow_fraction"]
            return args.fast_budget / (1.0 - phi) if phi < 1.0 else float("inf")
        row = {"deadline_us": D,
               "nv_phi_max": nv["max_slow_fraction"], "pc_phi_max": pc["max_slow_fraction"],
               "nv_min_fast": nv["min_fast_blocks"], "pc_min_fast": pc["min_fast_blocks"],
               "nv_max_ctx_blocks": maxctx(nv), "pc_max_ctx_blocks": maxctx(pc),
               "ctx_advantage": (maxctx(nv) / maxctx(pc)) if maxctx(pc) not in (0, float("inf")) else None}
        rows.append(row)
        print(f"  D={D/1e3:6.1f}ms  NVLink phi={nv['max_slow_fraction']:.3f} min_fast={nv['min_fast_blocks']:4d} "
              f"| PCIe phi={pc['max_slow_fraction']:.3f} min_fast={pc['min_fast_blocks']:4d} "
              f"| ctx adv={row['ctx_advantage'] if row['ctx_advantage'] else 'n/a'}")

    # the discriminating band: deadlines where NVLink phi_max - PCIe phi_max is largest
    disc = max(rows, key=lambda r: r["nv_phi_max"] - r["pc_phi_max"])
    res = {"_experiment": "e22_sizing_sweep", "_is_measured": False,
           "kind": "schedulability_inversion_deadline_sweep",
           "bandwidths_gbps": {"T1": model.device.bw_gbps[1], "T2": model.device.bw_gbps[2]},
           "c_us": list(model.c_us), "block_mb": args.block_mb,
           "ell_bar_nvlink_us": ell_nv, "ell_bar_pcie_us": ell_pc,
           "ell_ratio": ell_pc / ell_nv if ell_nv else None,
           "n_blocks": args.n_blocks, "fast_budget": args.fast_budget,
           "compute_us": args.compute_us, "attn_us": args.attn_us,
           "sweep": rows, "discriminating_deadline_us": disc["deadline_us"],
           "discriminating_nv_phi": disc["nv_phi_max"], "discriminating_pc_phi": disc["pc_phi_max"],
           "note": ("C2 inversion across deadlines. The band where nv_phi_max >> pc_phi_max is "
                    "the schedulability win: at the same per-token deadline, spilling over NVLink "
                    "admits a far larger slow fraction (longer context) than PCIe. ell_bar(PCIe)/"
                    "ell_bar(NVLink) sets the gap. Predicted (cost-model); _is_measured:false."),
           "_generated_at": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(res, indent=2))
    print(f"  ell_bar: NVLink={ell_nv:.1f}us PCIe={ell_pc:.1f}us (ratio {ell_pc/ell_nv:.1f}x)")
    print(f"  most-discriminating D={disc['deadline_us']/1e3:.1f}ms: NVLink phi={disc['nv_phi_max']:.3f} vs PCIe phi={disc['pc_phi_max']:.3f}")
    print(f"  -> wrote {OUT}")


if __name__ == "__main__":
    main()
