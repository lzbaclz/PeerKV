"""e39 -- fusion ceiling: does fusing the per-layer round-trip let full-model
compute-follows-KV (CFK) CROSS the TP line? PROJECTION from measured components.

NOT a new kernel (the full multi-device fused decoder is ~300-500 LOC CUDA, scoped
in docs/STRETCH_SCOPING.md + csrc/peer_fused_decoder.cu skeleton). This computes the
*ceiling* fusion can reach, from already-measured pieces:

  (a) byte-bound floor (round-trip = 0), from geometry: per token the compute GPU
      reads W + (ctx/2)*kvB for CFK; single reads W + ctx*kvB; TP reads W/2 +
      (ctx/2)*kvB. So:
        CFK/single bytes  = (W + ctx/2*kvB)/(W + ctx*kvB)   < 1  (CFK reads HALF the KV)
        CFK/TP bytes      = (W + ctx/2*kvB)/(W/2 + ctx/2*kvB) in (1,2]  (CFK reads FULL W)
      => with a perfect kernel CFK can BEAT single (it reads half the KV) but NEVER
         beats TP (TP halves the weight read; the C2 asymmetry) -- closes, not crosses.
  (b) measured dispatch recoverable by a CUDA graph: fused_kernel.json eager vs graph.
  (c) the eager full-model gaps to close: e27 (CFK vs single).

    python3 experiments/e39_fusion_ceiling.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from umallm.elastic_policy import Geometry  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _load(n):
    with open(os.path.join(RES, n)) as f:
        return json.load(f)


def byte_ratios(geom, ctx):
    W = geom.weight_bytes
    kv = geom.kv_bytes_per_token
    cfk = W + (ctx // 2) * kv
    single = W + ctx * kv
    tp = W / 2 + (ctx // 2) * kv
    return {"cfk_vs_single_bytes": round(cfk / single, 3),
            "cfk_vs_tp_bytes": round(cfk / tp, 3)}


def main():
    fk = _load("fused_kernel.json")
    # (b) dispatch recoverable per layer (eager -> graph), measured, attention-only
    disp = []
    for r in fk["rows"]:
        disp.append((r["peer_eager_ms"] - r["peer_graph_ms"]) / 32 * 1e3)  # us/layer
    recov_us = round(sum(disp) / len(disp), 1)

    geoms = {"MHA": Geometry.llama2_7b_mha(), "GQA": Geometry.gqa_8kv()}
    e27 = {
        "MHA": {"16384": _load("real_weights_L32_C16384.json"),
                "32768": _load("real_weights_L32_C32768.json")},
        "GQA": {"16384": _load("real_weights_L32_C16384_gqa8.json"),
                "32768": _load("real_weights_L32_C32768_gqa8.json")},
    }

    out = {"_experiment": "e39_fusion_ceiling",
           "_is_measured": "PROJECTION from measured (fused_kernel + e27 + geometry); NOT a new kernel",
           "_sources": ["fused_kernel.json", "real_weights_*.json"],
           "measured_attention_only_fused_vs_single": {
               r["ttot_per_layer"]: {"eager_x": r["peer_eager_x_vs_single"],
                                     "graph_x": r["peer_graph_x_vs_single"]}
               for r in fk["rows"]},
           "dispatch_recoverable_us_per_layer": recov_us,
           "byte_floor": {}, "eager_full_model_gap_vs_single": {},
           "conclusion": ("Fusion recovers ~%.0f us/layer dispatch (measured) and drives "
                          "the round-trip toward the ~50us 2-hop floor. BYTE FLOOR: a "
                          "perfect kernel makes CFK read HALF the KV -> CFK can match/beat "
                          "SINGLE at long context (cfk_vs_single_bytes < 1, consistent "
                          "with the measured attention-only 1.34-1.98x), BUT CFK reads "
                          "FULL weights while TP reads half -> cfk_vs_tp_bytes in (1,2]: "
                          "fusion CLOSES the gap to single but does NOT CROSS TP. CFK's "
                          "value stays enablement (>253K, where TP OOMs)." % recov_us)}

    print("=" * 74)
    print(f"FUSION CEILING (projection).  dispatch recoverable: {recov_us} us/layer (measured)")
    print("=" * 74)
    print("byte floor (round-trip=0):   ctx     CFK/single   CFK/TP")
    for g, geom in geoms.items():
        out["byte_floor"][g] = {}
        for ctx in (16384, 32768, 65536, 143360, 262144, 573440):
            br = byte_ratios(geom, ctx)
            out["byte_floor"][g][str(ctx)] = br
            print(f"  [{g}] {ctx:8d}      {br['cfk_vs_single_bytes']:.3f}      {br['cfk_vs_tp_bytes']:.3f}")
    print("\neager full-model CFK vs single (measured e27, the gap fusion would close):")
    for g in ("MHA", "GQA"):
        out["eager_full_model_gap_vs_single"][g] = {}
        for ctx in ("16384", "32768"):
            x = e27[g][ctx]["peer_parallel_vs_single"]
            out["eager_full_model_gap_vs_single"][g][ctx] = round(x, 3)
            print(f"  [{g}] ctx={ctx}: CFK {x:.2f}x single (eager)")
    print("\nCONCLUSION:", out["conclusion"])

    path = os.path.join(RES, "fusion_ceiling.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
