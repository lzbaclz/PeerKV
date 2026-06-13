"""e40 -- the full-weight-read asymmetry LAW (the non-collidable headline).

Closed-form, hardware-agnostic. Per token per layer, a memory-bound decode reads:
  single   : W + ctx*kvB                  (whole weights, whole KV)
  CFK/borrow: W + (ctx/2)*kvB             (whole weights, half KV; peer holds+computes the rest)
  TP       : W/2 + (ctx/2)*kvB            (sharded weights, sharded KV) + an all-reduce term

Let x = (ctx/2)*kvB / W (KV-to-weight read ratio). Then
  borrow_bytes / TP_bytes = (1 + 2x) / (1 + 2x - 1) ... = (W+(ctx/2)kvB)/(W/2+(ctx/2)kvB)
                          = (1 + x)/(0.5 + x)  in (1, 2],  decreasing in x.
So a whole-weight peer-borrow ALWAYS reads more bytes than TP (ratio>1): TP structurally
owns the bandwidth-bound FITTING region; the gap -> 1 as context grows (KV dominates).
And single > CFK in bytes (CFK drops half the KV), so CFK beats single when the peer's
compute is free. Ordering in byte-read: TP < CFK < single, hence in memory-bound decode
latency (modulo the TP all-reduce and CFK round-trip OVERHEAD terms, which set the small-
context crossovers the measured map locates). This script computes the law per geometry
and checks the predicted ordering against the committed measured latencies.
"""
from __future__ import annotations
import json
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"
OUT = RES / "asymmetry_law.json"

# (name, d_model, n_layers, heads, kv_heads, head_dim, d_ffn)
MODELS = {
    "MHA_llama2_7b": (4096, 32, 32, 32, 128, 11008),
    "GQA_llama3_8b": (4096, 32, 32, 8,  128, 14336),
}
FP16 = 2


def per_layer_weight_bytes(d, heads, kv_heads, hd, d_ffn):
    q = d * heads * hd
    kv = 2 * d * kv_heads * hd
    o = heads * hd * d
    ffn = 3 * d * d_ffn       # gate, up, down (SwiGLU)
    return (q + kv + o + ffn) * FP16


def kv_bytes_per_token_layer(kv_heads, hd):
    return 2 * kv_heads * hd * FP16     # K and V


def main():
    out = {"_experiment": "e40_asymmetry_law", "_is_measured": False,
           "kind": "closed_form_byte_read_asymmetry", "law": {}, "geometries": {}}
    for name, (d, L, H, Hkv, hd, dff) in MODELS.items():
        W = per_layer_weight_bytes(d, H, Hkv, hd, dff)
        kvB = kv_bytes_per_token_layer(Hkv, hd)
        rows = []
        for ctx in (8192, 16384, 32768, 65536, 131072, 262144, 524288):
            half_kv = (ctx // 2) * kvB
            single = W + ctx * kvB
            cfk = W + half_kv
            tp = W / 2 + half_kv
            x = half_kv / W
            rows.append({"ctx": ctx, "x_kv_over_w": round(x, 3),
                         "single_MB": round(single / 1e6, 1), "cfk_MB": round(cfk / 1e6, 1),
                         "tp_MB": round(tp / 1e6, 1),
                         "borrow_over_tp": round(cfk / tp, 3), "cfk_over_single": round(cfk / single, 3),
                         "ordering_holds": tp < cfk < single})
        ratios = [r["borrow_over_tp"] for r in rows]
        out["geometries"][name] = {
            "weight_MB_per_layer": round(W / 1e6, 1), "kvB_per_tok_layer": kvB, "n_layers": L,
            "rows": rows,
            "borrow_over_tp_range": [min(ratios), max(ratios)],
            "in_1_2_interval": all(1.0 < r <= 2.0 + 1e-9 for r in ratios),
            "ordering_always_TP_lt_CFK_lt_single": all(r["ordering_holds"] for r in rows),
        }
    out["law"] = {
        "statement": "borrow_bytes/TP_bytes = (1+x)/(0.5+x) in (1,2], decreasing in x=(ctx/2)kvB/W; "
                     "TP<CFK<single in byte-read always -> TP owns the bandwidth-bound fitting region, "
                     "CFK beats single (free peer compute), neither beats TP except via TP's all-reduce "
                     "overhead at small ctx (where the measured map locates the crossover).",
        "no_surveyed_system_compares_borrow_vs_TP_on_same_fitting_decode": True,
    }
    # cross-check ordering against committed measured latencies (e33 tp_available_best_corner)
    try:
        pr = json.load(open(RES / "policy_regret.json"))
        tp_rows = pr.get("tp_available_best_corner", {}).get("rows", [])
        chk = []
        for r in tp_rows:
            s, t, c = r.get("single"), r.get("TP"), r.get("CFK")
            if isinstance(s, (int, float)) and isinstance(t, (int, float)) and isinstance(c, (int, float)):
                chk.append({"ctx": r["ctx"], "single_ms": s, "TP_ms": t, "CFK_ms": c,
                            "measured_TP_fastest": t <= min(s, c), "measured_single_lt_CFK": s < c})
        out["measured_ordering_check"] = chk
    except Exception as e:
        out["measured_ordering_check"] = f"(unavailable: {e})"
    OUT.write_text(json.dumps(out, indent=2))
    for name, g in out["geometries"].items():
        print(f"{name}: W={g['weight_MB_per_layer']}MB/layer kvB={g['kvB_per_tok_layer']}B/tok "
              f"borrow/TP in {g['borrow_over_tp_range']} in(1,2]={g['in_1_2_interval']} "
              f"order(TP<CFK<single)={g['ordering_always_TP_lt_CFK_lt_single']}")
    print("law:", out["law"]["statement"][:120], "...")
    print("-> wrote", OUT)


if __name__ == "__main__":
    main()
