"""Relabel the existing per-block vs coalesced decode-loop as a Harvest-policy
vs PeerKV-policy head-to-head (ZERO new GPU runs).

Verified prior-art fact (web check): Harvest exposes idle peer-GPU HBM as an
opportunistic cache and reloads KV **per missing block** -- it does NOT coalesce.
That is exactly the policy measured in decode_loop_perblock.json (blocks_per_chunk
= 1, chunk_kb = 16), which LOSES to host (NVLink/host ~= 0.85x) because the NVLink
per-transfer setup c_T1 (23.6us) exceeds the PCIe c_T2 (12.6us) and dominates tiny
transfers. PeerKV's cost-model-selected C* coalescing (decode_loop.json, large
chunk) recovers 3.1-3.6x over host. So PeerKV's central negative result IS a
faithful prediction of Harvest's policy on this hardware -- relabel, don't rerun.

NOTE (citation accuracy): this relabel applies to HARVEST only. AQUA already
coalesces NVLink transfers, so the per-block arm must NOT be labeled "AQUA"; the
honest AQUA comparison is the fixed-large-chunk arm in e35_aqua_arm.py.

Run:  python3 experiments/analyze_harvest_relabel.py
"""
from __future__ import annotations

import json
import os

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def main():
    perblock = _load("decode_loop_perblock.json")   # Harvest policy (per-block)
    coalesced = _load("decode_loop.json")            # PeerKV policy (C* coalesced)

    rows = []
    for pb, co in zip(perblock["block_size_sweep"], coalesced["block_size_sweep"]):
        assert pb["block_tokens"] == co["block_tokens"]
        rows.append({
            "block_tokens": pb["block_tokens"],
            "harvest_perblock_nvlink_vs_host": round(pb["nvlink_vs_host_prefetch"], 3),
            "peerkv_coalesced_nvlink_vs_host": round(co["nvlink_vs_host_prefetch"], 3),
            "harvest_chunk_kb": perblock["chunk_kb"],
            "peerkv_chunk_kb": coalesced["chunk_kb"],
            "harvest_tpot_ms": round(pb["arms"]["nvlink_prefetch"]["model_tpot_p50_ms"], 1),
            "peerkv_tpot_ms": round(co["arms"]["nvlink_prefetch"]["model_tpot_p50_ms"], 1),
            "peerkv_speedup_over_harvest": round(
                pb["arms"]["nvlink_prefetch"]["model_tpot_p50_ms"]
                / co["arms"]["nvlink_prefetch"]["model_tpot_p50_ms"], 2),
        })

    out = {
        "_experiment": "harvest_relabel",
        "_is_measured": "relabel-of-measured (no new runs)",
        "_sources": ["decode_loop_perblock.json", "decode_loop.json"],
        "model": coalesced["model"], "geometry_HxDxL": coalesced["geometry_HxDxL"],
        "ctx_tokens": coalesced["ctx_tokens"], "overflow_frac": coalesced["overflow_frac"],
        "claim": ("Harvest's per-block peer reload (no coalescing) LOSES to host on "
                  "NVLink (~0.85x); PeerKV's cost-model C* coalescing wins ~3.1-3.6x "
                  "over host and 3.0-3.6x lower TPOT than the Harvest policy."),
        "citation_guard": ("Applies to Harvest only. AQUA already coalesces -> use "
                           "e35_aqua_arm.py (fixed-large-chunk) for the AQUA comparison."),
        "rows": rows,
    }
    path = os.path.join(RESULTS, "harvest_relabel.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print("=" * 78)
    print("Harvest-policy (per-block) vs PeerKV-policy (C* coalesced)  [relabel, no rerun]")
    print(f"  model={out['model']}  ctx={out['ctx_tokens']}  spill={out['overflow_frac']}")
    print("=" * 78)
    print(f"{'block_tok':>9s} {'Harvest x/host':>15s} {'PeerKV x/host':>14s} "
          f"{'Harvest TPOT':>13s} {'PeerKV TPOT':>12s} {'PeerKV/Harvest':>15s}")
    for r in rows:
        print(f"{r['block_tokens']:9d} {r['harvest_perblock_nvlink_vs_host']:15.3f} "
              f"{r['peerkv_coalesced_nvlink_vs_host']:14.3f} "
              f"{r['harvest_tpot_ms']:11.1f}ms {r['peerkv_tpot_ms']:10.1f}ms "
              f"{r['peerkv_speedup_over_harvest']:14.2f}x")
    print(f"\nwrote {path}")
    print("citation guard:", out["citation_guard"])


if __name__ == "__main__":
    main()
