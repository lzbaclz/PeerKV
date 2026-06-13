"""E-B1 -- enablement consolidation: feasible regions + measured OOM corners.

Track B spec (02_track_B_system_paper.md SS4 E-B1). This consolidator merges the
measured enablement evidence into one per-source-labeled artifact + the
"feasible region" figure:

  capacity lines (REAL vLLM 0.8.5, continuous batching, serve_m1_tp2.json):
      single-GPU ceiling 116,016 tok (MHA, util 0.9); TP-2 ceiling 253,536 tok
  OOM corner latencies (real-weights synthetic loop, e31 overflow_e2e*.json):
      MHA 143,360 tok: single = CUDA OOM; peer-parallel 57.7 ms/tok;
      copy-back 291.4; host 1506.8  (enablement + 5.05x / 26.1x)
      GQA 573,440 tok: single OOM; peer 817.2; copy-back 1167.3 (1.43x, the
      honest B1 margin); host 1521.6
  fitting-region do-no-harm (real vLLM serving, e2e_vllm.json):
      Llama-3.1-8B continuous batching under a concurrent 512 MB peer handoff:
      TPOT +11.7/11.9% (push/pull) -- the placement law's peer cost, direction
      null, on a REAL engine.

HONEST SCOPE (per-row ``source`` fields make this auditable): the capacity
lines and fitting-region serving numbers are real-vLLM; the OOM corner
latencies are the e31 real-weights loop, NOT a vLLM server -- the formal
connector (Track C Phase 1) is the remaining gap to a fully-vLLM E-B1, and
tests/test_no_harm.py::test_no_harm_ab_vs_vllm stays skipped until its
artifact exists.

    python3 experiments/eB1_enablement.py
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _load(name):
    return json.load(open(os.path.join(RES, name)))


def main():
    serve = _load("serve_m1_tp2.json")
    mha = _load("overflow_e2e.json")
    gqa = _load("overflow_e2e_G35_gqa8.json")
    e2e = _load("e2e_vllm.json")

    single_cap = serve["derived"]["single_gpu_ceiling_tokens_at_0.9"]
    tp_cap = serve["tp2_util0.90"]["kv_cache_tokens"]

    def corner_rows(d, geom):
        ctx = d["context"]
        rows = [{
            "geom": geom, "ctx_tokens": ctx, "corner": "single",
            "status": d["single_gpu"]["status"], "ms_per_token": None,
            "source": f"overflow_e2e{'_G35_gqa8' if geom == 'GQA' else ''}.json "
                      f"(e31 real-weights loop; {d['total_kv_gib']} GiB KV)"}]
        for arm, corner in (("peer_parallel", "cfk"), ("copyback", "copyback"),
                            ("host", "host")):
            rows.append({
                "geom": geom, "ctx_tokens": ctx, "corner": corner,
                "status": "ran", "ms_per_token": d[arm]["ms_per_token"],
                "iqr": d[arm].get("iqr"),
                "source": f"overflow_e2e{'_G35_gqa8' if geom == 'GQA' else ''}.json"})
        return rows

    rows = corner_rows(mha, "MHA") + corner_rows(gqa, "GQA")
    mha_peer = mha["peer_parallel"]["ms_per_token"]
    gqa_peer = gqa["peer_parallel"]["ms_per_token"]

    out = {
        "_experiment": "eB1_enablement",
        "_is_measured": "consolidated-from-measured (per-row source fields)",
        "_generated_at": datetime.now(timezone.utc).isoformat(),
        "capacity_lines": {
            "single_gpu_ceiling_tokens": single_cap,
            "tp2_ceiling_tokens": tp_cap,
            "engine": serve["engine"], "model": serve["model"],
            "source": "serve_m1_tp2.json (REAL vLLM continuous batching)",
        },
        "oom_corners": rows,
        "enablement_margins": {
            "MHA_peer_vs_copyback": round(mha["copyback"]["ms_per_token"] / mha_peer, 2),
            "MHA_peer_vs_host": round(mha["host"]["ms_per_token"] / mha_peer, 2),
            "GQA_peer_vs_copyback": round(gqa["copyback"]["ms_per_token"] / gqa_peer, 2),
            "GQA_peer_vs_host": round(gqa["host"]["ms_per_token"] / gqa_peer, 2),
            "note": "GQA margin 1.43x is the honest B1 boundary -- reported "
                    "front-and-center, never hidden behind the MHA margin",
        },
        "fitting_region_do_no_harm": {
            "engine_idle_tpot_ms": e2e["conditions"]["idle"]["tpot_ms_p50_mean"],
            "peer_handoff_overhead_pct": {
                k: e2e["conditions"][k]["tpot_inflation_pct_vs_idle"]
                for k in ("push", "pull")},
            "source": "e2e_vllm.json (REAL vLLM Llama-3.1-8B + concurrent "
                      "512MB handoff; direction null on a real engine)",
        },
        "honest_gaps": [
            "OOM corner latencies are the e31 real-weights loop, not a vLLM "
            "server; formal connector (Track C P1) is the remaining gap",
            "test_no_harm_ab_vs_vllm (CI gate d) skips until the connector "
            "A/B artifact exists",
        ],
    }
    path = os.path.join(RES, "eB1_enablement.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["enablement_margins"], indent=1))
    print("wrote", path)

    # ---- feasible-region figure -------------------------------------------- #
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        corners = ["single", "TP-2", "PeerKV 2-GPU", "host"]
        ypos = {c: i for i, c in enumerate(corners)}
        spans = {
            "single": (0, single_cap, "#7fb069",
                       "feasible (real vLLM ceiling, util 0.9)"),
            "TP-2": (0, tp_cap, "#8e44ad", "feasible (real vLLM TP-2 ceiling)"),
            "PeerKV 2-GPU": (0, 2 * single_cap, "#2e86de",
                             "feasible (2x HBM; corners measured at 143K)"),
            "host": (0, 1048576, "#c0392b", "feasible (slow tier)"),
        }
        for c, (lo, hi, color, _) in spans.items():
            ax.barh(ypos[c], hi - lo, left=lo, height=0.55, color=color, alpha=0.75)
        ax.axvline(single_cap, color="k", ls="--", lw=1)
        ax.text(single_cap * 1.02, 3.45, f"single cap {single_cap//1000}K",
                fontsize=8, rotation=0)
        ax.axvline(143360, color="k", ls=":", lw=1)
        ax.text(143360 * 1.02, 2.8,
                f"MHA OOM demo 143K:\nsingle=OOM, peer {mha_peer:.0f} ms/tok\n"
                f"(copy-back {mha['copyback']['ms_per_token']:.0f}, "
                f"host {mha['host']['ms_per_token']:.0f})", fontsize=7)
        ax.set_yticks(range(len(corners)))
        ax.set_yticklabels(corners, fontsize=9)
        ax.set_xscale("log")
        ax.set_xlim(8192, 1200000)
        ax.set_xlabel("context length (tokens, log)", fontsize=9)
        ax.set_title("E-B1 feasible regions (Llama-2-7B MHA): capacity lines from "
                     "real vLLM; OOM corners measured (e31)", fontsize=9)
        fig.tight_layout()
        pdf = os.path.join(RES, "eB1_enablement.pdf")
        fig.savefig(pdf, dpi=200)
        print("wrote", pdf)
    except ImportError:
        print("matplotlib unavailable -- figure skipped")


if __name__ == "__main__":
    main()
