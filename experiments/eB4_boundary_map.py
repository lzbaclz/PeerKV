"""E-B4 -- corner MAP with per-cell provenance and uncertainty (Track B headline).

Extends e34's tier-boundary map to meet the Track-B acceptance gate
(02_track_B_system_paper.md SS4 E-B4):

  * every cell carries a ``source`` field: a measured anchor (which JSON) or
    "cost-model" -- the boundary provenance is auditable per cell;
  * every cell carries a ROBUSTNESS verdict: the winner must survive the e33
    RQ1 held-out error bands (winner_pred*(1+err_w) < runnerup*(1-err_r));
    cells where the bands overlap are marked uncertain and rendered hatched;
  * decisions run through umallm.runtime.selector.online_select in STRICT
    mode -- the map is produced by the same do-no-harm single-exit path that
    serves requests (0 violations is asserted, not assumed);
  * TP stays an admissibility overlay (R3: never a latency point).

Outputs results/eB4_boundary_map.json + a publication PDF.

    python3 experiments/eB4_boundary_map.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from umallm.elastic_policy import (  # noqa: E402
    Deployment, Geometry, LinkState, OperatingPoint, PeerState,
    _single_capacity_tokens)
from umallm.runtime.selector import online_select  # noqa: E402
from experiments.e33_policy_regret import build_measured_table, calibrate  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

CTX_GRID = [8192, 16384, 32768, 65536, 116016, 143360, 262144,
            524288, 573440, 1048576]
GLYPH = {OperatingPoint.SINGLE: "S", OperatingPoint.CFK: "C",
         OperatingPoint.COPYBACK: "B", OperatingPoint.HOST: "H",
         OperatingPoint.TP: "T", OperatingPoint.INFEASIBLE: "."}

# measured anchors: ctx values whose boundary position is directly measured
ANCHORS = {
    116016: "serve_m1_tp2.json (single-GPU ceiling, vLLM 0.8.5)",
    253536: "serve_m1_tp2.json (TP-2 ceiling, vLLM 0.8.5)",
    143360: "overflow_e2e_G35.json (MHA OOM enablement)",
    573440: "overflow_e2e_G35_gqa8.json (GQA OOM enablement)",
    32768: "real_weights_L32_C32768[_gqa8].json (fitting anchor)",
    16384: "real_weights_L32_C16384[_gqa8].json (fitting anchor)",
}


def _err_bands(rq1: dict, geom_name: str) -> dict:
    """Per-corner relative error band from the e33 held-out cells (worst case
    per corner; 5% floor for in-sample corners; host uses the copy-back band
    -- same streamed mechanism, PCIe instead of NVLink)."""
    held = rq1[geom_name]["heldout"]
    worst = {"compute_follows_kv": 0.05, "copyback": 0.05}
    for key, cell in held.items():
        corner = key.split("@")[0]
        worst[corner] = max(worst.get(corner, 0.05), cell["abs_pct"] / 100.0)
    return {
        OperatingPoint.SINGLE: 0.05,
        OperatingPoint.CFK: worst["compute_follows_kv"],
        OperatingPoint.COPYBACK: worst["copyback"],
        OperatingPoint.HOST: worst["copyback"],
        OperatingPoint.TP: 0.05,
    }


def label_cell(ctx, geom, geom_name, model, deploy, idle, bands, measured):
    from umallm.elastic_policy import admissible_points
    peer = PeerState(compute_idle=idle)
    dec = online_select(ctx, geom, deploy, model, peer=peer,
                        link=LinkState.from_peer(peer), strict=True)
    # report predictions over the FULL admissible set (transparency) -- the
    # R1-fit re-pick narrows the decision's candidate set in the fitting
    # region, but the map cell should still show what the alternatives cost.
    full_adm = admissible_points(ctx, geom, peer, deploy)
    preds = {p: model.predict_ms(p, ctx, peer, deploy) for p in full_adm
             if model.predict_ms(p, ctx, peer, deploy) < float("inf")}
    cap = _single_capacity_tokens(geom, deploy)
    if ctx <= cap:
        # fitting region: SINGLE is chosen by the R1-fit invariant against a
        # MEASURED capacity anchor, not by a prediction comparison -- the
        # cost-model error bands are irrelevant to this boundary.
        robust = True
    else:
        legal = {p: v for p, v in preds.items()
                 if not (p is OperatingPoint.CFK and not idle)}
        robust = True
        if len(legal) >= 2 and dec.point in legal:
            win = legal[dec.point]
            runner_pt = min((p for p in legal if p is not dec.point),
                            key=lambda p: legal[p])
            runner = legal[runner_pt]
            robust = win * (1 + bands[dec.point]) < runner * (1 - bands[runner_pt])
    cell = {
        "ctx": ctx,
        "peer_state": "idle" if idle else "busy",
        "corner": dec.point.value,
        "glyph": GLYPH[dec.point],
        "predicted_ms": {p.value: round(v, 2) for p, v in preds.items()},
        "r1_restricted": "R1-restricted" in dec.reason,
        "robust_under_rq1_bands": bool(robust),
        "source": ANCHORS.get(ctx, "cost-model (calibrated on e27 anchors; "
                                    "RQ1 bands from policy_regret.json)"),
        "do_no_harm": "single" if dec.point is OperatingPoint.SINGLE else dec.reason,
    }
    # measured-anchor injection: where this exact (geom, ctx) was measured,
    # record the measured per-corner values and verify the model's RANKING
    # against measurement (the e33 claim: the model ranks, magnitudes drift).
    key = (geom_name, ctx)
    if key in measured and ctx > cap:
        meas = {p.value: v for p, v in measured[key].items()
                if isinstance(p, OperatingPoint) and v < float("inf")
                and not (p is OperatingPoint.CFK and not idle)}
        if meas:
            meas_winner = min(meas, key=meas.get)
            cell["measured_ms"] = {k: round(v, 2) for k, v in meas.items()}
            cell["measured_winner"] = meas_winner
            cell["model_ranking_correct"] = meas_winner == dec.point.value
            cell["robust_under_rq1_bands"] = meas_winner == dec.point.value
            cell["source"] += " [anchored: winner verified against measurement]"
    return cell


def main():
    table = build_measured_table()
    models, _ = calibrate(table)
    rq1 = json.load(open(os.path.join(RESULTS, "policy_regret.json")))[
        "rq1_costmodel_heldout_error"]

    out = {
        "_experiment": "eB4_boundary_map",
        "_capacity_rule": ("single/TP capacities are util-0.9 vLLM ceilings "
                           "(serve_m1_tp2); GQA single cap scales the measured "
                           "MHA ceiling by the per-token KV ratio; the GQA TP "
                           "ceiling uses the same 4x KV-ratio scaling "
                           "(253536*4) -- scaled, not directly measured"),
        "_is_measured": "derived-from-measured (offline; per-cell provenance)",
        "_generated_at": datetime.now(timezone.utc).isoformat(),
        "ctx_grid": CTX_GRID,
        "legend": {v: k.value for k, v in GLYPH.items()},
        "decision_path": "umallm.runtime.selector.online_select (strict; "
                         "do-no-harm single-exit; 0 violations asserted)",
        "uncertainty": "winner robust iff pred*(1+err) < runnerup*(1-err) "
                       "under e33 RQ1 held-out bands",
        "maps": {}, "cells": [], "boundaries_peer_idle": {},
    }

    scenarios = [
        ("MHA / TP-unavailable", "MHA", Deployment(tp_enabled=False)),
        ("GQA / TP-unavailable", "GQA", Deployment(tp_enabled=False)),
        ("MHA / TP-available", "MHA", Deployment(tp_enabled=True)),
        ("GQA / TP-available", "GQA",
         Deployment(tp_enabled=True, tp_capacity_tokens=int(253536 * 4),
                    single_capacity_tokens=116016)),
    ]
    for title, gname, dep in scenarios:
        geom = {"MHA": Geometry.llama2_7b_mha(), "GQA": Geometry.gqa_8kv()}[gname]
        model = models[gname]
        bands = _err_bands(rq1, gname)
        rows, robust_rows = {}, {}
        prev, bounds = None, []
        for idle in (True, False):
            cells = [label_cell(c, geom, gname, model, dep, idle, bands, table)
                     for c in CTX_GRID]
            for cell in cells:
                cell["scenario"] = title
            out["cells"].extend(cells)
            key = "peer_idle" if idle else "peer_busy"
            rows[key] = [c["glyph"] for c in cells]
            robust_rows[key] = [c["robust_under_rq1_bands"] for c in cells]
            if idle:
                for c in cells:
                    if c["corner"] != prev:
                        bounds.append({"at_ctx": c["ctx"], "becomes": c["corner"],
                                       "source": c["source"]})
                        prev = c["corner"]
        if dep.tp_enabled:
            rows["tp_overlay"] = ["T" if c <= dep.tp_capacity_tokens else "-"
                                  for c in CTX_GRID]
        out["maps"][title] = {"glyphs": rows, "robust": robust_rows}
        out["boundaries_peer_idle"][title] = bounds
        cap = _single_capacity_tokens(geom, dep)
        print(f"[{title}] cap={cap:,}")
        for k in ("peer_idle", "peer_busy"):
            marks = ["".join(g if r else g.lower())
                     for g, r in zip(rows[k], robust_rows[k])]
            print(f"  {k:9s} " + " ".join(f"{m:>4s}" for m in marks)
                  + "   (lowercase = uncertain under RQ1 bands)")

    path = os.path.join(RESULTS, "eB4_boundary_map.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", path)

    # ---- publication figure ------------------------------------------------ #
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch

        glyph2id = {"S": 0, "C": 1, "B": 2, "H": 3, "T": 4, ".": 5}
        labels = ["single", "CFK", "copy-back", "host", "TP", "infeasible"]
        cmap = ListedColormap(["#7fb069", "#2e86de", "#f0932b", "#c0392b",
                               "#8e44ad", "#cccccc"])
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 5.2))
        for ax, (title, m) in zip(axes.flat, out["maps"].items()):
            rows_ = m["glyphs"]
            grid = np.array([[glyph2id[g] for g in rows_["peer_idle"]],
                             [glyph2id[g] for g in rows_["peer_busy"]]])
            ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=5)
            for (r, key) in ((0, "peer_idle"), (1, "peer_busy")):
                for c_idx, ok in enumerate(m["robust"][key]):
                    if not ok:
                        ax.add_patch(plt.Rectangle(
                            (c_idx - .5, r - .5), 1, 1, fill=False,
                            hatch="///", edgecolor="k", linewidth=0.0))
            if "tp_overlay" in rows_:
                for c_idx, g in enumerate(rows_["tp_overlay"]):
                    if g == "T":
                        ax.plot(c_idx, -0.38, marker="v", color="#8e44ad",
                                markersize=5, clip_on=False)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["peer idle", "peer busy"], fontsize=8)
            ax.set_xticks(range(len(CTX_GRID)))
            ax.set_xticklabels([f"{c//1024}K" if c < 1024**2 else
                                f"{c//1024//1024}M" for c in CTX_GRID],
                               rotation=45, fontsize=7)
            ax.set_title(title, fontsize=9)
        handles = [Patch(color=cmap(i), label=l) for i, l in enumerate(labels)]
        handles.append(Patch(facecolor="w", hatch="///", edgecolor="k",
                             label="uncertain (RQ1 bands)"))
        fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=8,
                   frameon=False)
        fig.suptitle("PeerKV corner map: winning corner in (context x peer-state); "
                     "TP markers = admissibility overlay (R3)", fontsize=10)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        pdf = os.path.join(RESULTS, "eB4_boundary_map.pdf")
        fig.savefig(pdf, dpi=200)
        print("wrote", pdf)
    except ImportError:
        print("matplotlib unavailable -- JSON written, figure skipped")


if __name__ == "__main__":
    main()
