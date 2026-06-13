"""e34 -- measured tier-boundary map: which corner wins in (context x peer-state).

The science headline of the repositioned paper. For each geometry we sweep
context length x peer-compute-state and label each cell with the corner the
deadline-gated selector picks, anchored to committed measurements:

    SINGLE   x < single-GPU capacity                  (measured 116K MHA, serve_m1_tp2)
    CFK      overflow & peer compute-idle             (e27/e31)
    COPYBACK overflow & peer compute-busy             (e23 contention: peer busy =>
                                                        CFK inadmissible; copy-back keeps
                                                        99.7% borrower BW)
    HOST     beyond two-GPU KV capacity               (peer can't hold the spill)
    [TP overlay] where TP is deployable it OWNS the fitting region up to its
                 capacity (253K MHA) -- the full-weight-read asymmetry (C2); shown
                 as an overlay because TP latency here is admissibility-only.

Emits results/tier_boundary_map.json, an ASCII map, and a guarded matplotlib
script (experiments/plot_boundary_map.py) for a PNG.

Run:  python3 experiments/e34_boundary_map.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from umallm.elastic_policy import (  # noqa: E402
    DecodeStepModel, Deployment, Geometry, OperatingPoint, PeerState,
    _single_capacity_tokens, select_point,
)
from experiments.e33_policy_regret import build_measured_table, calibrate  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

GLYPH = {
    OperatingPoint.SINGLE: "S",
    OperatingPoint.CFK: "C",
    OperatingPoint.COPYBACK: "B",   # copy-Back
    OperatingPoint.HOST: "H",
    OperatingPoint.TP: "T",
    OperatingPoint.INFEASIBLE: ".",
}
CTX_GRID = [8192, 16384, 32768, 65536, 116016, 143360, 262144,
            524288, 573440, 1048576]


def label_grid(geom_name, models, deploy):
    geom = {"MHA": Geometry.llama2_7b_mha(), "GQA": Geometry.gqa_8kv()}[geom_name]
    model = models[geom_name]
    rows = {}
    for idle in (True, False):
        line = []
        for ctx in CTX_GRID:
            peer = PeerState(compute_idle=idle)
            dec = select_point(ctx, geom, peer, deploy, model)
            line.append(GLYPH[dec.point])
        rows["peer_idle" if idle else "peer_busy"] = line
    if deploy.tp_enabled:
        # admissibility overlay: where TP is deployable it OWNS this region and is
        # faster than the whole-weight corners (C2 full-weight-read asymmetry).
        # Latency here is admissibility/throughput-derived, NOT a single-stream
        # measurement -> shown as an overlay, not fed to the latency oracle.
        rows["tp_overlay"] = ["T" if ctx <= deploy.tp_capacity_tokens else "-"
                              for ctx in CTX_GRID]
    return rows


def boundaries(geom_name, models, deploy):
    """The ctx thresholds where the selected corner changes (peer-idle row)."""
    geom = {"MHA": Geometry.llama2_7b_mha(), "GQA": Geometry.gqa_8kv()}[geom_name]
    model = models[geom_name]
    out, prev = [], None
    for ctx in CTX_GRID:
        dec = select_point(ctx, geom, PeerState(compute_idle=True), deploy, model)
        if dec.point.value != prev:
            out.append({"at_ctx": ctx, "becomes": dec.point.value})
            prev = dec.point.value
    return out


def ascii_map(geom_name, rows, deploy):
    cols = "".join(f"{c//1024:>5d}K" if c < 1024**2 else f"{c//1024//1024:>4d}M " for c in CTX_GRID)
    single_cap = _single_capacity_tokens(Geometry.gqa_8kv() if geom_name == "GQA"
                                         else Geometry.llama2_7b_mha(), deploy)
    lines = [f"  ctx -> {cols}"]
    row_names = ("tp_overlay", "peer_idle", "peer_busy") if "tp_overlay" in rows \
        else ("peer_idle", "peer_busy")
    for name in row_names:
        lines.append(f"  {name:9s} " + "  ".join(f"{g:>4s}" for g in rows[name]))
    lines.append(f"  single-GPU cap ~= {single_cap:,} tok"
                 + (f" | TP cap ~= {deploy.tp_capacity_tokens:,} tok" if deploy.tp_enabled else ""))
    return "\n".join(lines)


def write_plot_script():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_boundary_map.py")
    code = '''"""Optional PNG of the tier-boundary map. Needs matplotlib.
   pip install matplotlib && python3 experiments/plot_boundary_map.py"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
d = json.load(open(os.path.join(R, "tier_boundary_map.json")))
glyph2id = {"S": 0, "C": 1, "B": 2, "H": 3, "T": 4, ".": 5}
labels = ["single", "CFK", "copy-back", "host", "TP", "infeasible"]
cmap = ListedColormap(["#7fb069", "#2e86de", "#f0932b", "#c0392b", "#8e44ad", "#cccccc"])
ctx = d["ctx_grid"]
fig, axes = plt.subplots(2, 2, figsize=(13, 6))
for ax, key in zip(axes.flat, d["maps"]):
    m = d["maps"][key]
    grid = np.array([[glyph2id[g] for g in m["peer_idle"]],
                     [glyph2id[g] for g in m["peer_busy"]]])
    ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=5)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["peer idle", "peer busy"])
    ax.set_xticks(range(len(ctx)))
    ax.set_xticklabels([f"{c//1024}K" if c < 1024**2 else f"{c//1024//1024}M" for c in ctx], rotation=45)
    ax.set_title(key)
fig.suptitle("PeerKV tier-boundary map: winning corner in (context x peer-state)")
fig.tight_layout()
out = os.path.join(R, "tier_boundary_map.png")
fig.savefig(out, dpi=130)
print("wrote", out)
'''
    with open(path, "w") as f:
        f.write(code)
    return path


def main():
    table = build_measured_table()
    models, _ = calibrate(table)
    out = {
        "_experiment": "e34_boundary_map",
        "_is_measured": "derived-from-measured (offline)",
        "ctx_grid": CTX_GRID,
        "legend": {v: k.value for k, v in GLYPH.items()},
        "measured_anchors": {
            "single_gpu_ceiling_tokens_MHA": 116016,
            "tp2_capacity_tokens_MHA": 253536,
            "demonstrated_overflow_MHA_tokens": 143360,
            "demonstrated_overflow_GQA_tokens": 573440,
            "contention_lender_flops_retained": 0.6676,
            "contention_borrower_bw_retained": 0.9974,
            "_src": ["serve_m1_tp2.json", "overflow_e2e_G35.json",
                     "overflow_e2e_G35_gqa8.json", "contention.json"],
        },
        "maps": {},
        "boundaries_peer_idle": {},
    }
    scenarios = [
        ("MHA / TP-unavailable", "MHA", Deployment(tp_enabled=False)),
        ("GQA / TP-unavailable", "GQA", Deployment(tp_enabled=False)),
        ("MHA / TP-available", "MHA", Deployment(tp_enabled=True)),
        ("GQA / TP-available", "GQA",
         Deployment(tp_enabled=True, tp_capacity_tokens=int(253536 * 4),
                    single_capacity_tokens=116016)),
    ]
    print("=" * 86)
    print("e34  TIER-BOUNDARY MAP  (S=single C=CFK B=copy-back H=host T=TP-overlay)")
    print("=" * 86)
    for title, g, dep in scenarios:
        rows = label_grid(g, models, dep)
        out["maps"][title] = rows
        out["boundaries_peer_idle"][title] = boundaries(g, models, dep)
        print(f"\n[{title}]")
        print(ascii_map(g, rows, dep))

    path = os.path.join(RESULTS, "tier_boundary_map.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    plot = write_plot_script()
    print(f"\nwrote {path}")
    print(f"wrote {plot}  (run with matplotlib for a PNG)")


if __name__ == "__main__":
    main()
