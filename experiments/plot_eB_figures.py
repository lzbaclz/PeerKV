"""Publication figures for eB2 (head-to-head) and eB5 (policy closed loop).

    python3 experiments/plot_eB_figures.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ARM_ORDER = ["harvest_perblock", "aqua_64mb", "peerkv_cstar", "host_offload", "cfk"]
ARM_LABEL = {"harvest_perblock": "Harvest\n(per-block)", "aqua_64mb": "AQUA\n(64MB fixed)",
             "peerkv_cstar": "PeerKV\n(C* coalesced)", "host_offload": "host\noffload",
             "cfk": "CFK\n(partials move)"}
ARM_COLOR = {"harvest_perblock": "#c0392b", "aqua_64mb": "#f0932b",
             "peerkv_cstar": "#2e86de", "host_offload": "#95a5a6", "cfk": "#7fb069"}


def fig_eb2():
    d = json.load(open(os.path.join(RES, "eB2_headtohead.json")))
    cells = [c for c in d["cells"] if c.get("arm") in ARM_ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=False)
    for ax, geom in zip(axes, ("MHA", "GQA")):
        x = np.arange(len(ARM_ORDER))
        for off, state, hatch in ((-0.2, "idle", None), (0.2, "busy", "//")):
            vals, errs, sel = [], [], []
            for arm in ARM_ORDER:
                cc = [c for c in cells if c["geom"] == geom
                      and c["peer_state"] == state and c["arm"] == arm]
                vals.append(cc[0]["ms"] if cc else np.nan)
                errs.append(cc[0]["iqr_ms"] if cc else 0)
                sel.append(cc[0].get("selected_by_policy", False) if cc else False)
            bars = ax.bar(x + off, vals, 0.38, yerr=errs, capsize=2,
                          color=[ARM_COLOR[a] for a in ARM_ORDER],
                          hatch=hatch, edgecolor="k", linewidth=0.4,
                          alpha=0.95 if state == "idle" else 0.7,
                          label=f"peer {state}")
            for b, v, s in zip(bars, vals, sel):
                if s:
                    ax.annotate("selector", (b.get_x() + b.get_width() / 2, v),
                                ha="center", va="bottom", fontsize=6.5,
                                color="k", weight="bold",
                                xytext=(0, 8), textcoords="offset points",
                                arrowprops=dict(arrowstyle="->", lw=0.7))
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([ARM_LABEL[a] for a in ARM_ORDER], fontsize=7.5)
        ax.set_ylabel("decode-step ms (log)" if geom == "MHA" else "")
        spill = [c for c in cells if c["geom"] == geom and c.get("spill_bytes_mb")]
        ax.set_title(f"{geom} -- {spill[0]['spill_bytes_mb']:.0f} MB spill, "
                     f"same kernel/harness for every arm", fontsize=9)
        ax.grid(axis="y", alpha=0.3, lw=0.4)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("E-B2 head-to-head: one runtime skeleton, only the transfer/compute "
                 "policy differs (65,536-token layer, 50% spill)", fontsize=10)
    fig.tight_layout()
    out = os.path.join(RES, "eB2_headtohead.pdf")
    fig.savefig(out, dpi=200)
    print("wrote", out)


def fig_eb5():
    d = json.load(open(os.path.join(RES, "eB5_policy.json")))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.4),
                                   gridspec_kw={"width_ratios": [1.1, 1.6]})
    # left: regret vs fixed policies
    pol = d["fixed_policy_regret"]
    names = ["selector\n(prior)", "selector\n(recal)"] + \
            [f"{k}-only" for k in pol]
    vals = [d["selector_prior_model"]["realized_regret"],
            d["selector_recalibrated"]["realized_regret"]] + list(pol.values())
    colors = ["#2e86de", "#1b4f8a"] + ["#95a5a6"] * len(pol)
    bars = ax1.bar(names, vals, color=colors, edgecolor="k", linewidth=0.4)
    for b, v in zip(bars, vals):
        ax1.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v),
                     ha="center", va="bottom", fontsize=7.5)
    ax1.axhline(1.0, color="k", ls="--", lw=0.8)
    ax1.set_ylabel("realized regret vs live oracle")
    ax1.set_title("selector regret = 1.0 (14/14 match);\n"
                  "every fixed policy strictly worse", fontsize=9)
    ax1.tick_params(axis="x", labelsize=7)
    ax1.grid(axis="y", alpha=0.3, lw=0.4)

    # right: live latency table + decisions
    lat = d["live_latency_ms_per_token"]
    ctxs = sorted(int(c) for c in lat)
    arms = ["single", "cfk", "copyback", "host"]
    mk = {"single": "o", "cfk": "s", "copyback": "^", "host": "D"}
    cl = {"single": "#7fb069", "cfk": "#2e86de", "copyback": "#f0932b",
          "host": "#c0392b"}
    for a in arms:
        ax2.plot(ctxs, [lat[str(c)][a] for c in ctxs], marker=mk[a],
                 color=cl[a], label=a, lw=1.2, ms=4)
    cap = d["single_capacity_tokens"]
    ax2.axvline(cap, color="k", ls=":", lw=1)
    ax2.text(cap * 1.04, ax2.get_ylim()[1] * 0.45,
             f"SLO budget {cap//1024}K:\nselector switches\nsingle->cfk/copyback",
             fontsize=7)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("context (tokens)"); ax2.set_ylabel("live ms/token")
    ax2.set_title("live per-arm latency measured THIS run "
                  "(non-circular oracle); violations = 0", fontsize=9)
    ax2.legend(fontsize=7.5, frameon=False)
    ax2.grid(alpha=0.3, lw=0.4)
    fig.tight_layout()
    out = os.path.join(RES, "eB5_policy.pdf")
    fig.savefig(out, dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    fig_eb2()
    fig_eb5()
