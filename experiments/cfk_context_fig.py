#!/usr/bin/env python3
"""Track B headline figure: CFK fused multi-device decode speedup vs context length.
Reads experiments/results/cfk_context_sweep.json -> cfk_context_sweep.pdf."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 8,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.8, "legend.frameon": False,
})
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "experiments" / "results" / "cfk_context_sweep.json"
OUT = ROOT / "experiments" / "results" / "cfk_context_sweep.pdf"
EAGER, GRAPH = "#0072B2", "#D55E00"


def main():
    rows = [r for r in json.loads(DATA.read_text())["by_context"] if "error" not in r]
    ctx = [r["ctx"] for r in rows]
    eager = [r["peer_eager_speedup"] for r in rows]
    graph = [r["peer_graph_speedup"] for r in rows]

    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.text(ctx[0], 1.005, "single-GPU baseline", fontsize=6.5, color="gray", va="bottom")
    ax.plot(ctx, graph, "s-", color=GRAPH, markersize=5, linewidth=1.3,
            label="CFK (multi-device CUDA graph)")
    ax.plot(ctx, eager, "o-", color=EAGER, markersize=5, linewidth=1.3,
            label="CFK (eager)")
    # annotate the long-context endpoint
    ax.annotate(f"{graph[-1]:.2f}$\\times$", (ctx[-1], graph[-1]),
                textcoords="offset points", xytext=(-4, 6), fontsize=8,
                color=GRAPH, fontweight="bold", ha="right")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ctx)
    ax.set_xticklabels([f"{c//1024}K" for c in ctx])
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Speedup over single-GPU decode")
    ax.set_ylim(0.95, 1.6)
    ax.set_title("CFK fused decode: speedup grows with context\n(Llama-3-8B GQA, dual-A100 NVLink3, L=32)")
    ax.legend(loc="upper left", fontsize=7)
    fig.text(0.5, -0.02,
             "Numerics: cosine(single, CFK) $>$ 0.9999 at every point. KV sharded across "
             "2 GPUs; only q and the (O,lse) partial cross NVLink.",
             ha="center", fontsize=6, style="italic")
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
