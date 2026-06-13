"""Optional PNG of the tier-boundary map. Needs matplotlib.
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
