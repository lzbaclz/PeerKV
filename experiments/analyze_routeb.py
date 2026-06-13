#!/usr/bin/env python3
"""Turn routeb_benchmark JSON into the paper's tables and figures.

Reads a results file from ``routeb_benchmark.py`` and emits:
  * summary.md / summary.csv -- the comparison table (one row per mode x context)
  * goodput_vs_context.pdf   -- goodput under SLO (the headline metric)
  * tpot_p99_vs_context.pdf  -- tail latency
  * capacity.pdf             -- served fraction (OOM ceiling) per mode
  * footprint.pdf            -- HBM high-water + Grace residency

Matplotlib is optional: tables are always written; figures are skipped (with a
note) if it's absent. Every figure is stamped MOCK or MEASURED from the JSON so
a synthetic run is never mistaken for hardware data.

Usage:
    python experiments/analyze_routeb.py experiments/results/routeb_gh200.json \
        --outdir experiments/results/figs
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:  # pragma: no cover - plotting is optional
    HAS_MPL = False

_MODE_ORDER = ("hbm_only", "vllm_offload", "passive_uvm", "routeb")
_MODE_LABEL = {
    "hbm_only": "HBM-only (no offload)",
    "vllm_offload": "vLLM CPU offload",
    "passive_uvm": "Passive UVM",
    "routeb": "Route B (ours)",
}
_TABLE_COLS = [
    ("mode", "mode"), ("context_tokens", "ctx"), ("concurrency", "conc"),
    ("served", "ok/total"), ("tpot_p50_ms", "TPOT p50 (ms)"),
    ("tpot_p99_ms", "TPOT p99 (ms)"), ("goodput_tok_s", "goodput (tok/s)"),
    ("hbm_high_water_gib", "HBM HW (GiB)"), ("grace_resident_gib", "Grace (GiB)"),
]


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _stamp(payload: dict) -> str:
    return "MEASURED" if payload.get("measured") else "MOCK (synthetic)"


def table_rows(payload: dict) -> list[dict]:
    """Flatten results into table rows, sorted by (context, mode order)."""
    rows = []
    for r in payload.get("results", []):
        rows.append({
            "mode": r.get("mode", "?"),
            "context_tokens": r.get("context_tokens", 0),
            "concurrency": r.get("concurrency", 0),
            "served": f"{r.get('n_ok', 0)}/{r.get('n_requests', 0)}",
            "served_frac": (r.get("n_ok", 0) / r["n_requests"]
                            if r.get("n_requests") else 0.0),
            "tpot_p50_ms": round(r.get("tpot_p50_ms", float("nan")), 1),
            "tpot_p99_ms": round(r.get("tpot_p99_ms", float("nan")), 1),
            "goodput_tok_s": round(r.get("goodput_tok_s", 0.0), 1),
            "hbm_high_water_gib": round(r.get("hbm_high_water_gib", 0.0), 1),
            "grace_resident_gib": round(r.get("grace_resident_gib", 0.0), 1),
        })

    def key(row):
        m = row["mode"]
        return (row["context_tokens"],
                _MODE_ORDER.index(m) if m in _MODE_ORDER else 99)
    return sorted(rows, key=key)


def markdown_table(payload: dict) -> str:
    rows = table_rows(payload)
    head = "| " + " | ".join(label for _, label in _TABLE_COLS) + " |"
    sep = "| " + " | ".join("---" for _ in _TABLE_COLS) + " |"
    lines = [f"# Route B benchmark -- {_stamp(payload)}", "",
             f"backend: `{payload.get('backend')}`  measured: "
             f"`{payload.get('measured')}`", "", head, sep]
    for row in rows:
        lines.append("| " + " | ".join(str(row[k]) for k, _ in _TABLE_COLS) + " |")
    return "\n".join(lines) + "\n"


def write_csv(payload: dict, path: str):
    rows = table_rows(payload)
    cols = [k for k, _ in _TABLE_COLS] + ["served_frac"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in cols})


def _series_by_mode(payload: dict, ykey: str):
    """{mode: ([contexts], [yvals])} sorted by context, for plotting."""
    by_mode: dict[str, list[tuple[int, float]]] = {}
    for r in payload.get("results", []):
        by_mode.setdefault(r.get("mode", "?"), []).append(
            (r.get("context_tokens", 0), r.get(ykey, float("nan"))))
    out = {}
    for mode, pts in by_mode.items():
        pts.sort()
        out[mode] = ([p[0] for p in pts], [p[1] for p in pts])
    return out


def _line_plot(payload, ykey, ylabel, title, outpath):  # pragma: no cover - needs mpl
    series = _series_by_mode(payload, ykey)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for mode in _MODE_ORDER:
        if mode not in series:
            continue
        xs, ys = series[mode]
        ax.plot(xs, ys, marker="o", label=_MODE_LABEL.get(mode, mode))
    ax.set_xscale("log", base=2)
    ax.set_xlabel("context tokens")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}  [{_stamp(payload)}]", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def _capacity_plot(payload, outpath):  # pragma: no cover - needs mpl
    series = _series_by_mode(payload, "n_ok")
    # served fraction needs n_requests too; recompute from rows
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    rows = payload.get("results", [])
    ctxs = sorted({r.get("context_tokens", 0) for r in rows})
    width = 0.2
    for j, mode in enumerate([m for m in _MODE_ORDER if any(
            r.get("mode") == m for r in rows)]):
        fracs = []
        for c in ctxs:
            rr = [r for r in rows if r.get("mode") == mode
                  and r.get("context_tokens") == c]
            fracs.append((rr[0]["n_ok"] / rr[0]["n_requests"])
                         if rr and rr[0].get("n_requests") else 0.0)
        xs = [i + j * width for i in range(len(ctxs))]
        ax.bar(xs, fracs, width=width, label=_MODE_LABEL.get(mode, mode))
    ax.set_xticks([i + 1.5 * width for i in range(len(ctxs))])
    ax.set_xticklabels([str(c) for c in ctxs], fontsize=7)
    ax.set_ylabel("served fraction (1 - OOM)")
    ax.set_xlabel("context tokens")
    ax.set_title(f"Capacity ceiling  [{_stamp(payload)}]", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def make_figures(payload: dict, outdir: str) -> list[str]:
    if not HAS_MPL:  # pragma: no cover
        print("matplotlib not available -- skipping figures (tables still written)")
        return []
    os.makedirs(outdir, exist_ok=True)
    written = []
    jobs = [
        ("goodput_tok_s", "goodput under SLO (tok/s)", "Goodput vs context",
         "goodput_vs_context.pdf"),
        ("tpot_p99_ms", "TPOT p99 (ms)", "Tail latency vs context",
         "tpot_p99_vs_context.pdf"),
        ("hbm_high_water_gib", "HBM high-water (GiB)", "HBM footprint",
         "hbm_footprint.pdf"),
    ]
    for ykey, ylabel, title, fname in jobs:
        p = os.path.join(outdir, fname)
        _line_plot(payload, ykey, ylabel, title, p)
        written.append(p)
    cap = os.path.join(outdir, "capacity.pdf")
    _capacity_plot(payload, cap)
    written.append(cap)
    return written


def analyze(path: str, outdir: str) -> dict:
    payload = load_results(path)
    os.makedirs(outdir, exist_ok=True)
    md_path = os.path.join(outdir, "summary.md")
    csv_path = os.path.join(outdir, "summary.csv")
    with open(md_path, "w") as f:
        f.write(markdown_table(payload))
    write_csv(payload, csv_path)
    figs = make_figures(payload, outdir)
    return {"summary_md": md_path, "summary_csv": csv_path, "figures": figs,
            "measured": payload.get("measured", False)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_json")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args(argv)
    outdir = args.outdir or (os.path.splitext(args.results_json)[0] + "_figs")
    out = analyze(args.results_json, outdir)
    print(f"[{('MEASURED' if out['measured'] else 'MOCK')}] wrote:")
    print(" ", out["summary_md"])
    print(" ", out["summary_csv"])
    for f in out["figures"]:
        print(" ", f)
    if not out["figures"]:
        print("  (no figures -- matplotlib absent)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
