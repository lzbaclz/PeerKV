"""Align DCGM dmon samples to G1 per-condition time windows and merge with timing.

Reads results/g1_dcgm.log (epoch-prefixed dmon lines), results/g1_markers.json
({cond:[t0,t1]}), results/g1_timing.json -> writes results/g1_attribution.json
with per-condition, per-GPU averages of DRAM_ACTIVE / SM_ACTIVE / NVLINK bytes.
"""
from __future__ import annotations
import json, statistics
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"


def parse_dmon(path: Path):
    """Yield (epoch, gpu_id, {field:val}). dmon line:
    '<epoch> GPU 0  <GRACT> <SMACT> <SMOCC> <DRAMA> <NVLTX> <NVLRX>'
    Header lines start with '#'."""
    rows = []
    cols = ["GRACT", "SMACT", "SMOCC", "DRAMA", "NVLTX", "NVLRX"]
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[1] != "GPU":
            continue
        try:
            epoch = float(parts[0]); gpu = int(parts[2])
        except ValueError:
            continue
        vals = parts[3:]
        rec = {}
        for c, v in zip(cols, vals):
            try:
                rec[c] = float(v)
            except ValueError:
                rec[c] = None
        rows.append((epoch, gpu, rec))
    return rows


def main():
    markers = json.loads((RES / "g1_markers.json").read_text())
    timing = json.loads((RES / "g1_timing.json").read_text())
    rows = parse_dmon(RES / "g1_dcgm.log")

    attribution = {}
    for cond, (t0, t1) in markers.items():
        # use the steady-spin tail half of the window (skip transient)
        mid = t0 + 0.5 * (t1 - t0)
        per_gpu = {0: {}, 1: {}}
        for gpu in (0, 1):
            sel = [rec for (e, g, rec) in rows if g == gpu and mid <= e <= t1]
            for field in ("GRACT", "SMACT", "SMOCC", "DRAMA", "NVLTX", "NVLRX"):
                vals = [r[field] for r in sel if r.get(field) is not None]
                per_gpu[gpu][field] = round(statistics.mean(vals), 4) if vals else None
            per_gpu[gpu]["_n_samples"] = len(sel)
        attribution[cond] = {
            "timing": timing["conditions"][cond],
            "holder_gpu1": per_gpu[1],
            "consumer_gpu0": per_gpu[0],
        }

    out = {
        "_experiment": "g1_attribution_merged",
        "geometry": timing.get("geometry"), "ctx": timing.get("ctx"), "batch": timing.get("batch"),
        "handoff_mb": timing.get("handoff_mb"),
        "fields": {"DRAMA": "DRAM interface active fraction (HBM port utilization)",
                   "SMACT": "SM active fraction", "SMOCC": "SM occupancy",
                   "NVLTX": "NVLink TX bytes/s", "NVLRX": "NVLink RX bytes/s"},
        "conditions": attribution,
    }
    (RES / "g1_attribution.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
