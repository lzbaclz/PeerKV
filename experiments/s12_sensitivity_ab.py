"""S12 -- the controlled A/B that resolves the s10-vs-s11 sensitivity mystery.

The open contradiction: the SAME engine (vLLM Llama-3.1-8B, conc=32, same
GPU) showed invert(5%) ~ 8-10 GB/s under s10's protocol (client restarted
per condition, shared prefix) but 52-70 GB/s under s11's (one continuous
client, unique prefixes) -- a ~7x sensitivity swing that an admission
controller must understand.  Candidate explanations:

  H1 prefix/KV    cached shared prefix => weight-bound decode vs unique
                  prefixes => real per-request KV reads
  H2 lifecycle    a freshly-started client front-loads 32 simultaneous
                  prefills + scheduler warm-up; the measurement window of a
                  restarted client is dominated by that ramp, and the ramp
                  is when ingress hurts
  H3 metric side  client-side TPOT amplifies via queueing (partially
                  excluded already: s10's engine logs showed ~12.7% too)

Design: ONE server instance for everything; fixed governed ingress at a
constant rate (default 30 GB/s, the s10 operating point); 2x2 grid of
{client lifecycle: continuous|restarted} x {prefix: cached|unique}, each
with an idle and an ingress cell; engine-side /metrics TPOT as the common
currency, sampled in 5s sub-windows so the restarted cells resolve
early(ramp) vs late(steady) inflation -- the smoking gun for H2.  Client-
side TPOT recorded where the protocol yields a per-cell client run
(restarted cells), matching s10's original metric.

Run via experiments/s12_sensitivity_ab.sh.  Output:
experiments/results/s12_sensitivity_ab.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _s_common import Train  # noqa: E402
from s11_engine_loop import EngineTpot  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "s12_sensitivity_ab.json"


class SubwindowMeter:
    """Engine-side TPOT in fixed sub-windows (time-resolved inflation)."""

    def __init__(self, base: str, sub_s: float = 5.0):
        self.tpot = EngineTpot(base)
        self.sub_s = sub_s

    def run(self, secs: float) -> list[dict]:
        rows = []
        t0 = time.monotonic()
        self.tpot.delta_ms()                      # reset origin
        while time.monotonic() - t0 < secs:
            time.sleep(self.sub_s)
            d = self.tpot.delta_ms()
            rows.append({"t_s": round(time.monotonic() - t0, 1),
                         "tpot_ms": round(d, 3) if d is not None else None})
        return rows


def med_tpot(rows, t_min=0.0, t_max=1e9):
    xs = [r["tpot_ms"] for r in rows
          if r["tpot_ms"] is not None and t_min <= r["t_s"] <= t_max]
    return round(statistics.median(xs), 3) if xs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--client-py", required=True,
                    help="python used to run e2e_vllm_client.py")
    ap.add_argument("--rate-gbs", type=float, default=30.0)
    ap.add_argument("--mb", type=int, default=512)
    ap.add_argument("--chunk-mb", type=int, default=64)
    ap.add_argument("--cell-secs", type=float, default=40.0)
    ap.add_argument("--conc", type=int, default=32)
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()

    from umallm.observability import gate_or_skip
    gate_or_skip("s12_sensitivity_ab")            # SKIP_IDLE_PROBE set by runner
    import torch
    from umallm.governor.native import NativePacedCopier

    n = args.mb * (1 << 20) // 2
    src = torch.randn(n, dtype=torch.float16, device="cuda:0")
    dst = torch.empty(n, dtype=torch.float16, device="cuda:1")
    train = Train(NativePacedCopier(device=0), src, dst, args.chunk_mb << 20)
    meter = SubwindowMeter(args.base)
    here = Path(__file__).resolve().parent

    def client_cmd(unique: bool, secs: float, label: str, out: Path | None):
        cmd = [args.client_py, str(here / "e2e_vllm_client.py"),
               "--model", args.model, "--concurrency", str(args.conc),
               "--secs", str(secs), "--max-tokens", "256", "--label", label]
        if unique:
            cmd.append("--unique-prefix")
        if out:
            cmd += ["--out", str(out)]
        return cmd

    cells = []
    t_exp0 = time.time()
    for rep in range(args.reps):
        for lifecycle in ("continuous", "restarted"):
            for prefix in ("cached", "unique"):
                uniq = prefix == "unique"
                tag = f"{lifecycle}-{prefix}-r{rep}"
                if lifecycle == "continuous":
                    # one client spans ramp + idle cell + ingress cell
                    span = 15 + 2 * (args.cell_secs + 6)
                    cl = subprocess.Popen(
                        client_cmd(uniq, span, tag, None),
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    time.sleep(15)                # past ramp by design
                    idle_rows = meter.run(args.cell_secs)
                    train.start(args.rate_gbs)
                    time.sleep(3)
                    ing_rows = meter.run(args.cell_secs)
                    train.stop()
                    cl.terminate(); cl.wait(timeout=20)
                    client_json = {}
                else:
                    # s10-style: fresh client per cell, measured FROM client
                    # start (ramp inside the window, like s10's 45s runs)
                    out_i = RESULTS / f"s12_cl_{tag}_idle.json"
                    cl = subprocess.Popen(
                        client_cmd(uniq, args.cell_secs + 5, tag + "-idle",
                                   out_i),
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    time.sleep(2)                 # streams connecting
                    idle_rows = meter.run(args.cell_secs)
                    cl.wait(timeout=60)
                    train.start(args.rate_gbs)    # ingress first, s10 order
                    time.sleep(3)
                    out_g = RESULTS / f"s12_cl_{tag}_ing.json"
                    cl = subprocess.Popen(
                        client_cmd(uniq, args.cell_secs + 5, tag + "-ing",
                                   out_g),
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    time.sleep(2)
                    ing_rows = meter.run(args.cell_secs)
                    cl.wait(timeout=60)
                    train.stop()
                    client_json = {}
                    for k, p in (("client_idle", out_i), ("client_ing", out_g)):
                        try:
                            client_json[k] = json.loads(p.read_text())
                        except Exception:
                            client_json[k] = None
                time.sleep(3)                     # settle between configs

                idle_med = med_tpot(idle_rows)
                ing_med = med_tpot(ing_rows)
                cell = {
                    "rep": rep, "lifecycle": lifecycle, "prefix": prefix,
                    "rate_gbs": args.rate_gbs,
                    "engine_idle_tpot_ms": idle_med,
                    "engine_ing_tpot_ms": ing_med,
                    "engine_inflation_pct": round(
                        (ing_med / idle_med - 1) * 100, 2)
                    if idle_med and ing_med else None,
                    # ramp split (meaningful for restarted cells)
                    "ing_early_med_ms": med_tpot(ing_rows, 0, 15),
                    "ing_late_med_ms": med_tpot(ing_rows, 15, 1e9),
                    "idle_early_med_ms": med_tpot(idle_rows, 0, 15),
                    "idle_late_med_ms": med_tpot(idle_rows, 15, 1e9),
                    "idle_series": idle_rows, "ing_series": ing_rows,
                }
                if client_json:
                    ci, cg = client_json.get("client_idle"), client_json.get(
                        "client_ing")
                    if ci and cg:
                        cell["client_inflation_pct"] = round(
                            (cg["tpot_ms_p50"] / ci["tpot_ms_p50"] - 1) * 100, 2)
                        cell["client_idle_p50"] = ci["tpot_ms_p50"]
                        cell["client_ing_p50"] = cg["tpot_ms_p50"]
                cells.append(cell)
                print(f"  {tag:24s} engine infl "
                      f"{cell['engine_inflation_pct']}% "
                      f"(idle {idle_med} -> ing {ing_med} ms) "
                      f"client infl {cell.get('client_inflation_pct', '-')}%",
                      flush=True)
    train.shutdown()

    out = {
        "_experiment": "s12_sensitivity_ab",
        "_is_measured": True,
        "_timing_method": ("ONE server instance; engine-side /metrics TPOT "
                           "in 5s sub-windows as common currency; client-side "
                           "p50 for restarted cells (s10's original metric); "
                           "fixed governed ingress via native paced train"),
        "victim": f"vLLM {args.model} conc={args.conc}, GPU1, receiver side",
        "rate_gbs": args.rate_gbs, "cell_secs": args.cell_secs,
        "reps": args.reps,
        "grid": "lifecycle{continuous,restarted} x prefix{cached,unique}",
        "cells": cells,
        "_generated_at": datetime.now(timezone.utc).isoformat(),
    }
    RESULTS.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
