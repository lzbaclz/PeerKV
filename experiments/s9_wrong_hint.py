"""S9 -- hints are advisory, never load-bearing: wrong-hint failure modes.

The static-rule advocate's per-class table proposal ({class: cap}) is the
hint API's zero-feedback equivalent -- and on this box the native census
defused its premise (receiver write-polarity cost is class-insensitive, so
the b1/b8 caps nearly coincide).  What remains decisive locally is the
RELIABILITY contrast when the class signal is wrong, using the strongest
class split that exists: idle vs busy.

Scenario: the engine believes GPU0 is idle (hint='idle') -- true for the
first half of the window, wrong for the second (a decode arrives).  A static
table acts on the hint; the governor treats unknown buckets as 'worst' by
design (calib.curve falls back), so a wrong hint costs it goodput, never the
contract.

Arms:
  static-table  trusts the hint: idle => uncapped ingress (this IS what a
                per-class table does with class='idle') -- fail-DEADLY when
                the decode arrives
  governor-ff   same wrong hint passed via workload_hint; unknown bucket =>
                pointwise-max fallback -- fail-SAFE, pays goodput in the
                truly-idle phase
  governor-fb   + victim feed once the decode exists (no feed while idle --
                the known feed-less limitation applies to phase A)

Phase A (0-8s, victim idle): metric = delivered GB/s (the price of safety).
Phase B (8-16s, b1 decode):  metric = victim median + violation_frac.
Protocol as s2; box gated; locked clocks.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _s_common import Train, make_decoder  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "s9_wrong_hint.json"
CALIB = RESULTS / "governor_calib.json"
EPOCH_S = 0.25


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--eps-pct", type=float, default=5.0)
    ap.add_argument("--chunk-mb", type=int, default=64)
    ap.add_argument("--buf-mb", type=int, default=512)
    ap.add_argument("--phase-secs", type=float, default=8.0)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--pacer", choices=["auto", "python", "native"],
                    default="auto")
    args = ap.parse_args()

    from umallm.observability import gate_or_skip
    gate_or_skip("s9_wrong_hint")
    import torch
    from umallm.governor import Governor, GovernorCalibration, TransferRequest
    from umallm.governor.pacer import PacedCopier

    pacer_cls, pacer_kind = PacedCopier, "python"
    if args.pacer in ("auto", "native"):
        try:
            from umallm.governor.native import NativePacedCopier, available
            if available():
                pacer_cls, pacer_kind = NativePacedCopier, "native"
            elif args.pacer == "native":
                raise RuntimeError("native pacer unavailable")
        except ImportError:
            if args.pacer == "native":
                raise
    print(f"pacer: {pacer_kind}")

    assert torch.cuda.device_count() >= 2
    cal = GovernorCalibration.load(CALIB)
    cap_worst = cal.curve("receiver", "peer", "worst").invert(args.eps_pct)
    print(f"worst cap {cap_worst:.1f} GB/s; per-class caps for context: "
          + ", ".join(f"{w}={cal.curve('receiver','peer',w).invert(args.eps_pct):.1f}"
                      for w in cal.workloads("receiver", "peer")))

    dt = torch.float16
    n = args.buf_mb * (1 << 20) // 2
    src = torch.randn(n, dtype=dt, device="cuda:1")
    dst = torch.empty(n, dtype=dt, device="cuda:0")
    chunk_bytes = args.chunk_mb << 20
    timed_iter = make_decoder("cuda:0", args.batch, args.ctx)
    HINT = {0: "idle"}                       # the wrong-in-phase-B signal

    cells = []
    for rep in range(args.reps):
        for arm in ["static-table", "governor-ff", "governor-fb"]:
            base = statistics.median([timed_iter() for _ in range(200)])
            gov, train, stop = None, None, None
            delivered = [0]
            if arm == "static-table":
                train = Train(pacer_cls(device=1), src, dst, chunk_bytes)
                train.start(10000.0)         # hint=idle => table says no cap
            else:
                gov = Governor(cal, eps_holder_pct=args.eps_pct,
                               eps_receiver_pct=args.eps_pct,
                               mode="ff" if arm.endswith("ff") else "fb",
                               chunk_mb=args.chunk_mb, epoch_s=EPOCH_S,
                               pacer=args.pacer)
                stop = threading.Event()

                def sender():
                    while not stop.is_set():
                        h = gov.submit(TransferRequest(
                            src=src, dst=dst, src_dev=1, dst_dev=0,
                            route="peer",
                            meta={"workload_hint": HINT}))
                        while not h.done.wait(0.05):
                            if stop.is_set():
                                return
                        if h.result is not None:
                            delivered[0] += h.result.bytes_launched
                if arm.endswith("fb"):       # clean-baseline epochs first
                    t_pre = time.monotonic() + 1.2
                    while time.monotonic() < t_pre:
                        gov.feed_victim(0, timed_iter())
                threading.Thread(target=sender, daemon=True).start()

            # ---- phase A: victim idle ---------------------------------
            time.sleep(0.4)
            tA0 = time.monotonic()
            time.sleep(args.phase_secs)
            tA1 = time.monotonic()
            if train is not None:
                a_gbs = train.bytes_launched / (tA1 - train.t_start) / 1e9
            else:
                a_gbs = delivered[0] / (tA1 - tA0) / 1e9

            # ---- phase B: decode arrives (hint now wrong) --------------
            during = []
            tB0 = time.monotonic()
            while time.monotonic() < tB0 + args.phase_secs:
                ms = timed_iter()
                during.append((time.monotonic(), ms))
                if gov is not None and arm.endswith("fb"):
                    gov.feed_victim(0, ms)
            tB1 = time.monotonic()

            if train is not None:
                train.stop()
                train.shutdown()
            else:
                stop.set()
                gov.shutdown()

            mss = [ms for _, ms in during]
            med = statistics.median(mss)
            epochs = collections.defaultdict(list)
            for t, ms in during:
                epochs[int((t - tB0) / EPOCH_S)].append(ms)
            viol = [statistics.median(v) / base - 1 > args.eps_pct / 100
                    for v in epochs.values() if len(v) >= 3]
            cell = {
                "rep": rep, "arm": arm,
                "phaseA_delivered_gbs": round(a_gbs, 1),
                "phaseB_victim_med_pct": round((med / base - 1) * 100, 2),
                "phaseB_viol_frac": round(sum(viol) / len(viol), 3) if viol else None,
                "baseline_ms": round(base, 4),
            }
            cells.append(cell)
            print(f"  rep{rep} {arm:13s} A_delivered={cell['phaseA_delivered_gbs']:6.1f}GB/s "
                  f"B_med +{cell['phaseB_victim_med_pct']:5.2f}% "
                  f"B_viol={cell['phaseB_viol_frac']}", flush=True)

    out = {
        "_experiment": "s9_wrong_hint",
        "_is_measured": True,
        "_timing_method": ("victim-decode-stream CUDA events; two-phase "
                           "window (idle then decode) under a frozen "
                           "hint='idle'; locked clocks; box gated"),
        "device": torch.cuda.get_device_name(0),
        "ctx": args.ctx, "batch": args.batch, "eps_pct": args.eps_pct,
        "pacer": pacer_kind, "phase_secs": args.phase_secs,
        "worst_cap_gbs": round(cap_worst, 1),
        "reps": args.reps,
        "cells": cells,
        "note": ("a per-class static table is fail-deadly under a wrong "
                 "class signal; the governor's unknown-bucket fallback "
                 "converts the same wrong hint into bounded goodput loss. "
                 "The reviewer's original b1-vs-b8 table premise dissolved "
                 "with the native census (receiver cost is class-insensitive"
                 ", caps nearly equal) -- recorded here for the report."),
        "_generated_at": datetime.now(timezone.utc).isoformat(),
    }
    RESULTS.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
