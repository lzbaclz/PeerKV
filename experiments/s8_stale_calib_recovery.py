"""S8 -- stale-calibration recovery: the closed loop's case against ALL
static tables.

The static-rule advocate's strongest remaining objection: "a static per-class
cap table derived offline from the same census matches your protection; the
feedback loop adds little."  True -- WHEN the table's constants are right.
This experiment makes them wrong the way they would actually be wrong: it
loads the g10 *injector* curves (the calibration this project would have
shipped before the s1 real-copy census), which under-price receiver-side
write traffic ~2x (g11: predicted +11.8% where +23.5% was measured).  Every
static artifact of that calibration -- cap tables, ff governors -- violates
persistently and has no mechanism to notice.  The fb governor starts from the
same wrong prior and must converge to <=eps using only engine-reported
iteration times.

Arms (b1 victim, sustained saturating ingress GPU1->GPU0):
  static-stale   Train paced at the stale worst-bucket inversion -- the
                 per-class static table built from the wrong census
  ff-stale       Governor(ff) admitting against the stale curves
  fb-stale       Governor(fb), same wrong prior + victim feed -- the claim
                 under test: backs off to <=eps within a few epochs
  fb-fresh       Governor(fb) with the real (native-actuator) calibration --
                 reference for where fb should land

Reported per arm: per-epoch victim-median timeline, violation_frac split
into early (first quarter) vs late (last half) windows, time-to-compliance
(first epoch from which all subsequent epoch medians stay <=eps), trim
trajectory.  Protocol as s1/s2 (victim-stream events, idle-machinery
baseline, no whole-device sync, locked clocks, box gated).
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
OUT = RESULTS / "s8_stale_calib_recovery.json"
CALIB = RESULTS / "governor_calib.json"
G10 = RESULTS / "g10_paced_sweep.json"
EPOCH_S = 0.25


def build_stale_calib(real):
    """GovernorCalibration from the g10 injector curves (pre-s1 world)."""
    from umallm.governor.calib import Curve, CurvePoint, GovernorCalibration
    g10 = json.loads(G10.read_text())
    cal = GovernorCalibration(
        device=g10.get("device", "A100 (g10 injector)"),
        hbm_peak_gbs=g10.get("hbm_peak_gbs", 2039.0),
        link_peak_gbs={"peer": real.link_peak_gbs["peer"]},
        meta={"source": "g10_paced_sweep (injector -- deliberately stale)"})
    mix_to_role = {"write": "receiver", "read": "holder"}
    pts: dict[str, list] = {"receiver": [], "holder": []}
    for k, v in g10["summary"].items():
        mix = k.split("@")[0]
        if mix in mix_to_role:
            pts[mix_to_role[mix]].append(CurvePoint(
                rate_gbs=v["achieved_gbs"],
                victim_pct=v["victim_slowdown_pct"], spread_pp=0.5))
    for role, ps in pts.items():
        cal.add_curve(Curve(role=role, route="peer", workload="b1",
                            points=sorted(ps, key=lambda p: p.rate_gbs)))
    return cal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--eps-pct", type=float, default=5.0)
    ap.add_argument("--chunk-mb", type=int, default=64)
    ap.add_argument("--buf-mb", type=int, default=512)
    ap.add_argument("--window-secs", type=float, default=16.0)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--pacer", choices=["auto", "python", "native"],
                    default="auto")
    args = ap.parse_args()

    from umallm.observability import gate_or_skip
    gate_or_skip("s8_stale_calib_recovery")
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
    assert CALIB.exists() and G10.exists(), "need s1 calib + g10 injector data"
    real = GovernorCalibration.load(CALIB)
    stale = build_stale_calib(real)
    cap_real = real.curve("receiver", "peer", "worst").invert(args.eps_pct)
    cap_stale = stale.curve("receiver", "peer", "worst").invert(args.eps_pct)
    pred_true_at_stale = real.curve("receiver", "peer", "worst").predict(
        cap_stale)
    print(f"caps: real={cap_real:.1f} stale={cap_stale:.1f} GB/s; true cost "
          f"at the stale cap (real curve): {pred_true_at_stale:.1f}% "
          f"(eps={args.eps_pct}%)")

    dt = torch.float16
    n = args.buf_mb * (1 << 20) // 2
    src = torch.randn(n, dtype=dt, device="cuda:1")
    dst = torch.empty(n, dtype=dt, device="cuda:0")
    chunk_bytes = args.chunk_mb << 20
    timed_iter = make_decoder("cuda:0", args.batch, args.ctx)

    def comply_epoch(epoch_meds, base):
        """First epoch index from which all later epoch medians <= eps."""
        ok = [statistics.median(v) / base - 1 <= args.eps_pct / 100
              for _, v in sorted(epoch_meds.items()) if len(v) >= 3]
        for i in range(len(ok)):
            if all(ok[i:]):
                return i
        return None

    cells = []
    for rep in range(args.reps):
        for arm in ["static-stale", "ff-stale", "fb-stale", "fb-fresh"]:
            base = statistics.median([timed_iter() for _ in range(200)])
            gov, train, stop, senders = None, None, None, []
            if arm == "static-stale":
                train = Train(pacer_cls(device=1), src, dst, chunk_bytes)
                train.start(cap_stale)
            else:
                cal = real if arm == "fb-fresh" else stale
                gov = Governor(cal, eps_holder_pct=args.eps_pct,
                               eps_receiver_pct=args.eps_pct,
                               mode="ff" if arm == "ff-stale" else "fb",
                               chunk_mb=args.chunk_mb, epoch_s=EPOCH_S,
                               pacer=args.pacer)
                stop = threading.Event()
                done_bytes = [0]

                def sender():
                    while not stop.is_set():
                        h = gov.submit(TransferRequest(
                            src=src, dst=dst, src_dev=1, dst_dev=0,
                            route="peer"))
                        while not h.done.wait(0.05):
                            if stop.is_set():
                                return
                        if h.result is not None:
                            done_bytes[0] += h.result.bytes_launched

                th = threading.Thread(target=sender, daemon=True)
                senders = [th]
                # fb needs a clean-baseline epoch before traffic: feed the
                # victim during a short pre-window (same as s2's baseline)
                if arm.startswith("fb"):
                    t_pre = time.monotonic() + 1.2
                    while time.monotonic() < t_pre:
                        gov.feed_victim(0, timed_iter())
                th.start()

            time.sleep(0.4)
            t0 = time.monotonic()
            during = []
            while time.monotonic() < t0 + args.window_secs:
                ms = timed_iter()
                during.append((time.monotonic(), ms))
                if gov is not None and arm.startswith("fb"):
                    gov.feed_victim(0, ms)
            t1 = time.monotonic()
            if train is not None:
                ach = train.stop()
                train.shutdown()
                trim_hist = None
            else:
                stop.set()
                for s_ in senders:
                    s_.join(timeout=10)
                gov.shutdown()
                ach = done_bytes[0] / (t1 - t0) / 1e9
                tr = gov.trims.get(0)
                trim_hist = tr.state.history[:120] if tr else None

            epochs = collections.defaultdict(list)
            for t, ms in during:
                epochs[int((t - t0) / EPOCH_S)].append(ms)
            meds = {e: statistics.median(v) / base - 1
                    for e, v in epochs.items() if len(v) >= 3}
            viols = {e: m > args.eps_pct / 100 for e, m in meds.items()}
            n_ep = len(viols)
            early = [viols[e] for e in sorted(viols)[: max(1, n_ep // 4)]]
            late = [viols[e] for e in sorted(viols)[n_ep // 2:]]
            cell = {
                "rep": rep, "arm": arm,
                "victim_med_slowdown_pct": round(
                    (statistics.median([m for _, m in during]) / base - 1)
                    * 100, 2),
                "viol_early_frac": round(sum(early) / len(early), 3),
                "viol_late_frac": round(sum(late) / len(late), 3),
                "compliance_epoch": comply_epoch(epochs, base),
                "delivered_gbs": round(ach, 1),
                "baseline_ms": round(base, 4),
                "epoch_med_pct_first16": [round(meds[e] * 100, 2)
                                          for e in sorted(meds)[:16]],
                "trim_history": trim_hist,
            }
            cells.append(cell)
            print(f"  rep{rep} {arm:13s} med +{cell['victim_med_slowdown_pct']:5.2f}% "
                  f"viol early={cell['viol_early_frac']} late={cell['viol_late_frac']} "
                  f"comply@ep{cell['compliance_epoch']} "
                  f"delivered={cell['delivered_gbs']}GB/s", flush=True)

    out = {
        "_experiment": "s8_stale_calib_recovery",
        "_is_measured": True,
        "_timing_method": ("victim-decode-stream CUDA events; saturating "
                           "submit loop; idle-machinery baseline; no "
                           "whole-device sync; clocks locked"),
        "device": torch.cuda.get_device_name(0),
        "ctx": args.ctx, "batch": args.batch, "eps_pct": args.eps_pct,
        "pacer": pacer_kind,
        "cap_real_gbs": round(cap_real, 1),
        "cap_stale_gbs": round(cap_stale, 1),
        "true_cost_at_stale_cap_pct": round(pred_true_at_stale, 1),
        "window_secs": args.window_secs, "reps": args.reps,
        "epoch_s": EPOCH_S,
        "cells": cells,
        "note": ("static artifacts of a wrong calibration violate forever; "
                 "fb converges from the same wrong prior using only the "
                 "victim feed -- the in-principle advantage no static table "
                 "has"),
        "_generated_at": datetime.now(timezone.utc).isoformat(),
    }
    RESULTS.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
