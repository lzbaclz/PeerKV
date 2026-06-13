"""S6 -- mixed-route fan-in: the one local experiment a static rule cannot pass.

Adversarial review (static-rule advocate) pointed out that on a 2-GPU box the
shared ledger never does anything a per-route static cap cannot -- except in
ONE configuration that needs no extra GPUs: two ROUTES converging on the same
victim HBM port.  NVLink peer ingress (GPU1->GPU0) and PCIe host ingress
(pinned->GPU0) both write the victim's HBM, but a GB/s of host traffic costs
~5-8x a GB/s of peer traffic (s1), so:

  independent-caps  each route paced at ITS OWN single-route inversion
                    (peer: invert_peer(eps); host: invert_host(eps)) -- each
                    cap is sound in isolation, and a per-route/per-transfer
                    static rule has no way to know the other route exists.
                    If victim costs add across routes (g8 law), the joint
                    cost runs ~2x eps.
  shared-ledger     the governor's cost-space ledger admits both flows
                    against ONE eps budget (sum of per-route costs <= eps).
  unpaced-both      both routes at full rate (context: the unprotected sum).

Victim: b1 decode on GPU0 (the sensitive bucket).  Protocol as s1/s2:
victim-stream CUDA events, idle-machinery baseline, no whole-device sync,
clocks locked, box gated.  Requires governor_calib.json (s1).
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _s_common import Train, make_decoder  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "s6_mixed_route_ledger.json"
CALIB = RESULTS / "governor_calib.json"
EPOCH_S = 0.25


def pctl(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--eps-pct", type=float, default=5.0)
    ap.add_argument("--chunk-mb", type=int, default=64)
    ap.add_argument("--buf-mb", type=int, default=512)
    ap.add_argument("--window-secs", type=float, default=8.0)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--arms", nargs="+",
                    default=["independent-caps", "shared-ledger",
                             "unpaced-both"])
    ap.add_argument("--pacer", choices=["auto", "python", "native"],
                    default="auto")
    args = ap.parse_args()

    from umallm.observability import gate_or_skip
    gate_or_skip("s6_mixed_route_ledger")
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
    assert CALIB.exists(), "run s1_governor_calib.py first"
    cal = GovernorCalibration.load(CALIB)
    cap_peer = cal.curve("receiver", "peer", "worst").invert(args.eps_pct)
    cap_host = cal.curve("receiver", "host", "worst").invert(args.eps_pct)
    pred_joint = (cal.curve("receiver", "peer", "worst").predict(cap_peer)
                  + cal.curve("receiver", "host", "worst").predict(cap_host))
    print(f"single-route caps: peer={cap_peer:.1f} host={cap_host:.1f} GB/s; "
          f"central predicted JOINT cost if both run: {pred_joint:.1f}% "
          f"(eps={args.eps_pct}%)")

    dt = torch.float16
    n = args.buf_mb * (1 << 20) // 2
    src_peer = torch.randn(n, dtype=dt, device="cuda:1")
    src_host = torch.randn(n, dtype=dt, device="cpu", pin_memory=True)
    dst_a = torch.empty(n, dtype=dt, device="cuda:0")
    dst_b = torch.empty(n, dtype=dt, device="cuda:0")
    chunk_bytes = args.chunk_mb << 20
    timed_iter = make_decoder("cuda:0", args.batch, args.ctx)

    cells = []
    for rep in range(args.reps):
        for arm in args.arms:
            base = statistics.median([timed_iter() for _ in range(200)])
            gov, trains, senders, stop = None, [], [], None
            if arm == "shared-ledger":
                gov = Governor(cal, eps_holder_pct=args.eps_pct,
                               eps_receiver_pct=args.eps_pct, mode="ff",
                               chunk_mb=args.chunk_mb, epoch_s=EPOCH_S,
                               pacer=args.pacer)
                import threading
                stop = threading.Event()
                delivered = collections.Counter()

                def sender(route, src, dst, src_dev):
                    while not stop.is_set():
                        h = gov.submit(TransferRequest(
                            src=src, dst=dst, src_dev=src_dev, dst_dev=0,
                            route=route))
                        while not h.done.wait(0.05):
                            if stop.is_set():
                                return
                        if h.result is not None:
                            delivered[route] += h.result.bytes_launched

                senders = [threading.Thread(target=sender, daemon=True,
                                            args=("peer", src_peer, dst_a, 1)),
                           threading.Thread(target=sender, daemon=True,
                                            args=("host", src_host, dst_b,
                                                  None))]
                for s in senders:
                    s.start()
            else:
                rate_p = 0.0 if arm == "unpaced-both" else cap_peer
                rate_h = 0.0 if arm == "unpaced-both" else cap_host
                t_p = Train(pacer_cls(device=1), src_peer, dst_a, chunk_bytes)
                t_h = Train(pacer_cls(device=0), src_host, dst_b, chunk_bytes)
                # rate 0.0 in Train means "no gap" only with unpaced runs;
                # emulate unpaced by pacing at link-exceeding rates
                t_p.start(rate_p if rate_p > 0 else 10000.0)
                t_h.start(rate_h if rate_h > 0 else 10000.0)
                trains = [t_p, t_h]

            time.sleep(0.6)
            t0 = time.monotonic()
            during = []
            while time.monotonic() < t0 + args.window_secs:
                during.append((time.monotonic(), timed_iter()))
            t1 = time.monotonic()

            rates = {}
            if trains:
                rates["peer_gbs"] = round(trains[0].stop(), 1)
                rates["host_gbs"] = round(trains[1].stop(), 1)
                for t in trains:
                    t.shutdown()
            else:
                stop.set()
                for s in senders:
                    s.join(timeout=10)
                gov.shutdown()
                win = t1 - t0 + 0.6
                rates["peer_gbs"] = round(delivered["peer"] / win / 1e9, 1)
                rates["host_gbs"] = round(delivered["host"] / win / 1e9, 1)
                rates["deferral_polls"] = gov.report()["stats"].get(
                    "deferral_polls", 0)

            mss = [ms for _, ms in during]
            med = statistics.median(mss)
            epochs = collections.defaultdict(list)
            for t, ms in during:
                epochs[int((t - t0) / EPOCH_S)].append(ms)
            viol = [statistics.median(v) / base - 1 > args.eps_pct / 100
                    for v in epochs.values() if len(v) >= 3]
            cell = {"rep": rep, "arm": arm,
                    "victim_med_slowdown_pct": round((med / base - 1) * 100, 2),
                    "victim_p99_slowdown_pct": round(
                        (pctl(mss, 0.99) / base - 1) * 100, 2),
                    "violation_frac": round(sum(viol) / len(viol), 3) if viol else None,
                    "baseline_ms": round(base, 4), **rates}
            cells.append(cell)
            print(f"  rep{rep} {arm:17s} victim med "
                  f"+{cell['victim_med_slowdown_pct']:6.2f}% "
                  f"p99 +{cell['victim_p99_slowdown_pct']:6.2f}% "
                  f"viol={cell['violation_frac']} "
                  f"peer={cell['peer_gbs']} host={cell['host_gbs']} GB/s",
                  flush=True)

    out = {
        "_experiment": "s6_mixed_route_ledger",
        "_is_measured": True,
        "_timing_method": ("victim-decode-stream CUDA events; two concurrent "
                           "ingress routes; idle-machinery baseline; no "
                           "whole-device sync; clocks locked"),
        "device": torch.cuda.get_device_name(0),
        "geometry": "Llama-3-8B GQA (random weights)",
        "ctx": args.ctx, "batch": args.batch, "eps_pct": args.eps_pct,
        "single_route_caps_gbs": {"peer": round(cap_peer, 1),
                                  "host": round(cap_host, 1)},
        "predicted_joint_cost_if_both_capped_pct": round(pred_joint, 1),
        "window_secs": args.window_secs, "reps": args.reps,
        "pacer": pacer_kind,
        "cells": cells,
        "note": ("independent-caps is sound per route and unsound jointly iff "
                 "victim costs add across routes; shared-ledger admits both "
                 "flows against one eps budget in cost space"),
        "_generated_at": datetime.now(timezone.utc).isoformat(),
    }
    RESULTS.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
