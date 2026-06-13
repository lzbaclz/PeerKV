"""S5 -- fan-in admission: K senders converge on one busy decode receiver.

The second regime where the static rule fails by construction (g8 additive
law): K transfers that are each individually "do-no-harm" sum their
footprints on the shared victim.  Per-transfer rules cannot be sound here --
only an admission control that accounts the AGGREGATE per-endpoint rate can.
The governor's ledger does exactly that: K leases on the same receiver split
invert(eps), so the sum stays under budget no matter K.

Arms:
  unpaced    every sender fires whole-payload copies immediately (K
             bandwidth-greedy engines that cannot see each other)
  static     every sender paces itself to the SINGLE-transfer static cap
             (the per-transfer rule, sound for K=1, unsound for K>1)
  governor   one shared ledger; senders get additive leases

Needs >= K+1 GPUs with peer access to GPU0 -- on the dual-A100 box this
degenerates to K=1 (a smoke test); its real target is the 8xH100 HGX
partition (scripts/run_governor_hgx.sh).  NVSwitch note: on HGX every sender
has a dedicated link to the switch, so aggregate ingress at the victim is
NOT capped by one link -- exactly why fan-in escapes the single-link
structural cap that protects holders on 2-GPU bridges.
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
from _s_common import make_decoder  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "s5_fanin_admission.json"
CALIB = RESULTS / "governor_calib.json"
EPOCH_S = 0.25


def pctl(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--payload-mb", type=int, default=512)
    ap.add_argument("--chunk-mb", type=int, default=64)
    ap.add_argument("--eps-pct", type=float, default=5.0)
    ap.add_argument("--ks", type=int, nargs="+", default=None,
                    help="fan-in degrees; default = 1..device_count-1")
    ap.add_argument("--arms", nargs="+",
                    default=["unpaced", "static", "governor"])
    ap.add_argument("--window-secs", type=float, default=8.0)
    ap.add_argument("--baseline-secs", type=float, default=2.5)
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    from umallm.observability import gate_or_skip
    gate_or_skip("s5_fanin_admission")
    import torch
    from umallm.governor import Governor, GovernorCalibration, TransferRequest
    from umallm.governor.pacer import PacedCopier

    ndev = torch.cuda.device_count()
    assert ndev >= 2
    ks = args.ks or list(range(1, ndev))
    assert max(ks) <= ndev - 1, f"K={max(ks)} needs {max(ks)+1} GPUs, have {ndev}"
    assert CALIB.exists(), "run s1_governor_calib.py on this box first"
    cal = GovernorCalibration.load(CALIB)
    link = cal.link_peak_gbs["peer"]
    r_static = cal.curve("receiver", "peer", "worst").invert(args.eps_pct)
    print(f"devices={ndev} ks={ks} link={link:.0f}GB/s "
          f"per-transfer static cap={r_static:.1f}GB/s")

    dt = torch.float16
    n = args.payload_mb * (1 << 20) // 2
    srcs = {d: torch.randn(n, dtype=dt, device=f"cuda:{d}")
            for d in range(1, max(ks) + 1)}
    dsts = {d: torch.empty(n, dtype=dt, device="cuda:0")
            for d in range(1, max(ks) + 1)}
    chunk_bytes = args.chunk_mb << 20
    timed_iter = make_decoder("cuda:0", args.batch, args.ctx)

    cells = []
    for K in ks:
        for seed in range(args.seeds):
            torch.manual_seed(5000 + seed)
            for arm in args.arms:
                gov = None
                if arm == "governor":
                    gov = Governor(cal, eps_holder_pct=args.eps_pct,
                                   eps_receiver_pct=args.eps_pct, mode="ff",
                                   chunk_mb=args.chunk_mb, epoch_s=EPOCH_S)

                base = statistics.median(
                    [timed_iter() for _ in range(200)])

                # each sender saturates its own path for the whole window
                stop = threading.Event()
                handles, copiers = [], []

                def sender(d):
                    if gov is not None:
                        while not stop.is_set():
                            h = gov.submit(TransferRequest(
                                src=srcs[d], dst=dsts[d],
                                src_dev=d, dst_dev=0, route="peer"))
                            handles.append(h)
                            while not h.done.wait(0.05):
                                if stop.is_set():
                                    return
                    else:
                        c = PacedCopier(device=d)
                        copiers.append(c)
                        rate = r_static if arm == "static" else 0.0
                        while not stop.is_set():
                            c.run(srcs[d], dsts[d], lambda: rate, chunk_bytes,
                                  cancel=stop, unpaced=(arm == "unpaced"))

                threads = [threading.Thread(target=sender, args=(d,),
                                            daemon=True)
                           for d in range(1, K + 1)]
                t0 = time.monotonic()
                for t in threads:
                    t.start()
                time.sleep(0.6)                       # warm
                during = []
                t_end = time.monotonic() + args.window_secs
                while time.monotonic() < t_end:
                    during.append((time.monotonic(), timed_iter()))
                stop.set()
                for t in threads:
                    t.join(timeout=15)
                if gov is not None:
                    gov.shutdown()
                for c in copiers:
                    c.stream.synchronize()

                mss = [ms for _, ms in during]
                med = statistics.median(mss)
                epochs = collections.defaultdict(list)
                for t, ms in during:
                    epochs[int((t - t0) / EPOCH_S)].append(ms)
                viol = [statistics.median(v) / base - 1 > args.eps_pct / 100
                        for v in epochs.values() if len(v) >= 3]
                cell = {
                    "K": K, "seed": seed, "arm": arm,
                    "victim_med_slowdown_pct": round((med / base - 1) * 100, 2),
                    "victim_p99_slowdown_pct": round(
                        (pctl(mss, 0.99) / base - 1) * 100, 2),
                    "violation_frac": round(sum(viol) / len(viol), 3) if viol else None,
                    "baseline_ms": round(base, 4),
                    "iters": len(mss),
                }
                cells.append(cell)
                print(f"  K={K} s{seed} {arm:9s} "
                      f"victim med +{cell['victim_med_slowdown_pct']:6.2f}% "
                      f"p99 +{cell['victim_p99_slowdown_pct']:6.2f}% "
                      f"viol={cell['violation_frac']}", flush=True)

    out = {
        "_experiment": "s5_fanin_admission",
        "_is_measured": True,
        "_timing_method": ("victim-decode-stream CUDA events; K saturating "
                           "senders; no whole-device sync"),
        "device": torch.cuda.get_device_name(0),
        "device_count": ndev, "ks": ks, "batch": args.batch, "ctx": args.ctx,
        "payload_mb": args.payload_mb, "eps_pct": args.eps_pct,
        "per_transfer_static_cap_gbs": round(r_static, 1),
        "window_secs": args.window_secs, "seeds": args.seeds,
        "cells": cells,
        "note": ("K=1 on a 2-GPU box is a smoke test; the claim under test "
                 "(static per-transfer caps sum past eps, governor's additive "
                 "ledger does not) needs K>=2 on an NVSwitch partition"),
        "_generated_at": datetime.now(timezone.utc).isoformat(),
    }
    RESULTS.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
