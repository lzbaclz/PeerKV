"""S1 -- governor calibration census: REAL paced-copy chunk trains, both
endpoints, both polarities, swept delivered rate.

Why this exists (the law repair the system is gated on): the g10 curves were
measured with a paced *injector* (local read/write kernels), and they mis-price
real copies at both endpoints -- g11 measured the receiver at +23.5% where the
g10 write curve predicts +11.8%, and g7 measured the sustained holder at +1.1%
where the read curve predicts ~4%.  A governor that admits against the
injector curve would violate its receiver budget 2x and strand half the holder
budget.  S1 re-measures the footprint->victim-cost law with the SAME actuator
the governor deploys (PacedCopier chunk trains of real cudaMemcpy), per

    role x route x rate x victim-workload
    role  = holder  (copy READS victim HBM;  victim decodes on the source)
            receiver(copy WRITES victim HBM; victim decodes on the destination)
    route = peer (GPU1->GPU0 NVLink) | host (PCIe, pinned)
    rate  = fractions of the measured route bandwidth
    workload = decode batch (b1 = maximally sensitive, b8 = loaded)

and emits (a) the census JSON with per-point g10-injector predictions so the
injector-vs-real-copy residual is itself a result, and (b) the fitted
GovernorCalibration consumed by the ledger (--fit, on by default).

Protocol: per-iteration CUDA events on the victim's decode stream only; the
pacing clock threads across buffer wraps (no boundary bursts); baseline is
taken with all train machinery (sender thread, estimator poller) alive but
idle, so the arms differ only in traffic; clocks locked; box gated.
"""
from __future__ import annotations

import argparse
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
OUT = RESULTS / "s1_governor_calib.json"
CALIB_OUT = RESULTS / "governor_calib.json"
G10 = RESULTS / "g10_paced_sweep.json"


def g10_prediction(frac: float, polarity: str):
    """Interpolate the g10 *injector* curve (read|write) at footprint frac."""
    if not G10.exists():
        return None
    g10 = json.loads(G10.read_text())
    pts = sorted((v["achieved_frac"], v["victim_slowdown_pct"])
                 for k, v in g10["summary"].items()
                 if k.startswith(f"{polarity}@"))
    if not pts:
        return None
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= frac <= x1:
            return round(y0 + (frac - x0) / (x1 - x0) * (y1 - y0), 2)
    return round(pts[0][1] * frac / pts[0][0] if frac < pts[0][0] else pts[-1][1], 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--peer-fracs", type=float, nargs="+",
                    default=[0.125, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--host-fracs", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--chunk-mb", type=int, default=64)
    ap.add_argument("--buf-mb", type=int, default=512)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--base-iters", type=int, default=200)
    ap.add_argument("--train-secs", type=float, default=4.0)
    ap.add_argument("--warm-secs", type=float, default=0.6)
    ap.add_argument("--no-fit", action="store_true")
    ap.add_argument("--pacer", choices=["auto", "python", "native"],
                    default="auto",
                    help="actuator for the census (must match deployment)")
    args = ap.parse_args()

    from umallm.observability import gate_or_skip
    gate_or_skip("s1_governor_calib")
    import torch
    from umallm.governor.pacer import PacedCopier
    from umallm.governor.calib import Curve, CurvePoint, GovernorCalibration

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

    assert torch.cuda.device_count() >= 2, "census needs both A100s"
    dt = torch.float16

    # ----- transfer buffers ------------------------------------------------
    n = args.buf_mb * (1 << 20) // 2
    chunk_bytes = args.chunk_mb << 20
    src_d1 = torch.randn(n, dtype=dt, device="cuda:1")
    dst_d0 = torch.empty(n, dtype=dt, device="cuda:0")
    src_host = torch.randn(n, dtype=dt, device="cpu", pin_memory=True)
    dst_host = torch.empty(n, dtype=dt, device="cpu", pin_memory=True)

    routes = {
        # (role, route): victim_dev, copier_dev, src, dst
        ("receiver", "peer"): ("cuda:0", 1, src_d1, dst_d0),
        ("holder", "peer"): ("cuda:1", 1, src_d1, dst_d0),
        ("receiver", "host"): ("cuda:0", 0, src_host, dst_d0),
        ("holder", "host"): ("cuda:1", 1, src_d1, dst_host),
    }

    # ----- route bandwidths (unpaced chunk train, 3 reps) -------------------
    def route_bw(copier_dev, src, dst) -> float:
        c = pacer_cls(device=copier_dev)
        for _ in range(2):
            c.run(src, dst, lambda: 0.0, chunk_bytes, unpaced=True)
        bws = []
        for _ in range(3):
            r = c.run(src, dst, lambda: 0.0, chunk_bytes, unpaced=True)
            bws.append(r.achieved_gbs)
        return statistics.median(bws)

    link_bw = {}
    for (role, route), (vdev, cdev, src, dst) in routes.items():
        bw_key = route if route == "peer" else f"host_{role}"
        if bw_key not in link_bw:
            link_bw[bw_key] = route_bw(cdev, src, dst)
    print(f"route bandwidths: " + ", ".join(
        f"{k}={v:.1f}GB/s" for k, v in link_bw.items()))

    hbm_peak = 2039.0
    points = []
    t_exp0 = time.time()

    for (role, route), (vdev, cdev, src, dst) in routes.items():
        bw_key = route if route == "peer" else f"host_{role}"
        bw = link_bw[bw_key]
        fracs = args.peer_fracs if route == "peer" else args.host_fracs
        copier = pacer_cls(device=cdev)
        train = Train(copier, src, dst, chunk_bytes)
        for batch in args.batches:
            timed_iter = make_decoder(vdev, batch, args.ctx)
            for seed in range(args.seeds):
                torch.manual_seed(7000 + seed)
                # baseline with machinery alive but idle
                base = statistics.median(
                    [timed_iter() for _ in range(args.base_iters)])
                for frac in fracs:
                    rate = frac * bw
                    train.start(rate)
                    time.sleep(args.warm_secs)
                    t_end = time.monotonic() + args.train_secs
                    during = []
                    while time.monotonic() < t_end:
                        during.append(timed_iter())
                    achieved = train.stop()
                    med = statistics.median(during)
                    during.sort()
                    p99 = during[min(len(during) - 1,
                                     int(0.99 * len(during)))]
                    polarity = "read" if role == "holder" else "write"
                    pt = {
                        "role": role, "route": route, "workload": f"b{batch}",
                        "seed": seed, "target_rate_gbs": round(rate, 1),
                        "achieved_rate_gbs": round(achieved, 1),
                        "footprint_frac": round(achieved / hbm_peak, 4),
                        "victim_slowdown_pct": round((med / base - 1) * 100, 2),
                        "victim_p99_slowdown_pct": round(
                            (p99 / base - 1) * 100, 2),
                        "iters_during": len(during),
                        "baseline_ms": round(base, 4),
                        "g10_injector_pred_pct": g10_prediction(
                            achieved / hbm_peak, polarity),
                    }
                    points.append(pt)
                    print(f"  {role:8s}/{route:4s} b{batch} s{seed} "
                          f"rate={achieved:6.1f}/{rate:6.1f}GB/s "
                          f"victim=+{pt['victim_slowdown_pct']:6.2f}% "
                          f"(p99 +{pt['victim_p99_slowdown_pct']:6.2f}%) "
                          f"g10pred={pt['g10_injector_pred_pct']}")
        train.shutdown()

    # ----- aggregate + fit --------------------------------------------------
    def cells():
        seen = {}
        for p in points:
            k = (p["role"], p["route"], p["workload"], p["target_rate_gbs"])
            seen.setdefault(k, []).append(p)
        return seen

    summary = {}
    cal = GovernorCalibration(
        device=torch.cuda.get_device_name(0), hbm_peak_gbs=hbm_peak,
        link_peak_gbs={"peer": round(link_bw["peer"], 1),
                       "host": round(min(link_bw["host_receiver"],
                                         link_bw["host_holder"]), 1)},
        meta={"source": "s1_governor_calib", "pacer": pacer_kind,
              "chunk_mb": args.chunk_mb,
              "ctx": args.ctx, "geometry": "Llama-3-8B GQA (random weights)"})
    curve_pts: dict[tuple, list] = {}
    for (role, route, wl, tgt), ps in sorted(cells().items()):
        slows = [p["victim_slowdown_pct"] for p in ps]
        rates = [p["achieved_rate_gbs"] for p in ps]
        mean_slow = statistics.mean(slows)
        spread = (max(slows) - min(slows)) if len(slows) > 1 else 0.0
        summary[f"{role}/{route}/{wl}@{tgt}"] = {
            "victim_slowdown_pct": round(mean_slow, 2),
            "spread_pp": round(spread, 2),
            "achieved_rate_gbs": round(statistics.mean(rates), 1),
            "g10_injector_pred_pct": ps[0]["g10_injector_pred_pct"],
        }
        curve_pts.setdefault((role, route, wl), []).append(
            CurvePoint(rate_gbs=round(statistics.mean(rates), 1),
                       victim_pct=round(mean_slow, 2),
                       spread_pp=round(spread, 2)))
        base_k = f"{role}/{wl}"
        cal.baseline_ms.setdefault(base_k, round(statistics.mean(
            p["baseline_ms"] for p in ps), 4))
    for (role, route, wl), cps in curve_pts.items():
        cal.add_curve(Curve(role=role, route=route, workload=wl, points=cps))

    out = {
        "_experiment": "s1_governor_calib",
        "_is_measured": True,
        "_timing_method": ("victim-decode-stream CUDA events; sustained "
                           "PacedCopier chunk train (the deployed actuator); "
                           "baseline taken with idle machinery; no "
                           "whole-device sync; clocks locked"),
        "device": torch.cuda.get_device_name(0),
        "geometry": "Llama-3-8B GQA (random weights)",
        "ctx": args.ctx, "batches": args.batches,
        "chunk_mb": args.chunk_mb, "buf_mb": args.buf_mb,
        "pacer": pacer_kind,
        "train_secs": args.train_secs, "seeds": args.seeds,
        "route_bandwidths_gbs": {k: round(v, 1) for k, v in link_bw.items()},
        "summary": summary,
        "points": points,
        "wall_minutes": round((time.time() - t_exp0) / 60, 1),
        "note": ("real-copy census replacing the g10 injector curves; the "
                 "g10_injector_pred_pct column is the injector-vs-real-copy "
                 "residual finding"),
        "_generated_at": datetime.now(timezone.utc).isoformat(),
    }
    RESULTS.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"-> {OUT}")
    if not args.no_fit:
        cal.save(CALIB_OUT)
        print(f"-> {CALIB_OUT} (curves: {sorted(cal.curves)})")


if __name__ == "__main__":
    main()
