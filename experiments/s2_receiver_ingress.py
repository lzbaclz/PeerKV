"""S2 -- the headline: continuous KV ingress into a busy decode RECEIVER.

The regime where the static rule measurably fails (and the one no shipping
transfer engine paces): in PD-disaggregation the decode GPU is a busy
*receiver* -- every admitted request lands GBs of KV in its HBM, the inbound
duty cycle scales with arrival rate, and the write polarity costs ~2x read.
S2 drives Poisson arrivals of fixed-size KV payloads GPU1->GPU0 at swept
offered loads while GPU0 decodes, under five transfer policies:

  unpaced       one whole-payload copy per arrival, issued immediately
                (the NIXL/Mooncake bandwidth-greedy default)
  static64      64 MB chunks, issued back-to-back (the ICCD paper's own
                static advice: "always peer + chunk")
  strongstatic  the critique-mandated strong baseline: 64 MB chunks paced at
                a FIXED offline rate cap = the census worst-bucket inversion
                (same constant the governor starts from), one in flight
  governor-ff   feedforward-only governor (calibrated curves, worst bucket,
                no victim feed) -- should track strongstatic
  governor-fb   two-timescale governor: + per-epoch trim from engine-reported
                victim iteration times -- opens the throttle when the victim
                measures insensitive, the only arm that can win BOTH metrics

Victim modes: b1 (batch=1 decode, maximally sensitive), b8 (loaded decode,
g6 says nearly insensitive), idle (no decode -- exposes that a feed-less
governor over-protects; an honest limitation, reported, not hidden).

Metrics per cell: victim median/p99 slowdown vs an idle-machinery baseline,
epoch-wise violation fraction (250 ms epochs with median slowdown > eps),
transfer drain latency (arrival->completion) p50/p99, delivered vs offered
bytes, deferral counts.  Protocol: victim-stream CUDA events only, no
whole-device sync, clocks locked, box gated.

Requires experiments/results/governor_calib.json (run s1 first).
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import statistics
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _s_common import make_decoder  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "s2_receiver_ingress.json"
CALIB = RESULTS / "governor_calib.json"

EPOCH_S = 0.25


def pctl(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


class QueueArm:
    """unpaced / static64 / strongstatic: FIFO worker over a PacedCopier."""

    def __init__(self, copier, mode: str, chunk_bytes: int,
                 rate_cap: float | None = None):
        assert mode in ("single", "chunks", "paced")
        self.copier, self.mode = copier, mode
        self.chunk_bytes, self.rate_cap = chunk_bytes, rate_cap
        self.q: collections.deque = collections.deque()
        self.records: list[dict] = []
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, src, dst):
        with self._lock:
            self.q.append((src, dst, time.monotonic()))

    def queue_len(self):
        with self._lock:
            return len(self.q)

    def _loop(self):
        while not self._stop.is_set():
            with self._lock:
                item = self.q.popleft() if self.q else None
            if item is None:
                time.sleep(0.001)
                continue
            src, dst, t_sub = item
            nbytes = src.numel() * src.element_size()
            if self.mode == "single":
                r = self.copier.run(src, dst, lambda: 0.0, chunk_bytes=nbytes,
                                    unpaced=True, cancel=self._stop)
            elif self.mode == "chunks":
                r = self.copier.run(src, dst, lambda: 0.0,
                                    chunk_bytes=self.chunk_bytes,
                                    unpaced=True, cancel=self._stop)
            else:
                cap = self.rate_cap
                r = self.copier.run(src, dst, lambda: cap,
                                    chunk_bytes=self.chunk_bytes,
                                    cancel=self._stop)
            if not r.cancelled:
                self.records.append({"submit_t": t_sub, "start_t": r.start_t,
                                     "done_t": r.done_t, "nbytes": nbytes})

    def shutdown(self):
        self._stop.set()
        self._thread.join(timeout=10)
        self.copier.stream.synchronize()   # no bleed into the next cell

    def stats(self, t0: float, t1: float) -> dict:
        recs = [r for r in self.records if r["submit_t"] >= t0]
        drains = [r["done_t"] - r["submit_t"] for r in recs if r["done_t"] <= t1 + 30]
        delivered = sum(r["nbytes"] for r in recs if r["done_t"] <= t1)
        return {"completed": len(drains),
                "drain_p50_s": round(pctl(drains, 0.5), 4) if drains else None,
                "drain_p99_s": round(pctl(drains, 0.99), 4) if drains else None,
                "delivered_gb_in_window": round(delivered / 1e9, 2),
                "left_in_queue": self.queue_len()}


class GovernorArm:
    def __init__(self, gov):
        self.gov = gov
        self.handles = []

    def submit(self, src, dst):
        from umallm.governor import TransferRequest
        self.handles.append(self.gov.submit(TransferRequest(
            src=src, dst=dst, src_dev=1, dst_dev=0, route="peer")))

    def queue_len(self):
        return sum(1 for h in self.handles if not h.done.is_set())

    def shutdown(self):
        # grace only -- under sustained overload the deferred queue is
        # unbounded by design (protection converts violations into queueing);
        # abandoned work is reported as left_in_queue, not hidden
        deadline = time.monotonic() + 3.0
        while self.queue_len() and time.monotonic() < deadline:
            time.sleep(0.05)
        self.gov.shutdown()

    def stats(self, t0: float, t1: float) -> dict:
        recs = [h for h in self.handles if h.submit_t >= t0]
        done = [h for h in recs if h.result is not None]
        drains = [h.total_latency_s for h in done]
        delivered = sum(h.result.bytes_launched for h in done
                        if h.result.done_t <= t1)
        rep = self.gov.report()
        return {"completed": len(done),
                "drain_p50_s": round(pctl(drains, 0.5), 4) if drains else None,
                "drain_p99_s": round(pctl(drains, 0.99), 4) if drains else None,
                "delivered_gb_in_window": round(delivered / 1e9, 2),
                "left_in_queue": self.queue_len(),
                "deferral_polls": rep["stats"].get("deferral_polls", 0),
                "lease_rates_gbs": [round(h.lease_rate_gbs, 1) for h in done[:8]
                                    if h.lease_rate_gbs],
                "trim": rep["trims"].get(0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--payload-mb", type=int, default=512)
    ap.add_argument("--chunk-mb", type=int, default=64)
    ap.add_argument("--eps-pct", type=float, default=5.0)
    ap.add_argument("--loads", type=float, nargs="+", default=[0.25, 0.5, 0.8],
                    help="offered ingress as fraction of peer link bw")
    ap.add_argument("--victims", nargs="+", default=["b1", "b8", "idle"])
    ap.add_argument("--arms", nargs="+",
                    default=["unpaced", "static64", "strongstatic",
                             "governor-ff", "governor-fb"])
    ap.add_argument("--window-secs", type=float, default=8.0)
    ap.add_argument("--baseline-secs", type=float, default=2.5)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--dst-ring", type=int, default=4)
    ap.add_argument("--pacer", choices=["auto", "python", "native"],
                    default="auto")
    args = ap.parse_args()

    from umallm.observability import gate_or_skip
    gate_or_skip("s2_receiver_ingress")
    import torch
    from umallm.governor import Governor, GovernorCalibration
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
    link = cal.link_peak_gbs["peer"]
    r_static = cal.curve("receiver", "peer", "worst").invert(args.eps_pct)
    print(f"link={link:.0f}GB/s  eps={args.eps_pct}%  "
          f"strong-static cap={r_static:.1f}GB/s "
          f"(worst bucket: {cal.curve('receiver','peer','worst').workload})")

    dt = torch.float16
    decoders = {v: make_decoder("cuda:0", int(v[1:]), args.ctx)
                for v in args.victims if v != "idle"}

    n = args.payload_mb * (1 << 20) // 2
    src = torch.randn(n, dtype=dt, device="cuda:1")
    dst_ring = [torch.empty(n, dtype=dt, device="cuda:0")
                for _ in range(args.dst_ring)]
    chunk_bytes = args.chunk_mb << 20
    payload_gb = args.payload_mb / 1024

    def make_arm(name: str):
        if name in ("unpaced", "static64", "strongstatic"):
            copier = pacer_cls(device=1)
            mode = {"unpaced": "single", "static64": "chunks",
                    "strongstatic": "paced"}[name]
            return QueueArm(copier, mode, chunk_bytes,
                            rate_cap=r_static if mode == "paced" else None)
        gov = Governor(cal, eps_holder_pct=args.eps_pct,
                       eps_receiver_pct=args.eps_pct,
                       mode="ff" if name.endswith("ff") else "fb",
                       chunk_mb=args.chunk_mb, epoch_s=EPOCH_S,
                       pacer=args.pacer)
        return GovernorArm(gov)

    cells = []
    t_exp0 = time.time()
    for victim in args.victims:
        seeds = range(args.seeds if victim != "idle" else 1)
        loads = args.loads if victim != "idle" else [0.5]
        for seed in seeds:
            for load_frac in loads:
                offered_gbs = load_frac * link
                lam = offered_gbs / payload_gb          # arrivals per second
                for arm_name in args.arms:
                    rng = random.Random(31337 + seed * 100 + int(load_frac * 100))
                    arm = make_arm(arm_name)
                    gov = arm.gov if isinstance(arm, GovernorArm) else None

                    # --- baseline: machinery alive, no traffic -------------
                    base_samples = []
                    if victim != "idle":
                        timed_iter = decoders[victim]
                        t_b_end = time.monotonic() + args.baseline_secs
                        while time.monotonic() < t_b_end:
                            ms = timed_iter()
                            base_samples.append(ms)
                            if gov is not None:
                                gov.feed_victim(0, ms)
                        base = statistics.median(base_samples)
                    else:
                        base = None
                        time.sleep(0.3)

                    # --- arrival generator -----------------------------
                    stop_arr = threading.Event()

                    def arrivals():
                        i = 0
                        while not stop_arr.is_set():
                            time.sleep(rng.expovariate(lam))
                            if stop_arr.is_set():
                                break
                            arm.submit(src, dst_ring[i % len(dst_ring)])
                            i += 1

                    arr_t = threading.Thread(target=arrivals, daemon=True)
                    t0 = time.monotonic()
                    arr_t.start()

                    # --- measurement window ----------------------------
                    during = []          # (t, ms)
                    if victim != "idle":
                        t_end = t0 + args.window_secs
                        while time.monotonic() < t_end:
                            ms = timed_iter()
                            during.append((time.monotonic(), ms))
                            if gov is not None:
                                gov.feed_victim(0, ms)
                    else:
                        time.sleep(args.window_secs)
                    t1 = time.monotonic()
                    stop_arr.set()
                    arr_t.join(timeout=5)
                    tstats = arm.stats(t0, t1)
                    arm.shutdown()

                    # --- victim metrics --------------------------------
                    vstats = {}
                    if victim != "idle":
                        mss = [ms for _, ms in during]
                        med = statistics.median(mss)
                        vstats = {
                            "victim_med_slowdown_pct": round(
                                (med / base - 1) * 100, 2),
                            "victim_p99_slowdown_pct": round(
                                (pctl(mss, 0.99) / base - 1) * 100, 2),
                            "victim_iters": len(mss),
                            "baseline_ms": round(base, 4),
                        }
                        # epoch-wise violations
                        epochs = collections.defaultdict(list)
                        for t, ms in during:
                            epochs[int((t - t0) / EPOCH_S)].append(ms)
                        viol = [statistics.median(v) / base - 1 > args.eps_pct / 100
                                for v in epochs.values() if len(v) >= 3]
                        vstats["violation_frac"] = (round(
                            sum(viol) / len(viol), 3) if viol else None)
                        vstats["epochs"] = len(viol)

                    cell = {"victim": victim, "seed": seed,
                            "offered_frac": load_frac,
                            "offered_gbs": round(offered_gbs, 1),
                            "arm": arm_name, **vstats, **tstats}
                    cells.append(cell)
                    msg = (f"  {victim:4s} s{seed} load={load_frac:.2f} "
                           f"{arm_name:13s} ")
                    if victim != "idle":
                        msg += (f"victim med +{cell['victim_med_slowdown_pct']:6.2f}% "
                                f"p99 +{cell['victim_p99_slowdown_pct']:6.2f}% "
                                f"viol={cell['violation_frac']} ")
                    msg += (f"done={cell['completed']:3d} "
                            f"drain p50={cell['drain_p50_s']} "
                            f"p99={cell['drain_p99_s']} "
                            f"qleft={cell['left_in_queue']}")
                    print(msg, flush=True)

    out = {
        "_experiment": "s2_receiver_ingress",
        "_is_measured": True,
        "_timing_method": ("victim-decode-stream CUDA events; Poisson "
                           "arrivals on a sender thread; baseline with idle "
                           "machinery; no whole-device sync; clocks locked"),
        "device": torch.cuda.get_device_name(0),
        "geometry": "Llama-3-8B GQA (random weights)",
        "ctx": args.ctx, "payload_mb": args.payload_mb,
        "chunk_mb": args.chunk_mb, "eps_pct": args.eps_pct,
        "link_gbs": link, "strong_static_cap_gbs": round(r_static, 1),
        "loads": args.loads, "victims": args.victims, "arms": args.arms,
        "window_secs": args.window_secs, "seeds": args.seeds,
        "pacer": pacer_kind,
        "cells": cells,
        "wall_minutes": round((time.time() - t_exp0) / 60, 1),
        "_generated_at": datetime.now(timezone.utc).isoformat(),
    }
    RESULTS.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
