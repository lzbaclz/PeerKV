"""S11 -- engine-in-the-loop: census, validation, and the fb TPOT feed.

s10's verdict: the synthetic-proxy calibration does not transfer to a real
engine (vLLM Llama-8B@conc32 inflates +17.7% at the synthetic eps=5% cap;
its true invert(5%) is ~8-10 GB/s).  This script closes that gap both ways
the architecture supports, using only the engine's OWN telemetry -- the
vLLM /metrics Prometheus endpoint (time_per_output_token histogram deltas),
i.e. the deployment-realistic feed: no client modification, no engine fork.

Phases (vLLM serving on physical GPU1, a background client driving constant
concurrency; this process runs on the side and owns the ingress):

  P0 baseline   no ingress; engine TPOT from metrics deltas (the loaded-
                engine-without-governed-traffic baseline -- the right zero).
  P1 census     sustained governed trains at low rates (the regime s10
                located); per rate: engine TPOT inflation -> an ENGINE curve,
                fitted and saved as governor_calib_engine.json
                (bucket 'engine', receiver/peer).
  P2 validate   Governor(ff) admitting against the engine curve; expect
                inflation ~ eps.
  P3 fb demo    Governor(fb) deliberately given the WRONG (synthetic) calib,
                victim feed = metrics poller (feed_victim(1, epoch TPOT));
                the trim must discover the engine's sensitivity online and
                converge to ~eps -- s8's recovery story, now with a real
                engine and the real feed path.

Output: experiments/results/s11_engine_loop.json
Run via experiments/s11_engine_loop.sh (starts server + background client).
"""
from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _s_common import Train  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "s11_engine_loop.json"
CALIB = RESULTS / "governor_calib.json"
ENGINE_CALIB = RESULTS / "governor_calib_engine.json"


class EngineTpot:
    """vLLM /metrics poller: TPOT histogram sum/count deltas per window."""

    def __init__(self, base: str):
        self.url = base.rstrip("/") + "/metrics"
        self._last = self._read()

    def _read(self) -> tuple[float, float]:
        txt = urllib.request.urlopen(self.url, timeout=3).read().decode()
        s = c = 0.0
        found = False
        for line in txt.splitlines():
            if "time_per_output_token_seconds_sum" in line and not line.startswith("#"):
                s += float(line.rsplit(" ", 1)[1]); found = True
            elif "time_per_output_token_seconds_count" in line and not line.startswith("#"):
                c += float(line.rsplit(" ", 1)[1])
        if not found:
            raise RuntimeError("no time_per_output_token metric at " + self.url)
        return s, c

    def delta_ms(self) -> float | None:
        """Mean TPOT (ms) over tokens completed since the previous call."""
        s, c = self._read()
        ds, dc = s - self._last[0], c - self._last[1]
        self._last = (s, c)
        return (ds / dc) * 1000.0 if dc > 0 else None

    def window_ms(self, secs: float) -> tuple[float, float]:
        """(mean TPOT ms, metric observations/s) over a window.

        The histogram count rate is observation-frequency (engine metric
        semantics), NOT client tokens/s -- recorded for load sanity only."""
        s0, c0 = self._read()
        time.sleep(secs)
        s1, c1 = self._read()
        self._last = (s1, c1)
        if c1 - c0 <= 0:
            raise RuntimeError("no tokens completed in window -- load gone "
                               "(check s11_client.log / s11_server.log)")
        return (s1 - s0) / (c1 - c0) * 1000.0, (c1 - c0) / secs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--mb", type=int, default=512)
    ap.add_argument("--chunk-mb", type=int, default=64)
    ap.add_argument("--eps-pct", type=float, default=5.0)
    ap.add_argument("--census-rates", type=float, nargs="+",
                    default=[8.0, 18.0, 40.0, 70.0, 110.0, 160.0])
    ap.add_argument("--census-secs", type=float, default=14.0)
    ap.add_argument("--baseline-secs", type=float, default=10.0)
    ap.add_argument("--validate-secs", type=float, default=25.0)
    ap.add_argument("--fb-secs", type=float, default=40.0)
    args = ap.parse_args()

    from umallm.observability import gate_or_skip
    gate_or_skip("s11_engine_loop")            # PEERKV_SKIP_IDLE_PROBE=1 set by runner
    import torch
    from umallm.governor import Governor, GovernorCalibration, TransferRequest
    from umallm.governor.calib import Curve, CurvePoint
    from umallm.governor.native import NativePacedCopier

    syn = GovernorCalibration.load(CALIB)
    link = syn.link_peak_gbs["peer"]
    n = args.mb * (1 << 20) // 2
    src = torch.randn(n, dtype=torch.float16, device="cuda:0")
    dst = torch.empty(n, dtype=torch.float16, device="cuda:1")
    chunk_bytes = args.chunk_mb << 20
    tpot = EngineTpot(args.base)
    out: dict = {"phases": {}}

    # ---- P0 baseline ------------------------------------------------------
    base_ms, base_toks = tpot.window_ms(args.baseline_secs)
    print(f"P0 baseline engine TPOT: {base_ms:.3f} ms ({base_toks:.1f} obs/s)")
    out["phases"]["P0_baseline_tpot_ms"] = round(base_ms, 3)
    out["phases"]["P0_baseline_tok_s"] = round(base_toks, 1)

    # ---- P1 engine census -------------------------------------------------
    train = Train(NativePacedCopier(device=0), src, dst, chunk_bytes)
    census = []
    for rate in args.census_rates:
        train.start(rate)
        time.sleep(1.0)                          # settle
        tpot.delta_ms()                          # reset the delta origin
        ms, toks = tpot.window_ms(args.census_secs)
        ach = train.stop()
        census.append({"target_gbs": rate, "achieved_gbs": round(ach, 2),
                       "tpot_ms": round(ms, 3), "tok_s": round(toks, 1),
                       "inflation_pct": round((ms / base_ms - 1) * 100, 2)})
        print(f"  census rate={ach:6.2f} GB/s -> TPOT {ms:.3f} ms "
              f"(+{census[-1]['inflation_pct']:.2f}%, {toks:.0f} tok/s)")
        # short cool-down + baseline drift check every other point
        time.sleep(1.0)
    train.shutdown()
    base2, _ = tpot.window_ms(6.0)
    out["phases"]["P1_census"] = census
    out["phases"]["P1_baseline_recheck_ms"] = round(base2, 3)
    print(f"  baseline recheck: {base2:.3f} ms (drift "
          f"{(base2/base_ms-1)*100:+.2f}%)")

    # fit + save the engine calibration (engine curve only; the synthetic
    # holder curve is retained so dual-ended admission still has a source
    # side -- the engine box charges the receiver, GPU0 the holder)
    eng = GovernorCalibration(
        device=syn.device, hbm_peak_gbs=syn.hbm_peak_gbs,
        link_peak_gbs=dict(syn.link_peak_gbs),
        meta={"source": "s11 engine-in-the-loop census",
              "victim": "vLLM Llama-3.1-8B-Instruct conc=32 (engine-reported "
                        "TPOT via /metrics)",
              "baseline_tpot_ms": round(base_ms, 3)})
    eng.add_curve(Curve(role="receiver", route="peer", workload="engine",
                        points=[CurvePoint(rate_gbs=c["achieved_gbs"],
                                           victim_pct=c["inflation_pct"],
                                           spread_pp=0.5)
                                for c in census], margin_pp=0.5))
    for k, c in syn.curves.items():
        if k.startswith("holder/"):
            eng.curves[k] = c
    eng.save(ENGINE_CALIB)
    cap_eng = eng.curve("receiver", "peer", "worst").invert(args.eps_pct)
    cap_syn = syn.curve("receiver", "peer", "worst").invert(args.eps_pct)
    out["cap_engine_gbs"] = round(cap_eng, 2)
    out["cap_synthetic_gbs"] = round(cap_syn, 2)
    print(f"engine invert({args.eps_pct}%) = {cap_eng:.2f} GB/s "
          f"(synthetic said {cap_syn:.1f})")

    # ---- P2 validate: ff against the engine curve -------------------------
    gov = Governor(eng, eps_holder_pct=args.eps_pct,
                   eps_receiver_pct=args.eps_pct, mode="ff",
                   chunk_mb=args.chunk_mb, pacer="auto")
    stop = threading.Event()
    delivered = [0]

    def sender(g):
        while not stop.is_set():
            h = g.submit(TransferRequest(src=src, dst=dst, src_dev=0,
                                         dst_dev=1, route="peer"))
            while not h.done.wait(0.05):
                if stop.is_set():
                    return
            if h.result is not None:
                delivered[0] += h.result.bytes_launched

    th = threading.Thread(target=sender, args=(gov,), daemon=True)
    th.start()
    try:
        time.sleep(2.0)
        tpot.delta_ms()
        t0 = time.monotonic()
        ms, toks = tpot.window_ms(args.validate_secs)
        rate_p2 = delivered[0] / (time.monotonic() - t0 + 2.0) / 1e9
    finally:
        stop.set(); th.join(timeout=10); gov.shutdown()
    out["phases"]["P2_validate_ff_engine_calib"] = {
        "tpot_ms": round(ms, 3), "tok_s": round(toks, 1),
        "inflation_pct": round((ms / base_ms - 1) * 100, 2),
        "delivered_gbs": round(rate_p2, 2)}
    print(f"P2 ff@engine-calib: +{out['phases']['P2_validate_ff_engine_calib']['inflation_pct']:.2f}% "
          f"at {rate_p2:.2f} GB/s (target ~{args.eps_pct}%)")

    # ---- P3 fb from the WRONG prior, fed by engine metrics ----------------
    gov = Governor(syn, eps_holder_pct=args.eps_pct,
                   eps_receiver_pct=args.eps_pct, mode="fb",
                   chunk_mb=args.chunk_mb, pacer="auto")
    feed_stop = threading.Event()

    def feeder():
        # Feed ADAPTER: the engine histogram updates in clumps at request
        # completions (~3-6 obs/s), far sparser than the 250ms trim epochs.
        # Telemetry burstiness is not victim death: keep-alive re-feeds the
        # last value while it is fresh (<5s); true silence (engine stalled,
        # no completions for >5s) stops the feed and lets the governor's
        # starved-grace backoff fire. Engine telemetry quirks belong here,
        # in the adapter -- the governor contract stays generic.
        et = EngineTpot(args.base)
        last_v, last_t = None, 0.0
        while not feed_stop.wait(0.25):
            try:
                d = et.delta_ms()
            except Exception:
                continue
            now = time.monotonic()
            if d is not None:
                last_v, last_t = d, now
                gov.feed_victim(1, d)            # dev1 = the vLLM GPU
            elif last_v is not None and now - last_t < 5.0:
                gov.feed_victim(1, last_v)       # keep-alive: fresh enough

    fth = threading.Thread(target=feeder, daemon=True)
    fth.start()
    stop = threading.Event()
    delivered = [0]
    th = threading.Thread(target=sender, args=(gov,), daemon=True)
    traj = []
    try:
        time.sleep(2.5)                          # clean-baseline epochs
        th.start()
        # trajectory: per-2s windows of (inflation, delivered rate, trim m)
        tpot.delta_ms()
        t0 = time.monotonic()
        last_b = 0
        while time.monotonic() - t0 < args.fb_secs:
            try:
                ms2, toks2 = tpot.window_ms(3.0)
                infl = round((ms2 / base_ms - 1) * 100, 2)
            except RuntimeError:
                infl, toks2 = None, 0.0      # engine stalled: completions
            tr = gov.trims.get(1)            # froze; the starved backoff
            d_now = delivered[0]             # must now act
            traj.append({"t_s": round(time.monotonic() - t0, 1),
                         "inflation_pct": infl, "stalled": infl is None,
                         "rate_gbs": round((d_now - last_b) / 3.0 / 1e9, 2),
                         "tok_s": round(toks2, 1),
                         "trim_m": round(tr.multiplier, 4) if tr else None})
            last_b = d_now
            print(f"  fb t={traj[-1]['t_s']:5.1f}s "
                  f"infl={'STALL' if infl is None else f'+{infl:.2f}%':>7s} "
                  f"rate={traj[-1]['rate_gbs']:5.2f} GB/s m={traj[-1]['trim_m']}")
    finally:
        stop.set()
        if th.is_alive():
            th.join(timeout=10)
        feed_stop.set(); fth.join(timeout=2)
    tr = gov.trims.get(1)
    out["phases"]["P3_fb_wrong_prior"] = {
        "trajectory": traj,
        "trim_final": tr.snapshot() if tr else None,
        "late_inflation_pct": round(statistics.mean(
            [p["inflation_pct"] for p in traj[len(traj) // 2:]
             if p["inflation_pct"] is not None] or [float("nan")]), 2),
        "stall_windows": sum(1 for p in traj if p.get("stalled")),
        "late_rate_gbs": round(statistics.mean(
            p["rate_gbs"] for p in traj[len(traj) // 2:]), 2)}
    gov.shutdown()
    print(f"P3 late-half: +{out['phases']['P3_fb_wrong_prior']['late_inflation_pct']}% "
          f"at {out['phases']['P3_fb_wrong_prior']['late_rate_gbs']} GB/s")

    blob = {
        "_experiment": "s11_engine_loop",
        "_is_measured": True,
        "_timing_method": ("engine-reported TPOT via vLLM /metrics histogram "
                           "deltas; background client at constant "
                           "concurrency; ingress GPU0->GPU1 (vLLM GPU)"),
        "victim": "vLLM Llama-3.1-8B-Instruct conc=32, GPU1, receiver side",
        "eps_pct": args.eps_pct, "chunk_mb": args.chunk_mb,
        "link_gbs": link,
        **out,
        "_generated_at": datetime.now(timezone.utc).isoformat(),
    }
    RESULTS.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(blob, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
