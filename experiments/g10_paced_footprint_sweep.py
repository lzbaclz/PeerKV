"""G10 -- paced-footprint sweep: turn the normalized tiering rule into a curve.

Sweeps holder-side HBM activity at target fractions of the device's peak HBM
bandwidth (default 0.05/0.10/0.20/0.40/0.60) for three traffic mixes
(read-only, write-only, read+write), and measures the busy holder's decode
slowdown at each point.

**Multi-stream burst design (v2):** to reach high footprint fractions that a
single stream cannot sustain (a 64 MB kernel tops out at ~138 GB/s on A100),
we launch N parallel activity streams, each operating on its own buffer.  N is
chosen so that N × single-stream-peak >= target bandwidth.  All N streams fire
concurrently and stay saturated; the aggregate achieved BW is measured from the
total bytes transferred over the measurement wall time.

Run on the A100 box (locked clocks) and the H100 pod (boost + ratio):
    python experiments/g10_paced_footprint_sweep.py
    python experiments/g10_paced_footprint_sweep.py --hbm-peak-gbs 3350
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "g10_paced_sweep.json"

KNOWN_PEAKS_GBS = {
    "A100": 2039.0,
    "H100": 3350.0,
}


def detect_peak(device_name: str) -> float | None:
    for key, peak in KNOWN_PEAKS_GBS.items():
        if key in device_name:
            return peak
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--chunk-mb", type=int, default=64)
    ap.add_argument("--fractions", type=float, nargs="+",
                    default=[0.05, 0.10, 0.20, 0.40, 0.60])
    ap.add_argument("--mixes", nargs="+",
                    default=["read", "write", "readwrite"])
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--hbm-peak-gbs", type=float, default=None)
    ap.add_argument("--max-streams", type=int, default=16,
                    help="max parallel activity streams for high-frac points")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    assert torch.cuda.device_count() >= 1, "need a GPU"
    dev_h = "cuda:1" if torch.cuda.device_count() >= 2 else "cuda:0"
    torch.cuda.set_device(dev_h)
    name = torch.cuda.get_device_name(dev_h)
    peak = args.hbm_peak_gbs or detect_peak(name)
    assert peak, f"unknown device {name!r}: pass --hbm-peak-gbs"
    print(f"device={name}  hbm_peak={peak} GB/s  max_streams={args.max_streams}")

    D, H, HKV, HD, DFF = 4096, 32, 8, 128, 14336
    B, S = args.batch, args.ctx
    dt = torch.float16
    W = {k: torch.randn(*s, dtype=dt, device=dev_h) * 0.02 for k, s in {
        "q": (D, H * HD), "k": (D, HKV * HD), "v": (D, HKV * HD), "o": (H * HD, D),
        "g": (D, DFF), "u": (D, DFF), "d": (DFF, D)}.items()}
    Kc = torch.randn(B, HKV, S, HD, dtype=dt, device=dev_h) * 0.02
    Vc = torch.randn(B, HKV, S, HD, dtype=dt, device=dev_h) * 0.02
    x = torch.randn(B, 1, D, dtype=dt, device=dev_h) * 0.02

    def decode_step():
        q = (x @ W["q"]).view(B, 1, H, HD).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, Kc, Vc, enable_gqa=True)
        return (o.transpose(1, 2).reshape(B, 1, H * HD) @ W["o"]) + (
            F.silu(x @ W["g"]) * (x @ W["u"])) @ W["d"]

    n = args.chunk_mb * 1024 * 1024 // 2  # elements per chunk (fp16)
    nbytes_chunk = n * 2

    # Pre-allocate buffers for up to max_streams parallel activities
    MAX_S = args.max_streams
    srcs = [torch.randn(n, dtype=dt, device=dev_h) for _ in range(MAX_S)]
    dsts = [torch.empty(n, dtype=dt, device=dev_h) for _ in range(MAX_S)]
    accs = [torch.zeros(1, dtype=dt, device=dev_h) for _ in range(MAX_S)]
    act_streams = [torch.cuda.Stream(device=dev_h) for _ in range(MAX_S)]

    dec_stream = torch.cuda.Stream(device=dev_h)
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)

    # Calibrate single-stream peak for each mix
    ce0 = torch.cuda.Event(enable_timing=True)
    ce1 = torch.cuda.Event(enable_timing=True)

    def measure_single_stream_bw(mix: str, reps: int = 20) -> float:
        """Returns single-stream achievable BW in GB/s for this mix."""
        s = act_streams[0]
        for _ in range(5):
            _fire_one(mix, 0)
        s.synchronize()
        times = []
        for _ in range(reps):
            ce0.record(s)
            _fire_one(mix, 0)
            ce1.record(s)
            ce1.synchronize()
            times.append(ce0.elapsed_time(ce1))
        med_ms = statistics.median(times)
        traffic = nbytes_chunk * (2 if mix == "readwrite" else 1)
        return traffic / (med_ms / 1e3) / 1e9

    def _fire_one(mix: str, idx: int):
        """Fire one chunk op on stream idx."""
        s = act_streams[idx]
        with torch.cuda.stream(s):
            if mix == "read":
                accs[idx].add_(srcs[idx].sum())
            elif mix == "write":
                dsts[idx].fill_(1.0)
            elif mix == "readwrite":
                dsts[idx].copy_(srcs[idx], non_blocking=True)

    def _fire_all(mix: str, n_streams: int):
        """Fire one chunk op on each of n_streams streams simultaneously."""
        for i in range(n_streams):
            _fire_one(mix, i)

    def _all_idle(n_streams: int) -> bool:
        return all(act_streams[i].query() for i in range(n_streams))

    for _ in range(args.warmup):
        with torch.cuda.stream(dec_stream):
            decode_step()
    dec_stream.synchronize()

    # Measure single-stream peaks (solo, no decode contention)
    ss_peaks_solo = {}
    for mix in args.mixes:
        ss_peaks_solo[mix] = measure_single_stream_bw(mix)
        print(f"  single-stream solo  [{mix:9s}]: {ss_peaks_solo[mix]:.1f} GB/s")

    # Measure single-stream achieved *under decode contention* (this is what
    # actually limits each stream when decode is running)
    ss_peaks_cont = {}
    for mix in args.mixes:
        # run one quick measurement: fire activity while decode runs
        _fire_all(mix, 1)
        act_streams[0].synchronize()
        fires_cal = 0
        t0_cal = time.monotonic()
        for _ in range(50):
            if act_streams[0].query():
                _fire_one(mix, 0)
                fires_cal += 1
            with torch.cuda.stream(dec_stream):
                decode_step()
            dec_stream.synchronize()
        act_streams[0].synchronize()
        wall_cal = time.monotonic() - t0_cal
        traffic_one = nbytes_chunk * (2 if mix == "readwrite" else 1)
        ss_peaks_cont[mix] = fires_cal * traffic_one / wall_cal / 1e9
        print(f"  single-stream cont. [{mix:9s}]: {ss_peaks_cont[mix]:.1f} GB/s")

    def time_decode(mix: str | None, frac: float = 0.0):
        """Median decode-iteration ms under multi-stream activity."""
        if mix is None:
            ts = []
            for _ in range(args.iters):
                with torch.cuda.stream(dec_stream):
                    e0.record(dec_stream)
                    decode_step()
                    e1.record(dec_stream)
                e1.synchronize()
                ts.append(e0.elapsed_time(e1))
            ts.sort()
            iqr = ts[int(0.75 * len(ts))] - ts[int(0.25 * len(ts))]
            return statistics.median(ts), iqr, 0.0

        target_gbs = frac * peak
        # Use the under-contention per-stream rate to decide how many streams
        contended_rate = ss_peaks_cont[mix]
        n_streams = max(1, min(MAX_S, int(target_gbs / contended_rate + 0.999)))
        traffic_per_fire = nbytes_chunk * (2 if mix == "readwrite" else 1) * n_streams

        # warmup the streams
        for _ in range(3):
            _fire_all(mix, n_streams)
        for i in range(n_streams):
            act_streams[i].synchronize()

        ts = []
        fires = 0
        t_start = time.monotonic()
        for _ in range(args.iters):
            if _all_idle(n_streams):
                _fire_all(mix, n_streams)
                fires += 1
            with torch.cuda.stream(dec_stream):
                e0.record(dec_stream)
                decode_step()
                e1.record(dec_stream)
            e1.synchronize()
            ts.append(e0.elapsed_time(e1))
        for i in range(n_streams):
            act_streams[i].synchronize()
        wall_s = time.monotonic() - t_start
        achieved = fires * traffic_per_fire / wall_s / 1e9

        ts.sort()
        iqr = ts[int(0.75 * len(ts))] - ts[int(0.25 * len(ts))]
        return statistics.median(ts), iqr, achieved

    points = []
    for seed in range(args.seeds):
        torch.manual_seed(1000 + seed)
        base, _, _ = time_decode(None)
        for mix in args.mixes:
            for frac in args.fractions:
                med, iqr, achieved = time_decode(mix, frac)
                pt = {
                    "seed": seed, "mix": mix, "target_frac": frac,
                    "target_gbs": round(frac * peak, 1),
                    "achieved_gbs": round(achieved, 1),
                    "achieved_frac": round(achieved / peak, 4),
                    "victim_slowdown_pct": round((med / base - 1) * 100, 2),
                    "iqr_ms": round(iqr, 4),
                    "n_streams": max(1, min(MAX_S,
                        int(frac * peak / ss_peaks_cont[mix] + 0.999))),
                }
                points.append(pt)
                print(f"  seed{seed} {mix:9s} f={frac:.2f} "
                      f"target={pt['target_gbs']:7.1f} "
                      f"achieved={pt['achieved_gbs']:7.1f} GB/s "
                      f"(frac={pt['achieved_frac']:.3f}, "
                      f"N={pt['n_streams']}) "
                      f"victim=+{pt['victim_slowdown_pct']:.1f}%")

    summary = {}
    for mix in args.mixes:
        for frac in args.fractions:
            sel = [p for p in points
                   if p["mix"] == mix and p["target_frac"] == frac]
            summary[f"{mix}@{frac:.2f}"] = {
                "victim_slowdown_pct": round(statistics.mean(
                    p["victim_slowdown_pct"] for p in sel), 2),
                "achieved_gbs": round(statistics.mean(
                    p["achieved_gbs"] for p in sel), 1),
                "achieved_frac": round(statistics.mean(
                    p["achieved_frac"] for p in sel), 4),
                "n_streams": sel[0]["n_streams"],
            }

    out = {
        "_experiment": "g10_paced_footprint_sweep",
        "_version": "v2_multistream",
        "_is_measured": True,
        "_timing_method": "decode-stream CUDA events; N parallel activity streams",
        "device": name, "hbm_peak_gbs": peak,
        "geometry": "Llama-3-8B GQA (random weights)",
        "ctx": S, "batch": B, "chunk_mb": args.chunk_mb,
        "fractions": args.fractions, "mixes": args.mixes,
        "max_streams": MAX_S,
        "single_stream_peaks_solo_gbs": {k: round(v, 1) for k, v in ss_peaks_solo.items()},
        "single_stream_peaks_contended_gbs": {k: round(v, 1) for k, v in ss_peaks_cont.items()},
        "iters": args.iters, "seeds": args.seeds,
        "summary": summary, "points": points,
        "_generated_at": datetime.now(timezone.utc).isoformat(),
    }
    RESULTS.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
