"""G13 -- DVFS forensics: cold/hot decode baselines + SM-clock telemetry.

Companion to g10.  Tests whether a DVFS-unlocked run deflates the paced-sweep
victim%%: clock-ramp lag inflates the solo decode baseline (the denominator),
so the same contended decode reads as a smaller *relative* slowdown.

Self-contained: stdlib + torch + the nvidia-smi binary.  No sudo, no network.

    python experiments/g13_dvfs_forensics.py
    # -> experiments/results/g13_dvfs_forensics.json

Phases (same GPU as g10: cuda:1 when >=2 GPUs):
  A. env capture       driver / persistence / clocks via nvidia-smi
  B. ramp probe        idle 6 s, then 300 timed decode iters back-to-back;
                       on a DVFS box the early iters are slow (clock ramp),
                       on a locked box the series is flat
  C. baselines         cold = median of ramp iters 10..60 (pre-ramp window)
                       hot  = 500 sustained warmup iters, then median of 200
                       hot2 = re-baseline after 10 s decode+activity
                              conditioning (PRIMARY denominator; the first
                              contended cells otherwise read low even at
                              locked clocks -- slow memory-subsystem
                              conditioning, suspected HBM thermal)
  D. contended sweep   read mix at 0.10/0.40/0.60 of HBM peak (g10 cells),
                       50 untimed conditioning iters per cell, victim%%
                       computed against ALL THREE baselines (use vs_hot2)
SM clock + GPU/HBM temperature + power stream at 10 Hz (nvidia-smi -lms 100)
for the whole run with phase markers, so every number aligns to the trace.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "g13_dvfs_forensics.json"

KNOWN_PEAKS_GBS = {
    "A100": 2039.0,
    "A800": 2039.0,
    "H100": 3350.0,
}


def detect_peak(device_name: str) -> float | None:
    for key, peak in KNOWN_PEAKS_GBS.items():
        if key in device_name:
            return peak
    return None


def smi_index_for_cuda(cuda_idx: int) -> int:
    """Map a torch cuda index to the nvidia-smi index (CUDA_VISIBLE_DEVICES)."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cvd:
        return cuda_idx
    ids = [s.strip() for s in cvd.split(",") if s.strip()]
    try:
        return int(ids[cuda_idx])
    except (ValueError, IndexError):
        return cuda_idx  # UUID-style CVD: best effort


class ClockTrace:
    """Stream clocks.sm at 10 Hz from a single nvidia-smi process."""

    def __init__(self, smi_idx: int):
        self.samples: list = []   # (t_rel_s, sm_mhz, gpu_C, mem_C, power_W)
        self.marks: list = []     # (t_rel_s, label)
        self.t0 = time.monotonic()
        self.proc = None
        try:
            self.proc = subprocess.Popen(
                ["nvidia-smi", "-i", str(smi_idx), "-lms", "100",
                 "--query-gpu=clocks.sm,temperature.gpu,temperature.memory,"
                 "power.draw",
                 "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            self.thread = threading.Thread(target=self._reader, daemon=True)
            self.thread.start()
        except Exception as e:
            print(f"WARN: clock trace unavailable ({e}); continuing without")

    @staticmethod
    def _num(s):
        try:
            return float(s)
        except ValueError:
            return None  # "N/A" (e.g. temperature.memory on some boxes)

    def _reader(self):
        for line in self.proc.stdout:
            parts = [p.strip() for p in line.strip().split(",")]
            if parts and parts[0].isdigit():
                self.samples.append(
                    (round(time.monotonic() - self.t0, 2), int(parts[0]),
                     self._num(parts[1]) if len(parts) > 1 else None,
                     self._num(parts[2]) if len(parts) > 2 else None,
                     self._num(parts[3]) if len(parts) > 3 else None))

    def mark(self, label: str):
        self.marks.append((round(time.monotonic() - self.t0, 2), label))

    def stop(self):
        if self.proc is not None:
            self.proc.terminate()
            self.thread.join(timeout=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--chunk-mb", type=int, default=64)
    ap.add_argument("--fractions", type=float, nargs="+",
                    default=[0.10, 0.40, 0.60])
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--ramp-iters", type=int, default=300)
    ap.add_argument("--idle-s", type=float, default=6.0)
    ap.add_argument("--hbm-peak-gbs", type=float, default=None)
    ap.add_argument("--max-streams", type=int, default=16)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    assert torch.cuda.device_count() >= 1, "need a GPU"
    cuda_idx = 1 if torch.cuda.device_count() >= 2 else 0
    dev_h = f"cuda:{cuda_idx}"
    torch.cuda.set_device(dev_h)
    name = torch.cuda.get_device_name(dev_h)
    peak = args.hbm_peak_gbs or detect_peak(name)
    assert peak, f"unknown device {name!r}: pass --hbm-peak-gbs"
    smi_idx = smi_index_for_cuda(cuda_idx)

    env = {"torch": torch.__version__, "cuda": torch.version.cuda}
    try:
        q = subprocess.run(
            ["nvidia-smi", "-i", str(smi_idx),
             "--query-gpu=name,driver_version,persistence_mode,"
             "clocks.sm,clocks.max.sm",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        env["nvidia_smi"] = q.stdout.strip()
    except Exception as e:
        env["nvidia_smi"] = f"unavailable: {e}"
    print(f"device={name} (smi -i {smi_idx})  hbm_peak={peak} GB/s")
    print(f"env: {env['nvidia_smi']}")
    print(f"     torch {env['torch']}  cuda {env['cuda']}")

    # --- model geometry: identical to g10 ---
    D, H, HKV, HD, DFF = 4096, 32, 8, 128, 14336
    B, S = 1, args.ctx
    dt = torch.float16
    torch.manual_seed(1000)
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

    n = args.chunk_mb * 1024 * 1024 // 2  # fp16 elements per chunk
    nbytes_chunk = n * 2
    MAX_S = args.max_streams
    srcs = [torch.randn(n, dtype=dt, device=dev_h) for _ in range(MAX_S)]
    accs = [torch.zeros(1, dtype=dt, device=dev_h) for _ in range(MAX_S)]
    act_streams = [torch.cuda.Stream(device=dev_h) for _ in range(MAX_S)]
    dec_stream = torch.cuda.Stream(device=dev_h)
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)

    def _fire_one(idx: int):  # read mix only
        with torch.cuda.stream(act_streams[idx]):
            accs[idx].add_(srcs[idx].sum())

    def _fire_all(k: int):
        for i in range(k):
            _fire_one(i)

    def _all_idle(k: int) -> bool:
        return all(act_streams[i].query() for i in range(k))

    def timed_iters(count: int) -> list:
        ts = []
        for _ in range(count):
            with torch.cuda.stream(dec_stream):
                e0.record(dec_stream)
                decode_step()
                e1.record(dec_stream)
            e1.synchronize()
            ts.append(e0.elapsed_time(e1))
        return ts

    trace = ClockTrace(smi_idx)
    try:
        # warm kernels/allocator once so the ramp probe measures clocks,
        # not first-call compilation
        trace.mark("jit_warm_start")
        timed_iters(30)
        dec_stream.synchronize()

        # --- B. ramp probe: idle, then back-to-back decode ---
        trace.mark("idle_start")
        time.sleep(args.idle_s)
        trace.mark("ramp_start")
        ramp = timed_iters(args.ramp_iters)
        trace.mark("ramp_end")

        lo, hi = (10, 60) if len(ramp) >= 60 else (2, max(3, len(ramp) // 2))
        cold_ms = statistics.median(ramp[lo:hi])
        early = statistics.mean(ramp[:30])
        late = statistics.mean(ramp[-100:])

        # --- C. hot baseline ---
        timed_iters(500)  # sustain load so DVFS (if any) is fully ramped
        trace.mark("hot_base_start")
        hot_ms = statistics.median(timed_iters(200))
        trace.mark("hot_base_end")
        print(f"baseline: cold={cold_ms:.4f} ms  hot={hot_ms:.4f} ms  "
              f"cold/hot={cold_ms / hot_ms:.3f}  "
              f"ramp early/late={early / late:.3f}")

        # calibrate single-stream contended read rate (as in g10)
        _fire_all(1)
        act_streams[0].synchronize()
        fires_cal = 0
        t0_cal = time.monotonic()
        for _ in range(50):
            if act_streams[0].query():
                _fire_one(0)
                fires_cal += 1
            with torch.cuda.stream(dec_stream):
                decode_step()
            dec_stream.synchronize()
        act_streams[0].synchronize()
        cont_rate = fires_cal * nbytes_chunk / (time.monotonic() - t0_cal) / 1e9
        print(f"single-stream contended read: {cont_rate:.1f} GB/s")

        # --- conditioning: ~10 s of decode+activity co-run.  The first
        # contended cells otherwise read low even at locked clocks (the
        # memory subsystem takes tens of seconds of sustained traffic to
        # reach the steady state g10 reaches via its long calibration
        # preamble; suspected HBM thermal effect -- see trace temps) ---
        trace.mark("conditioning_start")
        k0 = max(1, min(MAX_S, int(0.4 * peak / cont_rate + 0.999)))
        t_cond = time.monotonic()
        while time.monotonic() - t_cond < 10.0:
            if _all_idle(k0):
                _fire_all(k0)
            with torch.cuda.stream(dec_stream):
                decode_step()
            dec_stream.synchronize()
        for i in range(k0):
            act_streams[i].synchronize()
        trace.mark("conditioning_end")

        # hot2: re-baseline in the conditioned state; PRIMARY denominator
        trace.mark("hot2_base_start")
        hot2_ms = statistics.median(timed_iters(200))
        trace.mark("hot2_base_end")
        print(f"baseline (post-conditioning): hot2={hot2_ms:.4f} ms  "
              f"hot/hot2={hot_ms / hot2_ms:.3f}")

        # --- D. contended sweep (read mix), victim% vs all baselines ---
        points = []
        for seed in range(args.seeds):
            torch.manual_seed(2000 + seed)
            for frac in args.fractions:
                target = frac * peak
                k = max(1, min(MAX_S, int(target / cont_rate + 0.999)))
                # per-cell conditioning: untimed co-run at this cell's N
                for _ in range(50):
                    if _all_idle(k):
                        _fire_all(k)
                    with torch.cuda.stream(dec_stream):
                        decode_step()
                    dec_stream.synchronize()
                for i in range(k):
                    act_streams[i].synchronize()
                trace.mark(f"sweep_s{seed}_f{frac:.2f}_start")
                ts, fires = [], 0
                t_start = time.monotonic()
                for _ in range(args.iters):
                    if _all_idle(k):
                        _fire_all(k)
                        fires += 1
                    with torch.cuda.stream(dec_stream):
                        e0.record(dec_stream)
                        decode_step()
                        e1.record(dec_stream)
                    e1.synchronize()
                    ts.append(e0.elapsed_time(e1))
                for i in range(k):
                    act_streams[i].synchronize()
                wall = time.monotonic() - t_start
                trace.mark(f"sweep_s{seed}_f{frac:.2f}_end")
                med = statistics.median(ts)
                achieved = fires * nbytes_chunk * k / wall / 1e9
                pt = {
                    "seed": seed, "mix": "read", "target_frac": frac,
                    "achieved_gbs": round(achieved, 1),
                    "achieved_frac": round(achieved / peak, 4),
                    "n_streams": k,
                    "contended_med_ms": round(med, 4),
                    "victim_pct_vs_cold": round((med / cold_ms - 1) * 100, 2),
                    "victim_pct_vs_hot": round((med / hot_ms - 1) * 100, 2),
                    "victim_pct_vs_hot2": round((med / hot2_ms - 1) * 100, 2),
                }
                points.append(pt)
                print(f"  seed{seed} read f={frac:.2f} "
                      f"achieved={achieved:7.1f} GB/s (N={k}) "
                      f"victim: vs_cold=+{pt['victim_pct_vs_cold']:.1f}%  "
                      f"vs_hot=+{pt['victim_pct_vs_hot']:.1f}%  "
                      f"vs_hot2=+{pt['victim_pct_vs_hot2']:.1f}%")
    finally:
        trace.stop()

    clocks = [s[1] for s in trace.samples]
    out = {
        "_experiment": "g13_dvfs_forensics",
        "_version": "v2_conditioned",
        "_is_measured": True,
        "_timing_method": ("decode-stream CUDA events; "
                           "nvidia-smi -lms 100 clock trace"),
        "device": name, "hbm_peak_gbs": peak, "env": env,
        "ctx": S, "chunk_mb": args.chunk_mb, "iters": args.iters,
        "seeds": args.seeds, "fractions": args.fractions,
        "ramp_iter_ms": [round(t, 4) for t in ramp],
        "baseline_cold_ms": round(cold_ms, 4),
        "baseline_hot_ms": round(hot_ms, 4),
        "baseline_hot2_ms": round(hot2_ms, 4),
        "baseline_cold_over_hot": round(cold_ms / hot_ms, 4),
        "baseline_hot_over_hot2": round(hot_ms / hot2_ms, 4),
        "ramp_early_over_late": round(early / late, 4),
        "single_stream_contended_read_gbs": round(cont_rate, 1),
        "points": points,
        "clock_trace": {
            "fields": "t_s, sm_mhz, gpu_temp_c, mem_temp_c, power_w",
            "samples": trace.samples,
            "marks": trace.marks,
            "min_mhz": min(clocks) if clocks else None,
            "max_mhz": max(clocks) if clocks else None,
        },
        "_generated_at": datetime.now(timezone.utc).isoformat(),
    }
    RESULTS.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"-> {OUT}")
    temps = [s[2] for s in trace.samples if len(s) > 2 and s[2] is not None]
    mtemps = [s[3] for s in trace.samples if len(s) > 3 and s[3] is not None]
    print(f"trace: sm {out['clock_trace']['min_mhz']}-"
          f"{out['clock_trace']['max_mhz']} MHz | "
          f"gpu {min(temps) if temps else '?'}-{max(temps) if temps else '?'} C | "
          f"hbm {min(mtemps) if mtemps else '?'}-{max(mtemps) if mtemps else '?'} C "
          f"({len(trace.samples)} samples)")


if __name__ == "__main__":
    main()
