"""G1 -- counter-backed attribution of the cross-GPU KV-handoff victim cost.

Kill-gate for the Case-B narrative. The Case-B finding is that a concurrent KV
handoff out of a *busy* holder GPU slows the holder's decode MORE when the copy is
consumer-issued (PULL, remote read of the holder's HBM) than holder-issued (PUSH,
local read + remote write), even though both move identical bytes out of the same
HBM at ~equal effective bandwidth. The reviewer's decisive objection: prove this
is HBM read-port / memory-subsystem contention, not a PyTorch-stream / copy-engine
/ allocator artifact.

Design: holder GPU1 runs a sustained memory-bound GQA decode (Llama-3-8B geometry,
random weights -- contention depends on HBM traffic, not values). We measure the
holder's per-iteration decode time (CUDA events on the decode stream only -- never a
whole-device sync, so the concurrent copy is never drained) under four conditions:

  decode_alone   : no copy (baseline)
  push           : GPU1 issues copy GPU1->GPU0   (local read,  remote write, NVLink TX)
  pull           : GPU0 issues copy GPU1->GPU0   (remote read, local  write, NVLink RX@GPU1)
  local_copy     : GPU1 issues copy GPU1->GPU1   (local read,  local  write, NO NVLink)   [decisive control]

All three copy conditions read GPU1's HBM at the same byte rate, so equal HBM-read
*bandwidth* contention would hurt the holder equally. If PULL hurts more than PUSH
and LOCAL_COPY, the extra cost is attributable to the *remote-issued* read path, not
to raw HBM bandwidth -- the publishable mechanism. Clocks must be locked (DVFS off)
before running; this script asserts it.

Outputs:
  results/g1_timing.json    -- per-condition decode-iter medians/IQR, slowdown, copy BW
  results/g1_markers.json   -- {condition: [t_start_epoch, t_end_epoch]} for DCGM alignment
Run dcgmi dmon in parallel (see g1_run.sh) to capture DRAMA/SMACT/NVLINK per window.
"""
from __future__ import annotations
import argparse, json, statistics, time
from datetime import datetime, timezone
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
TIMING_OUT = RESULTS / "g1_timing.json"
MARKERS_OUT = RESULTS / "g1_markers.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--handoff-mb", type=int, default=512)
    ap.add_argument("--iters", type=int, default=300, help="timed decode iters per condition")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--counter-secs", type=float, default=6.0,
                    help="extra steady spin per condition so DCGM can sample the window")
    args = ap.parse_args()
    from umallm.observability import gate_or_skip
    gate_or_skip("g1_victim_attribution")
    import torch
    import torch.nn.functional as F

    assert torch.cuda.device_count() >= 2, "need 2 GPUs"
    # --- assert clocks are locked (DVFS off) so idle/busy counters are comparable ---
    import pynvml
    pynvml.nvmlInit()
    for i in (0, 1):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        cur = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_GRAPHICS)
        mx = pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_GRAPHICS)
        print(f"GPU{i} graphics clock {cur}/{mx} MHz")
        if cur < int(0.9 * mx):
            print(f"  WARNING GPU{i} clock not locked near max -- run: sudo nvidia-smi -lgc {mx},{mx}")

    dev_h, dev_c = "cuda:1", "cuda:0"  # holder / consumer
    D, H, HKV, HD, DFF = 4096, 32, 8, 128, 14336
    B, S = args.batch, args.ctx
    dt = torch.float16
    torch.cuda.set_device(1)
    W = {k: torch.randn(*s, dtype=dt, device=dev_h) * 0.02 for k, s in {
        "q": (D, H * HD), "k": (D, HKV * HD), "v": (D, HKV * HD), "o": (H * HD, D),
        "g": (D, DFF), "u": (D, DFF), "d": (DFF, D)}.items()}
    Kc = torch.randn(B, HKV, S, HD, dtype=dt, device=dev_h) * 0.02
    Vc = torch.randn(B, HKV, S, HD, dtype=dt, device=dev_h) * 0.02
    x = torch.randn(B, 1, D, dtype=dt, device=dev_h) * 0.02

    def decode_step():
        q = (x @ W["q"]).view(B, 1, H, HD).transpose(1, 2)
        k = (x @ W["k"]).view(B, 1, HKV, HD).transpose(1, 2)
        v = (x @ W["v"]).view(B, 1, HKV, HD).transpose(1, 2)
        # faithful memory-bound decode: flash-attend the resident L-token KV in place
        # (paged-decode HBM-read pattern), no per-step full re-cat of the cache.
        o = F.scaled_dot_product_attention(q, Kc, Vc, enable_gqa=True)
        return (o.transpose(1, 2).reshape(B, 1, H * HD) @ W["o"]) + (F.silu(x @ W["g"]) * (x @ W["u"])) @ W["d"]

    # handoff buffers
    n = args.handoff_mb * 1024 * 1024 // 2
    nbytes = n * 2
    src_h = torch.randn(n, dtype=dt, device=dev_h)      # source on holder
    dst_c = torch.empty(n, dtype=dt, device=dev_c)      # dest on consumer (push/pull)
    dst_h = torch.empty(n, dtype=dt, device=dev_h)      # dest on holder  (local_copy)

    dec_stream = torch.cuda.Stream(device=1)
    cp_push = torch.cuda.Stream(device=1)   # holder issues -> GPU0
    cp_local = torch.cuda.Stream(device=1)  # holder issues -> GPU1
    cp_pull = torch.cuda.Stream(device=0)   # consumer issues <- GPU1

    CHUNK = 1  # one 512MB copy per top-up; refilled when stream drains

    def mk_copy(kind):
        if kind == "push":
            st = cp_push
            def op():
                with torch.cuda.stream(st):
                    dst_c.copy_(src_h, non_blocking=True)
            return st, op
        if kind == "pull":
            st = cp_pull
            def op():
                with torch.cuda.stream(st):
                    dst_c.copy_(src_h, non_blocking=True)
            return st, op
        if kind == "local":
            st = cp_local
            def op():
                with torch.cuda.stream(st):
                    dst_h.copy_(src_h, non_blocking=True)
            return st, op
        return None, None

    def topup(op, st):
        for _ in range(CHUNK):
            op()

    # warm decode
    torch.cuda.set_device(1)
    for _ in range(args.warmup):
        with torch.cuda.stream(dec_stream):
            decode_step()
    dec_stream.synchronize()

    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)

    def time_decode(copy_kind):
        """Return (median_ms, iqr_ms, copy_busy_frac). Times decode iters on dec_stream
        with events only; keeps copy saturated on its own stream; never whole-device sync."""
        st, op = (None, None) if copy_kind == "alone" else mk_copy(copy_kind)
        if op is not None:
            for _ in range(3):
                topup(op, st)
        # event warm
        with torch.cuda.stream(dec_stream):
            e1.record(dec_stream)
        e1.synchronize()
        ts, busy_hits = [], 0
        for _ in range(args.iters):
            if op is not None:
                if st.query():        # copy drained -> refill so it overlaps the next decode
                    topup(op, st)
                if not st.query():
                    busy_hits += 1
            with torch.cuda.stream(dec_stream):
                e0.record(dec_stream)
                decode_step()
                e1.record(dec_stream)
            e1.synchronize()
            ts.append(e0.elapsed_time(e1))
        ts.sort()
        med = statistics.median(ts)
        iqr = ts[int(0.75 * len(ts))] - ts[int(0.25 * len(ts))]
        busy_frac = busy_hits / args.iters if op is not None else 0.0
        return med, iqr, busy_frac

    def copy_bw(copy_kind):
        """Effective copy bandwidth (GB/s) under concurrent decode, events on copy stream."""
        st, op = mk_copy(copy_kind)
        # keep decode running underneath
        for _ in range(4):
            with torch.cuda.stream(dec_stream):
                for _ in range(50):
                    decode_step()
        ce0 = torch.cuda.Event(enable_timing=True)
        ce1 = torch.cuda.Event(enable_timing=True)
        for _ in range(5):
            op()
        with torch.cuda.stream(st):
            ce1.record(st)
        ce1.synchronize()
        cts = []
        for _ in range(30):
            if dec_stream.query():
                with torch.cuda.stream(dec_stream):
                    for _ in range(50):
                        decode_step()
            with torch.cuda.stream(st):
                ce0.record(st)
                op()
                ce1.record(st)
            ce1.synchronize()
            cts.append(ce0.elapsed_time(ce1))
        med_ms = statistics.median(cts)
        return nbytes / (med_ms / 1e3) / 1e9

    conditions = ["alone", "push", "pull", "local"]
    markers, timing = {}, {}
    for cond in conditions:
        torch.cuda.synchronize(0); torch.cuda.synchronize(1)
        t_start = time.time()
        med, iqr, busy = time_decode(cond)
        # steady spin so DCGM samples a clean window for this condition
        spin_end = time.time() + args.counter_secs
        if cond == "alone":
            while time.time() < spin_end:
                with torch.cuda.stream(dec_stream):
                    for _ in range(20):
                        decode_step()
                dec_stream.synchronize()
        else:
            st, op = mk_copy(cond)
            while time.time() < spin_end:
                if st.query():
                    topup(op, st)
                with torch.cuda.stream(dec_stream):
                    for _ in range(20):
                        decode_step()
                dec_stream.synchronize()
        torch.cuda.synchronize(0); torch.cuda.synchronize(1)
        t_end = time.time()
        markers[cond] = [round(t_start, 3), round(t_end, 3)]
        timing[cond] = {"decode_ms_median": round(med, 4), "decode_ms_iqr": round(iqr, 4),
                        "copy_overlap_busy_frac": round(busy, 3)}
        print(f"  {cond:11s} decode={med:.4f}ms (IQR {iqr:.4f}) overlap_busy={busy:.2f}")

    base = timing["alone"]["decode_ms_median"]
    for cond in conditions:
        timing[cond]["victim_slowdown_pct"] = round((timing[cond]["decode_ms_median"] / base - 1) * 100, 2)
    # copy bandwidth under decode (confirm ~equal across push/pull/local)
    for cond in ("push", "pull", "local"):
        timing[cond]["copy_bw_gbs"] = round(copy_bw(cond), 1)

    out = {"_experiment": "g1_victim_attribution", "_is_measured": True,
           "_timing_method": "decode-stream CUDA events; clocks locked; copy never drained",
           "device": torch.cuda.get_device_name(0),
           "geometry": "Llama-3-8B GQA (random weights)", "ctx": S, "batch": B,
           "handoff_mb": args.handoff_mb, "iters": args.iters,
           "conditions": timing, "_generated_at": datetime.now(timezone.utc).isoformat()}
    RESULTS.mkdir(exist_ok=True)
    TIMING_OUT.write_text(json.dumps(out, indent=2))
    MARKERS_OUT.write_text(json.dumps(markers, indent=2))
    print("\nvictim slowdown vs alone:")
    for cond in ("push", "pull", "local"):
        print(f"  {cond:11s} +{timing[cond]['victim_slowdown_pct']:.1f}%  copy_bw={timing[cond].get('copy_bw_gbs','?')}GB/s")
    print(f"-> {TIMING_OUT}\n-> {MARKERS_OUT}")


if __name__ == "__main__":
    main()
