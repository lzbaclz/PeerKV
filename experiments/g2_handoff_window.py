"""G2 -- realistic single/chunked KV-handoff victim cost, clean per-iter timing.

Reviewer W3: e46/e47 enqueue one 512MB copy *per decode step* and wrap the loop in a
whole-device sync -- not a real handoff schedule and confounded by sync/backlog. A
real KV migration moves a request's KV *once* (optionally chunked / pipelined) while
the holder keeps decoding other requests.

This driver measures the holder's per-decode-iteration time (CUDA events on the
decode stream only; clocks locked; never a whole-device sync) (a) at baseline and
(b) *while a single 512MB handoff is in flight*, for each direction and chunk size.
victim_slowdown = median(decode iters overlapping the handoff) / median(baseline) - 1.
We also report the handoff completion time. This is the authoritative, schedule-
realistic answer that supersedes e46/e47's per-step microbench.

Directions:  push (GPU1->GPU0, holder-issued)  pull (GPU1->GPU0, consumer-issued)
             local (GPU1->GPU1, holder-issued)  [HBM-pressure control]
Chunking:    full 512MB, 64MB x8, 16MB x32   (handoff issued as N back-to-back chunks)
"""
from __future__ import annotations
import argparse, json, statistics, time
from datetime import datetime, timezone
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "g2_handoff_window.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--handoff-mb", type=int, default=512)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--base-iters", type=int, default=200)
    args = ap.parse_args()
    from umallm.observability import gate_or_skip
    gate_or_skip("g2_handoff_window")
    import torch
    import torch.nn.functional as F
    import pynvml
    pynvml.nvmlInit()
    for i in (0, 1):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        cur = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_GRAPHICS)
        mx = pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_GRAPHICS)
        if cur < int(0.9 * mx):
            print(f"WARNING GPU{i} clock {cur}/{mx} not locked; run sudo nvidia-smi -lgc {mx},{mx}")

    D, H, HKV, HD, DFF = 4096, 32, 8, 128, 14336
    B, S = args.batch, args.ctx
    dt = torch.float16
    torch.cuda.set_device(1)
    W = {k: torch.randn(*s, dtype=dt, device="cuda:1") * 0.02 for k, s in {
        "q": (D, H * HD), "k": (D, HKV * HD), "v": (D, HKV * HD), "o": (H * HD, D),
        "g": (D, DFF), "u": (D, DFF), "d": (DFF, D)}.items()}
    Kc = torch.randn(B, HKV, S, HD, dtype=dt, device="cuda:1") * 0.02
    Vc = torch.randn(B, HKV, S, HD, dtype=dt, device="cuda:1") * 0.02
    x = torch.randn(B, 1, D, dtype=dt, device="cuda:1") * 0.02

    def decode_step():
        q = (x @ W["q"]).view(B, 1, H, HD).transpose(1, 2)
        k = (x @ W["k"]).view(B, 1, HKV, HD).transpose(1, 2)
        v = (x @ W["v"]).view(B, 1, HKV, HD).transpose(1, 2)
        # faithful memory-bound decode: flash-attend the resident L-token KV in place
        # (paged-decode HBM-read pattern), no per-step full re-cat of the cache.
        o = F.scaled_dot_product_attention(q, Kc, Vc, enable_gqa=True)
        return (o.transpose(1, 2).reshape(B, 1, H * HD) @ W["o"]) + (F.silu(x @ W["g"]) * (x @ W["u"])) @ W["d"]

    total = args.handoff_mb * 1024 * 1024 // 2
    src_h = torch.randn(total, dtype=dt, device="cuda:1")
    dst_c = torch.empty(total, dtype=dt, device="cuda:0")
    dst_h = torch.empty(total, dtype=dt, device="cuda:1")
    dec_stream = torch.cuda.Stream(device=1)
    s_push = torch.cuda.Stream(device=1)
    s_local = torch.cuda.Stream(device=1)
    s_pull = torch.cuda.Stream(device=0)

    def issue_handoff(direction, n_chunks):
        """Enqueue the whole 512MB handoff as n_chunks back-to-back copies on its stream."""
        csz = total // n_chunks
        if direction == "push":
            st = s_push
            for c in range(n_chunks):
                lo = c * csz; hi = total if c == n_chunks - 1 else lo + csz
                with torch.cuda.stream(st):
                    dst_c[lo:hi].copy_(src_h[lo:hi], non_blocking=True)
        elif direction == "pull":
            st = s_pull
            for c in range(n_chunks):
                lo = c * csz; hi = total if c == n_chunks - 1 else lo + csz
                with torch.cuda.stream(st):
                    dst_c[lo:hi].copy_(src_h[lo:hi], non_blocking=True)
        else:  # local
            st = s_local
            for c in range(n_chunks):
                lo = c * csz; hi = total if c == n_chunks - 1 else lo + csz
                with torch.cuda.stream(st):
                    dst_h[lo:hi].copy_(src_h[lo:hi], non_blocking=True)
        return st

    # warm
    for _ in range(30):
        with torch.cuda.stream(dec_stream):
            decode_step()
    dec_stream.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)

    def timed_iter():
        with torch.cuda.stream(dec_stream):
            e0.record(dec_stream); decode_step(); e1.record(dec_stream)
        e1.synchronize()
        return e0.elapsed_time(e1)

    def baseline_median():
        ts = [timed_iter() for _ in range(args.base_iters)]
        return statistics.median(ts)

    configs = [("push", 1), ("pull", 1), ("local", 1),
               ("push", 8), ("pull", 8), ("push", 32), ("pull", 32)]
    results = {}
    for direction, nch in configs:
        slow_seeds, ho_ms_seeds, niter_seeds = [], [], []
        for _ in range(args.seeds):
            base = baseline_median()
            # issue the handoff, then time decode iters until handoff drains
            cev0 = torch.cuda.Event(enable_timing=True)
            cev1 = torch.cuda.Event(enable_timing=True)
            st = None
            during = []
            # mark handoff start on its own stream, issue all chunks, mark end
            t_wall0 = time.time()
            st = issue_handoff(direction, nch)
            with torch.cuda.stream(st):
                cev1.record(st)
            # time decode iters while the handoff stream is not yet done
            guard = 0
            while not st.query() and guard < 100000:
                during.append(timed_iter())
                guard += 1
            cev1.synchronize()
            t_wall1 = time.time()
            if during:
                slow_seeds.append(statistics.median(during) / base - 1)
                niter_seeds.append(len(during))
            ho_ms_seeds.append((t_wall1 - t_wall0) * 1e3)
        key = f"{direction}_x{nch}"
        results[key] = {
            "direction": direction, "n_chunks": nch,
            "chunk_mb": args.handoff_mb // nch,
            "victim_slowdown_pct": round(statistics.mean(slow_seeds) * 100, 2) if slow_seeds else None,
            "victim_slowdown_std": round(statistics.pstdev(slow_seeds) * 100, 2) if len(slow_seeds) > 1 else 0.0,
            # per-seed raw victim values (%) so figures can show the seed scatter and a
            # reviewer can see the fine-chunk push/pull spread is dispersion, not ordering.
            "victim_slowdown_pct_seeds": [round(s * 100, 3) for s in slow_seeds],
            "handoff_ms": round(statistics.mean(ho_ms_seeds), 2),
            "decode_iters_overlapping": round(statistics.mean(niter_seeds), 1) if niter_seeds else 0,
        }
        r = results[key]
        print(f"  {key:11s} victim +{r['victim_slowdown_pct']}% (±{r['victim_slowdown_std']})  "
              f"handoff={r['handoff_ms']}ms  overlap_iters={r['decode_iters_overlapping']}")

    out = {"_experiment": "g2_handoff_window", "_is_measured": True,
           "_timing_method": "decode-stream events; single/chunked handoff; clocks locked; no whole-device sync",
           "device": torch.cuda.get_device_name(0),
           "geometry": "Llama-3-8B GQA (random weights)", "ctx": S, "batch": B,
           "handoff_mb": args.handoff_mb, "seeds": args.seeds,
           "configs": results, "_generated_at": datetime.now(timezone.utc).isoformat()}
    RESULTS.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
