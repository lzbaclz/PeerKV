"""G9 -- Stream-priority sweep: does CUDA stream priority change the victim cost
or re-introduce a direction effect?

Reviewer question: decode and copy run on separate default-priority streams in
G1--G4; would elevating the decode stream (or the copy stream) change the busy
holder's slowdown under a concurrent peer handoff?

Protocol: identical to G3 (single 512MB handoff, per-iteration decode-stream CUDA
events, no whole-device sync, clocks locked), sweeping
  (decode_priority, copy_priority) in {(0,0), (HI,0), (0,HI), (HI,HI)}
for both directions {peer_push, peer_pull}. HI is the device's greatest (most
negative) CUDA stream priority; 0 is the least, which is also the default, so a
"copy low" cell below default does not exist on CUDA.

Pre-registered decision rule: if every cell is within ~1pp of the default cell,
the paper's "we do not sweep stream priority" limitation is replaced by a null
result; otherwise the paper is left unchanged and the effect is reported for
follow-up.
"""
from __future__ import annotations
import argparse, json, statistics, time
from datetime import datetime, timezone
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "g9_stream_priority.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--handoff-mb", type=int, default=512)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--base-iters", type=int, default=200)
    args = ap.parse_args()
    from umallm.observability import gate_or_skip
    gate_or_skip("g9_stream_priority")
    import torch
    import torch.nn.functional as F

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
        o = F.scaled_dot_product_attention(q, Kc, Vc, enable_gqa=True)
        return (o.transpose(1, 2).reshape(B, 1, H * HD) @ W["o"]) + (F.silu(x @ W["g"]) * (x @ W["u"])) @ W["d"]

    total = args.handoff_mb * 1024 * 1024 // 2
    src_h = torch.randn(total, dtype=dt, device="cuda:1")
    dst_c = torch.empty(total, dtype=dt, device="cuda:0")

    HI = -5  # torch clamps to the device's greatest priority
    cells = [("dec0_cp0", 0, 0), ("decHI_cp0", HI, 0), ("dec0_cpHI", 0, HI), ("decHI_cpHI", HI, HI)]
    directions = ["peer_push", "peer_pull"]

    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)

    results = {}
    for cell, dec_p, cp_p in cells:
        dec_stream = torch.cuda.Stream(device=1, priority=dec_p)
        s_push = torch.cuda.Stream(device=1, priority=cp_p)
        s_pull = torch.cuda.Stream(device=0, priority=cp_p)

        def timed_iter():
            with torch.cuda.stream(dec_stream):
                e0.record(dec_stream); decode_step(); e1.record(dec_stream)
            e1.synchronize()
            return e0.elapsed_time(e1)

        for _ in range(30):
            with torch.cuda.stream(dec_stream):
                decode_step()
        dec_stream.synchronize()

        for direction in directions:
            st = s_push if direction == "peer_push" else s_pull
            slow_seeds, ho_ms = [], []
            for _ in range(args.seeds):
                base = statistics.median([timed_iter() for _ in range(args.base_iters)])
                t0 = time.time()
                with torch.cuda.stream(st):
                    dst_c.copy_(src_h, non_blocking=True)
                during, guard = [], 0
                while not st.query() and guard < 200000:
                    during.append(timed_iter())
                    guard += 1
                st.synchronize()
                ho_ms.append((time.time() - t0) * 1e3)
                if during:
                    slow_seeds.append(statistics.median(during) / base - 1)
            key = f"{cell}_{direction}"
            results[key] = {
                "decode_priority": dec_p, "copy_priority": cp_p, "direction": direction,
                "victim_slowdown_pct": round(statistics.mean(slow_seeds) * 100, 2) if slow_seeds else None,
                "victim_slowdown_std": round(statistics.pstdev(slow_seeds) * 100, 2) if len(slow_seeds) > 1 else 0.0,
                "victim_slowdown_pct_seeds": [round(s * 100, 2) for s in slow_seeds],
                "handoff_ms": round(statistics.mean(ho_ms), 2),
            }
            r = results[key]
            print(f"  {key:22s} victim=+{r['victim_slowdown_pct']:5.2f}% (±{r['victim_slowdown_std']}) "
                  f"handoff={r['handoff_ms']:.2f}ms")

    base_cells = {d: results[f"dec0_cp0_{d}"]["victim_slowdown_pct"] for d in directions}
    max_dev = max(abs(results[f"{c}_{d}"]["victim_slowdown_pct"] - base_cells[d])
                  for c, _, _ in cells for d in directions)
    print(f"\nmax |victim - default| across priority cells: {max_dev:.2f}pp")

    out = {"_experiment": "g9_stream_priority", "_is_measured": True,
           "_timing_method": "G3 protocol; single 512MB handoff; decode-stream events; no whole-device sync",
           "device": __import__("torch").cuda.get_device_name(0),
           "geometry": "Llama-3-8B GQA (random weights)", "ctx": S, "batch": B,
           "handoff_mb": args.handoff_mb, "seeds": args.seeds,
           "priority_hi_requested": HI,
           "cells": results,
           "max_abs_deviation_from_default_pp": round(max_dev, 2),
           "note": "CUDA exposes no priority below default (0 = least), so there is no 'copy low' cell.",
           "_generated_at": datetime.now(timezone.utc).isoformat()}
    RESULTS.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
