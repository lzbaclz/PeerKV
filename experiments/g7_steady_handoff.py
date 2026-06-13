"""G7 -- sustained back-to-back handoffs: transient vs steady-state victim tax (M3).

G1/G3 measure a *single* 512 MB handoff overlapping a few decode iterations.
Serving pools issue handoffs continuously.  This probe keeps the copy stream
saturated with back-to-back peer handoffs while decode runs, and compares
steady-state per-iteration decode time to the alone baseline.
"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "g7_steady_handoff.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--handoff-mb", type=int, default=512)
    ap.add_argument("--steady-secs", type=float, default=8.0,
                    help="seconds of saturated handoff before sampling decode")
    ap.add_argument("--sample-iters", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    from umallm.observability import gate_or_skip
    gate_or_skip("g7_steady_handoff")

    import torch
    import torch.nn.functional as F
    import time

    assert torch.cuda.device_count() >= 2
    dev_h, dev_c = "cuda:1", "cuda:0"
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
        o = F.scaled_dot_product_attention(q, Kc, Vc, enable_gqa=True)
        return (o.transpose(1, 2).reshape(B, 1, H * HD) @ W["o"]) + (
            F.silu(x @ W["g"]) * (x @ W["u"])) @ W["d"]

    n = args.handoff_mb * 1024 * 1024 // 2
    nbytes = n * 2
    src_h = torch.randn(n, dtype=dt, device=dev_h)
    dst_c = torch.empty(n, dtype=dt, device=dev_c)
    dec_stream = torch.cuda.Stream(device=1)
    cp_stream = torch.cuda.Stream(device=1)

    def handoff():
        with torch.cuda.stream(cp_stream):
            dst_c.copy_(src_h, non_blocking=True)

    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)

    for _ in range(30):
        with torch.cuda.stream(dec_stream):
            decode_step()
    dec_stream.synchronize()

    def sample_decode(n_iters: int):
        ts = []
        for _ in range(n_iters):
            with torch.cuda.stream(dec_stream):
                e0.record(dec_stream)
                decode_step()
                e1.record(dec_stream)
            e1.synchronize()
            ts.append(e0.elapsed_time(e1))
        ts.sort()
        return statistics.median(ts), ts[int(0.75 * len(ts))] - ts[int(0.25 * len(ts))]

    seed_rows = []
    for seed in range(args.seeds):
        torch.manual_seed(2000 + seed)
        alone_med, alone_iqr = sample_decode(args.sample_iters)

        # saturate copy stream for steady_secs, decode underneath
        for _ in range(3):
            handoff()
        t_end = time.time() + args.steady_secs
        copies = 0
        while time.time() < t_end:
            if cp_stream.query():
                handoff()
                copies += 1
            with torch.cuda.stream(dec_stream):
                for _ in range(5):
                    decode_step()

        steady_med, steady_iqr = sample_decode(args.sample_iters)
        cp_stream.synchronize()
        dt_run = args.steady_secs
        sustained_gbs = copies * nbytes / dt_run / 1e9 if dt_run > 0 else 0
        row = {
            "seed": seed,
            "alone_ms": round(alone_med, 4),
            "steady_ms": round(steady_med, 4),
            "steady_slowdown_pct": round((steady_med / alone_med - 1) * 100, 2),
            "alone_iqr": round(alone_iqr, 4),
            "steady_iqr": round(steady_iqr, 4),
            "handoffs_in_warmup_window": copies,
            "sustained_copy_gbs": round(sustained_gbs, 1),
        }
        seed_rows.append(row)
        print(f"  seed {seed}: alone={alone_med:.3f}ms  steady=+{row['steady_slowdown_pct']:.1f}%  "
              f"({copies} handoffs / {args.steady_secs:.0f}s ~ {sustained_gbs:.0f} GB/s)")

    def agg(k):
        return [r[k] for r in seed_rows]

    summary = {
        "alone_ms": round(statistics.mean(agg("alone_ms")), 4),
        "steady_slowdown_pct": round(statistics.mean(agg("steady_slowdown_pct")), 2),
        "steady_slowdown_std": round(statistics.pstdev(agg("steady_slowdown_pct")), 2)
        if args.seeds > 1 else 0.0,
        "sustained_copy_gbs": round(statistics.mean(agg("sustained_copy_gbs")), 1),
    }

    out = {
        "_experiment": "g7_steady_handoff",
        "_is_measured": True,
        "_timing_method": "back-to-back peer_push; steady-state decode after warm saturation",
        "device": torch.cuda.get_device_name(0),
        "geometry": "Llama-3-8B GQA (random weights)",
        "ctx": S, "batch": B, "handoff_mb": args.handoff_mb,
        "steady_secs": args.steady_secs, "sample_iters": args.sample_iters,
        "seeds": args.seeds,
        "summary": summary,
        "per_seed": seed_rows,
        "note": "If steady_slowdown ≈ single-handoff G3 victim, the +9% is sustained not transient.",
        "_generated_at": datetime.now(timezone.utc).isoformat(),
    }
    RESULTS.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"-> {OUT}  steady victim +{summary['steady_slowdown_pct']:.1f}% "
          f"(±{summary['steady_slowdown_std']})")


if __name__ == "__main__":
    main()
