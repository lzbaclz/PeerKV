"""G15 -- decode-intensity sweep: victim%% vs holder batch size.

Reviewer ask: show the budget depends on the holder's own HBM headroom.
Protocol is g1-style (copy topped up before every timed iteration, never
drained) to avoid the g6 single-handoff window artifact at large batch.
Peer push + pull at each batch size; clocks locked.
"""
from __future__ import annotations
import argparse, json, statistics
from datetime import datetime, timezone
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "g15_decode_intensity.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--handoff-mb", type=int, default=512)
    ap.add_argument("--batches", type=str, default="1,4,8,16")
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    from umallm.observability import gate_or_skip
    gate_or_skip("g15_decode_intensity")
    import torch
    import torch.nn.functional as F

    D, H, HKV, HD, DFF = 4096, 32, 8, 128, 14336
    S, dt = args.ctx, torch.float16
    torch.cuda.set_device(1)
    W = {k: torch.randn(*s, dtype=dt, device="cuda:1") * 0.02 for k, s in {
        "q": (D, H * HD), "k": (D, HKV * HD), "v": (D, HKV * HD), "o": (H * HD, D),
        "g": (D, DFF), "u": (D, DFF), "d": (DFF, D)}.items()}

    total = args.handoff_mb * 1024 * 1024 // 2
    nbytes = total * 2
    src_h = torch.randn(total, dtype=dt, device="cuda:1")
    dst_c = torch.empty(total, dtype=dt, device="cuda:0")
    s_dev1 = torch.cuda.Stream(device=1)
    s_dev0 = torch.cuda.Stream(device=0)
    dec_stream = torch.cuda.Stream(device=1)
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)

    def copy_for(kind):
        st = s_dev1 if kind == "push" else s_dev0
        return st, (lambda: dst_c.copy_(src_h, non_blocking=True))

    results = {}
    for B in [int(x) for x in args.batches.split(",")]:
        Kc = torch.randn(B, HKV, S, HD, dtype=dt, device="cuda:1") * 0.02
        Vc = torch.randn(B, HKV, S, HD, dtype=dt, device="cuda:1") * 0.02
        x = torch.randn(B, 1, D, dtype=dt, device="cuda:1") * 0.02

        def decode_step():
            q = (x @ W["q"]).view(B, 1, H, HD).transpose(1, 2)
            k = (x @ W["k"]); v = (x @ W["v"])  # noqa
            o = F.scaled_dot_product_attention(q, Kc, Vc, enable_gqa=True)
            return (o.transpose(1, 2).reshape(B, 1, H * HD) @ W["o"]) + \
                   (F.silu(x @ W["g"]) * (x @ W["u"])) @ W["d"]

        def timed_iter():
            with torch.cuda.device(1), torch.cuda.stream(dec_stream):
                e0.record(dec_stream); decode_step(); e1.record(dec_stream)
            e1.synchronize()
            return e0.elapsed_time(e1)

        for _ in range(30):
            with torch.cuda.stream(dec_stream):
                decode_step()
        dec_stream.synchronize()

        row = {}
        base_ms = statistics.median([timed_iter() for _ in range(args.iters)])
        row["alone_ms"] = round(base_ms, 4)
        for kind in ("push", "pull"):
            st, op = copy_for(kind)
            slows = []
            for _ in range(args.seeds):
                base = statistics.median([timed_iter() for _ in range(args.iters)])
                # saturate: top up before each timed iter (g1 protocol)
                with torch.cuda.device(st.device), torch.cuda.stream(st):
                    op(); op()
                during, busy = [], 0
                for _ in range(args.iters):
                    if st.query():
                        with torch.cuda.device(st.device), torch.cuda.stream(st):
                            op()
                    if not st.query():
                        busy += 1
                    during.append(timed_iter())
                st.synchronize()
                slows.append(statistics.median(during) / base - 1)
            row[kind] = {
                "victim_slowdown_pct": round(statistics.mean(slows) * 100, 2),
                "victim_slowdown_std": round(statistics.pstdev(slows) * 100, 2) if len(slows) > 1 else 0.0,
                "copy_busy_frac": round(busy / args.iters, 3),
            }
        results[f"batch_{B}"] = row
        print(f"  B={B:2d} alone={row['alone_ms']:.3f}ms  "
              f"push=+{row['push']['victim_slowdown_pct']:.2f}% "
              f"pull=+{row['pull']['victim_slowdown_pct']:.2f}% "
              f"(busy {row['push']['copy_busy_frac']})")

    out = {"_experiment": "g15_decode_intensity", "_is_measured": True,
           "_timing_method": "decode-stream events; copy topped up per iteration (g1 protocol); clocks locked",
           "device": torch.cuda.get_device_name(0),
           "geometry": "Llama-3-8B GQA (random weights)", "ctx": S,
           "handoff_mb": args.handoff_mb, "iters": args.iters, "seeds": args.seeds,
           "results": results,
           "_generated_at": datetime.now(timezone.utc).isoformat()}
    RESULTS.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
