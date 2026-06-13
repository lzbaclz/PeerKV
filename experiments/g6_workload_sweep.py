"""G6 -- workload generality: victim cost vs batch size and context length (M5).

HBM-read intensity of decode changes with batch and context, which sets holder
headroom for a concurrent peer handoff.  We sweep (batch, context) on the same
Llama-3-8B-geometry holder and measure peer-push victim slowdown during a single
512 MB handoff (G3 protocol).
"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "g6_workload_sweep.json"


def run_point(batch: int, ctx: int, handoff_mb: int, seeds: int, base_iters: int):
    import torch
    import torch.nn.functional as F

    dev_h, dev_c = "cuda:1", "cuda:0"
    D, H, HKV, HD, DFF = 4096, 32, 8, 128, 14336
    B, S = batch, ctx
    dt = torch.float16
    torch.cuda.set_device(1)
    try:
        W = {k: torch.randn(*s, dtype=dt, device=dev_h) * 0.02 for k, s in {
            "q": (D, H * HD), "k": (D, HKV * HD), "v": (D, HKV * HD), "o": (H * HD, D),
            "g": (D, DFF), "u": (D, DFF), "d": (DFF, D)}.items()}
        Kc = torch.randn(B, HKV, S, HD, dtype=dt, device=dev_h) * 0.02
        Vc = torch.randn(B, HKV, S, HD, dtype=dt, device=dev_h) * 0.02
        x = torch.randn(B, 1, D, dtype=dt, device=dev_h) * 0.02
    except RuntimeError as e:
        return {"batch": B, "ctx": S, "status": f"OOM: {e}"}

    def decode_step():
        q = (x @ W["q"]).view(B, 1, H, HD).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, Kc, Vc, enable_gqa=True)
        return (o.transpose(1, 2).reshape(B, 1, H * HD) @ W["o"]) + (
            F.silu(x @ W["g"]) * (x @ W["u"])) @ W["d"]

    total = handoff_mb * 1024 * 1024 // 2
    src_h = torch.randn(total, dtype=dt, device=dev_h)
    dst_c = torch.empty(total, dtype=dt, device=dev_c)
    dec_stream = torch.cuda.Stream(device=1)
    cp_stream = torch.cuda.Stream(device=1)

    for _ in range(20):
        with torch.cuda.stream(dec_stream):
            decode_step()
    dec_stream.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)

    def timed_iter():
        with torch.cuda.stream(dec_stream):
            e0.record(dec_stream)
            decode_step()
            e1.record(dec_stream)
        e1.synchronize()
        return e0.elapsed_time(e1)

    slow_seeds, base_seeds = [], []
    for _ in range(seeds):
        base = statistics.median([timed_iter() for _ in range(base_iters)])
        base_seeds.append(base)
        with torch.cuda.stream(cp_stream):
            dst_c.copy_(src_h, non_blocking=True)
        cev1 = torch.cuda.Event(enable_timing=True)
        cev1.record(cp_stream)
        during = []
        guard = 0
        while not cp_stream.query() and guard < 300000:
            during.append(timed_iter())
            guard += 1
        cev1.synchronize()
        if during:
            slow_seeds.append(statistics.median(during) / base - 1)

    return {
        "batch": B, "ctx": S, "status": "ok",
        "decode_ms_median": round(statistics.mean(base_seeds), 4),
        "victim_slowdown_pct": round(statistics.mean(slow_seeds) * 100, 2) if slow_seeds else None,
        "victim_slowdown_std": round(statistics.pstdev(slow_seeds) * 100, 2) if len(slow_seeds) > 1 else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=str, default="1,4,8,16")
    ap.add_argument("--contexts", type=str, default="4096,16384,32768")
    ap.add_argument("--handoff-mb", type=int, default=512)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--base-iters", type=int, default=150)
    args = ap.parse_args()
    from umallm.observability import gate_or_skip
    gate_or_skip("g6_workload_sweep")
    import torch

    batches = [int(b) for b in args.batches.split(",")]
    contexts = [int(c) for c in args.contexts.split(",")]
    rows = []
    for ctx in contexts:
        for batch in batches:
            print(f"=== batch={batch} ctx={ctx} ===")
            row = run_point(batch, ctx, args.handoff_mb, args.seeds, args.base_iters)
            rows.append(row)
            if row.get("status") == "ok":
                print(f"  decode={row['decode_ms_median']:.3f}ms  "
                      f"victim=+{row['victim_slowdown_pct']:.1f}% "
                      f"(±{row['victim_slowdown_std']})")
            else:
                print(f"  {row['status']}")

    out = {
        "_experiment": "g6_workload_sweep",
        "_is_measured": True,
        "_timing_method": "G3 single-handoff protocol; peer_push only",
        "device": torch.cuda.get_device_name(0),
        "geometry": "Llama-3-8B GQA (random weights)",
        "handoff_mb": args.handoff_mb, "seeds": args.seeds,
        "sweep": rows,
        "note": "Victim cost should rise as holder headroom shrinks (higher batch / context).",
        "_generated_at": datetime.now(timezone.utc).isoformat(),
    }
    RESULTS.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
