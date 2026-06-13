"""G5 -- isolate HBM read vs write footprint on a busy decode holder (M1/M2).

Adds synthetic holder-side read-only and write-only activities alongside peer-push
and local-repack controls.  Two synthetic shapes:
  * *_vol   : one 512MB burst at full HBM speed (isolates R vs W at saturation)
  * *_paced : 512MB read/write spread over the same wall time as a peer copy
              (isolates R vs W at matched ~250 GB/s effective rate)

If victim cost at paced read-only matches peer-push but paced write-only is lower,
read-port contention dominates; if both match, total traffic / overlap duration wins.
"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "g5_hbm_isolation.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--handoff-mb", type=int, default=512)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    from umallm.observability import gate_or_skip
    gate_or_skip("g5_hbm_footprint_isolation")

    import torch
    import torch.nn.functional as F

    assert torch.cuda.device_count() >= 2, "need 2 GPUs"
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
    dst_h = torch.empty(n, dtype=dt, device=dev_h)
    scratch = torch.empty(n, dtype=dt, device=dev_h)
    acc = torch.zeros(1, dtype=dt, device=dev_h)

    dec_stream = torch.cuda.Stream(device=1)
    s_push = torch.cuda.Stream(device=1)
    s_local = torch.cuda.Stream(device=1)
    s_read = torch.cuda.Stream(device=1)
    s_write = torch.cuda.Stream(device=1)

    ce0 = torch.cuda.Event(enable_timing=True)
    ce1 = torch.cuda.Event(enable_timing=True)

    def median_op_ms(op, st, reps=20):
        for _ in range(3):
            op()
        st.synchronize()
        ts = []
        for _ in range(reps):
            with torch.cuda.stream(st):
                ce0.record(st)
                op()
                ce1.record(st)
            ce1.synchronize()
            ts.append(ce0.elapsed_time(ce1))
        return statistics.median(ts)

    def peer_op():
        with torch.cuda.stream(s_push):
            dst_c.copy_(src_h, non_blocking=True)

    peer_ms = median_op_ms(peer_op, s_push)
    peer_gbs = nbytes / (peer_ms / 1e3) / 1e9
    print(f"  peer copy: {peer_ms:.3f} ms  {peer_gbs:.1f} GB/s")

    chunk = n // 8  # 64 MB chunks

    def calibrate_paced(read: bool):
        """Find pass count so total bytes=nbytes and wall time ≈ peer_ms."""
        lo, hi = 1, 32
        best = 8
        for _ in range(12):
            passes = (lo + hi) // 2

            def op(passes=passes):
                with torch.cuda.stream(s_read if read else s_write):
                    for _ in range(passes):
                        if read:
                            acc.add_(src_h[:chunk].sum())
                        else:
                            scratch[:chunk].fill_(1.0)

            ms = median_op_ms(op, s_read if read else s_write, reps=12)
            if ms < peer_ms * 0.95:
                lo = passes + 1
            else:
                hi = passes - 1
                best = passes
        return max(1, best)

    read_passes = calibrate_paced(True)
    write_passes = calibrate_paced(False)
    print(f"  paced read/write passes={read_passes}/{write_passes} "
          f"(chunk={chunk*2//1024//1024}MB, total={read_passes*chunk*2//1024//1024}MB)")

    def mk_activity(kind: str):
        if kind == "alone":
            return None, None, 0, 0
        if kind == "peer_push":
            return s_push, peer_op, nbytes, 0
        if kind == "local_repack":
            def op():
                with torch.cuda.stream(s_local):
                    dst_h.copy_(src_h, non_blocking=True)
            return s_local, op, nbytes, nbytes
        if kind == "read_only_vol":
            def op():
                with torch.cuda.stream(s_read):
                    acc.add_(src_h.sum())
            return s_read, op, nbytes, 2
        if kind == "write_only_vol":
            def op():
                with torch.cuda.stream(s_write):
                    scratch.fill_(1.0)
            return s_write, op, 0, nbytes
        if kind == "read_only_paced":
            def op():
                with torch.cuda.stream(s_read):
                    for _ in range(read_passes):
                        acc.add_(src_h[:chunk].sum())
            rb = read_passes * chunk * 2
            return s_read, op, rb, 2
        if kind == "write_only_paced":
            def op():
                with torch.cuda.stream(s_write):
                    for _ in range(write_passes):
                        scratch[:chunk].fill_(1.0)
            wb = write_passes * chunk * 2
            return s_write, op, 0, wb
        raise ValueError(kind)

    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    for _ in range(args.warmup):
        with torch.cuda.stream(dec_stream):
            decode_step()
    dec_stream.synchronize()

    def time_decode(kind: str):
        st, op, _, _ = mk_activity(kind)
        if op:
            for _ in range(3):
                op()
        ts = []
        for _ in range(args.iters):
            if op and st.query():
                op()
            with torch.cuda.stream(dec_stream):
                e0.record(dec_stream)
                decode_step()
                e1.record(dec_stream)
            e1.synchronize()
            ts.append(e0.elapsed_time(e1))
        ts.sort()
        return statistics.median(ts), ts[int(0.75 * len(ts))] - ts[int(0.25 * len(ts))]

    def activity_bw(kind: str):
        st, op, rb, wb = mk_activity(kind)
        if not op:
            return 0.0, 0.0, 0.0
        ms = median_op_ms(op, st, reps=15)
        return round(rb / (ms / 1e3) / 1e9, 1), round(wb / (ms / 1e3) / 1e9, 1), round(ms, 3)

    conditions = [
        "peer_push", "read_only_paced", "write_only_paced",
        "read_only_vol", "write_only_vol", "local_repack",
    ]
    seed_rows = []
    for seed in range(args.seeds):
        torch.manual_seed(1000 + seed)
        base, _ = time_decode("alone")
        row = {"seed": seed, "alone_ms": round(base, 4)}
        for cond in conditions:
            med, iqr = time_decode(cond)
            row[f"{cond}_slowdown_pct"] = round((med / base - 1) * 100, 2)
            row[f"{cond}_iqr"] = round(iqr, 4)
        seed_rows.append(row)
        print(f"  seed {seed}: " + "  ".join(
            f"{c} +{row[f'{c}_slowdown_pct']:.1f}%" for c in conditions))

    def agg(k):
        return [r[k] for r in seed_rows]

    summary = {}
    for cond in conditions:
        rb, wb, op_ms = activity_bw(cond)
        summary[cond] = {
            "victim_slowdown_pct": round(statistics.mean(agg(f"{cond}_slowdown_pct")), 2),
            "victim_slowdown_std": round(statistics.pstdev(agg(f"{cond}_slowdown_pct")), 2)
            if args.seeds > 1 else 0.0,
            "read_bw_gbs": rb, "write_bw_gbs": wb, "op_ms": op_ms,
        }
        s = summary[cond]
        print(f"  {cond:22s} victim=+{s['victim_slowdown_pct']:.1f}%  "
              f"read={rb} write={wb} GB/s  op={op_ms}ms")

    out = {
        "_experiment": "g5_hbm_footprint_isolation",
        "_is_measured": True,
        "_timing_method": "decode-stream CUDA events; saturated activity stream",
        "device": torch.cuda.get_device_name(0),
        "geometry": "Llama-3-8B GQA (random weights)",
        "ctx": S, "batch": B, "handoff_mb": args.handoff_mb,
        "peer_copy_ms": round(peer_ms, 3), "peer_copy_gbs": round(peer_gbs, 1),
        "read_passes": read_passes, "write_passes": write_passes,
        "iters": args.iters, "seeds": args.seeds,
        "conditions": summary, "per_seed": seed_rows,
        "_generated_at": datetime.now(timezone.utc).isoformat(),
    }
    RESULTS.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
