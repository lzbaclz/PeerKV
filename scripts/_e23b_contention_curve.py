"""e23b -- the contention RESPONSE SURFACE (the falsifiable gate for CONTEND).

e23 measured one point (square GEMM matn=4096 -> borrower keeps 99.7% NVLink BW,
lender keeps 66.8% FLOPS). CONTEND claims the response is SHAPE-DEPENDENT / non-
monotonic in the lender's op, inducing a context-dependent copy-back<->CFK
crossover. The physical hypothesis: the borrower's NVLink read hits the LENDER's
HBM read port, so

  * a COMPUTE-bound lender (large square GEMM, high arithmetic intensity) leaves
    HBM slack -> borrower keeps ~all its BW, lender loses little;
  * a MEMORY-bound lender (skinny GEMM / decode GEMV, low AI) saturates HBM ->
    borrower BW drops AND/OR lender loses more.

If retention is FLAT across shapes (always ~99.7% borrower), CONTEND collapses to
a monotone roofline (DuetServe absorbs it). If it varies (esp. non-monotonically),
the shape-dependent crossover is real and falsifiable. We sweep lender op shape,
measure (borrower_BW_retained, lender_FLOPS_retained, arithmetic_intensity) per
shape under a SUSTAINED threaded lender + CUDA-event borrower BW, and reproduce
the committed matn=4096 point to settle the 99.7%-vs-48% contradiction.
"""
from __future__ import annotations
import argparse, json, statistics, threading, time
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "experiments" / "results" / "contention_curve.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=int, default=256, help="borrower peer-read buffer")
    ap.add_argument("--trials", type=int, default=40)
    args = ap.parse_args()

    import torch
    dev_b, dev_l = "cuda:0", "cuda:1"   # borrower, lender
    n = args.mb * 1024 * 1024 // 2
    nbytes = n * 2
    src = torch.ones(n, dtype=torch.float16, device=dev_l)   # borrower pulls FROM lender HBM
    dst = torch.empty(n, dtype=torch.float16, device=dev_b)

    def peer_bw():
        for _ in range(5):
            dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize(0)
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(args.trials):
            torch.cuda.synchronize(0); e0.record()
            dst.copy_(src, non_blocking=True); e1.record(); torch.cuda.synchronize(0)
            ts.append(e0.elapsed_time(e1) / 1e3)
        return nbytes / statistics.median(ts) / 1e9

    # ---- lender op zoo: (label, build, flops, bytes_moved, arithmetic_intensity) ----
    def mk_gemm(nn):
        A = torch.randn(nn, nn, dtype=torch.float16, device=dev_l)
        B = torch.randn(nn, nn, dtype=torch.float16, device=dev_l)
        flop = 2 * nn**3
        byts = 3 * nn * nn * 2
        return (lambda: torch.mm(A, B)), flop, byts
    def mk_skinny(m, k, nrhs):  # memory-bound: tall-skinny GEMM ~ batched decode GEMV
        A = torch.randn(m, k, dtype=torch.float16, device=dev_l)
        B = torch.randn(k, nrhs, dtype=torch.float16, device=dev_l)
        flop = 2 * m * k * nrhs
        byts = (m * k + k * nrhs + m * nrhs) * 2
        return (lambda: torch.mm(A, B)), flop, byts

    ops = []
    for nn in (1024, 2048, 4096, 8192):
        op, flop, byts = mk_gemm(nn)
        ops.append((f"gemm{nn}", op, flop, byts))
    # memory-bound lenders (decode-shaped): tall-skinny -> low arithmetic intensity
    for (m, k, r, tag) in [(16384, 16384, 8, "decode_gemv8"), (16384, 16384, 64, "decode_b64"),
                           (4096, 11008, 16, "ffn_b16")]:
        op, flop, byts = mk_skinny(m, k, r)
        ops.append((tag, op, flop, byts))

    # All concurrency via CUDA STREAMS on one thread (NO Python threads -> no GIL
    # starvation). lender ops run on lender_stream (cuda:1); borrower copies on
    # cuda:0's default stream. The device schedules them concurrently; the
    # borrower's peer-read and the lender op both hit cuda:1's HBM.
    lender_stream = torch.cuda.Stream(device=1)

    def lender_thru_alone(op, flop, iters=200):
        torch.cuda.set_device(1)
        for _ in range(3): op()
        torch.cuda.synchronize(1)
        t0 = time.perf_counter()
        for _ in range(iters): op()
        torch.cuda.synchronize(1)
        torch.cuda.set_device(0)
        return flop * iters / (time.perf_counter() - t0) / 1e12, (time.perf_counter()-t0)/iters

    def queue_lender(op, n):
        with torch.cuda.stream(lender_stream):
            for _ in range(n): op()

    bw_idle = peer_bw()
    rows = []
    for tag, op, flop, byts in ops:
        ai = flop / byts
        thru_alone, per_iter_s = lender_thru_alone(op, flop)
        # keep the lender busy ~250ms (covers the borrower BW measurement window),
        # then measure borrower BW while the lender stream drains concurrently.
        n_busy = max(50, int(0.25 / max(per_iter_s, 1e-5)))
        torch.cuda.synchronize(1)
        queue_lender(op, n_busy)
        bw_cont = peer_bw()                       # runs concurrently with the draining lender
        torch.cuda.synchronize(1)
        # lender throughput WHILE borrower pulls: queue many borrower copies on
        # cuda:0, then time the lender op sequence on cuda:1 (both concurrent).
        bstream = torch.cuda.Stream(device=0)
        with torch.cuda.stream(bstream):
            for _ in range(2000): dst.copy_(src, non_blocking=True)
        torch.cuda.set_device(1)
        for _ in range(3): op()
        torch.cuda.synchronize(1)
        t0 = time.perf_counter()
        it = 200
        for _ in range(it): op()
        torch.cuda.synchronize(1)
        thru_cont = flop * it / (time.perf_counter() - t0) / 1e12
        torch.cuda.set_device(0); torch.cuda.synchronize(0)
        row = {"op": tag, "arithmetic_intensity": round(ai, 2),
               "borrower_bw_idle": round(bw_idle, 1), "borrower_bw_contended": round(bw_cont, 1),
               "borrower_bw_retained": round(bw_cont / bw_idle, 4),
               "lender_tflops_alone": round(thru_alone, 1), "lender_tflops_contended": round(thru_cont, 1),
               "lender_flops_retained": round(thru_cont / thru_alone, 4) if thru_alone else None}
        rows.append(row)
        print(f"  {tag:14s} AI={ai:7.1f}  borrower_BW_ret={row['borrower_bw_retained']*100:5.1f}%  "
              f"lender_FLOPS_ret={row['lender_flops_retained']*100:5.1f}%  ({thru_alone:.0f}->{thru_cont:.0f} TF)")

    # monotonicity / shape-dependence verdict
    br = [r["borrower_bw_retained"] for r in rows]
    lr = [r["lender_flops_retained"] for r in rows]
    br_spread = max(br) - min(br); lr_spread = max(lr) - min(lr)
    # sort by AI; non-monotone if the retention sequence reverses direction
    by_ai = sorted(rows, key=lambda r: r["arithmetic_intensity"])
    def nonmono(seq):
        diffs = [b - a for a, b in zip(seq, seq[1:])]
        return any(d > 1e-3 for d in diffs) and any(d < -1e-3 for d in diffs)
    verdict = {
        "borrower_bw_retained_spread": round(br_spread, 4),
        "lender_flops_retained_spread": round(lr_spread, 4),
        "borrower_shape_dependent": br_spread > 0.10,        # >10pp swing = real shape dependence
        "lender_shape_dependent": lr_spread > 0.10,
        "borrower_nonmonotonic_in_AI": nonmono([r["borrower_bw_retained"] for r in by_ai]),
        "lender_nonmonotonic_in_AI": nonmono([r["lender_flops_retained"] for r in by_ai]),
        "matn4096_reproduces_committed": next((abs(r["borrower_bw_retained"] - 0.997) < 0.05
                                               and abs(r["lender_flops_retained"] - 0.668) < 0.10
                                               for r in rows if r["op"] == "gemm4096"), None),
    }
    contend_lives = verdict["borrower_shape_dependent"] or verdict["lender_shape_dependent"] \
        or verdict["borrower_nonmonotonic_in_AI"] or verdict["lender_nonmonotonic_in_AI"]
    res = {"_experiment": "e23b_contention_curve", "_is_measured": True,
           "device": torch.cuda.get_device_name(0), "trials": args.trials,
           "rows": rows, "verdict": verdict,
           "CONTEND_gate": "SURVIVES (shape-dependent / non-monotonic)" if contend_lives
                           else "COLLAPSES (flat -> monotone roofline, DuetServe absorbs)",
           "note": ("Borrower NVLink read contends for the LENDER's HBM read port. Compute-bound "
                    "lenders (high AI) leave HBM slack; memory-bound lenders (low AI, decode GEMV) "
                    "saturate it. If retention varies with op shape/AI, the copy-back<->CFK crossover "
                    "is shape-dependent and falsifiable; if flat, CONTEND collapses."),
           "_generated_at": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(res, indent=2))
    print(f"\n  borrower_BW spread={br_spread*100:.1f}pp  lender_FLOPS spread={lr_spread*100:.1f}pp")
    print(f"  matn4096 reproduces committed (99.7/66.8): {verdict['matn4096_reproduces_committed']}")
    print(f"  CONTEND gate: {res['CONTEND_gate']}")
    print(f"  -> wrote {OUT}")


if __name__ == "__main__":
    main()
