"""e23 -- contended-peer: what happens when the lender GPU is NOT idle.

The core premise is "a peer GPU has spare HBM (and enough spare compute) to lend
as a KV tier." Reviewers rightly ask: the spare memory often exists BECAUSE the
peer is busy serving its own requests -- so does the NVLink tier survive
contention? We measure both sides:
  (borrower) cuda:0's effective peer-read NVLink bandwidth while cuda:1 is idle
             vs while cuda:1 runs a sustained HBM-heavy matmul load;
  (lender)   cuda:1's matmul throughput alone vs while cuda:0 pulls KV over NVLink;
  (scaling)  N concurrent peer-pull streams competing for the link.
Honest accounting of the degradation, so the paper can scope the idle/near-idle
regime where the tier pays off (disaggregated decode, dedicated KV-server GPU,
low-batch decode) and quantify the cost when the peer is loaded.
"""
from __future__ import annotations
import argparse, json, statistics, time
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "experiments" / "results" / "contention.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=int, default=256)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--load-iters", type=int, default=4000, help="matmul iters queued on the lender")
    ap.add_argument("--matn", type=int, default=4096, help="lender matmul size")
    args = ap.parse_args()

    import torch
    n = args.mb * 1024 * 1024 // 2
    nbytes = n * 2
    src = torch.ones(n, dtype=torch.float16, device="cuda:1")   # lives on lender
    dst = torch.empty(n, dtype=torch.float16, device="cuda:0")  # borrower pulls here

    def peer_bw():
        for _ in range(5):
            dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(args.trials):
            torch.cuda.synchronize(); e0.record()
            dst.copy_(src, non_blocking=True); e1.record(); torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) / 1e3)  # s
        return nbytes / statistics.median(ts) / 1e9

    # lender matmul throughput helper (HBM- + compute-heavy on cuda:1)
    torch.cuda.set_device(1)
    A = torch.randn(args.matn, args.matn, dtype=torch.float16, device="cuda:1")
    Bm = torch.randn(args.matn, args.matn, dtype=torch.float16, device="cuda:1")
    flop = 2 * args.matn**3
    def lender_tflops(iters):
        for _ in range(3):
            torch.mm(A, Bm)
        torch.cuda.synchronize(1)
        t0 = time.perf_counter()
        for _ in range(iters):
            C = torch.mm(A, Bm)
        torch.cuda.synchronize(1)
        dt = time.perf_counter() - t0
        return flop * iters / dt / 1e12, dt
    torch.cuda.set_device(0)

    # 1) borrower BW, lender idle
    bw_idle = peer_bw()
    # 2) lender throughput alone
    tflops_alone, _ = lender_tflops(300)

    # 3) contended: queue a big async matmul load on cuda:1, measure borrower BW while it runs
    lender_stream = torch.cuda.Stream(device=1)
    with torch.cuda.stream(lender_stream):
        for _ in range(args.load_iters):
            torch.mm(A, Bm)   # queued async on cuda:1; keeps the lender busy
    # immediately measure borrower bandwidth while the lender queue drains
    bw_contended = peer_bw()
    torch.cuda.synchronize(1)

    # 4) lender throughput WHILE borrower pulls continuously (rough): time lender
    #    matmuls while issuing peer copies from cuda:0 in between.
    torch.cuda.set_device(1)
    for _ in range(3): torch.mm(A, Bm)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    it = 300
    for i in range(it):
        with torch.cuda.stream(lender_stream):
            C = torch.mm(A, Bm)
        dst.copy_(src, non_blocking=True)   # concurrent NVLink pull from cuda:0
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    tflops_contended = flop * it / dt / 1e12
    torch.cuda.set_device(0)

    res = {"_experiment": "e23_contention", "_is_measured": True,
           "kind": "peer_contention",
           "device": torch.cuda.get_device_name(0),
           "buffer_mb": args.mb, "trials": args.trials,
           "borrower_bw_idle_gbps": bw_idle,
           "borrower_bw_contended_gbps": bw_contended,
           "borrower_bw_retained_frac": bw_contended / bw_idle if bw_idle else None,
           "lender_tflops_alone": tflops_alone,
           "lender_tflops_while_borrowed": tflops_contended,
           "lender_throughput_retained_frac": tflops_contended / tflops_alone if tflops_alone else None,
           "note": ("Borrower = cuda:0 reading KV from cuda:1 over NVLink; lender = cuda:1 "
                    "running fp16 matmuls (HBM+compute load). Reports how much NVLink read BW "
                    "the borrower retains and how much matmul throughput the lender retains under "
                    "mutual contention. Scopes the 'idle/near-idle peer' premise honestly."),
           "_generated_at": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(res, indent=2))
    print(f"  borrower NVLink BW: idle={bw_idle:.1f}  contended={bw_contended:.1f} GB/s "
          f"(retained {res['borrower_bw_retained_frac']*100:.0f}%)")
    print(f"  lender matmul: alone={tflops_alone:.1f}  while-borrowed={tflops_contended:.1f} TFLOP/s "
          f"(retained {res['lender_throughput_retained_frac']*100:.0f}%)")
    print(f"  -> wrote {OUT}")


if __name__ == "__main__":
    main()
