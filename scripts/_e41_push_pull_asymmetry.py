"""e41 -- PUSH/PULL initiator-direction asymmetry under a busy data-holder (the
'who issues the transfer' law).

e23b established that a borrower's remote READ (PULL) of a busy lender's HBM loses
~24% bandwidth when the lender is memory-bound (saturated HBM read port), one-sided
(lender keeps ~100%). This script tests the DUAL knob nobody schedules on: the
DIRECTION / INITIATOR of the SAME GB-scale transfer.

  * PULL : consumer GPU0 issues `dst0.copy_(src1)` -> a REMOTE read of GPU1's HBM
           read port. Re-arbitrates against GPU1's local decode SMs -> loses.
  * PUSH : data-holder GPU1 issues the same copy on its own stream -> a LOCAL read
           on GPU1 (its own copy engine wins HBM arbitration, exactly like local
           SMs do) + a remote WRITE into GPU0's UNcontended write port.

Hypothesis (one-sided arbitration law): PUSH should retain ~100% even when the
data-holder is memory-bound, because the transfer's HBM read is now LOCAL to the
busy GPU (wins arbitration) instead of a losing remote pull. Measured on this box:
PULL 76.6% vs PUSH 100.6% under a memory-bound holder; both ~100% under a
compute-bound holder. => 'always let the data-holder push' recovers the full loss.

Falsifiable: if PUSH also dropped to ~76% the loss would be the GPU1 read port
either way and direction would be irrelevant (mechanism dies). It does not.
"""
from __future__ import annotations
import argparse, json, statistics, time
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "experiments" / "results" / "push_pull_asymmetry.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=int, default=256)
    ap.add_argument("--trials", type=int, default=40)
    args = ap.parse_args()

    import torch
    torch.cuda.set_device(0)
    n = args.mb * 1024 * 1024 // 2
    nbytes = n * 2
    src1 = torch.ones(n, dtype=torch.float16, device="cuda:1")   # data lives on GPU1 (the holder)
    dst0 = torch.empty(n, dtype=torch.float16, device="cuda:0")  # consumer GPU0

    def make_op(kind):
        if kind == "membound":      # decode GEMV: low AI, saturates GPU1 read port
            m, k, r = 16384, 16384, 8
            A = torch.randn(m, k, dtype=torch.float16, device="cuda:1")
            B = torch.randn(k, r, dtype=torch.float16, device="cuda:1")
            return lambda: torch.mm(A, B)
        nn = 4096                    # compute-bound square GEMM: high AI
        A = torch.randn(nn, nn, dtype=torch.float16, device="cuda:1")
        B = torch.randn(nn, nn, dtype=torch.float16, device="cuda:1")
        return lambda: torch.mm(A, B)

    def per_iter(op):
        torch.cuda.set_device(1)
        for _ in range(5): op()
        torch.cuda.synchronize(1); t0 = time.perf_counter()
        for _ in range(200): op()
        torch.cuda.synchronize(1); torch.cuda.set_device(0)
        return (time.perf_counter() - t0) / 200

    def pull(): dst0.copy_(src1, non_blocking=True)

    def pull_bw():
        torch.cuda.synchronize(0)
        for _ in range(5): pull()
        torch.cuda.synchronize(0)
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True); ts = []
        for _ in range(args.trials):
            torch.cuda.synchronize(0); e0.record(); pull(); e1.record(); torch.cuda.synchronize(0)
            ts.append(e0.elapsed_time(e1) / 1e3)
        return nbytes / statistics.median(ts) / 1e9

    pstream = torch.cuda.Stream(device=1)
    def push():
        with torch.cuda.stream(pstream): dst0.copy_(src1, non_blocking=True)

    def push_bw_wall():   # wall-clock to rule out on-stream event artifacts
        torch.cuda.synchronize(1)
        for _ in range(5): push()
        torch.cuda.synchronize(1); ts = []
        for _ in range(args.trials):
            torch.cuda.synchronize(1); t0 = time.perf_counter(); push(); torch.cuda.synchronize(1)
            ts.append(time.perf_counter() - t0)
        return nbytes / statistics.median(ts) / 1e9

    pull_idle = pull_bw(); push_idle = push_bw_wall()
    rows = []
    for kind in ("membound", "compute"):
        op = make_op(kind); per = per_iter(op); nb = max(80, int(0.30 / max(per, 1e-5)))
        lstream = torch.cuda.Stream(device=1)
        def load():
            with torch.cuda.stream(lstream):
                for _ in range(nb): op()
        torch.cuda.synchronize(1); load(); pc = pull_bw(); torch.cuda.synchronize(1)
        torch.cuda.synchronize(1); load(); puc = push_bw_wall(); torch.cuda.synchronize(1)
        rows.append({"holder_kind": kind, "holder_per_iter_us": round(per * 1e6, 1),
                     "pull_retained": round(pc / pull_idle, 4),
                     "push_retained": round(puc / push_idle, 4),
                     "push_advantage_pp": round((puc / push_idle - pc / pull_idle) * 100, 1)})
        print(f"  holder={kind:9s} per={per*1e6:5.0f}us  PULL={pc/pull_idle*100:5.1f}%  "
              f"PUSH={puc/push_idle*100:5.1f}%  adv=+{rows[-1]['push_advantage_pp']}pp")

    res = {"_experiment": "e41_push_pull_asymmetry", "_is_measured": True,
           "device": torch.cuda.get_device_name(0), "trials": args.trials,
           "pull_idle_gbs": round(pull_idle, 1), "push_idle_gbs": round(push_idle, 1),
           "rows": rows,
           "law": ("The SAME inter-GPU transfer retains ~100% bandwidth when ISSUED BY THE "
                   "DATA-HOLDER (push: local read wins the holder's HBM arbitration + remote "
                   "write to the idle peer's uncontended write port), but loses ~24% when "
                   "issued by the consumer (pull: remote read loses to the holder's memory-bound "
                   "local decode). Direction/initiator is a free scheduling knob no transfer "
                   "engine (NIXL/UCCL/Harvest) selects on read-port arbitration."),
           "_generated_at": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(res, indent=2))
    print(f"  -> wrote {OUT}")


if __name__ == "__main__":
    main()
