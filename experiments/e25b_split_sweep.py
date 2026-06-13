"""e25b -- split-ratio sweep with error bars (the inverted-U, RQ4 design space).

Per-step peer-parallel latency vs the local fraction, median + IQR over seeds, with
the single-GPU reference. Shows the cost model's load-balancing prediction: the
minimum-latency point is the even (50/50) split, where max_d(local HBM read) is
minimized and the two GPUs' bandwidths add. Writes split_sweep.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent / "results" / "split_sweep.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-blocks", type=int, default=2048)
    ap.add_argument("--block-tokens", type=int, default=256)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--fracs", default="0.125,0.25,0.5,0.75,0.875")
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from umallm.peer_parallel_attn import flash_partial, merge_partial

    ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if ngpu < 2:
        OUT.write_text(json.dumps({"_is_measured": False, "note": f"need 2 GPUs (have {ngpu})"}))
        return
    H, T, D, N = args.heads, args.block_tokens, args.head_dim, args.n_blocks
    scale = 1.0 / (D ** 0.5)

    def sync():
        torch.cuda.synchronize(0); torch.cuda.synchronize(1)

    def med_iqr(xs):
        xs = sorted(xs); return statistics.median(xs), xs[(3*len(xs))//4]-xs[len(xs)//4]

    q = torch.randn(1, H, 1, D, dtype=torch.float16, device="cuda:0")

    # single-GPU reference (full KV on cuda:0)
    Ka = torch.randn(1, H, N*T, D, dtype=torch.float16, device="cuda:0")
    Va = torch.randn(1, H, N*T, D, dtype=torch.float16, device="cuda:0")
    sync()
    ms = []
    for _ in range(args.seeds):
        for _ in range(5):
            F.scaled_dot_product_attention(q, Ka, Va, scale=scale)
        sync(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        for _ in range(args.trials):
            sync(); s.record(); F.scaled_dot_product_attention(q, Ka, Va, scale=scale); e.record(); sync()
            ms.append(s.elapsed_time(e))
    sg_ms, sg_iqr = med_iqr(ms)
    del Ka, Va; torch.cuda.empty_cache()
    print(f"  single_gpu {sg_ms:.3f} ms (IQR {sg_iqr:.3f})")

    rows = []
    for frac in [float(x) for x in args.fracs.split(",")]:
        L = max(1, min(N-1, round(frac*N))); P = N-L
        K0 = torch.randn(1, H, L*T, D, dtype=torch.float16, device="cuda:0")
        V0 = torch.randn(1, H, L*T, D, dtype=torch.float16, device="cuda:0")
        K1 = torch.randn(1, H, P*T, D, dtype=torch.float16, device="cuda:1")
        V1 = torch.randn(1, H, P*T, D, dtype=torch.float16, device="cuda:1")
        sync()

        def step():
            q1 = q.to("cuda:1", non_blocking=True)
            O1, l1 = flash_partial(q1, K1, V1, scale)
            O0, l0 = flash_partial(q, K0, V0, scale)
            o, _ = merge_partial(O0, l0, O1.to("cuda:0", non_blocking=True),
                                 l1.to("cuda:0", non_blocking=True))
            return o
        ms = []
        for _ in range(args.seeds):
            for _ in range(5):
                step()
            sync(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            for _ in range(args.trials):
                sync(); s.record(); step(); e.record(); sync()
                ms.append(s.elapsed_time(e))
        m, iqr = med_iqr(ms)
        rows.append({"local_frac": frac, "local_blocks": L, "peer_blocks": P,
                     "ms": m, "iqr_ms": iqr, "speedup_vs_single": sg_ms/m})
        print(f"  frac={frac:.3f}  L={L:>4}/P={P:<4}  {m:.3f} ms (IQR {iqr:.3f})  "
              f"{sg_ms/m:.2f}x vs single")
        del K0, V0, K1, V1; torch.cuda.empty_cache()

    out = {"_experiment": "e25b_split_sweep", "_is_measured": True,
           "n_blocks": N, "block_tokens": T, "heads": H, "head_dim": D,
           "trials": args.trials, "seeds": args.seeds,
           "single_gpu_ms": sg_ms, "single_gpu_iqr_ms": sg_iqr, "sweep": rows,
           "note": ("inverted-U: latency minimized at the even split (bandwidths add); "
                    "speedup_vs_single>1 only where it also fits one GPU."),
           "_generated_at": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(out, indent=2))
    print(f"  -> wrote {OUT}")


if __name__ == "__main__":
    main()
