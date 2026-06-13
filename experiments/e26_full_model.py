"""e26 -- FULL-MODEL multi-layer decode: does per-layer cross-GPU sync kill the win?

The reviewer's sharpest gap: every prior peer-parallel number is a SINGLE attention
op. A real decoder runs L layers serially per token, and peer-parallel pays a
cross-GPU round-trip PER LAYER (q broadcast to peer + partial returned = 2 device
syncs/layer, L times, serial because layer i+1 depends on layer i). This script
measures the honest full-model attention-path cost per token:

  * peer_parallel : L layers, each calls peer_parallel_attention over its sharded KV
  * single_gpu    : L layers, fused flash SDPA over full KV on cuda:0 (ref; OOMs big)
  * copyback      : L layers, chunked bounded-peak copy-back of peer KV + online merge

We report ms/token (the L-layer attention loop) with IQR over seeds, at a balanced
split, for one or more context lengths. This isolates the attention/KV path (the part
the design changes); MLP/projection compute is model-common and omitted.

    python experiments/e26_full_model.py --layers 32 --context 16384
"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent / "results" / "full_model.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--context", type=int, default=16384, help="KV tokens per layer")
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--local-frac", type=float, default=0.5)
    ap.add_argument("--chunk-tokens", type=int, default=4096)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from umallm.peer_parallel_attn import peer_parallel_attention, flash_partial, merge_partial

    ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if ngpu < 2:
        OUT.write_text(json.dumps({"_is_measured": False, "note": f"need 2 GPUs (have {ngpu})"}))
        return
    L, C, H, D = args.layers, args.context, args.heads, args.head_dim
    Lc = max(1, min(C-1, round(args.local_frac * C)))   # local tokens/layer
    Pc = C - Lc
    scale = 1.0 / (D ** 0.5)

    def sync():
        torch.cuda.synchronize(0); torch.cuda.synchronize(1)

    def med_iqr(xs):
        xs = sorted(xs); return statistics.median(xs), xs[(3*len(xs))//4]-xs[len(xs)//4]

    # per-layer query (independent; the serial sync structure is what we measure)
    qs = [torch.randn(1, H, 1, D, dtype=torch.float16, device="cuda:0") for _ in range(L)]

    # ---------- peer_parallel: KV sharded per layer (local on cuda:0, peer on cuda:1) ----------
    K0 = [torch.randn(1, H, Lc, D, dtype=torch.float16, device="cuda:0") for _ in range(L)]
    V0 = [torch.randn(1, H, Lc, D, dtype=torch.float16, device="cuda:0") for _ in range(L)]
    K1 = [torch.randn(1, H, Pc, D, dtype=torch.float16, device="cuda:1") for _ in range(L)]
    V1 = [torch.randn(1, H, Pc, D, dtype=torch.float16, device="cuda:1") for _ in range(L)]
    sync()

    def pp_token():
        for i in range(L):
            peer_parallel_attention(qs[i], [(K0[i], V0[i]), (K1[i], V1[i])], scale=scale)

    def bench(fn):
        ms = []
        for _ in range(args.seeds):
            for _ in range(3):
                fn()
            sync(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            for _ in range(args.trials):
                sync(); s.record(); fn(); e.record(); sync()
                ms.append(s.elapsed_time(e))
        return med_iqr(ms)

    pp_ms, pp_iqr = bench(pp_token)
    print(f"  peer_parallel  {pp_ms:.3f} ms/token (IQR {pp_iqr:.3f}) over {L} layers "
          f"= {pp_ms/L*1000:.1f} us/layer")

    # ---------- single_gpu: full KV per layer on cuda:0 ----------
    res = {"layers": L, "context": C, "local_tokens": Lc, "peer_tokens": Pc,
           "heads": H, "head_dim": D, "trials": args.trials, "seeds": args.seeds,
           "device": torch.cuda.get_device_name(0),
           "peer_parallel_ms_per_token": pp_ms, "peer_parallel_iqr": pp_iqr,
           "peer_parallel_us_per_layer": pp_ms/L*1000}
    try:
        Kf = [torch.cat([K0[i], K1[i].to("cuda:0")], dim=2) for i in range(L)]
        Vf = [torch.cat([V0[i], V1[i].to("cuda:0")], dim=2) for i in range(L)]
        sync()
        def sg_token():
            for i in range(L):
                F.scaled_dot_product_attention(qs[i], Kf[i], Vf[i], scale=scale)
        sg_ms, sg_iqr = bench(sg_token)
        res.update({"single_gpu_ms_per_token": sg_ms, "single_gpu_iqr": sg_iqr,
                    "speedup_vs_single": sg_ms/pp_ms,
                    "per_gpu_throughput_ratio_vs_single": (1.0/pp_ms/2)/(1.0/sg_ms)})
        print(f"  single_gpu     {sg_ms:.3f} ms/token (IQR {sg_iqr:.3f})  "
              f"-> peer_parallel {sg_ms/pp_ms:.2f}x vs single (per-GPU tput "
              f"{(1.0/pp_ms/2)/(1.0/sg_ms):.2f}x)")
        del Kf, Vf
    except RuntimeError as e:
        res["single_gpu_status"] = f"OOM: {str(e)[:60]}"
        print("  single_gpu     OOM")
    torch.cuda.empty_cache()

    # ---------- copyback: per-layer chunked bounded-peak copy-back, FAIR baseline ----------
    # Identical fairness treatment to e25: the peer KV is pre-stored as CONTIGUOUS
    # chunks (a paged KV pool is already chunk-contiguous, so a real copy-back never
    # pays a per-step .contiguous()), and each chunk is streamed back with
    # one-step-ahead DOUBLE-BUFFERED prefetch on a side stream so the NVLink transfer
    # overlaps the local+merge compute. (Copy-back is transfer-bound -- e19 shows
    # overlap hides <1% on NVLink -- so this matches the synchronous number; we use the
    # overlapped+pre-contiguous form purely so the comparison cannot be called unfair.)
    Ctok = args.chunk_tokens
    K1c = [[K1[i][:, :, s:s+Ctok, :].contiguous() for s in range(0, Pc, Ctok)] for i in range(L)]
    V1c = [[V1[i][:, :, s:s+Ctok, :].contiguous() for s in range(0, Pc, Ctok)] for i in range(L)]
    sync()
    copy_stream = torch.cuda.Stream(device="cuda:0")
    def cb_token():
        for i in range(L):
            O, l = flash_partial(qs[i], K0[i], V0[i], scale)
            ks, vs = K1c[i], V1c[i]; n = len(ks)
            buf = [None]*n; ev = [torch.cuda.Event() for _ in range(n)]
            with torch.cuda.stream(copy_stream):
                buf[0] = (ks[0].to("cuda:0", non_blocking=True),
                          vs[0].to("cuda:0", non_blocking=True))
                ev[0].record(copy_stream)
            for j in range(n):
                if j + 1 < n:
                    with torch.cuda.stream(copy_stream):
                        buf[j+1] = (ks[j+1].to("cuda:0", non_blocking=True),
                                    vs[j+1].to("cuda:0", non_blocking=True))
                        ev[j+1].record(copy_stream)
                torch.cuda.current_stream().wait_event(ev[j])
                Kc, Vc = buf[j]
                Oc, lc = flash_partial(qs[i], Kc, Vc, scale)
                O, l = merge_partial(O, l, Oc, lc)
                buf[j] = None
    cb_ms, cb_iqr = bench(cb_token)
    res.update({"copyback_ms_per_token": cb_ms, "copyback_iqr": cb_iqr,
                "copyback_baseline": "fair: pre-contiguous chunks + double-buffered overlap (matches e25)",
                "speedup_vs_copyback": cb_ms/pp_ms})
    print(f"  copyback       {cb_ms:.3f} ms/token (IQR {cb_iqr:.3f})  "
          f"-> peer_parallel {cb_ms/pp_ms:.2f}x vs copyback")

    res.update({"_experiment": "e26_full_model", "_is_measured": True,
                "note": ("L-layer serial attention path per token; peer_parallel pays "
                         "2 cross-GPU syncs/layer. Isolates the KV/attention path "
                         "(MLP omitted, model-common)."),
                "_generated_at": datetime.now(timezone.utc).isoformat()})
    # per-context file (avoid overwriting other contexts) + a stable alias
    out_ctx = OUT.parent / f"full_model_L{L}_C{C}.json"
    out_ctx.write_text(json.dumps(res, indent=2))
    OUT.write_text(json.dumps(res, indent=2))
    print(f"  -> wrote {out_ctx}")


if __name__ == "__main__":
    main()
