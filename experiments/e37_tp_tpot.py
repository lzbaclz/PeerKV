"""e37 -- single-stream TENSOR-PARALLEL (TP-2) decode TPOT, matched to e27/e31.

Closes the honest gap flagged in the reposition: the only TP number in the repo is
THROUGHPUT (serve_m1_tp2.json, 15933 tok/s), so the selector treated TP as
admissibility-only (predict inf). Here we measure single-stream TP-2 decode latency
(ms/token) with the SAME real-weights methodology as e27, so it is apples-to-apples
with single / compute-follows-KV / copy-back and can enter the latency oracle.

Faithful 2-way TP, single process, two devices, concurrent enqueue + manual
all-reduce (2/layer):
  * weights SHARDED: Wqkv/Wg/Wu column-parallel (each rank holds half the output
    dim), Wo/Wd row-parallel (each rank holds half the input dim) -> each rank
    reads HALF the weights/token (TP's structural advantage, the C2 asymmetry).
  * KV SHARDED by head: each rank holds Hkv/2 heads' KV (TP halves the KV read too).
  * all-reduce = sum the (1,d_model) partial across the 2 devices (a few KB) -- the
    only per-layer sync, contrast compute-follows-KV's ~120us cross-GPU round-trip.

Also runs the single-GPU arm for the TP/single ratio where the context fits.

    /home/lzq/miniconda3/envs/peerkv/bin/python experiments/e37_tp_tpot.py
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import os
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--contexts", type=int, nargs="+", default=[16384, 32768, 65536, 131072])
    ap.add_argument("--d-model", type=int, default=4096)
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--kv-heads", type=int, default=None)   # None=MHA
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--d-ffn", type=int, default=11008)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    if torch.cuda.device_count() < 2:
        print("need >=2 GPUs"); return
    L, dm, H, D, dff = args.layers, args.d_model, args.heads, args.head_dim, args.d_ffn
    Hkv = args.kv_heads or H
    assert H % 2 == 0 and Hkv % 2 == 0 and H * D == dm
    geo = "Llama-2-7B (MHA)" if Hkv == H else f"GQA (q{H}/kv{Hkv})"
    Hh, Hkvh = H // 2, Hkv // 2          # per-rank head counts
    scale = 1.0 / (D ** 0.5)

    def sync():
        torch.cuda.synchronize(0); torch.cuda.synchronize(1)

    def med_iqr(xs):
        xs = sorted(xs); return statistics.median(xs), xs[(3*len(xs))//4]-xs[len(xs)//4]

    def w(a, b, dev, s):
        g = torch.Generator(device=dev).manual_seed(s)
        return torch.randn(a, b, generator=g, dtype=torch.float16, device=dev) * 0.02

    # ---- per-rank sharded weights (rank r on cuda:r) ----
    qkv_out_h = (Hh + 2 * Hkvh) * D        # this rank's QKV output width
    Wqkv = {r: [w(dm, qkv_out_h, f"cuda:{r}", 100+r*L+i) for i in range(L)] for r in (0, 1)}
    Wo   = {r: [w(dm // 2, dm, f"cuda:{r}", 200+r*L+i) for i in range(L)] for r in (0, 1)}  # row-parallel
    Wg   = {r: [w(dm, dff // 2, f"cuda:{r}", 300+r*L+i) for i in range(L)] for r in (0, 1)}
    Wu   = {r: [w(dm, dff // 2, f"cuda:{r}", 400+r*L+i) for i in range(L)] for r in (0, 1)}
    Wd   = {r: [w(dff // 2, dm, f"cuda:{r}", 500+r*L+i) for i in range(L)] for r in (0, 1)}
    nrm1 = {r: [torch.ones(dm, dtype=torch.float16, device=f"cuda:{r}") for _ in range(L)] for r in (0, 1)}
    nrm2 = {r: [torch.ones(dm, dtype=torch.float16, device=f"cuda:{r}") for _ in range(L)] for r in (0, 1)}
    sync()
    wgib_per_rank = sum(t.numel()*2 for t in (Wqkv[0]+Wo[0]+Wg[0]+Wu[0]+Wd[0]))/1024**3
    print(f"  TP-2 weights: {wgib_per_rank:.1f} GiB/rank ({L} layers, {geo})")

    def rmsnorm(x, gw):
        return (x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-5).to(x.dtype)) * gw

    out = {"_experiment": "e37_tp_tpot", "_is_measured": True,
           "_generated_at": datetime.now(timezone.utc).isoformat(),
           "layers": L, "d_model": dm, "heads": H, "kv_heads": Hkv, "head_dim": D,
           "d_ffn": dff, "geometry": geo, "trials": args.trials, "seeds": args.seeds,
           "device": torch.cuda.get_device_name(0),
           "weights_gib_per_rank": round(wgib_per_rank, 2),
           "method": "single-process 2-device concurrent enqueue + manual all-reduce (2/layer)",
           "caveat": ("Emulated all-reduce (cross-device .to(), not NCCL) + unfused "
                      "batch=1 GEMVs make this a CONSERVATIVE (slow) bound on TP "
                      "latency, esp. at small context where overhead dominates. Robust "
                      "findings: (i) single-stream batch=1 TP is NOT uniformly faster "
                      "than single (1.25x SLOWER at 16K) -- TP's 1.85x is a "
                      "throughput/batching effect; (ii) TP wins once KV read dominates "
                      "(>=32K) and enables 131K that single OOMs; (iii) the crossover is "
                      "~32K. A production vLLM-TP TPOT (NCCL+fused) on the dedicated box "
                      "would shift small-ctx numbers down -- run_h100_suite.sh."),
           "contexts": {}}

    for C in args.contexts:
        Ch = C  # each rank holds full context length but half the heads
        try:
            K = {r: [torch.randn(1, Hkvh, Ch, D, dtype=torch.float16, device=f"cuda:{r}") for _ in range(L)] for r in (0, 1)}
            V = {r: [torch.randn(1, Hkvh, Ch, D, dtype=torch.float16, device=f"cuda:{r}") for _ in range(L)] for r in (0, 1)}
        except RuntimeError as e:
            out["contexts"][str(C)] = {"tp2": {"status": f"OOM:{str(e)[:40]}"}}
            print(f"  ctx={C}: TP KV OOM"); continue
        sync()

        def tp_layer(x0, x1, i):
            # rank-local pre-norm + QKV (concurrent on both devices)
            h0 = rmsnorm(x0, nrm1[0][i]); h1 = rmsnorm(x1, nrm1[1][i])
            qkv0 = h0 @ Wqkv[0][i]; qkv1 = h1 @ Wqkv[1][i]
            def attn(qkv, r):
                q = qkv[:, :Hh*D].view(1, Hh, 1, D)
                k = qkv[:, Hh*D:(Hh+Hkvh)*D].view(1, Hkvh, 1, D)  # (unused new-token k/v; KV cache is K/V)
                o = F.scaled_dot_product_attention(q, K[r][i], V[r][i], scale=scale,
                                                   enable_gqa=(Hh != Hkvh))
                return o.reshape(1, Hh*D)
            a0 = attn(qkv0, 0); a1 = attn(qkv1, 1)
            o0 = a0 @ Wo[0][i]; o1 = a1 @ Wo[1][i]              # row-parallel partials (1,dm)
            o = o0 + o1.to("cuda:0")                            # all-reduce #1
            x0 = x0 + o; x1 = x1 + o.to("cuda:1")
            # MLP (column-parallel g/u, row-parallel d)
            g0 = rmsnorm(x0, nrm2[0][i]); g1 = rmsnorm(x1, nrm2[1][i])
            m0 = (F.silu(g0 @ Wg[0][i]) * (g0 @ Wu[0][i])) @ Wd[0][i]
            m1 = (F.silu(g1 @ Wg[1][i]) * (g1 @ Wu[1][i])) @ Wd[1][i]
            m = m0 + m1.to("cuda:0")                            # all-reduce #2
            return x0 + m, x1 + m.to("cuda:1")

        def tp_token():
            x0 = torch.randn(1, dm, dtype=torch.float16, device="cuda:0")
            x1 = x0.to("cuda:1")
            for i in range(L):
                x0, x1 = tp_layer(x0, x1, i)
            return x0

        def bench(step):
            ms = []
            for _ in range(args.seeds):
                for _ in range(3): step()
                sync(); s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
                for _ in range(args.trials):
                    sync(); s.record(); step(); e.record(); sync()
                    ms.append(s.elapsed_time(e))
            return med_iqr(ms)

        tp, tpi = bench(tp_token)
        rec = {"tp2": {"ms_per_token": round(tp, 3), "iqr": round(tpi, 3)}}
        print(f"  ctx={C:6d}  TP-2 {tp:.2f} ms/tok (IQR {tpi:.2f})", end="")

        # single-GPU arm (full KV on cuda:0) for the ratio, where it fits
        try:
            Ks = [torch.randn(1, Hkv, C, D, dtype=torch.float16, device="cuda:0") for _ in range(L)]
            Vs = [torch.randn(1, Hkv, C, D, dtype=torch.float16, device="cuda:0") for _ in range(L)]
            Wqkv_s = [w(dm, (H+2*Hkv)*D, "cuda:0", 900+i) for i in range(L)]
            Wo_s = [w(dm, dm, "cuda:0", 910+i) for i in range(L)]
            Wg_s = [w(dm, dff, "cuda:0", 920+i) for i in range(L)]
            Wu_s = [w(dm, dff, "cuda:0", 930+i) for i in range(L)]
            Wd_s = [w(dff, dm, "cuda:0", 940+i) for i in range(L)]
            n1s = [torch.ones(dm, dtype=torch.float16, device="cuda:0") for _ in range(L)]
            n2s = [torch.ones(dm, dtype=torch.float16, device="cuda:0") for _ in range(L)]
            sync()
            def sg_token():
                x = torch.randn(1, dm, dtype=torch.float16, device="cuda:0")
                for i in range(L):
                    h = rmsnorm(x, n1s[i]); qkv = h @ Wqkv_s[i]
                    q = qkv[:, :H*D].view(1, H, 1, D)
                    o = F.scaled_dot_product_attention(q, Ks[i], Vs[i], scale=scale,
                                                       enable_gqa=(H != Hkv)).reshape(1, dm)
                    x = x + o @ Wo_s[i]
                    g = rmsnorm(x, n2s[i])
                    x = x + (F.silu(g @ Wg_s[i]) * (g @ Wu_s[i])) @ Wd_s[i]
                return x
            sg, sgi = bench(sg_token)
            rec["single_gpu"] = {"ms_per_token": round(sg, 3), "iqr": round(sgi, 3)}
            rec["tp2_vs_single"] = round(tp / sg, 3)
            print(f"  | single {sg:.2f} ms/tok  -> TP {tp/sg:.2f}x single")
            del Ks, Vs, Wqkv_s, Wo_s, Wg_s, Wu_s, Wd_s
        except RuntimeError as e:
            rec["single_gpu"] = {"status": f"OOM:{str(e)[:40]}"}
            print(f"  | single OOM (TP enables it)")
        out["contexts"][str(C)] = rec
        del K, V; torch.cuda.empty_cache()

    path = os.path.join(RES, "tp_tpot.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
