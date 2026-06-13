"""e27 -- REAL-WEIGHTS end-to-end decode: does the MLP amortize the per-layer
cross-GPU round-trip?

e26 measured the attention path only and found peer-parallel 0.32-0.61x vs single
GPU at full model (the per-layer cross-GPU round-trip dominates a bare attention op).
But a real decoder layer is attention PLUS a large MLP + QKV/O projections, all pure
compute on the compute GPU and identical for every KV-placement. Decode is
weight-bandwidth-bound: reading ~13 GB of weights/token dwarfs the attention. So the
honest question is whether peer-parallel's fixed ~120 us/layer round-trip is
amortized once the real per-layer compute is present.

We run a faithful Llama-2-7B-geometry decoder (MHA: L=32, d=4096, 32 heads,
head_dim=128, d_ffn=11008, fp16 random weights on cuda:0 -- ~13 GB) and time
ms/token for the full 32-layer step (RMSNorm, QKV proj, attention, O proj, RMSNorm,
SwiGLU MLP), with attention done three ways:
  * single_gpu     : full KV on cuda:0, fused flash SDPA (ref; OOMs at large ctx)
  * peer_parallel  : KV sharded local/peer, partials-only over NVLink (ours)
  * copyback       : KV on peer, fair pre-contiguous + double-buffered overlap copy-back

    python experiments/e27_real_weights.py --layers 32 --context 16384
"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--context", type=int, default=16384, help="KV tokens/layer")
    ap.add_argument("--d-model", type=int, default=4096)
    ap.add_argument("--heads", type=int, default=32)        # q heads (MHA: Llama-2-7B)
    ap.add_argument("--kv-heads", type=int, default=None,   # None=MHA; 8=GQA (Llama-3-8B)
                    help="KV heads; <heads => GQA (lighter KV, expanded in-kernel)")
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--d-ffn", type=int, default=11008)
    ap.add_argument("--local-frac", type=float, default=0.5)
    ap.add_argument("--chunk-tokens", type=int, default=4096)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from umallm.peer_parallel_attn import peer_parallel_attention, flash_partial, merge_partial

    ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if ngpu < 2:
        (RES / "real_weights.json").write_text(json.dumps({"_is_measured": False,
            "note": f"need 2 GPUs (have {ngpu})"})); return

    L, C, dm, H, D, dff = (args.layers, args.context, args.d_model, args.heads,
                           args.head_dim, args.d_ffn)
    Hkv = args.kv_heads or H                 # GQA: Hkv<H (KV cache lighter)
    assert H * D == dm, "q heads*head_dim must equal d_model"
    assert H % Hkv == 0, "heads must be a multiple of kv-heads"
    geo = "Llama-2-7B (MHA)" if Hkv == H else f"GQA (q{H}/kv{Hkv})"
    Lc = max(1, min(C-1, round(args.local_frac * C))); Pc = C - Lc
    scale = 1.0 / (D ** 0.5)
    dev = "cuda:0"

    def sync():
        torch.cuda.synchronize(0); torch.cuda.synchronize(1)

    def med_iqr(xs):
        xs = sorted(xs); return statistics.median(xs), xs[(3*len(xs))//4]-xs[len(xs)//4]

    g = torch.Generator(device="cuda:0").manual_seed(0)
    def w(a, b):  # fp16 weight on cuda:0, scaled small for numeric sanity
        return (torch.randn(a, b, generator=g, dtype=torch.float16, device="cuda:0") * 0.02)

    # per-layer weights on cuda:0. qkv proj = (q + 2*kv) heads (GQA-correct).
    qkv_out = (H + 2*Hkv) * D
    Wqkv = [w(dm, qkv_out) for _ in range(L)]
    Wo   = [w(dm, dm) for _ in range(L)]
    Wg   = [w(dm, dff) for _ in range(L)]
    Wu   = [w(dm, dff) for _ in range(L)]
    Wd   = [w(dff, dm) for _ in range(L)]
    nrm1 = [torch.ones(dm, dtype=torch.float16, device="cuda:0") for _ in range(L)]
    nrm2 = [torch.ones(dm, dtype=torch.float16, device="cuda:0") for _ in range(L)]
    torch.cuda.synchronize(0)
    wgib = sum(t.numel()*2 for t in (Wqkv+Wo+Wg+Wu+Wd))/1024**3
    print(f"  weights: {wgib:.1f} GiB on cuda:0 ({L} layers, {geo})")

    # KV cache, sharded per layer (Hkv heads). local on cuda:0, peer on cuda:1.
    K0 = [torch.randn(1, Hkv, Lc, D, dtype=torch.float16, device="cuda:0") for _ in range(L)]
    V0 = [torch.randn(1, Hkv, Lc, D, dtype=torch.float16, device="cuda:0") for _ in range(L)]
    K1 = [torch.randn(1, Hkv, Pc, D, dtype=torch.float16, device="cuda:1") for _ in range(L)]
    V1 = [torch.randn(1, Hkv, Pc, D, dtype=torch.float16, device="cuda:1") for _ in range(L)]
    sync()

    def rmsnorm(x, gw):
        return (x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-5).to(x.dtype)) * gw

    def mlp(x, i):
        h = rmsnorm(x, nrm2[i])
        return (F.silu(h @ Wg[i]) * (h @ Wu[i])) @ Wd[i]

    def layer(x, i, attn_fn):
        h = rmsnorm(x, nrm1[i])
        qkv = h @ Wqkv[i]                         # [1, (H+2*Hkv)*D]
        q = qkv[:, :H*D].view(1, H, 1, D)         # decode: 1 query token (H q-heads)
        x = x + (attn_fn(i, q).reshape(1, dm) @ Wo[i])
        x = x + mlp(x, i)
        return x

    # ---- attention variants ----
    # single_gpu: the FULL KV is RESIDENT on cuda:0 (a real single-GPU decoder holds
    # all KV locally -- no per-step cross-device copy). Built once, not per token.
    Kfull = [torch.cat([K0[i], K1[i].to("cuda:0")], dim=2).contiguous() for i in range(L)]
    Vfull = [torch.cat([V0[i], V1[i].to("cuda:0")], dim=2).contiguous() for i in range(L)]
    sync()
    def a_single(i, q):
        return F.scaled_dot_product_attention(q, Kfull[i], Vfull[i], scale=scale,
                                              enable_gqa=(H != Hkv))

    def a_peer(i, q):
        return peer_parallel_attention(q, [(K0[i], V0[i]), (K1[i], V1[i])], scale=scale)

    Ctok = args.chunk_tokens
    K1c = [[K1[i][:, :, s:s+Ctok, :].contiguous() for s in range(0, Pc, Ctok)] for i in range(L)]
    V1c = [[V1[i][:, :, s:s+Ctok, :].contiguous() for s in range(0, Pc, Ctok)] for i in range(L)]
    cps = torch.cuda.Stream(device="cuda:0")
    def a_copyback(i, q):
        O, l = flash_partial(q, K0[i], V0[i], scale)
        ks, vs = K1c[i], V1c[i]; n = len(ks); buf=[None]*n; ev=[torch.cuda.Event() for _ in range(n)]
        with torch.cuda.stream(cps):
            buf[0]=(ks[0].to("cuda:0",non_blocking=True),vs[0].to("cuda:0",non_blocking=True)); ev[0].record(cps)
        for j in range(n):
            if j+1<n:
                with torch.cuda.stream(cps):
                    buf[j+1]=(ks[j+1].to("cuda:0",non_blocking=True),vs[j+1].to("cuda:0",non_blocking=True)); ev[j+1].record(cps)
            torch.cuda.current_stream().wait_event(ev[j]); Kc,Vc=buf[j]
            Oc,lc=flash_partial(q,Kc,Vc,scale); O,l=merge_partial(O,l,Oc,lc); buf[j]=None
        return O.to(torch.float16)

    def token(attn_fn):
        x = torch.randn(1, dm, dtype=torch.float16, device="cuda:0")
        for i in range(L):
            x = layer(x, i, attn_fn)
        return x

    def bench(attn_fn):
        ms = []
        for _ in range(args.seeds):
            for _ in range(3): token(attn_fn)
            sync(); s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
            for _ in range(args.trials):
                sync(); s.record(); token(attn_fn); e.record(); sync()
                ms.append(s.elapsed_time(e))
        return med_iqr(ms)

    res = {"_experiment": "e27_real_weights", "_is_measured": True,
           "layers": L, "context": C, "d_model": dm, "heads": H, "kv_heads": Hkv,
           "head_dim": D, "d_ffn": dff, "local_tokens": Lc, "peer_tokens": Pc,
           "weights_gib": wgib, "trials": args.trials, "seeds": args.seeds,
           "device": torch.cuda.get_device_name(0), "geometry": geo}

    pp, pp_i = bench(a_peer)
    res["peer_parallel"] = {"ms_per_token": pp, "iqr": pp_i}
    print(f"  peer_parallel  {pp:.2f} ms/tok (IQR {pp_i:.2f})")
    try:
        sg, sg_i = bench(a_single)
        res["single_gpu"] = {"ms_per_token": sg, "iqr": sg_i}
        res["peer_parallel_vs_single"] = pp/sg          # >1 means slower than single
        res["slowdown_pct_vs_single"] = (pp/sg - 1)*100
        print(f"  single_gpu     {sg:.2f} ms/tok (IQR {sg_i:.2f})  -> peer_parallel "
              f"{pp/sg:.2f}x single ({(pp/sg-1)*100:+.0f}%)")
    except RuntimeError as e:
        res["single_gpu"] = {"status": f"OOM: {str(e)[:60]}"}; print("  single_gpu OOM")
    torch.cuda.empty_cache()
    cb, cb_i = bench(a_copyback)
    res["copyback"] = {"ms_per_token": cb, "iqr": cb_i}
    res["speedup_vs_copyback"] = cb/pp
    print(f"  copyback       {cb:.2f} ms/tok (IQR {cb_i:.2f})  -> peer_parallel {cb/pp:.2f}x vs copyback")

    res["_generated_at"] = datetime.now(timezone.utc).isoformat()
    RES.mkdir(parents=True, exist_ok=True)
    tag = "" if Hkv == H else f"_gqa{Hkv}"
    (RES / f"real_weights_L{L}_C{C}{tag}.json").write_text(json.dumps(res, indent=2))
    if not tag:
        (RES / "real_weights.json").write_text(json.dumps(res, indent=2))
    print(f"  -> wrote real_weights_L{L}_C{C}{tag}.json")


if __name__ == "__main__":
    main()
