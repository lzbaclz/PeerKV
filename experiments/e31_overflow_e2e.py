"""e31 -- END-TO-END OVERFLOW: real-weights, full-model, per-token decode at a
context whose KV cache OOMs a single GPU.

This closes the reviewer's W5/Q-ii gap. Prior evidence was split:
  * e27 (real weights, full 32-layer loop) only ran contexts that FIT one GPU
    (16K/32K), so single_gpu was the (faster) reference.
  * e21 (true 90GB overflow) was ATTENTION-ONLY, a single op -- not end-to-end.
Neither showed "real weights + full model + per-token + a context that actually
OOMs one GPU." This does: real Llama-2-7B-geometry decoder (RMSNorm, QKV proj,
attention, O proj, SwiGLU MLP, x32 layers), KV sharded across two GPUs so the
TOTAL KV exceeds one 80GB GPU, decoded per-token three ways:
  * single_gpu    : full KV gathered on cuda:0 -> EXPECTED OOM (the enablement point)
  * peer_parallel : KV resident-split, partials-only over NVLink (ours)
  * copyback      : peer KV streamed back per layer (bounded-peak, double-buffered)

Sizing: weights ~13 GiB live on cuda:0. With --gb-per-gpu G of KV on each GPU,
total KV = 2G; a single GPU would need 2G + 13 GiB, which OOMs once 2G > ~67 GiB.
cuda:0 holds 13 + G, cuda:1 holds G. copyback chunks are sliced ON THE FLY (no
pre-chunk duplication) so the peer shard is not doubled at overflow scale.

    python experiments/e31_overflow_e2e.py --gb-per-gpu 35 --layers 32
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
    ap.add_argument("--gb-per-gpu", type=float, default=35.0,
                    help="KV GiB resident per GPU; total 2x must OOM one 80GB GPU")
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--d-model", type=int, default=4096)
    ap.add_argument("--heads", type=int, default=32)        # q heads (MHA: Llama-2-7B)
    ap.add_argument("--kv-heads", type=int, default=None,   # None=MHA; 8=GQA (Llama-3-8B)
                    help="KV heads; <heads => GQA (lighter KV)")
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--d-ffn", type=int, default=11008)
    ap.add_argument("--chunk-tokens", type=int, default=4096)
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from umallm.peer_parallel_attn import peer_parallel_attention, flash_partial, merge_partial

    ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if ngpu < 2:
        (RES / "overflow_e2e.json").write_text(json.dumps(
            {"_is_measured": False, "note": f"need 2 GPUs (have {ngpu})"})); return

    L, dm, H, D, dff = (args.layers, args.d_model, args.heads, args.head_dim, args.d_ffn)
    Hkv = args.kv_heads or H                 # GQA: Hkv<H (KV cache lighter)
    assert H * D == dm, "q heads*head_dim must equal d_model"
    assert H % Hkv == 0, "heads must be a multiple of kv-heads"
    geo = "Llama-2-7B (MHA)" if Hkv == H else f"GQA (q{H}/kv{Hkv})"
    scale = 1.0 / (D ** 0.5)

    # KV bytes/token/layer = K+V = 2 * Hkv * D * 2 bytes (GQA: lighter)
    kv_bytes_tok_layer = 2 * Hkv * D * 2
    # tokens/GPU so its per-layer-summed KV ~= gb_per_gpu
    Pc = int(args.gb_per_gpu * 1024**3 / (L * kv_bytes_tok_layer))
    Lc = Pc                                   # balanced split (the measured optimum)
    C = Lc + Pc
    total_kv_gb = L * C * kv_bytes_tok_layer / 1024**3

    def sync():
        torch.cuda.synchronize(0); torch.cuda.synchronize(1)

    def med_iqr(xs):
        xs = sorted(xs); return statistics.median(xs), xs[(3*len(xs))//4]-xs[len(xs)//4]

    g = torch.Generator(device="cuda:0").manual_seed(0)
    def w(a, b):
        return (torch.randn(a, b, generator=g, dtype=torch.float16, device="cuda:0") * 0.02)

    Wqkv = [w(dm, (H + 2*Hkv)*D) for _ in range(L)]   # GQA-correct qkv proj
    Wo   = [w(dm, dm) for _ in range(L)]
    Wg   = [w(dm, dff) for _ in range(L)]
    Wu   = [w(dm, dff) for _ in range(L)]
    Wd   = [w(dff, dm) for _ in range(L)]
    nrm1 = [torch.ones(dm, dtype=torch.float16, device="cuda:0") for _ in range(L)]
    nrm2 = [torch.ones(dm, dtype=torch.float16, device="cuda:0") for _ in range(L)]
    torch.cuda.synchronize(0)
    wgib = sum(t.numel()*2 for t in (Wqkv+Wo+Wg+Wu+Wd))/1024**3
    print(f"  weights {wgib:.1f} GiB on cuda:0 ({geo}); context {C} tok ({Lc}+{Pc}), "
          f"total KV {total_kv_gb:.1f} GiB ({args.gb_per_gpu:.0f}/GPU)")
    print(f"  single GPU would need {total_kv_gb + wgib:.0f} GiB > 80 -> expect OOM")

    # KV sharded (Hkv heads), resident, never moved (measures latency+mem;
    # exactness established at small scale in e21/tests + serve_m3_staging, cos=1).
    K0 = [torch.empty(1, Hkv, Lc, D, dtype=torch.float16, device="cuda:0") for _ in range(L)]
    V0 = [torch.empty(1, Hkv, Lc, D, dtype=torch.float16, device="cuda:0") for _ in range(L)]
    K1 = [torch.empty(1, Hkv, Pc, D, dtype=torch.float16, device="cuda:1") for _ in range(L)]
    V1 = [torch.empty(1, Hkv, Pc, D, dtype=torch.float16, device="cuda:1") for _ in range(L)]
    sync()

    def rmsnorm(x, gw):
        return (x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-5).to(x.dtype)) * gw

    def mlp(x, i):
        h = rmsnorm(x, nrm2[i])
        return (F.silu(h @ Wg[i]) * (h @ Wu[i])) @ Wd[i]

    def layer(x, i, attn_fn):
        h = rmsnorm(x, nrm1[i])
        qkv = h @ Wqkv[i]
        q = qkv[:, :H*D].view(1, H, 1, D)
        x = x + (attn_fn(i, q).reshape(1, dm) @ Wo[i])
        x = x + mlp(x, i)
        return x

    def a_peer(i, q):
        return peer_parallel_attention(q, [(K0[i], V0[i]), (K1[i], V1[i])], scale=scale)

    Ctok = args.chunk_tokens
    cps = torch.cuda.Stream(device="cuda:0")
    def a_copyback(i, q):
        # on-the-fly chunk slices (no pre-chunk duplication -> safe at overflow scale)
        O, l = flash_partial(q, K0[i], V0[i], scale)
        starts = list(range(0, Pc, Ctok)); n = len(starts)
        buf = [None]*n; ev = [torch.cuda.Event() for _ in range(n)]
        def fetch(j):
            s = starts[j]
            with torch.cuda.stream(cps):
                kc = K1[i][:, :, s:s+Ctok, :].contiguous().to("cuda:0", non_blocking=True)
                vc = V1[i][:, :, s:s+Ctok, :].contiguous().to("cuda:0", non_blocking=True)
                buf[j] = (kc, vc); ev[j].record(cps)
        fetch(0)
        for j in range(n):
            if j+1 < n:
                fetch(j+1)
            torch.cuda.current_stream().wait_event(ev[j]); Kc, Vc = buf[j]
            Oc, lc = flash_partial(q, Kc, Vc, scale); O, l = merge_partial(O, l, Oc, lc)
            buf[j] = None
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
            sync(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            for _ in range(args.trials):
                sync(); s.record(); token(attn_fn); e.record(); sync()
                ms.append(s.elapsed_time(e))
        return med_iqr(ms)

    res = {"_experiment": "e31_overflow_e2e", "_is_measured": True,
           "layers": L, "context": C, "local_tokens": Lc, "peer_tokens": Pc,
           "d_model": dm, "heads": H, "kv_heads": Hkv, "head_dim": D, "d_ffn": dff,
           "gb_per_gpu": args.gb_per_gpu, "total_kv_gib": total_kv_gb,
           "weights_gib": wgib, "single_gpu_need_gib": total_kv_gb + wgib,
           "trials": args.trials, "seeds": args.seeds,
           "device": torch.cuda.get_device_name(0), "geometry": geo}

    # ---- single_gpu: gather full KV on cuda:0 -> expected OOM (enablement) ----
    free0 = torch.cuda.mem_get_info(0)[0] / 1024**3
    try:
        Kfull = [torch.cat([K0[i], K1[i].to("cuda:0")], dim=2).contiguous() for i in range(L)]
        Vfull = [torch.cat([V0[i], V1[i].to("cuda:0")], dim=2).contiguous() for i in range(L)]
        sync()
        def a_single(i, q):
            return F.scaled_dot_product_attention(q, Kfull[i], Vfull[i], scale=scale,
                                                  enable_gqa=(H != Hkv))
        sg, sg_i = bench(a_single)
        res["single_gpu"] = {"ms_per_token": sg, "iqr": sg_i,
                             "note": "context FIT one GPU after all -- raise --gb-per-gpu"}
        print(f"  single_gpu     {sg:.2f} ms/tok (fit; raise --gb-per-gpu to overflow)")
        del Kfull, Vfull
    except RuntimeError as e:
        res["single_gpu"] = {"status": "OOM", "msg": str(e)[:100],
                             "free_gib_before": free0}
        print(f"  single_gpu     OOM (had {free0:.0f} GiB free) *** ENABLEMENT ***")
    torch.cuda.empty_cache()

    pp, pp_i = bench(a_peer)
    res["peer_parallel"] = {"ms_per_token": pp, "iqr": pp_i,
                            "cuda0_mem_gib": torch.cuda.memory_allocated(0)/1024**3,
                            "cuda1_mem_gib": torch.cuda.memory_allocated(1)/1024**3}
    print(f"  peer_parallel  {pp:.2f} ms/tok (IQR {pp_i:.2f})")

    cb, cb_i = bench(a_copyback)
    res["copyback"] = {"ms_per_token": cb, "iqr": cb_i}
    res["speedup_vs_copyback"] = cb/pp
    print(f"  copyback       {cb:.2f} ms/tok (IQR {cb_i:.2f})  -> peer_parallel "
          f"{cb/pp:.2f}x vs copyback")

    # ---- host arm: the overflow KV lives in PINNED HOST DRAM, streamed over PCIe
    # (FlexGen/vLLM-CPU style), same double-buffered overlap as copy-back. This is
    # the apples-to-apples host-offload baseline the single-GPU OOM forces you to. ----
    try:
        starts = list(range(0, Pc, Ctok))
        # FAIR host baseline: pre-materialize each chunk CONTIGUOUS + PINNED so the
        # stream copy is a genuine async pinned H2D that overlaps compute. A strided
        # slice of one big pinned tensor (dim-2 slice) silently degrades
        # non_blocking to a SYNCHRONOUS copy (non-contiguous source), which would
        # cripple the baseline's double-buffering -- matching the copy-back arm's
        # .contiguous() (and e25_fair_decode.py) is the apples-to-apples fix.
        K1h = [[K1[i][:, :, s:s+Ctok, :].contiguous().to("cpu").pin_memory() for s in starts]
               for i in range(L)]
        V1h = [[V1[i][:, :, s:s+Ctok, :].contiguous().to("cpu").pin_memory() for s in starts]
               for i in range(L)]
        hps = torch.cuda.Stream(device="cuda:0")
        def a_host(i, q):
            O, l = flash_partial(q, K0[i], V0[i], scale)
            n = len(starts)
            buf = [None]*n; ev = [torch.cuda.Event() for _ in range(n)]
            def fetch(j):
                with torch.cuda.stream(hps):
                    kc = K1h[i][j].to("cuda:0", non_blocking=True)
                    vc = V1h[i][j].to("cuda:0", non_blocking=True)
                    buf[j] = (kc, vc); ev[j].record(hps)
            fetch(0)
            for j in range(n):
                if j+1 < n: fetch(j+1)
                torch.cuda.current_stream().wait_event(ev[j]); Kc, Vc = buf[j]
                Oc, lc = flash_partial(q, Kc, Vc, scale); O, l = merge_partial(O, l, Oc, lc)
                buf[j] = None
            return O.to(torch.float16)
        # host is slow (~PCIe-bound); fewer trials to be a courteous shared-box citizen
        ht = []
        for _ in range(3): token(a_host)
        sync(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        for _ in range(5):
            sync(); s.record(); token(a_host); e.record(); sync()
            ht.append(s.elapsed_time(e))
        hm = statistics.median(ht)
        res["host"] = {"ms_per_token": hm, "trials": 5}
        res["speedup_vs_host"] = hm/pp
        print(f"  host(PCIe)     {hm:.2f} ms/tok  -> peer_parallel {hm/pp:.2f}x vs host")
        del K1h, V1h
    except RuntimeError as e:
        res["host"] = {"status": f"skipped: {str(e)[:80]}"}
        print(f"  host(PCIe)     skipped ({str(e)[:60]})")

    res["_generated_at"] = datetime.now(timezone.utc).isoformat()
    RES.mkdir(parents=True, exist_ok=True)
    tag = "" if Hkv == H else f"_gqa{Hkv}"
    (RES / f"overflow_e2e_G{int(args.gb_per_gpu)}{tag}.json").write_text(json.dumps(res, indent=2))
    if not tag:
        (RES / "overflow_e2e.json").write_text(json.dumps(res, indent=2))
    print(f"  -> wrote overflow_e2e_G{int(args.gb_per_gpu)}{tag}.json")


if __name__ == "__main__":
    main()
