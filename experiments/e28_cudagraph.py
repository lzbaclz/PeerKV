"""e28 -- can CUDA graphs cut the per-layer cross-GPU round-trip?

e27 shows peer-parallel is ~13-18% slower than single-GPU at full model; the gap is
the per-layer round-trip, most of which is kernel-launch + Python overhead (the
genuine cross-device data is ~4 KB/layer: q out + partial back). CUDA-graph capture
replays the whole L-layer attention step as one graph, removing per-op launch
overhead; the floor is the real cross-device copy latency.

We measure the L-layer peer-parallel ATTENTION loop (isolating the round-trip, no
weights) three ways:
  * eager        : current Python loop of peer_parallel_attention
  * cudagraph    : the same loop captured + replayed as one CUDA graph (if capturable)
  * floor        : bare per-layer cross-device round-trip (q.to(peer) + 1-block flash
                   + partial.to(home) + merge) -- the irreducible data-movement cost
and the single-GPU loop eager vs cudagraph (graphs help it too).

    python experiments/e28_cudagraph.py --layers 32 --context 16384
"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent / "results" / "cudagraph.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--context", type=int, default=16384)
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--trials", type=int, default=30)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from umallm.peer_parallel_attn import flash_partial, merge_partial

    ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if ngpu < 2:
        OUT.write_text(json.dumps({"_is_measured": False, "note": f"need 2 GPUs (have {ngpu})"})); return
    L, C, H, D = args.layers, args.context, args.heads, args.head_dim
    Lc = C // 2; Pc = C - Lc
    scale = 1.0 / (D ** 0.5)

    def sync(): torch.cuda.synchronize(0); torch.cuda.synchronize(1)
    def med(xs): xs=sorted(xs); return statistics.median(xs), xs[(3*len(xs))//4]-xs[len(xs)//4]

    qs = [torch.randn(1, H, 1, D, dtype=torch.float16, device="cuda:0") for _ in range(L)]
    K0=[torch.randn(1,H,Lc,D,dtype=torch.float16,device="cuda:0") for _ in range(L)]
    V0=[torch.randn(1,H,Lc,D,dtype=torch.float16,device="cuda:0") for _ in range(L)]
    K1=[torch.randn(1,H,Pc,D,dtype=torch.float16,device="cuda:1") for _ in range(L)]
    V1=[torch.randn(1,H,Pc,D,dtype=torch.float16,device="cuda:1") for _ in range(L)]
    sync()

    def pp_step():
        outs=[]
        for i in range(L):
            q1=qs[i].to("cuda:1",non_blocking=True)
            O1,l1=flash_partial(q1,K1[i],V1[i],scale)
            O0,l0=flash_partial(qs[i],K0[i],V0[i],scale)
            o,_=merge_partial(O0,l0,O1.to("cuda:0",non_blocking=True),l1.to("cuda:0",non_blocking=True))
            outs.append(o)
        return outs

    def bench(fn, warmup=5):
        for _ in range(warmup): fn()
        sync(); s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
        ts=[]
        for _ in range(args.trials):
            sync(); s.record(); fn(); e.record(); sync(); ts.append(s.elapsed_time(e))
        return med(ts)

    res={"_experiment":"e28_cudagraph","_is_measured":True,"layers":L,"context":C,
         "heads":H,"head_dim":D,"trials":args.trials,"device":torch.cuda.get_device_name(0)}

    eager_ms, eager_iqr = bench(pp_step)
    res["peer_parallel_eager"]={"ms":eager_ms,"iqr":eager_iqr,"us_per_layer":eager_ms/L*1000}
    print(f"  peer_parallel eager     {eager_ms:.3f} ms  ({eager_ms/L*1000:.1f} us/layer)")

    # ---- attempt CUDA graph capture of the peer-parallel loop ----
    try:
        sidestream=torch.cuda.Stream(device="cuda:0")
        sidestream.wait_stream(torch.cuda.current_stream("cuda:0"))
        with torch.cuda.stream(sidestream):
            for _ in range(3): pp_step()
        torch.cuda.current_stream("cuda:0").wait_stream(sidestream); sync()
        gph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(gph):
            static_out=pp_step()
        sync()
        def replay(): gph.replay()
        g_ms,g_iqr=bench(replay, warmup=5)
        res["peer_parallel_cudagraph"]={"ms":g_ms,"iqr":g_iqr,"us_per_layer":g_ms/L*1000,
                                        "speedup_vs_eager":eager_ms/g_ms}
        print(f"  peer_parallel cudagraph {g_ms:.3f} ms  ({g_ms/L*1000:.1f} us/layer)  "
              f"{eager_ms/g_ms:.2f}x vs eager")
    except Exception as ex:
        res["peer_parallel_cudagraph"]={"status":f"capture failed: {str(ex)[:140]}"}
        print(f"  peer_parallel cudagraph CAPTURE FAILED: {str(ex)[:140]}")

    # ---- single-GPU loop eager vs cudagraph (graphs help it too) ----
    Kf=[torch.cat([K0[i],K1[i].to('cuda:0')],dim=2).contiguous() for i in range(L)]
    Vf=[torch.cat([V0[i],V1[i].to('cuda:0')],dim=2).contiguous() for i in range(L)]
    sync()
    def sg_step():
        return [F.scaled_dot_product_attention(qs[i],Kf[i],Vf[i],scale=scale) for i in range(L)]
    sg_ms,sg_iqr=bench(sg_step)
    res["single_gpu_eager"]={"ms":sg_ms,"iqr":sg_iqr}
    print(f"  single_gpu eager        {sg_ms:.3f} ms")
    try:
        ss=torch.cuda.Stream(device="cuda:0"); ss.wait_stream(torch.cuda.current_stream("cuda:0"))
        with torch.cuda.stream(ss):
            for _ in range(3): sg_step()
        torch.cuda.current_stream("cuda:0").wait_stream(ss); sync()
        g2=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g2): sg_static=sg_step()
        sync()
        sgg_ms,_=bench(lambda: g2.replay())
        res["single_gpu_cudagraph"]={"ms":sgg_ms,"speedup_vs_eager":sg_ms/sgg_ms}
        print(f"  single_gpu cudagraph    {sgg_ms:.3f} ms  {sg_ms/sgg_ms:.2f}x vs eager")
    except Exception as ex:
        res["single_gpu_cudagraph"]={"status":f"failed: {str(ex)[:100]}"}
        print(f"  single_gpu cudagraph failed: {str(ex)[:100]}")

    # ---- irreducible floor: bare per-layer cross-device round-trip (1 block) ----
    kb=[torch.randn(1,H,256,D,dtype=torch.float16,device="cuda:1") for _ in range(L)]
    vb=[torch.randn(1,H,256,D,dtype=torch.float16,device="cuda:1") for _ in range(L)]
    sync()
    def floor_step():
        for i in range(L):
            q1=qs[i].to("cuda:1",non_blocking=True)
            O1,l1=flash_partial(q1,kb[i],vb[i],scale)
            O0,l0=flash_partial(qs[i],K0[i][:,:,:256,:],V0[i][:,:,:256,:],scale)
            merge_partial(O0,l0,O1.to("cuda:0",non_blocking=True),l1.to("cuda:0",non_blocking=True))
    fl_ms,_=bench(floor_step)
    res["roundtrip_floor"]={"ms":fl_ms,"us_per_layer":fl_ms/L*1000,
                            "note":"per-layer cross-device round-trip with minimal (1-block) KV"}
    print(f"  roundtrip floor         {fl_ms:.3f} ms  ({fl_ms/L*1000:.1f} us/layer)")

    res["_generated_at"]=datetime.now(timezone.utc).isoformat()
    OUT.write_text(json.dumps(res,indent=2)); print(f"  -> wrote {OUT}")


if __name__ == "__main__":
    main()
