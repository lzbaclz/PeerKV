"""e29 -- solve the per-layer cross-GPU round-trip: true home||peer overlap.

e28 ruled out CUDA graphs (cross-device uncapturable). The real lever: in the
current peer_parallel step the peer chain (q->peer, peer flash, partial->home) and
the home local flash are issued back-to-back on default streams and may NOT overlap,
so the round-trip is exposed. If we put the peer chain on its OWN cuda:1 stream and
the home flash on cuda:0 concurrently, syncing via a CUDA event only at the merge,
the two half-KV flashes run in parallel and the step approaches max(home, peer)
instead of home+peer+roundtrip -- potentially FASTER than a single full-KV flash.

Compares, over L layers (attention-only, MHA Llama geometry):
  * single_gpu      : full KV on cuda:0, one fused flash/layer
  * pp_baseline     : current peer_parallel_attention (default streams)
  * pp_overlapped   : peer chain on a dedicated cuda:1 stream + event sync (this file)

    python experiments/e29_overlap.py --layers 32 --context 16384 --heads 32
"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent / "results" / "overlap.json"


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
    from umallm.peer_parallel_attn import flash_partial, merge_partial, peer_parallel_attention

    if (torch.cuda.device_count() if torch.cuda.is_available() else 0) < 2:
        OUT.write_text(json.dumps({"_is_measured": False, "note": "need 2 GPUs"})); return
    L, C, H, D = args.layers, args.context, args.heads, args.head_dim
    Lc = C // 2; Pc = C - Lc; scale = 1.0 / (D ** 0.5)

    def sync(): torch.cuda.synchronize(0); torch.cuda.synchronize(1)
    def med(xs): xs=sorted(xs); return statistics.median(xs), xs[(3*len(xs))//4]-xs[len(xs)//4]

    qs=[torch.randn(1,H,1,D,dtype=torch.float16,device="cuda:0") for _ in range(L)]
    K0=[torch.randn(1,H,Lc,D,dtype=torch.float16,device="cuda:0") for _ in range(L)]
    V0=[torch.randn(1,H,Lc,D,dtype=torch.float16,device="cuda:0") for _ in range(L)]
    K1=[torch.randn(1,H,Pc,D,dtype=torch.float16,device="cuda:1") for _ in range(L)]
    V1=[torch.randn(1,H,Pc,D,dtype=torch.float16,device="cuda:1") for _ in range(L)]
    sync()

    peer_stream = torch.cuda.Stream(device="cuda:1")

    def pp_overlapped(i):
        # issue the WHOLE peer chain on peer_stream FIRST (async), then the home
        # flash on cuda:0 default stream concurrently; event-sync only at merge.
        with torch.cuda.stream(peer_stream):
            q1 = qs[i].to("cuda:1", non_blocking=True)
            O1, l1 = flash_partial(q1, K1[i], V1[i], scale)
            O1h = O1.to("cuda:0", non_blocking=True)
            l1h = l1.to("cuda:0", non_blocking=True)
            ev = torch.cuda.Event(); ev.record(peer_stream)
        O0, l0 = flash_partial(qs[i], K0[i], V0[i], scale)   # cuda:0, concurrent
        torch.cuda.current_stream("cuda:0").wait_event(ev)
        return merge_partial(O0, l0, O1h, l1h)[0]

    Kf=[torch.cat([K0[i],K1[i].to('cuda:0')],dim=2).contiguous() for i in range(L)]
    Vf=[torch.cat([V0[i],V1[i].to('cuda:0')],dim=2).contiguous() for i in range(L)]
    sync()

    def loop(fn):
        for i in range(L): fn(i)
    def bench(fn, warmup=5):
        for _ in range(warmup): loop(fn)
        sync(); s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True);ts=[]
        for _ in range(args.trials):
            sync(); s.record(); loop(fn); e.record(); sync(); ts.append(s.elapsed_time(e))
        return med(ts)

    sg=lambda i: F.scaled_dot_product_attention(qs[i],Kf[i],Vf[i],scale=scale)
    base=lambda i: peer_parallel_attention(qs[i],[(K0[i],V0[i]),(K1[i],V1[i])],scale=scale)

    sg_ms,_=bench(sg);          print(f"  single_gpu        {sg_ms:.3f} ms ({sg_ms/L*1000:.1f} us/layer)")
    b_ms,_=bench(base);         print(f"  pp_baseline       {b_ms:.3f} ms ({b_ms/L*1000:.1f} us/layer)  {sg_ms/b_ms:.2f}x vs single")
    o_ms,_=bench(pp_overlapped);print(f"  pp_overlapped     {o_ms:.3f} ms ({o_ms/L*1000:.1f} us/layer)  {sg_ms/o_ms:.2f}x vs single, {b_ms/o_ms:.2f}x vs baseline")

    # exactness of overlapped path vs dense
    ref=F.scaled_dot_product_attention(qs[0],Kf[0],Vf[0],scale=scale)
    out=pp_overlapped(0); sync()
    cos=float(F.cosine_similarity(ref.reshape(-1).float(),out.reshape(-1).float(),dim=0))
    print(f"  overlapped exactness cos={cos:.6f}")

    res={"_experiment":"e29_overlap","_is_measured":True,"layers":L,"context":C,
         "heads":H,"head_dim":D,"trials":args.trials,"device":torch.cuda.get_device_name(0),
         "single_gpu_ms":sg_ms,"pp_baseline_ms":b_ms,"pp_overlapped_ms":o_ms,
         "overlapped_vs_single":sg_ms/o_ms,"overlapped_vs_baseline":b_ms/o_ms,
         "baseline_vs_single":sg_ms/b_ms,"overlapped_exactness_cos":cos,
         "_generated_at":datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(res,indent=2)); print(f"  -> wrote {OUT}")


if __name__ == "__main__":
    main()
